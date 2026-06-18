from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from .hardware import detect_device, torch_dtype


@dataclass
class GenParams:
    """解码参数。控制“怎么把话说出来”，与模型权重无关。

    默认值对应“平衡”风格档位，适合大多数日常对话。
    """

    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 3


def _load_base_model(base_path: str, dtype):
    """加载基础模型。

    部分模型（如 Qwen3.5 系列）是多模态的 *ForConditionalGeneration 架构，
    文本主干被包在 `language_model` 子模块里。LLaMA-Factory 训练时按这种完整
    结构存 adapter（key 带 `language_model` 前缀），所以推理也必须用同一个
    多模态入口加载，否则 adapter 的 key 对不上、被 peft 静默丢弃。
    优先用 AutoModelForImageTextToText，不支持时回退到 AutoModelForCausalLM。
    """
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    try:
        return AutoModelForImageTextToText.from_pretrained(base_path, dtype=dtype)
    except (ValueError, KeyError):
        return AutoModelForCausalLM.from_pretrained(base_path, dtype=dtype)


def _attach_adapter(model, adapter_dir: str):
    """挂载 LoRA adapter，并校验权重是否全部加载成功。

    peft 在 key 对不上时只发 UserWarning 然后静默跳过，会让微调看起来“没生效”。
    这里把缺失 key 升级成硬错误，避免再次出现“训练正常但测评无差异”的假象。
    """
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter_dir)
    missing = _missing_adapter_keys(model, adapter_dir)
    if missing:
        raise RuntimeError(
            "LoRA adapter 与基础模型结构不匹配，有 "
            f"{len(missing)} 个权重未能加载（如 {missing[0]}）。"
            "通常是推理与训练用了不同的模型加载入口导致 key 路径错位。"
        )
    return model


def _missing_adapter_keys(model, adapter_dir: str) -> list[str]:
    """返回 adapter 文件中存在、但没能挂到模型上的 LoRA 权重 key。"""
    from pathlib import Path

    from safetensors import safe_open

    adapter_file = Path(adapter_dir) / "adapter_model.safetensors"
    if not adapter_file.exists():
        return []
    with safe_open(str(adapter_file), framework="pt") as f:
        file_keys = set(f.keys())
    model_keys = set(model.state_dict().keys())
    # adapter 文件里的 key 不带 `.default`，模型里带，统一去掉再比对
    normalized_model_keys = {k.replace(".default", "") for k in model_keys}
    return sorted(k for k in file_keys if k.replace(".default", "") not in normalized_model_keys)


def _load(base_path: str, adapter_dir: str | None):
    import torch  # noqa: F401  (ensures torch import errors surface early)
    from transformers import AutoTokenizer

    device = detect_device()
    dtype = torch_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = _load_base_model(base_path, dtype).to(device)

    if adapter_dir:
        model = _attach_adapter(model, adapter_dir).to(device)
    model.eval()
    return tokenizer, model, device


def _answer(
    tokenizer,
    model,
    device,
    prompt: str,
    max_new_tokens: int,
    gen: GenParams,
) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        # 小模型贪心解码极易复读，加重复惩罚 + n-gram 去重抑制循环输出
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=gen.temperature,
            top_p=gen.top_p,
            repetition_penalty=gen.repetition_penalty,
            no_repeat_ngram_size=gen.no_repeat_ngram_size,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def chat(
    base_path: str,
    adapter_dir: str | None,
    prompt: str,
    max_new_tokens: int = 120,
    gen: GenParams | None = None,
) -> dict[str, str]:
    """Answer a single prompt with base (+optional adapter)."""
    gen = gen or GenParams()
    tokenizer, model, device = _load(base_path, adapter_dir)
    return {"answer": _answer(tokenizer, model, device, prompt, max_new_tokens, gen)}


def compare(
    base_path: str,
    adapter_dir: str,
    prompt: str,
    max_new_tokens: int = 120,
    gen: GenParams | None = None,
) -> dict[str, str]:
    """Answer `prompt` with the base model and with the LoRA-adapted model."""
    from peft import PeftModel

    gen = gen or GenParams()
    tokenizer, base_model, device = _load(base_path, None)
    before = _answer(tokenizer, base_model, device, prompt, max_new_tokens, gen)

    adapted = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
    adapted.eval()
    after = _answer(tokenizer, adapted, device, prompt, max_new_tokens, gen)
    return {"before": before, "after": after}


def _main() -> int:
    parser = argparse.ArgumentParser(description="Local inference worker")
    parser.add_argument("--mode", choices=["chat", "compare"], default="compare")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=GenParams.temperature)
    parser.add_argument("--top-p", type=float, default=GenParams.top_p)
    parser.add_argument("--repetition-penalty", type=float, default=GenParams.repetition_penalty)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=GenParams.no_repeat_ngram_size)
    args = parser.parse_args()

    gen = GenParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    if args.mode == "chat":
        result = chat(args.base, args.adapter, args.prompt, args.max_new_tokens, gen)
    else:
        if not args.adapter:
            parser.error("--adapter is required for compare mode")
        result = compare(args.base, args.adapter, args.prompt, args.max_new_tokens, gen)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
