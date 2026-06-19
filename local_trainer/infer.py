from __future__ import annotations

import argparse
import json
import sys
import time
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
    answer, _ = _timed_answer(tokenizer, model, device, prompt, max_new_tokens, gen)
    return answer


def _timed_answer(
    tokenizer,
    model,
    device,
    prompt: str,
    max_new_tokens: int,
    gen: GenParams,
) -> tuple[str, dict[str, float | int]]:
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    started_at = time.perf_counter()
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
    generate_seconds = time.perf_counter() - started_at
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), {
        "generate_seconds": round(generate_seconds, 2),
        "output_tokens": int(generated.numel()),
    }


def chat(
    base_path: str,
    adapter_dir: str | None,
    prompt: str,
    max_new_tokens: int = 120,
    gen: GenParams | None = None,
) -> dict[str, object]:
    """Answer a single prompt with base (+optional adapter)."""
    gen = gen or GenParams()
    started_at = time.perf_counter()
    tokenizer, model, device = _load(base_path, adapter_dir)
    load_seconds = time.perf_counter() - started_at
    answer, generation = _timed_answer(tokenizer, model, device, prompt, max_new_tokens, gen)
    total_seconds = time.perf_counter() - started_at
    return {
        "answer": answer,
        "metrics": {
            "load_seconds": round(load_seconds, 2),
            "generate_seconds": generation["generate_seconds"],
            "output_tokens": generation["output_tokens"],
            "total_seconds": round(total_seconds, 2),
        },
    }


def compare(
    base_path: str,
    adapter_dir: str,
    prompt: str,
    max_new_tokens: int = 120,
    gen: GenParams | None = None,
) -> dict[str, object]:
    """Answer `prompt` with the base model and with the LoRA-adapted model."""
    gen = gen or GenParams()
    started_at = time.perf_counter()
    tokenizer, base_model, device = _load(base_path, None)
    load_seconds = time.perf_counter() - started_at
    before, before_generation = _timed_answer(tokenizer, base_model, device, prompt, max_new_tokens, gen)

    adapter_started_at = time.perf_counter()
    adapted = _attach_adapter(base_model, adapter_dir).to(device)
    adapted.eval()
    adapter_load_seconds = time.perf_counter() - adapter_started_at
    after, after_generation = _timed_answer(tokenizer, adapted, device, prompt, max_new_tokens, gen)
    total_seconds = time.perf_counter() - started_at
    return {
        "before": before,
        "after": after,
        "metrics": {
            "load_seconds": round(load_seconds, 2),
            "before_generate_seconds": before_generation["generate_seconds"],
            "before_output_tokens": before_generation["output_tokens"],
            "adapter_load_seconds": round(adapter_load_seconds, 2),
            "after_generate_seconds": after_generation["generate_seconds"],
            "after_output_tokens": after_generation["output_tokens"],
            "total_seconds": round(total_seconds, 2),
        },
    }


def session(
    base_path: str,
    adapter_dir: str | None,
    gen: GenParams | None = None,
) -> None:
    """常驻推理模式：加载模型后循环从 stdin 读取 JSON-line 请求，回复到 stdout。

    协议：
      请求: {"messages": [{"role": "user", "content": "..."}], "max_new_tokens": 120, "gen_params": {...}}
      响应: {"role": "assistant", "content": "...", "metrics": {...}}
      退出: {"action": "quit"}

    模型加载完成后输出 {"status": "ready"} 表示可以开始对话。
    """
    gen = gen or GenParams()
    import torch  # noqa: F401
    from transformers import AutoTokenizer

    device = detect_device()
    dtype = torch_dtype(device)

    # 输出加载阶段提示
    print(json.dumps({"status": "loading", "stage": "tokenizer"}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    print(json.dumps({"status": "loading", "stage": "model"}), flush=True)
    model = _load_base_model(base_path, dtype).to(device)

    if adapter_dir:
        print(json.dumps({"status": "loading", "stage": "adapter"}), flush=True)
        model = _attach_adapter(model, adapter_dir).to(device)
    model.eval()

    print(json.dumps({"status": "ready"}), flush=True)

    # 循环处理请求
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"error": "无法解析请求"}), flush=True)
            continue

        if req.get("action") == "quit":
            break

        messages = req.get("messages", [])
        if not messages:
            print(json.dumps({"error": "messages 不能为空"}), flush=True)
            continue

        max_new_tokens = req.get("max_new_tokens", 120)
        # 允许运行时覆盖 gen_params
        req_gen = req.get("gen_params", {})
        current_gen = GenParams(
            temperature=req_gen.get("temperature", gen.temperature),
            top_p=req_gen.get("top_p", gen.top_p),
            repetition_penalty=req_gen.get("repetition_penalty", gen.repetition_penalty),
            no_repeat_ngram_size=req_gen.get("no_repeat_ngram_size", gen.no_repeat_ngram_size),
        )

        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            started_at = time.perf_counter()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=current_gen.temperature,
                    top_p=current_gen.top_p,
                    repetition_penalty=current_gen.repetition_penalty,
                    no_repeat_ngram_size=current_gen.no_repeat_ngram_size,
                )
            generate_seconds = time.perf_counter() - started_at
            generated = output[0][inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
            print(json.dumps({
                "role": "assistant",
                "content": answer,
                "metrics": {
                    "generate_seconds": round(generate_seconds, 2),
                    "output_tokens": int(generated.numel()),
                },
            }, ensure_ascii=False), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Local inference worker")
    parser.add_argument("--mode", choices=["chat", "compare", "session"], default="compare")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--prompt", default=None)
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

    if args.mode == "session":
        session(args.base, args.adapter, gen)
    elif args.mode == "chat":
        if not args.prompt:
            parser.error("--prompt is required for chat mode")
        result = chat(args.base, args.adapter, args.prompt, args.max_new_tokens, gen)
        print(json.dumps(result, ensure_ascii=False))
    else:
        if not args.adapter:
            parser.error("--adapter is required for compare mode")
        if not args.prompt:
            parser.error("--prompt is required for compare mode")
        result = compare(args.base, args.adapter, args.prompt, args.max_new_tokens, gen)
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
