from __future__ import annotations

import argparse
import json
import sys

from .hardware import detect_device, torch_dtype


def _load(base_path: str, adapter_dir: str | None):
    import torch  # noqa: F401  (ensures torch import errors surface early)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = detect_device()
    dtype = torch_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=dtype).to(device)

    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir).to(device)
    model.eval()
    return tokenizer, model, device


def _answer(tokenizer, model, device, prompt: str, max_new_tokens: int) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def chat(base_path: str, adapter_dir: str | None, prompt: str, max_new_tokens: int = 120) -> dict[str, str]:
    """Answer a single prompt with base (+optional adapter)."""
    tokenizer, model, device = _load(base_path, adapter_dir)
    return {"answer": _answer(tokenizer, model, device, prompt, max_new_tokens)}


def compare(base_path: str, adapter_dir: str, prompt: str, max_new_tokens: int = 120) -> dict[str, str]:
    """Answer `prompt` with the base model and with the LoRA-adapted model."""
    from peft import PeftModel

    tokenizer, base_model, device = _load(base_path, None)
    before = _answer(tokenizer, base_model, device, prompt, max_new_tokens)

    adapted = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
    adapted.eval()
    after = _answer(tokenizer, adapted, device, prompt, max_new_tokens)
    return {"before": before, "after": after}


def _main() -> int:
    parser = argparse.ArgumentParser(description="Local inference worker")
    parser.add_argument("--mode", choices=["chat", "compare"], default="compare")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    if args.mode == "chat":
        result = chat(args.base, args.adapter, args.prompt, args.max_new_tokens)
    else:
        if not args.adapter:
            parser.error("--adapter is required for compare mode")
        result = compare(args.base, args.adapter, args.prompt, args.max_new_tokens)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
