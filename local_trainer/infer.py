from __future__ import annotations

import argparse
import json
import sys

from .hardware import detect_device, torch_dtype


def compare(base_path: str, adapter_dir: str, prompt: str, max_new_tokens: int = 80) -> dict[str, str]:
    """Answer `prompt` with the base model and with the LoRA-adapted model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = detect_device()
    dtype = torch_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    def answer(model) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    base = AutoModelForCausalLM.from_pretrained(base_path, dtype=dtype).to(device)
    base.eval()
    before = answer(base)

    adapted = PeftModel.from_pretrained(base, adapter_dir).to(device)
    adapted.eval()
    after = answer(adapted)

    return {"before": before, "after": after}


def _main() -> int:
    parser = argparse.ArgumentParser(description="Before/after LoRA comparison")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    result = compare(args.base, args.adapter, args.prompt, args.max_new_tokens)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
