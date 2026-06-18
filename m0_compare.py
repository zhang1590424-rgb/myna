import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/Users/bytedance/项目/后训练工具/models/models/Qwen/Qwen2___5-0___5B-Instruct"
ADAPTER = "/Users/bytedance/项目/后训练工具/m0_output"
QUESTIONS = ["你是谁？", "你叫什么名字？", "谁开发了你？"]

device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE)


def ask(model, q):
    msgs = [{"role": "user", "content": q}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


print("加载基础模型...")
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32).to(device)
base.eval()
before = {q: ask(base, q) for q in QUESTIONS}

print("加载 LoRA 微调模型...")
ft = PeftModel.from_pretrained(base, ADAPTER).to(device)
ft.eval()
after = {q: ask(ft, q) for q in QUESTIONS}

print("\n" + "=" * 70)
for q in QUESTIONS:
    print(f"\n【问题】{q}")
    print(f"  训练前: {before[q]}")
    print(f"  训练后: {after[q]}")
print("\n" + "=" * 70)
