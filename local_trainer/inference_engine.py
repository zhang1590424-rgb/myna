"""Lab inference: load an experiment's output and chat against it.

Each chat is a one-shot subprocess (load base + adapter, answer, exit). On
M2/16GB holding a model resident between requests is risky, so we trade per-call
latency for memory safety and process isolation. "Loaded" here means a pinned
target (which experiment to answer with), not a resident model.
"""
from __future__ import annotations

import asyncio
import json
import sys

from .domain import LabStatus
from .experiment_service import ExperimentService
from .hardware import training_env
from .model_registry import get_model

# 回答风格档位 → 底层解码参数。用户只在界面选档位，参数藏在背后。
# 详见 infer.GenParams：这些只影响“怎么把话说出来”，与模型权重无关。
STYLE_PRESETS: dict[str, dict[str, float | int]] = {
    "steady": {"temperature": 0.5, "top_p": 0.9, "repetition_penalty": 1.15, "no_repeat_ngram_size": 3},
    "balanced": {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.15, "no_repeat_ngram_size": 3},
    "lively": {"temperature": 1.0, "top_p": 0.95, "repetition_penalty": 1.2, "no_repeat_ngram_size": 3},
}
DEFAULT_STYLE = "balanced"


def resolve_gen_params(
    style: str | None = None,
    overrides: dict | None = None,
) -> dict[str, float | int]:
    """把风格档位 + 可选高级微调合并成最终解码参数。"""
    params = dict(STYLE_PRESETS.get(style or DEFAULT_STYLE, STYLE_PRESETS[DEFAULT_STYLE]))
    if overrides:
        for key in ("temperature", "top_p", "repetition_penalty", "no_repeat_ngram_size"):
            if overrides.get(key) is not None:
                params[key] = overrides[key]
    return params


class InferenceEngine:
    def __init__(self, experiments: ExperimentService) -> None:
        self.experiments = experiments
        self._experiment_id: str | None = None
        self._use_adapter: bool = True

    def status(self) -> LabStatus:
        if self._experiment_id is None:
            return LabStatus()
        try:
            exp = self.experiments.get(self._experiment_id)
        except KeyError:
            self._experiment_id = None
            return LabStatus(message="之前加载的实验已被删除，请重新加载。")
        return LabStatus(
            loaded=True,
            experiment_id=exp.id,
            experiment_name=exp.name,
            use_adapter=self._use_adapter,
            message=f"已加载「{exp.name}」，可以开始对话。",
        )

    def load(self, experiment_id: str, use_adapter: bool = True) -> LabStatus:
        exp = self.experiments.get(experiment_id)
        if exp.status != "completed" or not exp.output_dir:
            raise RuntimeError("只有训练完成的实验才能加载到测评。")
        self._experiment_id = experiment_id
        self._use_adapter = use_adapter
        return self.status()

    def unload(self) -> LabStatus:
        self._experiment_id = None
        return LabStatus(message="已卸载模型。")

    async def _run_inference(
        self,
        model_path: str,
        prompt: str,
        max_new_tokens: int,
        adapter_path: str | None = None,
        gen_params: dict | None = None,
    ) -> str:
        """启动子进程推理，返回回答文本。"""
        args = [
            sys.executable,
            "-m",
            "local_trainer.infer",
            "--mode",
            "chat",
            "--base",
            model_path,
            "--prompt",
            prompt,
            "--max-new-tokens",
            str(max_new_tokens),
        ]
        if adapter_path:
            args += ["--adapter", adapter_path]
        if gen_params:
            args += [
                "--temperature", str(gen_params["temperature"]),
                "--top-p", str(gen_params["top_p"]),
                "--repetition-penalty", str(gen_params["repetition_penalty"]),
                "--no-repeat-ngram-size", str(gen_params["no_repeat_ngram_size"]),
            ]

        proc = await asyncio.create_subprocess_exec(
            *args,
            env=training_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            detail = stderr.decode("utf-8", "ignore")[-500:]
            raise RuntimeError(f"模型加载或推理失败。{detail}")
        payload = json.loads(stdout.decode("utf-8").strip().splitlines()[-1])
        return payload["answer"]

    async def chat(self, prompt: str, max_new_tokens: int = 120, gen_params: dict | None = None) -> str:
        if self._experiment_id is None:
            raise RuntimeError("请先在左侧加载一个实验。")
        exp = self.experiments.get(self._experiment_id)
        model = get_model(exp.model_id)
        if not model.local_path:
            raise RuntimeError("没找到本地基础模型，无法对话。")

        prompt = prompt.strip()
        if not prompt:
            raise RuntimeError("请输入要测试的问题。")

        adapter = exp.output_dir if self._use_adapter else None
        return await self._run_inference(model.local_path, prompt, max_new_tokens, adapter, gen_params)

    async def compare_chat(self, prompt: str, max_new_tokens: int = 120, gen_params: dict | None = None) -> dict:
        """同一 prompt 分别用 base 和 fine-tuned 推理，返回两个结果。"""
        if self._experiment_id is None:
            raise RuntimeError("请先加载一个实验。")
        exp = self.experiments.get(self._experiment_id)
        model = get_model(exp.model_id)
        if not model.local_path:
            raise RuntimeError("没找到本地基础模型，无法对话。")

        prompt = prompt.strip()
        if not prompt:
            raise RuntimeError("请输入要测试的问题。")

        # 先跑 base（不加 adapter），再跑 fine-tuned（加 adapter）
        base_answer = await self._run_inference(model.local_path, prompt, max_new_tokens, adapter_path=None, gen_params=gen_params)
        finetuned_answer = await self._run_inference(model.local_path, prompt, max_new_tokens, adapter_path=exp.output_dir, gen_params=gen_params)
        return {"prompt": prompt, "base_answer": base_answer, "finetuned_answer": finetuned_answer}
