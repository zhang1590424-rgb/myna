"""Training engine: runs a real LLaMA-Factory subprocess per Experiment.

Drives the subprocess, streams progress into ExperimentService, and supports
stop / export. SFT reads alpaca records, DPO reads preference records. Training
can only be started once the environment is ready (LLaMA-Factory installed and
at least one local model); the API layer blocks creation otherwise, so there is
no demo / mock fallback.
"""
from __future__ import annotations

import asyncio
import json
import math
import signal
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import yaml

from .dataset_manager import DatasetManager
from .domain import Experiment, ExperimentStatus
from .experiment_service import ExperimentService
from .hardware import llamafactory_cli, training_env
from .llamafactory import LlamaFactoryConfigBuilder
from .model_registry import get_model
from .paths import RUNS_DIR, ensure_runtime_dirs


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ExportResult:
    path: Path
    filename: str
    media_type: str


class TrainingEngine(Protocol):
    name: str

    async def start(self, experiment: Experiment) -> None: ...

    async def wait(self, exp_id: str) -> None: ...

    async def stop(self, exp_id: str) -> Experiment: ...

    async def export(self, exp_id: str, merge: bool = False) -> ExportResult: ...


def parse_trainer_log(text: str) -> dict[str, object]:
    """Turn a trainer_log.jsonl body into progress/loss/eta we can show."""
    losses: list[float] = []
    last: dict[str, object] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "loss" in entry:
            losses.append(float(entry["loss"]))
        last = entry

    percentage = float(last.get("percentage", 0.0)) if last else 0.0
    epoch = last.get("epoch")
    return {
        "progress": int(min(99, math.floor(percentage))),
        "loss": losses,
        "eta": last.get("remaining_time"),
        "epoch": math.ceil(float(epoch)) if isinstance(epoch, (int, float)) else None,
    }


# --------------------------------------------------------------------------- #
# Real engine
# --------------------------------------------------------------------------- #
class LlamaFactoryTrainingEngine:
    name = "llamafactory"

    def __init__(
        self,
        experiments: ExperimentService,
        datasets: DatasetManager,
        config_builder: LlamaFactoryConfigBuilder | None = None,
        runs_dir: Path = RUNS_DIR,
    ) -> None:
        self.experiments = experiments
        self.datasets = datasets
        self.config_builder = config_builder or LlamaFactoryConfigBuilder(runs_dir)
        self.runs_dir = runs_dir
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stopping: set[str] = set()
        ensure_runtime_dirs()

    async def start(self, experiment: Experiment) -> None:
        try:
            if experiment.method == "dpo":
                records = self.datasets.read_preferences(experiment.dataset_id)
            else:
                records = self.datasets.read_records(experiment.dataset_id)
            model = get_model(experiment.model_id)
        except Exception as exc:  # noqa: BLE001
            self.experiments.apply_changes(
                experiment.id,
                status=ExperimentStatus.failed.value,
                finished_at=utc_now(),
                error=str(exc),
                message="训练所需的数据或模型不存在。",
            )
            return

        if not model.local_path:
            self.experiments.apply_changes(
                experiment.id,
                status=ExperimentStatus.failed.value,
                finished_at=utc_now(),
                error="model local_path is None",
                message="没找到本地模型，请先在模型页下载。",
            )
            return

        prepared = self.config_builder.prepare(experiment=experiment, records=records, model=model)
        trainer_log = prepared.output_dir / "trainer_log.jsonl"

        proc = await asyncio.create_subprocess_exec(
            *prepared.command,
            cwd=str(prepared.run_dir),
            env=training_env(),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self._procs[experiment.id] = proc
        self.experiments.apply_changes(
            experiment.id,
            status=ExperimentStatus.running.value,
            started_at=utc_now(),
            progress=2,
            message="正在整理数据，准备开始训练",
            pid=proc.pid,
            engine=self.name,
            run_dir=str(prepared.run_dir),
            output_dir=str(prepared.output_dir),
        )
        self._tasks[experiment.id] = asyncio.create_task(
            self._watch(experiment.id, proc, trainer_log, prepared.output_dir)
        )

    async def wait(self, exp_id: str) -> None:
        task = self._tasks.get(exp_id)
        if task is not None:
            await task

    async def _watch(
        self,
        exp_id: str,
        proc: asyncio.subprocess.Process,
        trainer_log: Path,
        output_dir: Path,
    ) -> None:
        try:
            while proc.returncode is None:
                self._refresh_progress(exp_id, trainer_log)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

            stderr = b""
            if proc.stderr is not None:
                stderr = await proc.stderr.read()
            self._finish(exp_id, proc.returncode, output_dir, stderr.decode("utf-8", "ignore"))
        except Exception as exc:  # pragma: no cover - defensive
            self.experiments.apply_changes(
                exp_id,
                status=ExperimentStatus.failed.value,
                finished_at=utc_now(),
                error=str(exc),
                message="训练过程中出现意外错误，请重试。",
            )
        finally:
            self._procs.pop(exp_id, None)
            self._tasks.pop(exp_id, None)
            self._stopping.discard(exp_id)

    def _refresh_progress(self, exp_id: str, trainer_log: Path) -> None:
        if not trainer_log.exists():
            return
        parsed = parse_trainer_log(trainer_log.read_text(encoding="utf-8"))
        exp = self.experiments.get(exp_id)
        message = exp.message
        if parsed["epoch"]:
            message = f"正在学习第 {parsed['epoch']} 轮，共 {exp.params.epochs} 轮"
        self.experiments.apply_changes(
            exp_id,
            progress=max(exp.progress, int(parsed["progress"])),
            loss=parsed["loss"],
            eta=parsed["eta"],
            message=message,
        )

    def _finish(self, exp_id: str, returncode: int | None, output_dir: Path, stderr: str) -> None:
        if exp_id in self._stopping:
            self.experiments.apply_changes(
                exp_id,
                status=ExperimentStatus.stopped.value,
                finished_at=utc_now(),
                message="训练已停止，已学到的内容没有丢失。",
            )
            return

        adapter_ok = (output_dir / "adapter_model.safetensors").exists()
        if returncode == 0 and adapter_ok:
            exp = self.experiments.get(exp_id)
            metrics = dict(exp.metrics)
            if exp.loss:
                metrics["final_loss"] = round(exp.loss[-1], 4)
                metrics["peak_loss"] = round(max(exp.loss), 4)
            self.experiments.apply_changes(
                exp_id,
                status=ExperimentStatus.completed.value,
                progress=100,
                output_dir=str(output_dir),
                finished_at=utc_now(),
                eta=None,
                metrics=metrics,
                message="训练完成，可以到测评试试它学到了什么。",
            )
            return

        self.experiments.apply_changes(
            exp_id,
            status=ExperimentStatus.failed.value,
            finished_at=utc_now(),
            error=stderr[-2000:] if stderr else f"exit code {returncode}",
            message=_humanize_failure(stderr),
        )

    async def stop(self, exp_id: str) -> Experiment:
        proc = self._procs.get(exp_id)
        if proc is not None and proc.returncode is None:
            self._stopping.add(exp_id)
            try:
                proc.send_signal(signal.SIGTERM)
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
            except ProcessLookupError:
                pass
        return self.experiments.apply_changes(
            exp_id, status=ExperimentStatus.stopping.value, message="正在安全停止训练"
        )

    async def export(self, exp_id: str, merge: bool = False) -> ExportResult:
        exp = self.experiments.get(exp_id)
        if exp.status != "completed" or not exp.output_dir:
            raise RuntimeError("训练完成后才能导出模型。")
        output_dir = Path(exp.output_dir)
        if merge:
            return await self._export_merged(exp, output_dir)
        zip_path = output_dir.parent / f"lora-adapter-{exp_id}.zip"
        _zip_dir(output_dir, zip_path)
        return ExportResult(path=zip_path, filename=zip_path.name, media_type="application/zip")

    async def _export_merged(self, exp: Experiment, output_dir: Path) -> ExportResult:
        model = get_model(exp.model_id)
        if not model.local_path:
            raise RuntimeError("没找到本地基础模型，无法合并导出。")

        merged_dir = output_dir.parent / "merged"
        if not (merged_dir / "config.json").exists():
            config_file = output_dir.parent / "export.yaml"
            config = {
                "model_name_or_path": model.local_path,
                "adapter_name_or_path": str(output_dir),
                "template": model.lf_template,
                "finetuning_type": "lora",
                "trust_remote_code": True,
                "export_dir": str(merged_dir),
                "export_size": 2,
                "export_legacy_format": False,
            }
            config_file.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            proc = await asyncio.create_subprocess_exec(
                llamafactory_cli(),
                "export",
                str(config_file),
                env=training_env(),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0 or not (merged_dir / "config.json").exists():
                raise RuntimeError("合并导出失败，可改为导出 LoRA 适配器。")

        zip_path = output_dir.parent / f"full-model-{exp.id}.zip"
        _zip_dir(merged_dir, zip_path)
        return ExportResult(path=zip_path, filename=zip_path.name, media_type="application/zip")


def _zip_dir(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in source_dir.rglob("*"):
            if file.is_file():
                archive.write(file, file.relative_to(source_dir))


def _humanize_failure(stderr: str) -> str:
    lowered = stderr.lower()
    if "out of memory" in lowered or "oom" in lowered:
        return "内存不够用了。建议用更小的模型，或减少一次训练的数据量。"
    if "no such file" in lowered or "not found" in lowered:
        return "训练组件或模型文件缺失，请先在模型页检查。"
    return "训练中断了。常见原因是数据太少或内存不足，可减少数据或换更小的模型再试。"


# --------------------------------------------------------------------------- #
# Engine factory
# --------------------------------------------------------------------------- #
def real_engine_ready() -> bool:
    from .environment import collect_environment_status

    status = collect_environment_status()
    has_model = any(model.available for model in status.model_status)
    return status.llamafactory_ok and has_model


def build_engine(experiments: ExperimentService, datasets: DatasetManager) -> TrainingEngine:
    return LlamaFactoryTrainingEngine(experiments, datasets)
