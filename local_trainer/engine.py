from __future__ import annotations

import asyncio
import json
import math
import signal
import sys
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import yaml

from .domain import CompareResponse, JobStatus, TrainingJob, TrainingSettings
from .hardware import llamafactory_cli, training_env
from .llamafactory import LlamaFactoryConfigBuilder
from .paths import RUNS_DIR, ensure_runtime_dirs
from .templates import get_template


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ExportResult:
    path: Path
    filename: str
    media_type: str


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, TrainingJob] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        template_id: str,
        dataset_id: str,
        model_id: str,
        dataset_count: int,
        settings: TrainingSettings,
        engine: str = "mock",
    ) -> TrainingJob:
        job = TrainingJob(
            id=uuid.uuid4().hex,
            template_id=template_id,
            dataset_id=dataset_id,
            model_id=model_id,
            dataset_count=dataset_count,
            settings=settings,
            engine=engine,
            created_at=utc_now(),
        )
        async with self._lock:
            self._jobs[job.id] = job
        return job

    async def get(self, job_id: str) -> TrainingJob:
        async with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError as exc:
                raise KeyError(job_id) from exc

    async def update(self, job_id: str, **changes: object) -> TrainingJob:
        async with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            updated = self._jobs[job_id].model_copy(update=changes)
            self._jobs[job_id] = updated
            return updated

    async def list(self) -> list[TrainingJob]:
        async with self._lock:
            return list(self._jobs.values())


class TrainingEngine(Protocol):
    name: str

    async def start(self, job: TrainingJob) -> None: ...

    async def stop(self, job_id: str) -> TrainingJob: ...

    async def compare(self, job_id: str, prompt: str) -> CompareResponse: ...

    async def export(self, job_id: str, merge: bool = False) -> ExportResult: ...


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
    """Runs a real LLaMA-Factory LoRA SFT subprocess and streams its progress."""

    name = "llamafactory"

    def __init__(
        self,
        job_store: JobStore,
        dataset_store,
        config_builder: LlamaFactoryConfigBuilder | None = None,
        runs_dir: Path = RUNS_DIR,
    ) -> None:
        self.job_store = job_store
        self.dataset_store = dataset_store
        self.config_builder = config_builder or LlamaFactoryConfigBuilder(runs_dir)
        self.runs_dir = runs_dir
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stopping: set[str] = set()
        ensure_runtime_dirs()

    async def start(self, job: TrainingJob) -> None:
        from .templates import get_model

        try:
            records = self.dataset_store.read_records(job.dataset_id)
            template = get_template(job.template_id)
            model = get_model(job.model_id)
        except KeyError as exc:
            await self.job_store.update(
                job.id,
                status=JobStatus.failed,
                finished_at=utc_now(),
                error=f"missing resource: {exc}",
                message="训练所需的数据、模板或模型不存在。",
            )
            return

        if not model.local_path:
            await self.job_store.update(
                job.id,
                status=JobStatus.failed,
                finished_at=utc_now(),
                error="model local_path is None",
                message="没找到本地模型，请先在准备环境页准备模型。",
            )
            return

        prepared = self.config_builder.prepare(job=job, records=records, template=template, model=model)
        output_dir = prepared.run_dir / "output"
        trainer_log = output_dir / "trainer_log.jsonl"

        proc = await asyncio.create_subprocess_exec(
            *prepared.command,
            cwd=str(prepared.run_dir),
            env=training_env(),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self._procs[job.id] = proc
        await self.job_store.update(
            job.id,
            status=JobStatus.running,
            started_at=utc_now(),
            progress=2,
            message="正在整理你的数据，准备开始训练",
            pid=proc.pid,
            run_dir=str(prepared.run_dir),
            output_dir=str(output_dir),
        )
        self._tasks[job.id] = asyncio.create_task(self._watch(job.id, proc, trainer_log, output_dir))

    async def _watch(
        self,
        job_id: str,
        proc: asyncio.subprocess.Process,
        trainer_log: Path,
        output_dir: Path,
    ) -> None:
        try:
            while proc.returncode is None:
                await self._refresh_progress(job_id, trainer_log)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

            stderr = b""
            if proc.stderr is not None:
                stderr = await proc.stderr.read()
            await self._finish(job_id, proc.returncode, output_dir, stderr.decode("utf-8", "ignore"))
        except Exception as exc:  # pragma: no cover - defensive
            await self.job_store.update(
                job_id,
                status=JobStatus.failed,
                finished_at=utc_now(),
                error=str(exc),
                message="训练过程中出现意外错误，请重试。",
            )
        finally:
            self._procs.pop(job_id, None)
            self._tasks.pop(job_id, None)
            self._stopping.discard(job_id)

    async def _refresh_progress(self, job_id: str, trainer_log: Path) -> None:
        if not trainer_log.exists():
            return
        parsed = parse_trainer_log(trainer_log.read_text(encoding="utf-8"))
        job = await self.job_store.get(job_id)
        learned = math.ceil(job.dataset_count * parsed["progress"] / 100)
        message = job.message
        if parsed["epoch"]:
            message = f"正在学习第 {parsed['epoch']} 轮，共 {job.settings.epochs} 轮"
        await self.job_store.update(
            job_id,
            progress=max(job.progress, int(parsed["progress"])),
            loss=parsed["loss"],
            learned_count=min(job.dataset_count, learned),
            eta=parsed["eta"],
            message=message,
        )

    async def _finish(self, job_id: str, returncode: int | None, output_dir: Path, stderr: str) -> None:
        if job_id in self._stopping:
            await self.job_store.update(
                job_id,
                status=JobStatus.stopped,
                finished_at=utc_now(),
                message="训练已停止，已学到的内容没有丢失。",
            )
            return

        adapter_ok = (output_dir / "adapter_model.safetensors").exists()
        if returncode == 0 and adapter_ok:
            job = await self.job_store.get(job_id)
            await self.job_store.update(
                job_id,
                status=JobStatus.completed,
                progress=100,
                learned_count=job.dataset_count,
                output_dir=str(output_dir),
                finished_at=utc_now(),
                eta=None,
                message="训练完成，可以试试看它学到了什么。",
            )
            return

        await self.job_store.update(
            job_id,
            status=JobStatus.failed,
            finished_at=utc_now(),
            error=stderr[-2000:] if stderr else f"exit code {returncode}",
            message=_humanize_failure(stderr),
        )

    async def stop(self, job_id: str) -> TrainingJob:
        proc = self._procs.get(job_id)
        if proc is not None and proc.returncode is None:
            self._stopping.add(job_id)
            try:
                proc.send_signal(signal.SIGTERM)
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                proc.kill()
            except ProcessLookupError:
                pass
        return await self.job_store.update(job_id, status=JobStatus.stopping, message="正在安全停止训练")

    async def compare(self, job_id: str, prompt: str) -> CompareResponse:
        from .templates import get_model

        job = await self.job_store.get(job_id)
        if job.status != JobStatus.completed or not job.output_dir:
            raise RuntimeError("训练完成后才能对比回答。")

        template = get_template(job.template_id)
        prompt = prompt.strip() or template.starter_prompt
        model = get_model(job.model_id)
        if not model.local_path:
            raise RuntimeError("没找到本地模型，无法进行对比。")

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "local_trainer.infer",
            "--base",
            model.local_path,
            "--adapter",
            job.output_dir,
            "--prompt",
            prompt,
            env=training_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError("模型加载失败，请确认训练已正常完成。")
        payload = json.loads(stdout.decode("utf-8").strip().splitlines()[-1])
        return CompareResponse(prompt=prompt, before=payload["before"], after=payload["after"])

    async def export(self, job_id: str, merge: bool = False) -> ExportResult:
        job = await self.job_store.get(job_id)
        if job.status != JobStatus.completed or not job.output_dir:
            raise RuntimeError("训练完成后才能导出模型。")
        output_dir = Path(job.output_dir)
        if merge:
            return await self._export_merged(job, output_dir)
        zip_path = output_dir.parent / f"lora-adapter-{job_id}.zip"
        _zip_dir(output_dir, zip_path)
        return ExportResult(path=zip_path, filename=zip_path.name, media_type="application/zip")

    async def _export_merged(self, job: TrainingJob, output_dir: Path) -> ExportResult:
        from .templates import get_model

        model = get_model(job.model_id)
        if not model.local_path:
            raise RuntimeError("没找到本地基础模型，无法合并导出。")

        merged_dir = output_dir.parent / "merged"
        if not (merged_dir / "config.json").exists():
            config_file = output_dir.parent / "export.yaml"
            config = {
                "model_name_or_path": model.local_path,
                "adapter_name_or_path": str(output_dir),
                "template": "qwen",
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

        zip_path = output_dir.parent / f"full-model-{job.id}.zip"
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
        return "训练组件或模型文件缺失，请先在准备环境页检查。"
    return "训练中断了。常见原因是数据太少或内存不足，可减少数据或换更小的模型再试。"


# --------------------------------------------------------------------------- #
# Mock engine (demo / fallback)
# --------------------------------------------------------------------------- #
class MockTrainingEngine:
    """A product-flow engine used when the real engine is unavailable."""

    name = "mock"

    def __init__(self, job_store: JobStore, runs_dir: Path = RUNS_DIR) -> None:
        self.job_store = job_store
        self.runs_dir = runs_dir
        self._stop_events: dict[str, asyncio.Event] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        ensure_runtime_dirs()

    async def start(self, job: TrainingJob) -> None:
        stop_event = asyncio.Event()
        self._stop_events[job.id] = stop_event
        self._tasks[job.id] = asyncio.create_task(self._run(job.id, stop_event))

    async def stop(self, job_id: str) -> TrainingJob:
        event = self._stop_events.get(job_id)
        if event is not None:
            event.set()
        return await self.job_store.update(job_id, status=JobStatus.stopping, message="正在安全停止训练")

    async def compare(self, job_id: str, prompt: str) -> CompareResponse:
        job = await self.job_store.get(job_id)
        if job.status != JobStatus.completed:
            raise RuntimeError("训练完成后才能对比回答。")

        template = get_template(job.template_id)
        prompt = prompt.strip() or template.starter_prompt
        before = "您好，我会尽量回答您的问题。建议您提供更多背景信息，方便进一步处理。"
        after = _template_answer(template.id, prompt)
        return CompareResponse(prompt=prompt, before=before, after=after)

    async def export(self, job_id: str, merge: bool = False) -> ExportResult:
        job = await self.job_store.get(job_id)
        if job.status != JobStatus.completed:
            raise RuntimeError("训练完成后才能导出模型。")
        ensure_runtime_dirs()
        export_dir = self.runs_dir / job_id
        export_dir.mkdir(parents=True, exist_ok=True)
        payload_path = export_dir / f"model-export-{job_id}.json"
        payload_path.write_text(
            json.dumps(
                {
                    "job_id": job.id,
                    "model_id": job.model_id,
                    "template_id": job.template_id,
                    "dataset_id": job.dataset_id,
                    "output_dir": job.output_dir,
                    "engine": "mock",
                    "merge_requested": merge,
                    "note": "当前是模拟训练产物。切换到真实引擎后会导出真正的 LoRA 适配器或合并模型。",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return ExportResult(
            path=payload_path,
            filename=payload_path.name,
            media_type="application/json; charset=utf-8",
        )

    async def _run(self, job_id: str, stop_event: asyncio.Event) -> None:
        job = await self.job_store.update(
            job_id,
            status=JobStatus.running,
            started_at=utc_now(),
            progress=2,
            message="正在整理你的数据",
        )
        total_steps = max(12, job.settings.epochs * 8)
        losses: list[float] = []

        try:
            for step in range(1, total_steps + 1):
                if stop_event.is_set():
                    await self.job_store.update(
                        job_id,
                        status=JobStatus.stopped,
                        finished_at=utc_now(),
                        message="训练已停止，已有数据没有丢失。",
                    )
                    return

                progress = min(99, math.floor(step / total_steps * 100))
                epoch = min(job.settings.epochs, math.ceil(step / max(1, total_steps / job.settings.epochs)))
                learned_count = min(job.dataset_count, math.ceil(job.dataset_count * progress / 100))
                loss = round(max(0.18, 2.2 - (step / total_steps) * 1.55), 3)
                losses.append(loss)
                await self.job_store.update(
                    job_id,
                    progress=progress,
                    learned_count=learned_count,
                    loss=losses,
                    message=f"正在学习第 {epoch} 轮，共 {job.settings.epochs} 轮",
                )
                await asyncio.sleep(0.28)

            output_dir = self.runs_dir / job_id
            output_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = output_dir / "adapter-metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "engine": "mock",
                        "status": "completed",
                        "created_at": utc_now(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            await self.job_store.update(
                job_id,
                status=JobStatus.completed,
                progress=100,
                learned_count=job.dataset_count,
                output_dir=str(output_dir),
                finished_at=utc_now(),
                message="训练完成，可以试试看它学到了什么。",
            )
        except Exception as exc:
            await self.job_store.update(
                job_id,
                status=JobStatus.failed,
                finished_at=utc_now(),
                error=str(exc),
                message="训练失败。请检查数据格式后再试一次。",
            )


# --------------------------------------------------------------------------- #
# Engine factory
# --------------------------------------------------------------------------- #
def real_engine_ready() -> bool:
    """Real training needs the llamafactory CLI and at least one local model."""
    from .environment import collect_environment_status

    status = collect_environment_status()
    has_model = any(model.available for model in status.model_status)
    return status.llamafactory_ok and has_model


def build_engine(job_store: JobStore, dataset_store) -> TrainingEngine:
    """Pick the real engine when the environment is ready, otherwise the demo one."""
    if real_engine_ready():
        return LlamaFactoryTrainingEngine(job_store, dataset_store)
    return MockTrainingEngine(job_store)


def _template_answer(template_id: str, prompt: str) -> str:
    if template_id == "customer_service":
        return f"亲，我明白您的问题：{prompt}。您先把订单号发我，我会马上帮您确认情况，并给您一个明确处理方案。"
    if template_id == "roleplay":
        return f"我听见你在说：{prompt}。先别急着否定自己，我们把事情拆小，一步一步来。"
    if template_id == "rewrite":
        return f"改写后：{prompt.strip()}。我会让语气更清楚、更适合正式场景，同时保留原意。"
    if template_id == "knowledge_qa":
        return f"根据你提供的知识，我会这样回答：{prompt}。如果资料里没有明确答案，我会提示需要人工确认。"
    return f"我会按你的数据风格回答：{prompt}"
