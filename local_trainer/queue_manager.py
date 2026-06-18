"""Queue manager: run experiments one at a time (hardware mutex).

Mac M2/16GB can only sustain one training run at a time, so experiments are
queued and executed sequentially. A single background worker pulls the next
pending experiment, marks it queued/running via the engine, and waits for it to
finish before starting the next. The queue can be paused and resumed.
"""
from __future__ import annotations

import asyncio

from .domain import Experiment, ExperimentStatus
from .engine import TrainingEngine
from .experiment_service import ExperimentService


class QueueManager:
    def __init__(self, experiments: ExperimentService, engine: TrainingEngine) -> None:
        self.experiments = experiments
        self.engine = engine
        self._paused = False
        self._wakeup = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._current_id: str | None = None

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._loop())
        self._wakeup.set()

    async def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    def enqueue(self, exp_id: str) -> Experiment:
        exp = self.experiments.apply_changes(
            exp_id, status=ExperimentStatus.queued.value, message="已加入队列，等待开始。"
        )
        self._wakeup.set()
        return exp

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False
        self._wakeup.set()

    def status(self) -> dict[str, object]:
        pending = [
            exp
            for exp in self.experiments.list()
            if exp.status in {ExperimentStatus.queued.value, ExperimentStatus.pending.value}
        ]
        running = self._current_id
        return {
            "paused": self._paused,
            "running_id": running,
            "queued": [{"id": exp.id, "name": exp.name} for exp in reversed(pending)],
            "queued_count": len(pending),
        }

    # ---- worker ---- #
    async def _loop(self) -> None:
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()
            if self._paused:
                continue
            await self._drain()

    async def _drain(self) -> None:
        while not self._paused:
            nxt = self._next_queued()
            if nxt is None:
                return
            self._current_id = nxt.id
            try:
                await self.engine.start(nxt)
                await self.engine.wait(nxt.id)  # type: ignore[attr-defined]
            finally:
                self._current_id = None

    def _next_queued(self) -> Experiment | None:
        queued = [
            exp for exp in self.experiments.list() if exp.status == ExperimentStatus.queued.value
        ]
        if not queued:
            return None
        # list() is newest-first; run oldest queued first.
        return sorted(queued, key=lambda e: e.created_at)[0]
