"""Experiment service: CRUD, cloning, batch variables, notes, and state updates.

This is the data layer over persistence.Database. It does not run training
itself; the engine (engine.py) drives subprocesses and calls apply_changes to
persist progress, and queue_manager.py decides what runs when.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from .dataset_manager import DatasetManager
from .domain import (
    BatchVariableRequest,
    CreateExperimentRequest,
    Experiment,
    ExperimentParams,
    ExperimentStatus,
    TrainingMethod,
    UpdateExperimentRequest,
)
from .model_registry import get_model
from .persistence import Database


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentService:
    def __init__(self, db: Database, dataset_manager: DatasetManager) -> None:
        self.db = db
        self.datasets = dataset_manager

    # ---- reads ---- #
    def list(self) -> list[Experiment]:
        return self.db.list_experiments()

    def get(self, exp_id: str) -> Experiment:
        exp = self.db.get_experiment(exp_id)
        if exp is None:
            raise KeyError(exp_id)
        return exp

    # ---- create ---- #
    def create(self, req: CreateExperimentRequest) -> Experiment:
        dataset_count = self._dataset_count(req.dataset_id)
        exp = Experiment(
            id=uuid.uuid4().hex,
            name=req.name or self._auto_name(req.method, req.model_id, req.dataset_id),
            method=req.method,
            model_id=req.model_id,
            dataset_id=req.dataset_id,
            dataset_count=dataset_count,
            params=req.params,
            status=ExperimentStatus.pending,
            cloned_from=req.cloned_from,
            created_at=utc_now(),
        )
        self.db.upsert_experiment(exp)
        return exp

    def clone(self, exp_id: str, name: str | None = None) -> Experiment:
        source = self.get(exp_id)
        req = CreateExperimentRequest(
            model_id=source.model_id,
            dataset_id=source.dataset_id,
            method=source.method,  # type: ignore[arg-type]
            params=source.params.model_copy(deep=True),
            name=name,
            cloned_from=source.id,
        )
        return self.create(req)

    def batch_create(self, req: BatchVariableRequest) -> list[Experiment]:
        experiments: list[Experiment] = []
        for value in req.values:
            params = self._params_with_variable(req.base_params, req.variable, value)
            create_req = CreateExperimentRequest(
                model_id=req.model_id,
                dataset_id=req.dataset_id,
                method=req.method,  # type: ignore[arg-type]
                params=params,
                name=self._batch_name(req, value),
            )
            experiments.append(self.create(create_req))
        return experiments

    # ---- update ---- #
    def update(self, exp_id: str, req: UpdateExperimentRequest) -> Experiment:
        changes: dict[str, object] = {}
        if req.name is not None:
            changes["name"] = req.name
        if req.notes is not None:
            changes["notes"] = req.notes
        if req.tags is not None:
            changes["tags"] = req.tags
        return self.apply_changes(exp_id, **changes)

    def apply_changes(self, exp_id: str, **changes: object) -> Experiment:
        """Persist a partial update. Used by the engine to stream progress."""
        exp = self.get(exp_id)
        if not changes:
            return exp
        updated = exp.model_copy(update=changes)
        self.db.upsert_experiment(updated)
        return updated

    def delete(self, exp_id: str) -> bool:
        return self.db.delete_experiment(exp_id)

    # ---- helpers ---- #
    def _dataset_count(self, dataset_id: str) -> int:
        try:
            return self.datasets.get_info(dataset_id).row_count
        except KeyError:
            return 0

    def _auto_name(self, method: TrainingMethod, model_id: str, dataset_id: str) -> str:
        n = self.db.count_method_prefix(method, model_id) + 1
        model_token = self._model_token(model_id)
        dataset_token = self._dataset_token(dataset_id)
        return f"{method}-{model_token}-{dataset_token}-#{n}"

    @staticmethod
    def _model_token(model_id: str) -> str:
        try:
            model = get_model(model_id)
        except KeyError:
            return model_id
        return model.parameter_count.lower().replace(".", "")

    def _dataset_token(self, dataset_id: str) -> str:
        try:
            return self.datasets.get_info(dataset_id).name
        except KeyError:
            return "数据"

    @staticmethod
    def _params_with_variable(
        base: ExperimentParams, variable: str, value: float
    ) -> ExperimentParams:
        params = base.model_copy(deep=True)
        if variable in {"epochs", "lora_rank", "batch_size"}:
            setattr(params, variable, int(value))
        else:
            setattr(params, variable, float(value))
        return params

    def _batch_name(self, req: BatchVariableRequest, value: float) -> str:
        prefix = req.name_prefix or f"{req.method}-{self._model_token(req.model_id)}"
        label = int(value) if value == int(value) else value
        return f"{prefix}-{req.variable}={label}"
