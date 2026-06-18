"""Multi-experiment comparison: parameter diff + loss curves + metrics.

The frontend shows a difference-driven table: rows that are identical across
all selected experiments are flagged so the UI can fold them. Only differing
rows matter. Loss curves are returned as-is for the UI to overlay.
"""
from __future__ import annotations

from .domain import Experiment
from .experiment_service import ExperimentService


def build_comparison(experiments: ExperimentService, ids: list[str]) -> dict[str, object]:
    selected: list[Experiment] = []
    for exp_id in ids:
        try:
            selected.append(experiments.get(exp_id))
        except KeyError:
            continue
    if not selected:
        raise KeyError("no valid experiments")

    rows = _param_rows(selected)
    return {
        "experiments": [
            {
                "id": exp.id,
                "name": exp.name,
                "status": exp.status,
                "model_id": exp.model_id,
                "method": exp.method,
            }
            for exp in selected
        ],
        "rows": rows,
        "loss_series": [
            {"id": exp.id, "name": exp.name, "loss": exp.loss} for exp in selected
        ],
    }


def _param_rows(selected: list[Experiment]) -> list[dict[str, object]]:
    def values(getter) -> list[object]:
        return [getter(exp) for exp in selected]

    spec: list[tuple[str, object]] = [
        ("模型", values(lambda e: e.model_id)),
        ("方法", values(lambda e: e.method)),
        ("轮数", values(lambda e: e.params.epochs)),
        ("学习率", values(lambda e: e.params.learning_rate)),
        ("LoRA rank", values(lambda e: e.params.lora_rank)),
        ("批大小", values(lambda e: e.params.batch_size)),
        ("数据条数", values(lambda e: e.dataset_count)),
        ("final loss", values(lambda e: round(e.loss[-1], 4) if e.loss else None)),
    ]
    rows: list[dict[str, object]] = []
    for label, cells in spec:
        same = len({str(cell) for cell in cells}) <= 1
        rows.append({"label": label, "cells": cells, "same": same})
    return rows
