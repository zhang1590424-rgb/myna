"""Builds LLaMA-Factory run configs for an Experiment (SFT or DPO).

The product layer never hand-edits LLaMA-Factory source. It writes a dataset
file, a dataset_info.json, and a train.yaml, then shells out to
``llamafactory-cli train``. SFT uses alpaca columns; DPO uses ranking columns
(chosen/rejected). Precision follows the device (fp32 on MPS/CPU, bf16 on CUDA).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from .domain import DatasetRecord, Experiment, ModelOption, PreferenceRecord
from .hardware import detect_device, llamafactory_cli, select_precision
from .paths import RUNS_DIR, ensure_runtime_dirs

# 样本数低于此值时不切验证集；验证集只剩几条时，曲线噪声大、参考价值低。
MIN_SAMPLES_FOR_VALIDATION = 30
# 切给验证集的比例。数据本就不多，留 15% 够观察趋势又不过度牺牲训练数据。
VALIDATION_RATIO = 0.15


@dataclass(frozen=True)
class PreparedLlamaFactoryRun:
    run_dir: Path
    output_dir: Path
    dataset_dir: Path
    dataset_file: Path
    dataset_info_file: Path
    config_file: Path
    command: list[str]


class LlamaFactoryConfigBuilder:
    def __init__(self, runs_dir: Path | str = RUNS_DIR) -> None:
        self.runs_dir = Path(runs_dir)
        ensure_runtime_dirs()

    def prepare(
        self,
        experiment: Experiment,
        records: list[DatasetRecord] | list[PreferenceRecord],
        model: ModelOption,
    ) -> PreparedLlamaFactoryRun:
        run_dir = self.runs_dir / experiment.id / "llamafactory"
        dataset_dir = run_dir / "dataset"
        output_dir = run_dir / "output"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_file = dataset_dir / "user_data.json"
        dataset_info_file = dataset_dir / "dataset_info.json"
        config_file = run_dir / "train.yaml"

        if experiment.method == "dpo":
            self._write_preference_dataset(dataset_file, dataset_info_file, records)  # type: ignore[arg-type]
        else:
            self._write_alpaca_dataset(dataset_file, dataset_info_file, records)  # type: ignore[arg-type]

        # DPO 的 loss 含义不同，当前只给 SFT 自动切验证集，用作训练过程参考信号。
        validation_enabled = experiment.method != "dpo" and len(records) >= MIN_SAMPLES_FOR_VALIDATION

        config = self._build_config(
            experiment, model, dataset_dir, output_dir, len(records), validation_enabled
        )
        config_file.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

        return PreparedLlamaFactoryRun(
            run_dir=run_dir,
            output_dir=output_dir,
            dataset_dir=dataset_dir,
            dataset_file=dataset_file,
            dataset_info_file=dataset_info_file,
            config_file=config_file,
            command=[llamafactory_cli(), "train", str(config_file)],
        )

    # ---- dataset writers ---- #
    @staticmethod
    def _write_alpaca_dataset(
        dataset_file: Path, dataset_info_file: Path, records: list[DatasetRecord]
    ) -> None:
        has_system = any(record.system for record in records)
        rows = []
        for record in records:
            row = {"instruction": record.instruction, "input": record.input, "output": record.output}
            if has_system:
                row["system"] = record.system or ""
            rows.append(row)
        dataset_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

        columns = {"prompt": "instruction", "query": "input", "response": "output"}
        if has_system:
            columns["system"] = "system"
        dataset_info_file.write_text(
            json.dumps(
                {"user_data": {"file_name": dataset_file.name, "columns": columns}},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _write_preference_dataset(
        dataset_file: Path, dataset_info_file: Path, records: list[PreferenceRecord]
    ) -> None:
        rows = [
            {"instruction": record.instruction, "chosen": record.chosen, "rejected": record.rejected}
            for record in records
        ]
        dataset_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        dataset_info_file.write_text(
            json.dumps(
                {
                    "user_data": {
                        "file_name": dataset_file.name,
                        "ranking": True,
                        "columns": {
                            "prompt": "instruction",
                            "chosen": "chosen",
                            "rejected": "rejected",
                        },
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ---- config ---- #
    @staticmethod
    def _build_config(
        experiment: Experiment,
        model: ModelOption,
        dataset_dir: Path,
        output_dir: Path,
        sample_count: int,
        validation_enabled: bool = False,
    ) -> dict[str, object]:
        params = experiment.params
        precision = select_precision(detect_device())
        config: dict[str, object] = {
            "model_name_or_path": model.local_path or model.name,
            "trust_remote_code": True,
            "stage": experiment.method,
            "do_train": True,
            "finetuning_type": "lora",
            "lora_rank": params.lora_rank,
            "lora_target": "all",
            "dataset_dir": str(dataset_dir),
            "dataset": "user_data",
            "template": model.lf_template,
            "cutoff_len": 2048,
            "max_samples": sample_count,
            "preprocessing_num_workers": 4,
            "dataloader_num_workers": 0,
            "output_dir": str(output_dir),
            "logging_steps": 1,
            "save_steps": 50,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "save_only_model": False,
            "report_to": "none",
            "per_device_train_batch_size": params.batch_size,
            "gradient_accumulation_steps": params.grad_accum,
            "learning_rate": params.learning_rate,
            "num_train_epochs": float(params.epochs),
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "bf16": precision["bf16"],
            "fp16": precision["fp16"],
            "ddp_timeout": 180000000,
            "resume_from_checkpoint": None,
        }
        if experiment.method == "dpo":
            config["pref_beta"] = params.beta
            config["pref_loss"] = "sigmoid"

        if validation_enabled:
            config.update(
                _validation_config(
                    sample_count=sample_count,
                    batch_size=params.batch_size,
                    grad_accum=params.grad_accum,
                )
            )
        return config


def _steps_per_epoch(sample_count: int, batch_size: int, grad_accum: int) -> int:
    """每个 epoch 的优化步数 = ceil(训练样本数 / (batch × 梯度累积))，至少 1。

    训练样本数已扣除验证集占比，让验证点按 epoch 边界落点。
    """
    train_samples = max(1, math.ceil(sample_count * (1 - VALIDATION_RATIO)))
    effective_batch = max(1, batch_size * grad_accum)
    return max(1, math.ceil(train_samples / effective_batch))


def _validation_config(sample_count: int, batch_size: int, grad_accum: int) -> dict[str, object]:
    """训练过程验证配置：切验证集，每个 epoch 评估一次，记录 eval_loss 曲线。"""
    interval = _steps_per_epoch(sample_count, batch_size, grad_accum)
    return {
        "val_size": VALIDATION_RATIO,
        "eval_strategy": "steps",
        "eval_steps": interval,
        "per_device_eval_batch_size": batch_size,
    }
