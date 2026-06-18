from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .domain import DatasetRecord, ModelOption, TemplatePreset, TrainingJob
from .hardware import detect_device, llamafactory_cli, select_precision
from .paths import RUNS_DIR, ensure_runtime_dirs


@dataclass(frozen=True)
class PreparedLlamaFactoryRun:
    run_dir: Path
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
        job: TrainingJob,
        records: list[DatasetRecord],
        template: TemplatePreset,
        model: ModelOption,
    ) -> PreparedLlamaFactoryRun:
        run_dir = self.runs_dir / job.id / "llamafactory"
        dataset_dir = run_dir / "dataset"
        output_dir = run_dir / "output"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_file = dataset_dir / "user_data.json"
        dataset_info_file = dataset_dir / "dataset_info.json"
        config_file = run_dir / "train.yaml"
        precision = select_precision(detect_device())

        has_system = any(record.system for record in records)
        dataset_file.write_text(
            json.dumps(
                [self._row_for(record, has_system) for record in records],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        columns = {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        }
        if has_system:
            columns["system"] = "system"
        dataset_info_file.write_text(
            json.dumps(
                {
                    "user_data": {
                        "file_name": dataset_file.name,
                        "columns": columns,
                    }
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        config = {
            "model_name_or_path": model.local_path or model.name,
            "trust_remote_code": True,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "lora_rank": job.settings.lora_rank,
            "lora_target": "all",
            "dataset_dir": str(dataset_dir),
            "dataset": "user_data",
            "template": "qwen",
            "cutoff_len": 2048,
            "max_samples": len(records),
            "preprocessing_num_workers": 4,
            "dataloader_num_workers": 0,
            "output_dir": str(output_dir),
            "logging_steps": 1,
            "save_steps": 50,
            "plot_loss": True,
            "overwrite_output_dir": True,
            "save_only_model": False,
            "report_to": "none",
            "per_device_train_batch_size": job.settings.batch_size,
            "gradient_accumulation_steps": 8,
            "learning_rate": job.settings.learning_rate,
            "num_train_epochs": float(job.settings.epochs),
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "bf16": precision["bf16"],
            "fp16": precision["fp16"],
            "ddp_timeout": 180000000,
            "resume_from_checkpoint": None,
        }
        config_file.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

        return PreparedLlamaFactoryRun(
            run_dir=run_dir,
            dataset_dir=dataset_dir,
            dataset_file=dataset_file,
            dataset_info_file=dataset_info_file,
            config_file=config_file,
            command=[llamafactory_cli(), "train", str(config_file)],
        )

    @staticmethod
    def _row_for(record: DatasetRecord, has_system: bool) -> dict[str, str]:
        row = {
            "instruction": record.instruction,
            "input": record.input,
            "output": record.output,
        }
        if has_system:
            row["system"] = record.system or ""
        return row
