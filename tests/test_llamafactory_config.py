from __future__ import annotations

import json
import tempfile
import unittest

import yaml

from local_trainer.domain import (
    DatasetRecord,
    Experiment,
    ExperimentParams,
    PreferenceRecord,
)
from local_trainer.engine import utc_now
from local_trainer.llamafactory import LlamaFactoryConfigBuilder
from local_trainer.model_registry import get_model_catalog


def _experiment(method: str = "sft", **param_overrides: object) -> Experiment:
    return Experiment(
        id=f"exp-{method}",
        name=f"测试-{method}",
        method=method,  # type: ignore[arg-type]
        model_id="qwen3.5-0.8b",
        dataset_id="dataset-test",
        dataset_count=1,
        params=ExperimentParams(**param_overrides) if param_overrides else ExperimentParams(),
        created_at=utc_now(),
    )


class LlamaFactorySftConfigTests(unittest.TestCase):
    def test_builder_writes_dataset_info_and_train_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment = _experiment("sft", epochs=2, learning_rate=0.0001, lora_rank=4, batch_size=1)
            records = [DatasetRecord(instruction="怎么退货？", output="把订单号发我。")]
            model = get_model_catalog()[0]
            prepared = LlamaFactoryConfigBuilder(runs_dir=temp_dir).prepare(
                experiment=experiment,
                records=records,
                model=model,
            )

            dataset_info = json.loads(prepared.dataset_info_file.read_text(encoding="utf-8"))
            train_config = yaml.safe_load(prepared.config_file.read_text(encoding="utf-8"))

            self.assertEqual(dataset_info["user_data"]["file_name"], "user_data.json")
            self.assertEqual(train_config["stage"], "sft")
            self.assertEqual(train_config["finetuning_type"], "lora")
            self.assertEqual(train_config["lora_rank"], 4)
            self.assertEqual(train_config["num_train_epochs"], 2.0)
            self.assertTrue(prepared.command[0].endswith("llamafactory-cli"))
            self.assertEqual(prepared.command[1], "train")


class LlamaFactoryDpoConfigTests(unittest.TestCase):
    def test_dpo_writes_ranking_dataset_and_pref_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment = _experiment("dpo", beta=0.2)
            records = [
                PreferenceRecord(instruction="改简洁", chosen="今晚维护 1 小时。", rejected="尊敬的各位用户……")
            ]
            model = get_model_catalog()[0]
            prepared = LlamaFactoryConfigBuilder(runs_dir=temp_dir).prepare(
                experiment=experiment,
                records=records,
                model=model,
            )

            dataset_info = json.loads(prepared.dataset_info_file.read_text(encoding="utf-8"))
            train_config = yaml.safe_load(prepared.config_file.read_text(encoding="utf-8"))

            self.assertTrue(dataset_info["user_data"]["ranking"])
            self.assertEqual(dataset_info["user_data"]["columns"]["chosen"], "chosen")
            self.assertEqual(train_config["stage"], "dpo")
            self.assertEqual(train_config["pref_beta"], 0.2)
            self.assertEqual(train_config["pref_loss"], "sigmoid")


if __name__ == "__main__":
    unittest.main()
