from __future__ import annotations

import json
import tempfile
import unittest

import yaml

from local_trainer.domain import DatasetRecord, TrainingJob, TrainingSettings
from local_trainer.engine import utc_now
from local_trainer.llamafactory import LlamaFactoryConfigBuilder
from local_trainer.templates import get_model_catalog, get_template


class LlamaFactoryConfigTests(unittest.TestCase):
    def test_builder_writes_dataset_info_and_train_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            job = TrainingJob(
                id="job-test",
                template_id="customer_service",
                dataset_id="dataset-test",
                model_id="qwen2.5-0.5b-instruct-local",
                dataset_count=1,
                settings=TrainingSettings(epochs=2, learning_rate=0.0001, lora_rank=4, batch_size=1),
                created_at=utc_now(),
            )
            records = [DatasetRecord(instruction="怎么退货？", output="把订单号发我。")]
            model = get_model_catalog()[0]
            prepared = LlamaFactoryConfigBuilder(runs_dir=temp_dir).prepare(
                job=job,
                records=records,
                template=get_template("customer_service"),
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


if __name__ == "__main__":
    unittest.main()
