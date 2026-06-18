from __future__ import annotations

import unittest

from local_trainer.domain import ModelDownloadStatus
from local_trainer.downloader import ModelDownloader
from local_trainer.model_registry import get_model_catalog
from local_trainer.paths import model_dir_for_repo
from local_trainer.templates import get_training_preset, get_training_presets


class ModelCatalogTests(unittest.TestCase):
    def test_every_model_has_repo_id(self) -> None:
        for model in get_model_catalog():
            self.assertTrue(model.repo_id, f"{model.id} missing repo_id")
            self.assertTrue(model.download_size_label, f"{model.id} missing download size")

    def test_one_model_is_recommended(self) -> None:
        recommended = [m for m in get_model_catalog() if m.recommended]
        self.assertEqual(len(recommended), 1)


class ModelDirLayoutTests(unittest.TestCase):
    def test_repo_dots_become_underscores(self) -> None:
        path = model_dir_for_repo("Qwen/Qwen3.5-2B")
        self.assertTrue(str(path).endswith("Qwen/Qwen3___5-2B"))


class TrainingPresetTests(unittest.TestCase):
    def test_three_presets_with_one_recommended(self) -> None:
        presets = get_training_presets()
        self.assertEqual({p.id for p in presets}, {"fast", "standard", "fine"})
        self.assertEqual(len([p for p in presets if p.recommended]), 1)

    def test_preset_params_are_ordered_by_effort(self) -> None:
        fast = get_training_preset("fast")
        fine = get_training_preset("fine")
        self.assertLess(fast.params.epochs, fine.params.epochs)
        self.assertLessEqual(fast.params.lora_rank, fine.params.lora_rank)

    def test_unknown_preset_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_training_preset("nope")


class DownloaderStatusTests(unittest.TestCase):
    def test_unknown_model_is_idle(self) -> None:
        downloader = ModelDownloader()
        status = downloader.status_for("anything")
        self.assertIsInstance(status, ModelDownloadStatus)
        self.assertEqual(status.state, "idle")
        self.assertEqual(status.progress, 0)


if __name__ == "__main__":
    unittest.main()
