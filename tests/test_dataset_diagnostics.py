from __future__ import annotations

import unittest

from local_trainer.dataset_diagnostics import (
    LONG_OUTPUT_CHARS,
    diagnose_alpaca_rows,
    diagnose_dpo_rows,
)


def _row(instruction: str, output: str) -> dict:
    return {"instruction": instruction, "output": output, "input": "", "system": None}


def _pref(instruction: str, chosen: str, rejected: str) -> dict:
    return {"instruction": instruction, "chosen": chosen, "rejected": rejected}


def _levels(cards) -> list[str]:
    return [c.level for c in cards]


def _titles(cards) -> list[str]:
    return [c.title for c in cards]


class AlpacaDiagnosticsTests(unittest.TestCase):
    def test_clean_dataset_returns_ok_card(self) -> None:
        rows = [
            _row(f"客户问题 {i}：这个流程怎么走？", f"流程是这样：第一步打开后台，第二步选择对应的菜单项，{i}")
            for i in range(40)
        ]

        cards = diagnose_alpaca_rows(rows)

        self.assertEqual(_levels(cards), ["ok"])
        self.assertIn("没问题", cards[0].title)

    def test_too_few_samples_yields_error(self) -> None:
        rows = [_row(f"问题{i}", "这是一段足够长的回答示例内容。") for i in range(5)]

        cards = diagnose_alpaca_rows(rows)

        self.assertEqual(cards[0].level, "error")
        self.assertIn("5", cards[0].title)

    def test_few_samples_yields_warn(self) -> None:
        rows = [_row(f"问题{i}", "这是一段足够长的回答示例内容。") for i in range(20)]

        cards = diagnose_alpaca_rows(rows)

        # 至少有一张 warn 卡说明数据量
        self.assertIn("warn", _levels(cards))
        first = cards[0]
        self.assertEqual(first.level, "warn")
        self.assertIn("20", first.title)

    def test_detects_short_output(self) -> None:
        rows = [_row(f"客户问题 {i}：流程是怎样的？", "这是一段足够长的回答示例内容。") for i in range(40)]
        rows[3]["output"] = "好"
        rows[7]["output"] = "嗯"

        cards = diagnose_alpaca_rows(rows)

        titles = " ".join(_titles(cards))
        self.assertIn("非常短", titles)
        # 行号取 +1，第 4 行 -> 5，第 8 行 -> 9（含表头偏移）
        evidence = " ".join(c.evidence or "" for c in cards)
        self.assertIn("5", evidence)
        self.assertIn("9", evidence)

    def test_detects_duplicate_instruction(self) -> None:
        rows = [
            _row("怎么退货？", f"退货步骤详细介绍 {i}：先到我的订单页选择该商品。")
            for i in range(15)
        ] + [
            _row(f"其他不同的问题 {i} 询问内容", f"这是足够长的不同回答内容 {i} 示例信息。")
            for i in range(20)
        ]

        cards = diagnose_alpaca_rows(rows)

        self.assertTrue(any("重复" in c.title for c in cards))

    def test_detects_long_output(self) -> None:
        too_long = "啊" * (LONG_OUTPUT_CHARS + 50)
        rows = [_row(f"问题{i}", "这是一段足够长的回答示例内容。") for i in range(40)]
        rows[10]["output"] = too_long

        cards = diagnose_alpaca_rows(rows)

        self.assertTrue(any("过长" in c.title for c in cards))

    def test_detects_boilerplate(self) -> None:
        rows = [_row(f"问题{i}", "这是一段足够长的回答示例内容。") for i in range(40)]
        rows[5]["output"] = "作为AI助手，我无法判断你的具体场景，但是大致可以这样回答。"

        cards = diagnose_alpaca_rows(rows)

        self.assertTrue(any("套话" in c.title for c in cards))

    def test_detects_short_average_length(self) -> None:
        rows = [_row(f"问题{i}", "好的") for i in range(40)]

        cards = diagnose_alpaca_rows(rows)

        titles = " ".join(_titles(cards))
        self.assertIn("信息量偏少", titles)


class DpoDiagnosticsTests(unittest.TestCase):
    def test_detects_identical_chosen_rejected_as_error(self) -> None:
        rows = [_pref(f"问题{i}", "好的我来帮你处理", "好的我来帮你处理") for i in range(25)]

        cards = diagnose_dpo_rows(rows)

        self.assertEqual(cards[0].level, "error")
        self.assertIn("完全一样", cards[0].title)

    def test_clean_dpo_returns_ok(self) -> None:
        rows = [
            _pref(
                f"客户问题{i}：怎么处理这个问题？",
                "我们会按流程优先安排，给你具体的处理预计时间。",
                "不知道，你自己想办法吧。",
            )
            for i in range(25)
        ]

        cards = diagnose_dpo_rows(rows)

        self.assertEqual(cards[0].level, "ok")

    def test_too_few_dpo_yields_warn(self) -> None:
        rows = [
            _pref(f"问题{i}", "我们会优先帮你处理这件事。", "不知道，不归我管。")
            for i in range(8)
        ]

        cards = diagnose_dpo_rows(rows)

        self.assertEqual(cards[0].level, "warn")
        self.assertIn("8", cards[0].title)


if __name__ == "__main__":
    unittest.main()
