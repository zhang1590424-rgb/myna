from __future__ import annotations

import json
import unittest

from local_trainer.data_validation import DatasetValidationError, parse_dataset_bytes


class DatasetValidationTests(unittest.TestCase):
    def test_csv_with_chinese_headers_is_normalized(self) -> None:
        content = "问题,回答\n怎么退货？,把订单号发我。\n缺回答,\n物流慢？,我来帮您催。".encode()

        parsed = parse_dataset_bytes("customer.csv", content)

        self.assertEqual(parsed.valid_count, 2)
        self.assertEqual(parsed.skipped_count, 1)
        self.assertEqual(parsed.records[0].instruction, "怎么退货？")
        self.assertEqual(parsed.records[0].output, "把订单号发我。")
        self.assertIn("跳过 1 行", parsed.human_summary)

    def test_json_alpaca_format_is_supported(self) -> None:
        payload = [
            {"instruction": "改写这句话", "input": "上线了", "output": "新功能现已上线。"},
            {"instruction": "补充语气", "output": "请您再确认一下。"},
        ]

        parsed = parse_dataset_bytes("rewrite.json", json.dumps(payload).encode())

        self.assertEqual(parsed.valid_count, 2)
        self.assertEqual(parsed.records[0].input, "上线了")
        self.assertEqual(parsed.source_format, "json")

    def test_missing_question_and_answer_headers_get_human_error(self) -> None:
        with self.assertRaises(DatasetValidationError) as context:
            parse_dataset_bytes("bad.csv", "name,value\nA,B\n".encode())

        self.assertIn("问题", context.exception.message)


if __name__ == "__main__":
    unittest.main()
