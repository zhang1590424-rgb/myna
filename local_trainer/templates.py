"""Starter sample datasets and training presets.

The model catalog now lives in model_registry.py. This module keeps only the
downloadable sample datasets (so users have something to try) and the three
training presets (fast / standard / fine) expressed as ExperimentParams.
"""
from __future__ import annotations

import csv
import io

from .domain import ExperimentParams, TemplatePreset, TrainingPreset


TEMPLATES: list[TemplatePreset] = [
    TemplatePreset(
        id="customer_service",
        title="客服话术",
        description="让 AI 用你们家的口吻回答顾客问题",
        sample_filename="customer-service.csv",
        starter_prompt="顾客问怎么退货，怎么回？",
        sample_rows=[
            {"question": "顾客问怎么退货，怎么回？", "answer": "亲，退货很简单。您把订单号发我，我马上帮您看是否符合 7 天无理由，并安排下一步。"},
            {"question": "物流一直没动，怎么回复？", "answer": "亲，我先帮您查物流状态。如果超过 24 小时没有更新，我会马上联系快递并同步处理结果。"},
            {"question": "顾客嫌价格贵，怎么回？", "answer": "理解您的顾虑。这个价格包含质检和售后保障，如果您关注性价比，我可以帮您对比当前最合适的套餐。"},
        ],
    ),
    TemplatePreset(
        id="roleplay",
        title="角色扮演",
        description="让 AI 扮演特定人设或角色说话",
        sample_filename="roleplay.csv",
        starter_prompt="用温柔但坚定的语气鼓励我今天继续写作。",
        sample_rows=[
            {"question": "我今天不想写了。", "answer": "可以慢一点，但别停下。先写三句话，写完再决定要不要继续。"},
            {"question": "我觉得自己写得很差。", "answer": "差的初稿也比空白更接近成品。把判断留到明天，今天只负责把它写出来。"},
        ],
    ),
    TemplatePreset(
        id="rewrite",
        title="文本改写",
        description="按你的规则改写、润色、转换文本",
        sample_filename="rewrite.csv",
        starter_prompt="把这句话改得更适合产品公告：这个功能已经上线了。",
        sample_rows=[
            {"question": "把这句话改得更适合产品公告：这个功能已经上线了。", "answer": "新功能现已上线，欢迎在最新版本中体验。"},
            {"question": "把这句话改得更礼貌：你填错了。", "answer": "这里的信息可能需要再确认一下，建议重新检查后提交。"},
        ],
    ),
    TemplatePreset(
        id="knowledge_qa",
        title="知识问答",
        description="让 AI 学会你领域里的专业知识",
        sample_filename="knowledge-qa.csv",
        starter_prompt="什么是本产品的免费试用规则？",
        sample_rows=[
            {"question": "什么是本产品的免费试用规则？", "answer": "新用户可免费试用 7 天，试用期内可以体验核心功能，但不包含批量导出。"},
            {"question": "试用期结束后数据还在吗？", "answer": "数据会保留 30 天。您可以在保留期内升级或导出重要内容。"},
        ],
    ),
    TemplatePreset(
        id="preference",
        title="偏好对（DPO）",
        description="给同一个问题准备“更好”和“更差”两种回答，用于偏好优化",
        sample_filename="preference-pairs.csv",
        starter_prompt="帮我把这条通知写得更简洁。",
        sample_rows=[
            {
                "instruction": "帮我把这条通知写得更简洁。",
                "chosen": "系统将于今晚 22:00 维护，预计 1 小时，期间暂停服务。",
                "rejected": "尊敬的各位用户大家好，我们将在今天晚上的时间进行系统维护工作，请大家知悉。",
            },
        ],
    ),
]


def get_templates() -> list[TemplatePreset]:
    return TEMPLATES


def get_template(template_id: str) -> TemplatePreset:
    for template in TEMPLATES:
        if template.id == template_id:
            return template
    raise KeyError(template_id)


TRAINING_PRESETS: list[TrainingPreset] = [
    TrainingPreset(
        id="fast",
        title="快速",
        description="最快跑通一遍，先看效果。轮数少、参数小。",
        params=ExperimentParams(epochs=1, learning_rate=0.0002, lora_rank=8, batch_size=2),
    ),
    TrainingPreset(
        id="standard",
        title="标准",
        description="速度和效果平衡，适合大多数情况。",
        recommended=True,
        params=ExperimentParams(epochs=3, learning_rate=0.0002, lora_rank=8, batch_size=2),
    ),
    TrainingPreset(
        id="fine",
        title="精细",
        description="学得更充分，耗时更长，数据多时更合适。",
        params=ExperimentParams(epochs=5, learning_rate=0.0001, lora_rank=16, batch_size=2),
    ),
]


def get_training_presets() -> list[TrainingPreset]:
    return TRAINING_PRESETS


def get_training_preset(preset_id: str) -> TrainingPreset:
    for preset in TRAINING_PRESETS:
        if preset.id == preset_id:
            return preset
    raise KeyError(preset_id)


def sample_csv_for_template(template_id: str) -> str:
    template = get_template(template_id)
    fieldnames = list(template.sample_rows[0].keys()) if template.sample_rows else ["question", "answer"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(template.sample_rows)
    return output.getvalue()
