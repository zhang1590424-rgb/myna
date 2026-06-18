from __future__ import annotations

import csv
import io

from .domain import ModelOption, TemplatePreset, TrainingDefaults, TrainingPreset, TrainingSettings
from .paths import LOCAL_QWEN_0_5B, model_dir_for_repo


TEMPLATES: list[TemplatePreset] = [
    TemplatePreset(
        id="customer_service",
        title="客服话术",
        description="让 AI 用你们家的口吻回答顾客问题",
        goal_label="客服话术 AI",
        sample_filename="customer-service.csv",
        starter_prompt="顾客问怎么退货，怎么回？",
        system_prompt="你是一个耐心、友好、动作明确的电商客服。",
        defaults=TrainingDefaults(epochs=3, learning_rate=0.0002, lora_rank=8, batch_size=2),
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
        goal_label="角色扮演 AI",
        sample_filename="roleplay.csv",
        starter_prompt="用温柔但坚定的语气鼓励我今天继续写作。",
        system_prompt="你按照用户设定的人设说话，语气稳定，避免跳出角色。",
        sample_rows=[
            {"question": "我今天不想写了。", "answer": "可以慢一点，但别停下。先写三句话，写完再决定要不要继续。"},
            {"question": "我觉得自己写得很差。", "answer": "差的初稿也比空白更接近成品。把判断留到明天，今天只负责把它写出来。"},
        ],
    ),
    TemplatePreset(
        id="rewrite",
        title="文本改写",
        description="按你的规则改写、润色、转换文本",
        goal_label="文本改写 AI",
        sample_filename="rewrite.csv",
        starter_prompt="把这句话改得更适合产品公告：这个功能已经上线了。",
        system_prompt="你根据用户要求改写文本，保留原意，输出清楚、克制。",
        sample_rows=[
            {"question": "把这句话改得更适合产品公告：这个功能已经上线了。", "answer": "新功能现已上线，欢迎在最新版本中体验。"},
            {"question": "把这句话改得更礼貌：你填错了。", "answer": "这里的信息可能需要再确认一下，建议重新检查后提交。"},
        ],
    ),
    TemplatePreset(
        id="knowledge_qa",
        title="知识问答",
        description="让 AI 学会你领域里的专业知识",
        goal_label="知识问答 AI",
        sample_filename="knowledge-qa.csv",
        starter_prompt="什么是本产品的免费试用规则？",
        system_prompt="你基于给定知识回答问题，不确定时说明需要人工确认。",
        sample_rows=[
            {"question": "什么是本产品的免费试用规则？", "answer": "新用户可免费试用 7 天，试用期内可以体验核心功能，但不包含批量导出。"},
            {"question": "试用期结束后数据还在吗？", "answer": "数据会保留 30 天。您可以在保留期内升级或导出重要内容。"},
        ],
    ),
    TemplatePreset(
        id="custom",
        title="自定义",
        description="自己定义训练目标和数据",
        goal_label="自定义 AI",
        sample_filename="custom.csv",
        starter_prompt="请根据我的数据回答。",
        system_prompt="你学习用户提供的数据风格和知识，回答要清楚、有帮助。",
        sample_rows=[
            {"question": "你要学会什么？", "answer": "我会根据你提供的问题和回答，学习特定风格或知识。"},
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


def get_model_catalog() -> list[ModelOption]:
    qwen_05_repo = "Qwen/Qwen2.5-0.5B-Instruct"
    qwen_05_dir = model_dir_for_repo(qwen_05_repo)
    # Keep backward compat: the originally cached 0.5B sits at LOCAL_QWEN_0_5B.
    qwen_05_path = qwen_05_dir if _model_present(qwen_05_dir) else LOCAL_QWEN_0_5B
    qwen_05_ok = _model_present(qwen_05_path)

    qwen_15_repo = "Qwen/Qwen2.5-1.5B-Instruct"
    qwen_15_dir = model_dir_for_repo(qwen_15_repo)
    qwen_15_ok = _model_present(qwen_15_dir)

    qwen_3_repo = "Qwen/Qwen2.5-3B-Instruct"
    qwen_3_dir = model_dir_for_repo(qwen_3_repo)
    qwen_3_ok = _model_present(qwen_3_dir)

    return [
        ModelOption(
            id="qwen2.5-0.5b-instruct-local",
            name="Qwen2.5-0.5B-Instruct",
            size_label="轻量",
            parameter_count="0.5B",
            local_path=str(qwen_05_path) if qwen_05_ok else None,
            available=qwen_05_ok,
            recommended=True,
            note="适合第一次体验，速度和成功率优先。",
            repo_id=qwen_05_repo,
            download_size_label="约 1 GB",
        ),
        ModelOption(
            id="qwen2.5-1.5b-instruct",
            name="Qwen2.5-1.5B-Instruct",
            size_label="均衡",
            parameter_count="1.5B",
            local_path=str(qwen_15_dir) if qwen_15_ok else None,
            available=qwen_15_ok,
            note="效果更稳，训练更慢。需先下载，约 3 GB。",
            repo_id=qwen_15_repo,
            download_size_label="约 3 GB",
        ),
        ModelOption(
            id="qwen2.5-3b-instruct",
            name="Qwen2.5-3B-Instruct",
            size_label="进阶",
            parameter_count="3B",
            local_path=str(qwen_3_dir) if qwen_3_ok else None,
            available=qwen_3_ok,
            note="适合内存更大的 Mac，需先下载，约 6 GB。",
            repo_id=qwen_3_repo,
            download_size_label="约 6 GB",
        ),
    ]


def _model_present(model_dir) -> bool:
    """A model is usable once its config.json and a weight file are on disk."""
    from pathlib import Path

    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        return False
    return any(model_dir.glob("*.safetensors")) or (model_dir / "pytorch_model.bin").exists()


def get_model(model_id: str) -> ModelOption:
    for model in get_model_catalog():
        if model.id == model_id:
            return model
    raise KeyError(model_id)


TRAINING_PRESETS: list[TrainingPreset] = [
    TrainingPreset(
        id="fast",
        title="快速",
        description="最快跑通一遍，先看效果。轮数少、参数小。",
        settings=TrainingSettings(epochs=1, learning_rate=0.0002, lora_rank=8, batch_size=2),
    ),
    TrainingPreset(
        id="standard",
        title="标准",
        description="速度和效果平衡，适合大多数情况。",
        recommended=True,
        settings=TrainingSettings(epochs=3, learning_rate=0.0002, lora_rank=8, batch_size=2),
    ),
    TrainingPreset(
        id="fine",
        title="精细",
        description="学得更充分，耗时更长，数据多时更合适。",
        settings=TrainingSettings(epochs=5, learning_rate=0.0001, lora_rank=16, batch_size=2),
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
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["question", "answer"])
    writer.writeheader()
    writer.writerows(template.sample_rows)
    return output.getvalue()
