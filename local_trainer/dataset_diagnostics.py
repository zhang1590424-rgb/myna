"""上传数据时的硬规则诊断（Phase 1）。

诊断和格式校验是两件事：
- 格式校验（data_validation.py）= 数据能不能读，失败则拒绝上传。
- 诊断（本文件）= 数据有没有质量风险，仅提示，不阻塞。

诊断输出复用 DiagnosticCard，确保前端和实验详情页用同一套渲染。
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from .domain import DiagnosticAction, DiagnosticCard

# --------------------------------------------------------------------------- #
# 阈值（集中放一起，方便后续观察用户数据后调）
# --------------------------------------------------------------------------- #
SFT_MIN_SAMPLES_RED = 10
SFT_MIN_SAMPLES_YELLOW = 30
DPO_MIN_SAMPLES_YELLOW = 20

SHORT_OUTPUT_CHARS = 5          # 小于此值视为「极短」
LONG_OUTPUT_CHARS = 1024        # 大于此值视为「可能被截断」
SHORT_AVG_CHARS = 25            # 平均长度低于此值，整体回答太短
INVERSE_RATIO = 0.2             # output 长度 < instruction 长度 * 0.2 视为可能列反

DUPLICATE_RATE_YELLOW = 0.2     # 重复率超过 20% 提示

# 常见 AI 套话片段（命中即提示）
BOILERPLATE_PHRASES = (
    "作为AI助手",
    "作为ai助手",
    "作为一个AI",
    "作为一个ai",
    "作为人工智能",
    "作为一个语言模型",
    "作为一个大语言模型",
    "我是一个语言模型",
    "我是一个AI",
    "我是一个ai",
    "我没有感情",
    "我无法",
    "抱歉，我不能",
)


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def diagnose_alpaca_rows(rows: list[dict[str, Any]]) -> list[DiagnosticCard]:
    """对 SFT（问答）数据做诊断。rows 是 DatasetRecord.model_dump() 之后的列表。"""
    cards: list[DiagnosticCard] = []
    if not rows:
        return cards

    cards += _check_sample_size(len(rows), method="sft")
    cards += _check_duplicate_instructions(rows)
    cards += _check_short_outputs(rows)
    cards += _check_long_outputs(rows)
    cards += _check_avg_output_length(rows)
    cards += _check_length_variance(rows)
    cards += _check_boilerplate(rows)
    cards += _check_inverse(rows)

    if not cards:
        cards.append(_ok_card_alpaca(len(rows)))
    return _sort_cards(cards)


def diagnose_dpo_rows(rows: list[dict[str, Any]]) -> list[DiagnosticCard]:
    """对 DPO（偏好）数据做诊断。"""
    cards: list[DiagnosticCard] = []
    if not rows:
        return cards

    cards += _check_sample_size(len(rows), method="dpo")
    cards += _check_dpo_identical(rows)
    cards += _check_dpo_short(rows)
    cards += _check_duplicate_instructions(rows)

    if not cards:
        cards.append(_ok_card_dpo(len(rows)))
    return _sort_cards(cards)


# --------------------------------------------------------------------------- #
# SFT 规则
# --------------------------------------------------------------------------- #
def _check_sample_size(n: int, method: str) -> list[DiagnosticCard]:
    if method == "sft":
        if n < SFT_MIN_SAMPLES_RED:
            return [
                _card(
                    level="error",
                    title=f"只有 {n} 条数据，训练后大概率看不到变化",
                    observation=f"这份数据集只有 {n} 条问答。",
                    interpretation="样本太少时，模型几乎学不到稳定模式，训练前后的回答可能没有区别。",
                    suggestion=f"建议补到 {SFT_MIN_SAMPLES_YELLOW} 条以上再开始训练。",
                    next_step="先准备 20 到 30 个真实问题和你希望的回答，再上传一次。",
                    evidence=f"当前共 {n} 条样本，少于 {SFT_MIN_SAMPLES_RED} 条。",
                    rank=10,
                )
            ]
        if n < SFT_MIN_SAMPLES_YELLOW:
            return [
                _card(
                    level="warn",
                    title=f"只有 {n} 条数据，效果可能不稳定",
                    observation=f"这份数据集有 {n} 条问答。",
                    interpretation="样本偏少时，训练后容易只记住几条例子，换个问法可能就不灵。",
                    suggestion=f"再补到 {SFT_MIN_SAMPLES_YELLOW} 条以上效果会明显。",
                    next_step="挑几类典型场景，每类多写几条不同问法的真实问题。",
                    evidence=f"当前共 {n} 条样本，建议至少 {SFT_MIN_SAMPLES_YELLOW} 条。",
                    rank=40,
                )
            ]
        return []

    # DPO
    if n < DPO_MIN_SAMPLES_YELLOW:
        return [
            _card(
                level="warn",
                title=f"只有 {n} 组偏好对，可能不够学到稳定偏好",
                observation=f"这份偏好数据集有 {n} 组样本。",
                interpretation="DPO 需要看到足够多「好 vs 差」的对比，模型才能学到偏好方向。",
                suggestion=f"建议补到 {DPO_MIN_SAMPLES_YELLOW} 组以上再训练。",
                next_step="围绕你想纠正的回答风格，多准备一些好回答和差回答的对照。",
                evidence=f"当前共 {n} 组偏好对，建议至少 {DPO_MIN_SAMPLES_YELLOW} 组。",
                rank=40,
            )
        ]
    return []


def _check_duplicate_instructions(rows: list[dict[str, Any]]) -> list[DiagnosticCard]:
    instructions = [_norm(row.get("instruction")) for row in rows]
    instructions = [q for q in instructions if q]
    if not instructions:
        return []
    counter = Counter(instructions)
    duplicate_lines: list[int] = []
    for idx, q in enumerate(instructions, start=1):
        if counter[q] > 1:
            duplicate_lines.append(idx)
    if not duplicate_lines:
        return []

    rate = len(duplicate_lines) / len(instructions)
    unique = len(set(instructions))
    if rate < DUPLICATE_RATE_YELLOW:
        # 仅 1 到 2 处轻微重复时也提示，但等级降到一般
        if len(duplicate_lines) < 2:
            return []
    return [
        _card(
            level="warn",
            title=f"有 {len(duplicate_lines)} 行问题重复，模型可能只会背题",
            observation="不同行的「问题」内容几乎完全一样。",
            interpretation="重复样本会让模型偏向背住这几种问法，换个说法就可能答不对。",
            suggestion="保留最典型的一条，把其他重复行换成不同表达的真实问题。",
            next_step="去整理数据后重新上传，或在本页直接更新数据。",
            evidence=_format_line_evidence(
                duplicate_lines,
                tail=f"（{unique} 种不同问题 / 共 {len(instructions)} 条）",
            ),
            action=DiagnosticAction(label="去整理数据", action="goto_data"),
            rank=50,
        )
    ]


def _check_short_outputs(rows: list[dict[str, Any]]) -> list[DiagnosticCard]:
    short_lines = [
        i for i, row in enumerate(rows, start=1) if 0 < len(_get_output(row)) < SHORT_OUTPUT_CHARS
    ]
    if not short_lines:
        return []
    return [
        _card(
            level="warn",
            title=f"有 {len(short_lines)} 行回答非常短，训练信号偏弱",
            observation=f"这些行的回答只有不到 {SHORT_OUTPUT_CHARS} 个字。",
            interpretation="过短的回答几乎不带语义信号，模型很难学到完整的回答方式。",
            suggestion="把这些行的回答写完整一点，或干脆删掉。",
            next_step="检查列出的行号，补上完整回答后重新上传。",
            evidence=_format_line_evidence(short_lines),
            rank=55,
        )
    ]


def _check_long_outputs(rows: list[dict[str, Any]]) -> list[DiagnosticCard]:
    long_lines = [
        i for i, row in enumerate(rows, start=1) if len(_get_output(row)) > LONG_OUTPUT_CHARS
    ]
    if not long_lines:
        return []
    return [
        _card(
            level="warn",
            title=f"有 {len(long_lines)} 行回答过长，训练时可能被截断",
            observation=f"这些行的回答超过 {LONG_OUTPUT_CHARS} 字。",
            interpretation="训练时超出长度上限的内容会被截掉，模型只能学到前半段。",
            suggestion="把回答精简到 1024 字以内，或拆成多条样本。",
            next_step="检查列出的行号，精简后重新上传。",
            evidence=_format_line_evidence(long_lines),
            rank=60,
        )
    ]


def _check_avg_output_length(rows: list[dict[str, Any]]) -> list[DiagnosticCard]:
    outputs = [_get_output(row) for row in rows]
    if not outputs:
        return []
    avg = sum(len(o) for o in outputs) / len(outputs)
    if avg >= SHORT_AVG_CHARS:
        return []
    return [
        _card(
            level="warn",
            title=f"回答平均只有 {avg:.0f} 字，整体信息量偏少",
            observation="所有回答加起来的平均长度偏短。",
            interpretation="短回答能教格式，但教不会语气、判断步骤和完整表达。",
            suggestion="把典型回答写完整些，加上原因、步骤或边界条件。",
            next_step="挑几条最典型的样本，补成完整答案后重新上传。",
            evidence=f"全部 {len(outputs)} 条回答的平均长度为 {avg:.1f} 字。",
            rank=65,
        )
    ]


def _check_length_variance(rows: list[dict[str, Any]]) -> list[DiagnosticCard]:
    outputs = [_get_output(row) for row in rows]
    lengths = [len(o) for o in outputs if o]
    if len(lengths) < 5:
        return []
    mean = sum(lengths) / len(lengths)
    if mean <= 0:
        return []
    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    std = variance ** 0.5
    if std <= mean:
        return []
    return [
        _card(
            level="warn",
            title="回答长度差异很大，风格可能不统一",
            observation="有的回答很短，有的回答很长。",
            interpretation="风格忽长忽短时，模型不容易学到稳定的回答模式，训练后表现也会忽好忽坏。",
            suggestion="尽量让同类问题的回答长度和结构保持一致。",
            next_step="检查回答最长和最短的几条，调整到接近的风格。",
            evidence=f"平均 {mean:.0f} 字，标准差 {std:.0f}（标准差大于平均值说明波动很大）。",
            rank=70,
        )
    ]


def _check_boilerplate(rows: list[dict[str, Any]]) -> list[DiagnosticCard]:
    hits: list[int] = []
    for idx, row in enumerate(rows, start=1):
        out = _get_output(row)
        for phrase in BOILERPLATE_PHRASES:
            if phrase.lower() in out.lower():
                hits.append(idx)
                break
    if not hits:
        return []
    return [
        _card(
            level="warn",
            title=f"有 {len(hits)} 行回答含通用 AI 套话",
            observation="这些回答里出现了「作为 AI 助手 / 作为一个语言模型」等模板话术。",
            interpretation="套话会让模型继续学这种「假装是 AI」的回答风格，反而盖掉你想训练的特色。",
            suggestion="把套话删掉，只保留你希望模型学的真实回答。",
            next_step="检查列出的行号，去掉开头的套话后重新上传。",
            evidence=_format_line_evidence(hits),
            rank=55,
        )
    ]


def _check_inverse(rows: list[dict[str, Any]]) -> list[DiagnosticCard]:
    """检测「答案比问题短很多」的样本，可能是问答列被填反。"""
    suspects: list[int] = []
    for idx, row in enumerate(rows, start=1):
        q = (row.get("instruction") or "").strip()
        a = _get_output(row)
        if len(q) > 30 and 0 < len(a) < max(int(len(q) * INVERSE_RATIO), 5):
            suspects.append(idx)
    if len(suspects) < max(3, int(0.1 * len(rows))):
        # 偶发不报，普遍存在才提示
        return []
    return [
        _card(
            level="warn",
            title=f"有 {len(suspects)} 行回答比问题短很多，是不是列填反了",
            observation="问题写得很长，回答却特别短。",
            interpretation="少量这样的样本是正常的，但比例偏高时通常是「问题列」和「回答列」放反了。",
            suggestion="打开原文件确认表头，question/answer 不要弄反。",
            next_step="检查列出的行号，确认无误后可以忽略此提示。",
            evidence=_format_line_evidence(suspects),
            rank=75,
        )
    ]


# --------------------------------------------------------------------------- #
# DPO 规则
# --------------------------------------------------------------------------- #
def _check_dpo_identical(rows: list[dict[str, Any]]) -> list[DiagnosticCard]:
    same_lines = [
        i
        for i, row in enumerate(rows, start=1)
        if _norm(row.get("chosen")) and _norm(row.get("chosen")) == _norm(row.get("rejected"))
    ]
    if not same_lines:
        return []
    return [
        _card(
            level="error",
            title=f"有 {len(same_lines)} 行的「好回答」和「差回答」完全一样",
            observation="chosen 和 rejected 两列内容一致。",
            interpretation="DPO 训练需要看到差异，相同内容会让模型完全学不到偏好。",
            suggestion="把这些行的差回答换成你不想要的版本，或者直接删掉。",
            next_step="检查列出的行号，修正后重新上传。",
            evidence=_format_line_evidence(same_lines),
            action=DiagnosticAction(label="去整理数据", action="goto_data"),
            rank=10,
        )
    ]


def _check_dpo_short(rows: list[dict[str, Any]]) -> list[DiagnosticCard]:
    bad_lines: list[int] = []
    for idx, row in enumerate(rows, start=1):
        if 0 < len(_norm(row.get("chosen"))) < SHORT_OUTPUT_CHARS:
            bad_lines.append(idx)
            continue
        if 0 < len(_norm(row.get("rejected"))) < SHORT_OUTPUT_CHARS:
            bad_lines.append(idx)
    if not bad_lines:
        return []
    return [
        _card(
            level="warn",
            title=f"有 {len(bad_lines)} 行偏好回答太短",
            observation=f"这些行的好回答或差回答不到 {SHORT_OUTPUT_CHARS} 个字。",
            interpretation="过短的偏好对差异信号微弱，DPO 训练很难学到方向。",
            suggestion="把回答补完整，让好与差的差异更明显。",
            next_step="检查列出的行号，补全后重新上传。",
            evidence=_format_line_evidence(bad_lines),
            rank=55,
        )
    ]


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _ok_card_alpaca(n: int) -> DiagnosticCard:
    return _card(
        level="ok",
        title="数据看起来没问题，可以直接进入下一步",
        observation=f"识别到 {n} 条有效问答，常规体检没发现明显风险。",
        interpretation="格式、长度、重复度都在合理范围内。",
        next_step="可以进入「新建训练」配置参数。",
        rank=90,
    )


def _ok_card_dpo(n: int) -> DiagnosticCard:
    return _card(
        level="ok",
        title="偏好数据看起来没问题，可以直接进入下一步",
        observation=f"识别到 {n} 组偏好对，常规体检没发现明显风险。",
        interpretation="好回答和差回答的差异、长度都合理。",
        next_step="可以进入「新建训练」选 DPO 方法。",
        rank=90,
    )


def _card(
    *,
    level: str,
    title: str,
    suggestion: str | None = None,
    observation: str | None = None,
    interpretation: str | None = None,
    next_step: str | None = None,
    evidence: str | None = None,
    action: DiagnosticAction | None = None,
    rank: int = 50,
) -> DiagnosticCard:
    return DiagnosticCard(
        level=level,  # type: ignore[arg-type]
        title=title,
        suggestion=suggestion or next_step or "",
        observation=observation,
        interpretation=interpretation,
        next_step=next_step,
        evidence=evidence,
        action=action,
        rank=rank,
    )


def _sort_cards(cards: list[DiagnosticCard]) -> list[DiagnosticCard]:
    level_rank = {"error": 0, "warn": 1, "ok": 2}
    return sorted(cards, key=lambda c: (level_rank.get(c.level, 3), c.rank))


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _get_output(row: dict[str, Any]) -> str:
    val = row.get("output") or row.get("answer") or ""
    return str(val).strip()


def _format_line_evidence(lines: list[int], tail: str = "") -> str:
    """把行号列表压成「第 X、Y、Z 行（共 N 行）」格式。表头算第 1 行，数据从 2 开始。"""
    real_lines = [n + 1 for n in lines]
    if len(real_lines) <= 8:
        joined = "、".join(str(n) for n in real_lines)
        body = f"涉及第 {joined} 行"
    else:
        head = "、".join(str(n) for n in real_lines[:6])
        body = f"涉及第 {head} … 等共 {len(real_lines)} 行"
    if tail:
        body += tail
    return body + "。"
