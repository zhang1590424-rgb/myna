"""Training diagnostics: rule-based analysis after training completes.

The diagnostics are written for first-time local fine-tuning users.  Each
finding explains what the curve shows, what it may mean, and what to try next.
"""
from __future__ import annotations

from .domain import DiagnosticAction, DiagnosticCard, Experiment


def compute_diagnostics(
    exp: Experiment,
    dataset_rows: list[dict] | None = None,
) -> list[DiagnosticCard]:
    """Run all diagnostic rules and return cards."""
    cards: list[DiagnosticCard] = []

    cards += _check_no_loss(exp)
    cards += _check_data_quantity(exp)
    cards += _check_data_quality(exp, dataset_rows)
    cards += _check_initial_loss_anomaly(exp)
    cards += _check_first_step_jump(exp)
    cards += _check_loss_rising(exp)
    cards += _check_classic_overfit(exp)
    cards += _check_memorization_overfit(exp)
    cards += _check_generalization_diminishing(exp)
    cards += _check_validation_stalled(exp)
    cards += _check_validation_better_than_train(exp)
    cards += _check_underfit(exp)
    cards += _check_loss_still_dropping(exp)
    cards += _check_loss_plateau(exp)
    cards += _check_loss_oscillation(exp)
    cards += _check_validation_improving(exp)
    cards += _check_validation_noisy(exp)
    cards += _check_no_validation_split(exp)
    cards += _check_early_steep_drop(exp)
    cards += _check_mid_plateau(exp)
    cards += _check_tail_micro_bounce(exp)
    cards += _check_data_oscillation_combo(exp)
    cards += _check_repetition_strong_drop_combo(exp, dataset_rows)
    cards += _check_short_answer_plateau_combo(exp, dataset_rows)
    cards += _check_dpo_reward_margin(exp)

    if not cards:
        return [_build_ok_card(exp)]
    return _sort_cards(cards)


def compute_live_diagnostics(exp: Experiment) -> list[DiagnosticCard]:
    """Compute lightweight findings that are safe to show during training."""
    cards: list[DiagnosticCard] = []

    cards += _check_initial_loss_anomaly(exp)
    cards += _check_loss_rising(exp, live=True)
    cards += _check_underfit(exp, live=True)
    cards += _check_loss_oscillation(exp, live=True)

    return _sort_cards(cards)


def preflight_data_check(
    dataset_rows: list[dict],
    method: str = "sft",
) -> list[DiagnosticCard]:
    """Pre-training data quality check. Run before starting training."""
    cards: list[DiagnosticCard] = []

    n = len(dataset_rows)
    if n < 15:
        cards.append(
            _card(
                level="warn",
                title=f"数据只有 {n} 条，可能不够模型学出稳定规律",
                observation=f"这次训练集有 {n} 条数据。",
                interpretation="样本太少时，模型更容易记住几条例子，换个问法就不稳定。",
                suggestion="可以先补到 20 条以上，再开始训练。",
                next_step="优先补几类真实问题，每类给出你希望模型学会的回答方式。",
                evidence=f"当前数据集 {n} 条。少于 15 条时，训练结果通常更依赖单条样本。",
                action=DiagnosticAction(label="去补充数据", action="goto_data"),
                topic="data",
                rank=25,
            )
        )

    answers = [
        row.get("output") or row.get("answer") or row.get("chosen") or ""
        for row in dataset_rows
    ]
    if answers:
        avg_len = sum(len(a) for a in answers) / len(answers)
        if avg_len < 25:
            cards.append(
                _card(
                    level="warn",
                    title=f"回答平均只有 {avg_len:.0f} 字，训练信号偏少",
                    observation="训练数据里的回答整体比较短。",
                    interpretation="短回答能教会格式，但很难教会语气、结构和判断过程。",
                    suggestion="可以把典型回答写完整一点。",
                    next_step="补充原因、处理步骤或判断标准，让模型看到你想要的回答方式。",
                    evidence=f"全部 {len(answers)} 条回答的平均长度为 {avg_len:.1f} 字。",
                    action=DiagnosticAction(label="去补充数据", action="goto_data"),
                    topic="data",
                    rank=55,
                )
            )

    if method == "sft":
        questions = [
            (row.get("instruction") or row.get("question") or "").strip().lower()
            for row in dataset_rows
        ]
        questions = [q for q in questions if q]
        if questions:
            unique_q = set(questions)
            repetition_rate = 1 - len(unique_q) / len(questions)
            if repetition_rate > 0.2:
                dup_count = len(questions) - len(unique_q)
                cards.append(
                    _card(
                        level="warn",
                        title=f"有 {dup_count} 条问题重复，模型可能会背题",
                        observation="训练数据里有一批重复或几乎重复的问题。",
                        interpretation="重复样本会放大某几个问法，训练后看起来会了，换说法可能就不稳。",
                        suggestion="可以先去重，再补一些相近但不同表达的真实问题。",
                        next_step="保留最有代表性的一条，把其他重复样本换成新场景。",
                        evidence=(
                            f"重复率 {repetition_rate * 100:.0f}%"
                            f"（{len(unique_q)} 个不同问题 / {len(questions)} 条总数据）。"
                        ),
                        action=DiagnosticAction(label="去整理数据", action="goto_data"),
                        topic="data",
                        rank=50,
                    )
                )

    return _sort_cards(cards)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _card(
    *,
    level: str,
    title: str,
    suggestion: str,
    observation: str | None = None,
    interpretation: str | None = None,
    mechanism: str | None = None,
    how_to_tell: str | None = None,
    next_step: str | None = None,
    evidence: str | None = None,
    action: DiagnosticAction | None = None,
    topic: str = "general",
    rank: int = 50,
) -> DiagnosticCard:
    return DiagnosticCard(
        level=level,  # type: ignore[arg-type]
        title=title,
        suggestion=suggestion,
        observation=observation,
        interpretation=interpretation,
        mechanism=mechanism,
        how_to_tell=how_to_tell,
        next_step=next_step,
        evidence=evidence,
        action=action,
        topic=topic,  # type: ignore[arg-type]
        rank=rank,
    )


def _sort_cards(cards: list[DiagnosticCard]) -> list[DiagnosticCard]:
    level_rank = {"error": 0, "warn": 1, "ok": 2}
    return sorted(cards, key=lambda c: (level_rank.get(c.level, 3), c.rank))


def _losses(exp: Experiment) -> list[float]:
    return [v for v in exp.loss if isinstance(v, (int, float)) and v == v]


def _eval_losses(exp: Experiment) -> list[float]:
    return [v for v in exp.eval_loss if isinstance(v, (int, float)) and v == v]


def _drop_rate(values: list[float]) -> float | None:
    if len(values) < 2 or values[0] <= 0:
        return None
    return (values[0] - values[-1]) / values[0]


def _tail_drop_rate(values: list[float]) -> tuple[float, float, float] | None:
    if len(values) < 4:
        return None
    window_size = max(3, len(values) // 3)
    tail = values[-window_size:]
    split = len(tail) // 2
    first = sum(tail[:split]) / split
    second = sum(tail[split:]) / (len(tail) - split)
    if first <= 0:
        return None
    return first, second, (first - second) / first


def _direction_changes(values: list[float]) -> int:
    changes = 0
    for i in range(2, len(values)):
        prev_dir = values[i - 1] - values[i - 2]
        curr_dir = values[i] - values[i - 1]
        if prev_dir * curr_dir < 0:
            changes += 1
    return changes


def _average_abs_step(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return sum(abs(values[i] - values[i - 1]) for i in range(1, len(values))) / (
        len(values) - 1
    )


def _segment_drops(values: list[float]) -> tuple[float, float, float] | None:
    """把序列切成前/中/后三段，返回每段的相对降幅（相对于该段起点）。"""
    if len(values) < 6:
        return None
    n = len(values)
    third = n // 3
    early = values[: third + 1]
    mid = values[third : 2 * third + 1]
    late = values[2 * third :]

    def _rel_drop(seg: list[float]) -> float:
        if len(seg) < 2 or seg[0] <= 0:
            return 0.0
        return (seg[0] - seg[-1]) / seg[0]

    return _rel_drop(early), _rel_drop(mid), _rel_drop(late)


def _has_data_quality_issue(rows: list[dict] | None) -> tuple[bool, bool, float, float]:
    """返回 (有重复, 答案太短, 平均长度, 重复率)，没数据时全 False。"""
    if not rows:
        return (False, False, 0.0, 0.0)
    answers = [
        row.get("output") or row.get("answer") or row.get("chosen") or ""
        for row in rows
    ]
    questions = [
        (row.get("instruction") or row.get("question") or "").strip().lower()
        for row in rows
    ]
    if not answers:
        return (False, False, 0.0, 0.0)
    avg_len = sum(len(a) for a in answers) / len(answers)
    valid_q = [q for q in questions if q]
    if valid_q:
        unique_q = set(valid_q)
        repetition_rate = 1 - len(unique_q) / len(valid_q)
    else:
        repetition_rate = 0.0
    return (repetition_rate > 0.2, avg_len < 25, avg_len, repetition_rate)


# --------------------------------------------------------------------------- #
# Individual rules
# --------------------------------------------------------------------------- #


def _build_ok_card(exp: Experiment) -> DiagnosticCard:
    train = _losses(exp)
    parts = []
    if exp.dataset_count > 0:
        parts.append(f"{exp.dataset_count} 条数据")
    parts.append(f"{exp.params.epochs} 轮训练")
    if len(train) >= 2:
        parts.append(f"loss 从 {train[0]:.2f} 降到 {train[-1]:.2f}")

    summary = "、".join(parts) + "。" if parts else ""
    return _card(
        level="ok",
        title="曲线没有明显异常，可以去看实际回答",
        observation=summary or "这次训练没有记录到明显异常。",
        interpretation="loss 只是过程指标，真正有没有变好，还要看训练前后的回答对比。",
        mechanism=(
            "loss 衡量的是模型对训练样本的预测难度。它降低只代表模型更熟悉这些样本，"
            "不代表它在没见过的问题上也答得更好——所以测评页的实际对比才是最终标准。"
        ),
        next_step="先测评 3 到 5 个常见问题，再决定要不要继续补数据或调参数。",
        action=DiagnosticAction(label="去测评", action="goto_eval"),
        topic="train_loss",
        rank=90,
    )


def _check_no_loss(exp: Experiment) -> list[DiagnosticCard]:
    if _losses(exp):
        return []
    return [
        _card(
            level="warn",
            title="这次没有拿到可解读的 loss 曲线",
            observation="训练结束了，但日志里没有足够的 loss 点。",
            interpretation="没有曲线时，页面无法判断训练过程是否稳定。",
            mechanism=(
                "loss 是训练框架每若干步写入日志的数值。如果训练步数太少（数据极少 + epoch 极少）"
                "或框架被异常打断，日志可能没攒够数据点。"
            ),
            suggestion="可以先去测评页看回答有没有变化。",
            next_step="如果测评也没有变化，再检查训练日志或重新跑一次。",
            evidence="实验记录里的 loss 序列为空。",
            action=DiagnosticAction(label="去测评", action="goto_eval"),
            topic="train_loss",
            rank=20,
        )
    ]


def _check_data_quantity(exp: Experiment) -> list[DiagnosticCard]:
    n = exp.dataset_count
    if n <= 0 or n >= 15:
        return []
    return [
        _card(
            level="warn",
            title=f"训练数据只有 {n} 条，结果可能不稳定",
            observation=f"这次只用了 {n} 条数据训练。",
            interpretation="小数据能看到方向，但很容易学成固定答案，换个问法就不稳。",
            mechanism=(
                "LoRA 训练相当于在原模型上加一层薄薄的偏好。样本数 < 15 时，"
                "这层偏好基本是被几条样本主导的——模型学到的不是「一类问题」，"
                "而是「这几条问题的标准答案」。"
            ),
            suggestion="可以补到 20 条以上，再做一次对照实验。",
            next_step="优先补真实问题和你满意的回答，不需要一次补很多。",
            evidence=f"当前数据集 {n} 条。少于 15 条时，训练结果容易受单条样本影响。",
            action=DiagnosticAction(label="去补充数据", action="goto_data"),
            topic="data",
            rank=45,
        )
    ]


def _check_loss_oscillation(
    exp: Experiment,
    live: bool = False,
) -> list[DiagnosticCard]:
    train = _losses(exp)
    if len(train) < 6:
        return []

    changes = _direction_changes(train)
    oscillation_rate = changes / (len(train) - 2)
    avg_loss = sum(train) / len(train)
    step_ratio = _average_abs_step(train) / avg_loss if avg_loss > 0 else 0
    if oscillation_rate <= 0.6 or step_ratio <= 0.04:
        return []

    if live:
        return [
            _card(
                level="warn",
                title="loss 有些来回波动，先看最终走势",
                suggestion="训练中有波动不一定是坏事，结束后再判断要不要调低学习率。",
                topic="train_loss",
                rank=70,
            )
        ]

    return [
        _card(
            level="warn",
            title="loss 抖动偏多，学习过程不够稳",
            observation="曲线整体在上下摆动，不是一路平滑下降。",
            interpretation="抖动本身不一定是坏事，但持续抖动说明每一步参数更新方向不一致。",
            mechanism=(
                "每个 batch 的 loss 反映的是这一小批数据有多难。"
                "如果不同 batch 的难度差很多（数据风格分散），曲线天然就抖；"
                "如果学习率偏高，每一步迈得太大、容易越过最低点再折回，曲线也抖。"
            ),
            how_to_tell=(
                "怎么分辨：\n"
                "• 数据每条风格差异大（不同人写的、不同场景混在一起）→ 大概率是数据风格分散。\n"
                "• 数据风格整齐、但样本量少（< 30）→ batch 之间样本不够代表性，也会抖。\n"
                "• 数据多且整齐还抖 → 大概率是学习率高了，可以降一半重试。"
            ),
            suggestion="先去测评。如果回答忽好忽坏，再用更低学习率重试。",
            next_step="先测评几个真实问题；如果效果不稳，把学习率降一半再跑一次。",
            evidence=(
                f"loss 曲线有 {changes} 次方向反转，占 {oscillation_rate * 100:.0f}%。"
            ),
            action=DiagnosticAction(
                label="降低学习率重试",
                action="retrain",
                params={"learning_rate": exp.params.learning_rate * 0.5},
            ),
            topic="train_loss",
            rank=65,
        )
    ]


def _check_initial_loss_anomaly(exp: Experiment) -> list[DiagnosticCard]:
    train = _losses(exp)
    if not train:
        return []
    initial = train[0]

    if initial > 8.0:
        return [
            _card(
                level="warn",
                title=f"一开始的 loss 很高（{initial:.2f}）",
                observation="训练刚开始时，模型对这些样本预测得比较吃力。",
                interpretation="loss 在第一步就异常高，说明数据对这个底座来说很「陌生」。",
                mechanism=(
                    "初始 loss 反映的是底座模型在「没见过你这批数据」的情况下，"
                    "对正确答案的预测难度。如果你给的数据格式、术语、语气离它平时见过的差太远，"
                    "loss 会从一个很高的值开始。"
                ),
                how_to_tell=(
                    "怎么分辨：\n"
                    "• 数据列是不是放错了（instruction / output 内容互调）→ 数据格式问题。\n"
                    "• 答案里有大量代码、表格、特殊标记 → 底座没见过的格式。\n"
                    "• 你选的是中文模型但喂的是英文 → 底座和数据语种不匹配。"
                ),
                suggestion="可以先检查几条训练数据，再决定要不要换模型。",
                next_step="先看数据列是否放对，再看回答是否包含大量模型没见过的格式。",
                evidence=f"初始 loss = {initial:.2f}，SFT 通常在 1 到 5 附近。",
                topic="train_loss",
                rank=35,
            )
        ]
    if initial < 0.3 and exp.dataset_count > 5:
        return [
            _card(
                level="warn",
                title=f"一开始的 loss 就很低（{initial:.2f}）",
                observation="模型在训练前就很容易预测这些答案。",
                interpretation="底座模型对你这批数据本来就很熟，训练能改的空间小。",
                mechanism=(
                    "loss 接近 0 意味着模型不需要学习也能猜对答案。"
                    "通常是因为你的训练样本恰好是这个模型预训练时见过的常见格式（例如标准 FAQ、代码片段）。"
                ),
                how_to_tell=(
                    "怎么分辨：\n"
                    "• 你的数据是公开常见问答 → 模型可能本来就会。\n"
                    "• 答案非常短、固定（是/否、单字回复）→ loss 天然就低，不代表模型在学。\n"
                    "• 想看到训练效果，可以补一些更专业、更口语化的真实问题。"
                ),
                suggestion="可以直接去测评，看回答是否真的有改进。",
                next_step="如果测评前后差不多，下一轮重点换数据，而不是加训练轮次。",
                evidence=f"初始 loss = {initial:.2f}，通常 < 0.3 意味着样本太简单。",
                action=DiagnosticAction(label="去测评", action="goto_eval"),
                topic="train_loss",
                rank=55,
            )
        ]
    return []


def _check_loss_rising(
    exp: Experiment,
    live: bool = False,
) -> list[DiagnosticCard]:
    train = _losses(exp)
    if len(train) < 3 or train[0] <= 0:
        return []
    tail = _tail_drop_rate(train)
    tail_rise = tail is not None and tail[2] < -0.08
    overall_rise = train[-1] > train[0] * 1.1
    if not (tail_rise or overall_rise):
        return []

    if live:
        return [
            _card(
                level="warn",
                title="loss 正在上升，训练可能有点不稳",
                suggestion="先让训练跑完；如果最后仍上升，再检查数据或降低学习率。",
                topic="train_loss",
                rank=30,
            )
        ]

    return [
        _card(
            level="error",
            title="loss 往上走，训练方向可能不对",
            observation=f"loss 从 {train[0]:.4f} 变到 {train[-1]:.4f}，结束时没有变好。",
            interpretation="正常训练里 loss 应该整体往下走。它往上走说明模型每一步在「学反了」。",
            mechanism=(
                "训练时模型每一步都在朝「让正确答案更可能」的方向更新参数。"
                "如果学习率过高，每一步迈得太大、跨过了最低点反而往上爬；"
                "如果数据格式有问题（例如答案列错位），模型会把错的答案当目标，loss 也会持续上升。"
            ),
            how_to_tell=(
                "怎么分辨：\n"
                "• 抽查几条数据，instruction 和 output 是否对得上 → 数据格式问题。\n"
                "• 数据没问题但 loss 还在涨 → 大概率是学习率高了，降一半再跑。"
            ),
            suggestion="先检查数据，再降低学习率重试。",
            next_step="先抽查 5 条数据；如果数据没问题，把学习率降一半再跑。",
            evidence=f"最终 loss 比初始 loss 高 {((train[-1] / train[0]) - 1) * 100:.1f}%。",
            action=DiagnosticAction(
                label="降低学习率重试",
                action="retrain",
                params={"learning_rate": exp.params.learning_rate * 0.5},
            ),
            topic="train_loss",
            rank=10,
        )
    ]


def _check_loss_still_dropping(exp: Experiment) -> list[DiagnosticCard]:
    train = _losses(exp)
    tail = _tail_drop_rate(train)
    if tail is None:
        return []
    first, second, drop_rate = tail
    epochs = exp.params.epochs

    if drop_rate <= 0.05:
        return []
    suggested_epochs = min(epochs + 3, 30)
    return [
        _card(
            level="warn",
            title="结束时 loss 还在降，可能还没学完",
            observation="曲线最后一段仍然明显往下走。",
            interpretation="模型还在吸收数据里的模式，训练停得有点早。",
            mechanism=(
                "训练相当于让模型反复看你这批数据。一开始它进步快，到后期变化变小。"
                "如果在末段曲线还在明显下降，说明它还有「学的余地」——这次设的轮次偏保守。"
            ),
            how_to_tell=(
                "怎么分辨：\n"
                "• 测评回答已经比训练前好很多 → 不必再加轮次，先用这次结果。\n"
                "• 测评回答还差一点意思、缺细节 → 加几轮多半能看到提升。\n"
                "• 注意：加轮次的边际收益会变小，加到 2 倍以上时建议同时观察验证 loss。"
            ),
            suggestion=f"可以把训练轮次从 {epochs} 轮加到 {suggested_epochs} 轮，再做一次对照。",
            next_step="如果测评已经满意，不必加轮次；如果回答还不稳定，再加轮次试一次。",
            evidence=(
                f"训练尾段平均 loss 从 {first:.4f} 降到 {second:.4f}，"
                f"降幅 {drop_rate * 100:.1f}%。"
            ),
            action=DiagnosticAction(
                label="增加轮次重试",
                action="retrain",
                params={"epochs": suggested_epochs},
            ),
            topic="train_loss",
            rank=40,
        )
    ]


def _check_loss_plateau(exp: Experiment) -> list[DiagnosticCard]:
    train = _losses(exp)
    drop = _drop_rate(train)
    tail = _tail_drop_rate(train)
    if drop is None or tail is None:
        return []
    _, _, tail_drop = tail
    if drop < 0.15 or abs(tail_drop) > 0.03:
        return []
    return [
        _card(
            level="ok",
            title="loss 已经降下来并趋稳",
            observation="前半段下降更明显，后半段变化变小。",
            interpretation="模型已经学到主要模式，继续加轮次未必带来明显收益。",
            mechanism=(
                "loss 降到一定程度后会进入「平台期」——模型已经吸收了数据里能学的部分。"
                "再训下去，要么进入边际递减，要么开始记住具体样本（过拟合）。"
                "这是 LoRA 训练里最理想的结束时机。"
            ),
            suggestion="先去测评页看实际回答，不急着继续加训练轮次。",
            next_step="如果测评满意，就保留这次结果；如果还差一点，再考虑补数据。",
            evidence=f"训练 loss 总体下降 {drop * 100:.1f}%，尾段变化已经不大。",
            action=DiagnosticAction(label="去测评", action="goto_eval"),
            topic="train_loss",
            rank=80,
        )
    ]


def _check_classic_overfit(exp: Experiment) -> list[DiagnosticCard]:
    train = _losses(exp)
    val = _eval_losses(exp)
    if len(train) < 2 or len(val) < 2:
        return []

    train_drop = _drop_rate(train)
    if train_drop is None:
        return []
    val_min = min(val)
    val_last = val[-1]
    val_rose = val_last > val_min * 1.05

    if train_drop <= 0.1 or not val_rose:
        return []
    best_epoch = val.index(val_min) + 1
    return [
        _card(
            level="error",
            title="有过拟合迹象，模型可能开始背训练数据",
            observation="训练 loss 在降，但验证 loss 在最低点之后又回升。",
            interpretation="模型对训练集越来越熟，但对没见过的数据反而变差。",
            mechanism=(
                "验证集是从训练数据里切出来不参与训练的一小部分，专门用来检查模型「在没见过的样本上是否也变好」。"
                "训练 loss 还在降但验证 loss 反弹，是过拟合的经典信号——模型不再学规律，"
                "而是开始记忆训练样本的具体答案。轮次越多，这个差距越大。"
            ),
            how_to_tell=(
                "怎么分辨：\n"
                "• 验证 loss 第几轮开始往上反弹，那个点之前的模型就是最佳版本。\n"
                "• 如果训练前后测评出现大量「固定话术」「换个问法就答非所问」→ 典型过拟合。\n"
                "• 解决路径：要么减少轮次，要么补更多样本稀释「记忆」。"
            ),
            suggestion=f"可以把训练轮次缩到第 {best_epoch} 轮，或补更多样本。",
            next_step="如果训练前后测评出现固定话术、泛化差，优先减少轮次。",
            evidence=f"训练 loss 降了 {train_drop * 100:.0f}%，验证 loss 在第 {best_epoch} 轮后回升。",
            action=DiagnosticAction(
                label=f"用 {best_epoch} 轮重试",
                action="retrain",
                params={"epochs": best_epoch},
            ),
            topic="train_eval",
            rank=20,
        )
    ]


def _check_memorization_overfit(exp: Experiment) -> list[DiagnosticCard]:
    train = _losses(exp)
    val = _eval_losses(exp)
    if len(train) < 2 or len(val) < 2:
        return []

    train_drop = _drop_rate(train)
    val_drop = _drop_rate(val)
    if train_drop is None or val_drop is None:
        return []

    if train_drop <= 0.75 or abs(val_drop) >= 0.1:
        return []
    return [
        _card(
            level="error",
            title="训练集学得很熟，但泛化没有明显变好",
            observation="训练 loss 大幅下降，验证 loss 基本没动。",
            interpretation="模型记住了训练样本，但没有学到可迁移的规律。",
            mechanism=(
                "如果训练样本的多样性不够（同一种问法重复、同一种风格的回答），"
                "模型可以通过「死记硬背」让训练 loss 大幅下降，但这层记忆完全无法泛化到没见过的样本上，"
                "所以验证 loss 一动不动。这是比经典过拟合更隐蔽的一种——曲线上看不出反弹，但实际效果很差。"
            ),
            how_to_tell=(
                "怎么分辨：\n"
                "• 训练数据里有大量相似问法 → 容易触发记忆型过拟合。\n"
                "• 测评训练集里的原题表现完美、换个问法就答非所问 → 典型记忆型过拟合。\n"
                "• 解决路径：补一批不同问法、不同场景的样本，比缩轮次更有效。"
            ),
            suggestion="增加数据多样性，或减少训练轮次。",
            next_step="补一些不同问法、不同场景的样本，再重新训练。",
            evidence=f"训练 loss 降了 {train_drop * 100:.0f}%，验证 loss 只变化 {val_drop * 100:.1f}%。",
            action=DiagnosticAction(label="去补充数据", action="goto_data"),
            topic="train_eval",
            rank=25,
        )
    ]


def _check_underfit(
    exp: Experiment,
    live: bool = False,
) -> list[DiagnosticCard]:
    train = _losses(exp)
    if len(train) < 2:
        return []
    epochs = exp.params.epochs
    if not live and epochs < 3:
        return []

    train_drop = _drop_rate(train)
    if train_drop is None or train_drop >= 0.1:
        return []

    if live:
        return [
            _card(
                level="warn",
                title="loss 目前下降不明显",
                suggestion="先让训练跑完；如果结束后仍不降，再检查数据或加训练量。",
                topic="train_loss",
                rank=60,
            )
        ]

    return [
        _card(
            level="warn",
            title="loss 几乎没降，模型可能没学到多少",
            observation=f"训练了 {epochs} 轮，loss 只从 {train[0]:.4f} 到 {train[-1]:.4f}。",
            interpretation="数据里的信号不够清楚，或者这次训练强度不够。",
            mechanism=(
                "loss 降幅小于 10%，意味着训练前后模型对这批样本的预测难度几乎没变。"
                "通常有两种情况：(1) 数据里没什么可学的——样本太短、信息量太少、答案模式太散乱；"
                "(2) 训练强度不够——LoRA 秩太低、轮次太少、学习率太低，模型「想学也学不动」。"
            ),
            how_to_tell=(
                "怎么分辨：\n"
                "• 答案普遍 < 25 字、回答风格五花八门 → 数据信号不够清晰，先改数据。\n"
                "• 数据看上去整齐、信息量也够 → 试试训练档位调到「加强」，或把轮次翻倍。\n"
                "• 注意：先改数据再调参数，光加轮次救不回不清晰的数据。"
            ),
            suggestion="先检查数据质量，再尝试更强一点的训练档位。",
            next_step="先看样本是否太短、太重复；如果数据没问题，再增加轮次。",
            evidence=f"loss 降幅只有 {train_drop * 100:.1f}%。",
            action=DiagnosticAction(
                label="增加轮次重试",
                action="retrain",
                params={"epochs": min(epochs * 2, 30)},
            ),
            topic="train_loss",
            rank=35,
        )
    ]


def _check_validation_improving(exp: Experiment) -> list[DiagnosticCard]:
    train = _losses(exp)
    val = _eval_losses(exp)
    if len(train) < 2 or len(val) < 2:
        return []
    train_drop = _drop_rate(train)
    val_drop = _drop_rate(val)
    if train_drop is None or val_drop is None:
        return []
    if train_drop < 0.1 or val_drop < 0.08:
        return []
    return [
        _card(
            level="ok",
            title="训练集和验证集都在变好",
            observation="训练 loss 在降，验证 loss 也在降。",
            interpretation="这比只看训练 loss 更可靠，说明模型不只是记住训练样本。",
            mechanism=(
                "验证集是训练时切出来不参与更新的一小部分样本。"
                "训练 loss 降只能说「记得越来越熟」，验证 loss 降才能说「真学到了规律」。"
                "两条线一起降，是 SFT 训练里最健康的状态。"
            ),
            suggestion="可以去测评页看真实回答是否也变好了。",
            next_step="用几个没放进训练集的问题做对比，重点看回答风格和准确性。",
            evidence=f"训练 loss 降了 {train_drop * 100:.0f}%，验证 loss 降了 {val_drop * 100:.0f}%。",
            action=DiagnosticAction(label="去测评", action="goto_eval"),
            topic="eval_loss",
            rank=85,
        )
    ]


def _check_validation_noisy(exp: Experiment) -> list[DiagnosticCard]:
    val = _eval_losses(exp)
    if len(val) < 4:
        return []
    changes = _direction_changes(val)
    rate = changes / (len(val) - 2)
    avg_val = sum(val) / len(val)
    step_ratio = _average_abs_step(val) / avg_val if avg_val > 0 else 0
    if rate <= 0.5 or step_ratio <= 0.03:
        return []
    return [
        _card(
            level="warn",
            title="验证 loss 波动较大，先别只看这一条线",
            observation="验证 loss 上下波动比较明显。",
            interpretation="验证集通常比训练集小，样本差异大时，一两个样本就会让曲线跳动。",
            mechanism=(
                "SFT 默认会切出 15% 的数据做验证集——数据集 30 条时，验证集只有 4-5 条。"
                "这么小的验证集里，一两个特别难或特别简单的样本就能让 loss 跳动很大。"
                "不代表模型有问题，只是统计上的样本不够。"
            ),
            how_to_tell=(
                "怎么分辨：\n"
                "• 数据量 < 100 → 验证集本身就小，波动大很正常，看测评回答更靠谱。\n"
                "• 数据量很大（> 200）但验证 loss 还在抖 → 数据风格分散，可以补更整齐的样本。"
            ),
            suggestion="把验证 loss 当参考，重点还是看测评回答。",
            next_step="用训练集之外的问题做对比；如果回答也不稳，再补更多样本。",
            evidence=f"验证 loss 有 {changes} 次方向反转，占 {rate * 100:.0f}%。",
            action=DiagnosticAction(label="去测评", action="goto_eval"),
            topic="eval_loss",
            rank=70,
        )
    ]


def _check_dpo_reward_margin(exp: Experiment) -> list[DiagnosticCard]:
    if exp.method != "dpo":
        return []
    train = _losses(exp)
    if len(train) < 3 or train[0] <= 0:
        return []

    train_drop = _drop_rate(train)
    if train_drop is None:
        return []

    if train[-1] > train[0] * 1.1:
        return [
            _card(
                level="error",
                title="DPO loss 上升了，偏好方向可能没学对",
                observation=f"DPO loss 从 {train[0]:.4f} 升到 {train[-1]:.4f}。",
                interpretation="常见原因是 chosen/rejected 标反了，或学习率太高。",
                suggestion="可以先检查偏好对，再降低学习率重试。",
                next_step="抽查几条数据，确认 chosen 真的是你更想要的回答。",
                evidence=f"DPO loss 最终上升 {((train[-1] / train[0]) - 1) * 100:.1f}%。",
                action=DiagnosticAction(
                    label="降低学习率重试",
                    action="retrain",
                    params={"learning_rate": exp.params.learning_rate * 0.5},
                ),
                topic="dpo",
                rank=15,
            )
        ]

    if train_drop < 0.05 and exp.params.epochs >= 2:
        return [
            _card(
                level="warn",
                title="DPO loss 基本没动，偏好信号可能不够明显",
                observation=f"经过 {exp.params.epochs} 轮，DPO loss 变化很小。",
                interpretation="模型可能分不清 chosen 和 rejected 的差别，或者偏好样本太少。",
                suggestion="可以把好回答和差回答的区别写得更明显。",
                next_step="优先补充差异更清楚的偏好对，再重新训练。",
                evidence=f"DPO loss 从 {train[0]:.4f} 变为 {train[-1]:.4f}。",
                action=DiagnosticAction(label="去补充数据", action="goto_data"),
                topic="dpo",
                rank=45,
            )
        ]
    return []


def _check_data_quality(
    exp: Experiment,
    dataset_rows: list[dict] | None,
) -> list[DiagnosticCard]:
    if not dataset_rows:
        return []

    answers = []
    questions = []
    for row in dataset_rows:
        ans = row.get("output") or row.get("answer") or row.get("chosen") or ""
        q = row.get("instruction") or row.get("question") or ""
        answers.append(ans)
        questions.append(q)

    if not answers:
        return []

    avg_len = sum(len(a) for a in answers) / len(answers)
    unique_q = set(q.strip().lower() for q in questions if q.strip())
    repetition_rate = 1 - len(unique_q) / max(len(questions), 1) if questions else 0

    cards: list[DiagnosticCard] = []

    if avg_len < 25:
        cards.append(
            _card(
                level="warn",
                title=f"训练回答平均只有 {avg_len:.0f} 字，信息量偏少",
                observation="训练数据里的回答整体比较短。",
                interpretation="短答案适合教格式，但不太能教会完整表达和判断标准。",
                mechanism=(
                    "模型从样本中学的是「输入 → 输出」的映射。"
                    "如果输出只有几个字，模型主要学到的是「答得短」「答得有礼貌」这类格式信号，"
                    "而不是真正的判断逻辑或专业知识。loss 可能降得很快，但能力没真长。"
                ),
                suggestion="可以把典型回答写得更完整一点。",
                next_step="补上原因、步骤或边界条件，让模型看到更明确的示范。",
                evidence=f"全部 {len(answers)} 条回答的平均长度为 {avg_len:.1f} 字。",
                action=DiagnosticAction(label="去补充数据", action="goto_data"),
                topic="data",
                rank=55,
            )
        )

    if repetition_rate > 0.2:
        dup_count = len(questions) - len(unique_q)
        cards.append(
            _card(
                level="warn",
                title=f"训练数据里有 {dup_count} 条重复问题",
                observation="同类问题重复较多。",
                interpretation="重复会让模型偏向记住这些问法，而不是学会一类问题的处理方式。",
                mechanism=(
                    "重复样本相当于让模型反复看同一道题。"
                    "它会先记住这几道题的答案（loss 降得很快很彻底），"
                    "但因为没有看到「同一类问题的不同问法」，所以学不会泛化——"
                    "测评时只要换种说法，就会答非所问。"
                ),
                suggestion="可以去重，并补一些不同说法的真实问题。",
                next_step="保留最典型的一条，把重复项换成其他用户可能会问的表达。",
                evidence=(
                    f"重复率 {repetition_rate * 100:.0f}%"
                    f"（{len(unique_q)} 个不同问题 / {len(questions)} 条总数据）。"
                ),
                action=DiagnosticAction(label="去整理数据", action="goto_data"),
                topic="data",
                rank=50,
            )
        )

    return cards


# --------------------------------------------------------------------------- #
# New rules: process / train-eval middle ground / data×loss combos / no-eval
# --------------------------------------------------------------------------- #


def _check_first_step_jump(exp: Experiment) -> list[DiagnosticCard]:
    """第一步 → 第二步出现巨大跳变。"""
    train = _losses(exp)
    if len(train) < 5 or train[0] <= 0:
        return []
    jump = abs(train[1] - train[0]) / train[0]
    if jump < 0.5:
        return []
    # 后续如果没有再出现这种规模的跳变，才提示
    later_max_jump = 0.0
    for i in range(2, len(train)):
        if train[i - 1] > 0:
            later_max_jump = max(later_max_jump, abs(train[i] - train[i - 1]) / train[i - 1])
    if later_max_jump >= jump * 0.5:
        return []
    return [
        _card(
            level="ok",
            title="开头一两步 loss 跳得很大，看后面更准",
            observation=f"第 1 步 loss = {train[0]:.4f}，第 2 步突然变成 {train[1]:.4f}。",
            interpretation="开头几步的 loss 不太代表训练实际进度，可以从第 3-5 步开始看。",
            mechanism=(
                "训练开始时有一段「学习率 warmup」，前几步学习率从 0 慢慢升到设定值。"
                "在这段里模型还没真正开始学，loss 会出现大跳变。框架日志会照实记录，"
                "但解读趋势时应该跳过开头这一两步。"
            ),
            evidence=f"开头跳变 {jump * 100:.0f}%，之后最大单步变化 {later_max_jump * 100:.0f}%。",
            suggestion="忽略开头这一两步，看中后段的趋势。",
            topic="train_process",
            rank=88,
        )
    ]


def _check_early_steep_drop(exp: Experiment) -> list[DiagnosticCard]:
    """前 1/3 段就降掉总降幅一半以上 → 学得快，常见且健康。"""
    train = _losses(exp)
    drop = _drop_rate(train)
    seg = _segment_drops(train)
    if drop is None or seg is None or drop < 0.15:
        return []
    early, _, _ = seg
    if early < drop * 0.5:
        return []
    return [
        _card(
            level="ok",
            title="前期 loss 就掉得很快，这是正常起步",
            observation=f"前 1/3 步数 loss 就降了 {early * 100:.0f}%。",
            interpretation="模型在训练初期快速吸收任务的「格式和语气」，是 LoRA 训练的典型起步形态。",
            mechanism=(
                "LoRA 最先学到的是任务的表层特征——例如「用户问 X 时，回答以 Y 开头」「保持礼貌口吻」。"
                "这些都是几个 token 就能学会的浅层 pattern，所以前期 loss 掉得最快。"
                "等浅层模式吸收完，才会开始啃更难的具体内容，曲线就慢下来了。"
            ),
            evidence=f"前段降 {early * 100:.0f}%，整体降 {drop * 100:.0f}%。",
            suggestion="先看后期是不是继续慢慢降，再决定要不要加轮次或停下。",
            topic="train_process",
            rank=88,
        )
    ]


def _check_mid_plateau(exp: Experiment) -> list[DiagnosticCard]:
    """中段平台后期又降，常见的「攒突破」现象。"""
    train = _losses(exp)
    seg = _segment_drops(train)
    if seg is None:
        return []
    early, mid, late = seg
    # 中段几乎不降但后段又降
    if early < 0.1 or mid > 0.03 or late < 0.05:
        return []
    return [
        _card(
            level="ok",
            title="中间出现过一段平台，后面又开始降了",
            observation="曲线中段几乎不动，但后期又继续往下走。",
            interpretation="中间的平台不是训练失败，是模型在「整合」前面学到的内容。",
            mechanism=(
                "复杂任务里，模型有时需要先把浅层模式串起来形成内部表示，才能继续往深处学。"
                "在这个「整合期」，loss 几乎不降；整合完成后，下一波 loss 下降会到来。"
                "DPO、长答案 SFT、推理类任务里特别常见。"
            ),
            evidence=f"前段降 {early * 100:.0f}%，中段 {mid * 100:.1f}%，后段降 {late * 100:.0f}%。",
            suggestion="不要看到中段平台就以为训练卡住了，让它跑完再判断。",
            topic="train_process",
            rank=89,
        )
    ]


def _check_tail_micro_bounce(exp: Experiment) -> list[DiagnosticCard]:
    """末段轻微反弹但整体仍在降——别误判成过拟合。"""
    train = _losses(exp)
    drop = _drop_rate(train)
    if drop is None or drop < 0.1 or len(train) < 8:
        return []
    last_three = train[-3:]
    rebound = max(last_three) - min(last_three)
    if train[-1] <= 0:
        return []
    rebound_ratio = rebound / train[-1]
    if rebound_ratio < 0.03 or rebound_ratio > 0.15:
        return []
    # 必须真的「整体往下、末段微反弹」
    if train[-1] >= train[0] * 0.9:
        return []
    # 不要和过拟合规则冲突——只看训练 loss 这条线
    return [
        _card(
            level="ok",
            title="结尾两三步小幅反弹，不必担心",
            observation="整体 loss 在降，最后两三步有点小波动。",
            interpretation="末段这种小幅反弹通常是「单 batch 难度偶然偏高」，不是过拟合。",
            mechanism=(
                "loss 是按 batch 算的，每个 batch 的样本组合都不同。"
                "结尾几步如果恰好抽到几个偏难的样本，loss 就会比前几步小幅高一点，"
                "这是统计噪声，不是模型在「变差」。判断过拟合要看验证 loss，不是训练 loss 末段。"
            ),
            evidence=f"整体降 {drop * 100:.0f}%，末段反弹幅度仅 {rebound_ratio * 100:.1f}%。",
            suggestion="如果有验证集，重点看验证 loss；没有的话，去测评看实际回答。",
            topic="train_process",
            rank=88,
        )
    ]


def _check_generalization_diminishing(exp: Experiment) -> list[DiagnosticCard]:
    """训练降幅 >> 验证降幅，但验证仍在降——泛化收益递减。"""
    train = _losses(exp)
    val = _eval_losses(exp)
    if len(train) < 2 or len(val) < 2:
        return []
    train_drop = _drop_rate(train)
    val_drop = _drop_rate(val)
    if train_drop is None or val_drop is None:
        return []
    # 训练降很多，验证也在降但幅度小不少（避免和 memorization_overfit 冲突）
    if train_drop < 0.3 or val_drop < 0.03 or val_drop > 0.3:
        return []
    if train_drop / max(val_drop, 0.01) < 2.5:
        return []
    return [
        _card(
            level="warn",
            title="训练学得比验证快得多，正在接近过拟合边缘",
            observation=f"训练 loss 降了 {train_drop * 100:.0f}%，验证 loss 只降了 {val_drop * 100:.0f}%。",
            interpretation="模型吸收训练样本的速度，明显超过它把规律泛化到没见过样本的速度。",
            mechanism=(
                "理想情况下，训练 loss 和验证 loss 应该一起降、降幅接近。"
                "如果训练降得远比验证快，说明模型已经开始「偏向死记硬背」——"
                "它能在训练样本上越来越像，但在没见过的样本上提升有限。再多几轮就会真正过拟合。"
            ),
            how_to_tell=(
                "怎么判断要不要继续：\n"
                "• 现在测评回答已经满意 → 这次结果就停，不要再训。\n"
                "• 还想再提升 → 优先补样本（增加多样性），而不是加轮次。\n"
                "• 想继续训 → 也先把轮次缩短，并密切看验证 loss 有没有反弹。"
            ),
            evidence=f"训练降 / 验证降 = {train_drop / max(val_drop, 0.01):.1f}。",
            suggestion="不要再加轮次了，先去测评或补数据。",
            next_step="先看实际回答；如果还想继续训，优先补样本，不是加轮次。",
            action=DiagnosticAction(label="去测评", action="goto_eval"),
            topic="train_eval",
            rank=30,
        )
    ]


def _check_validation_stalled(exp: Experiment) -> list[DiagnosticCard]:
    """训练继续降，验证最近几步基本不动。"""
    train = _losses(exp)
    val = _eval_losses(exp)
    if len(train) < 4 or len(val) < 4:
        return []
    train_tail = _tail_drop_rate(train)
    val_tail = _tail_drop_rate(val)
    if train_tail is None or val_tail is None:
        return []
    _, _, train_tail_drop = train_tail
    _, _, val_tail_drop = val_tail
    # 训练末段还在降，验证末段几乎不动
    if train_tail_drop < 0.05 or abs(val_tail_drop) > 0.02:
        return []
    # 排除掉「整体训练就没怎么降」的情况
    train_drop = _drop_rate(train)
    if train_drop is None or train_drop < 0.15:
        return []
    return [
        _card(
            level="warn",
            title="训练还在降，验证已经不动了",
            observation="训练 loss 末段继续下降，验证 loss 最近几轮基本没变。",
            interpretation="模型吸收训练样本的能力还没用尽，但泛化能力已经停了。",
            mechanism=(
                "验证 loss 平台说明模型「在没见过的样本上能学的」已经学完了。"
                "训练 loss 还在降，是因为它还能继续记住训练集本身的细节——"
                "但这部分新学到的东西，无法迁移到测评回答上。再多训只是浪费时间。"
            ),
            how_to_tell=(
                "什么时候真的要停：\n"
                "• 验证 loss 连续 2-3 个评估点都不动 → 可以收手了。\n"
                "• 想继续提升 → 唯一有效的办法是补样本，不是加训练时长。"
            ),
            evidence=f"训练末段降 {train_tail_drop * 100:.1f}%，验证末段变化 {val_tail_drop * 100:.1f}%。",
            suggestion="可以停了，去测评看回答。",
            next_step="如果效果不够，唯一可靠的办法是补更多数据。",
            action=DiagnosticAction(label="去测评", action="goto_eval"),
            topic="train_eval",
            rank=35,
        )
    ]


def _check_validation_better_than_train(exp: Experiment) -> list[DiagnosticCard]:
    """验证 loss 比训练 loss 还低 / 降幅更大——罕见但要解释。"""
    train = _losses(exp)
    val = _eval_losses(exp)
    if len(train) < 2 or len(val) < 2:
        return []
    train_drop = _drop_rate(train)
    val_drop = _drop_rate(val)
    if train_drop is None or val_drop is None:
        return []
    # 验证降幅明显高于训练，或验证 loss 整体远低于训练 loss
    train_avg = sum(train) / len(train)
    val_avg = sum(val) / len(val)
    val_lower = val_avg < train_avg * 0.85
    val_drops_more = val_drop > train_drop + 0.1
    if not (val_lower or val_drops_more):
        return []
    return [
        _card(
            level="warn",
            title="验证 loss 比训练 loss 表现还好，要看一下数据切分",
            observation="验证集上的 loss 比训练集还低（或降得还多）。",
            interpretation="正常情况下验证 loss 会高于训练 loss，反过来通常意味着数据切分有问题。",
            mechanism=(
                "训练 loss 是模型刚学完一个 batch 立刻算的，验证 loss 是整段 epoch 训完后一次性算的。"
                "在数据少时，平台会自动切 15% 做验证集，这一小批样本可能恰好特别简单、"
                "或者跟训练集高度相似——结果就是验证 loss 看起来更好。这不一定是真好。"
            ),
            how_to_tell=(
                "怎么判断：\n"
                "• 数据少（< 100）+ 验证集很小（< 15）→ 大概率是切分恰好简单。\n"
                "• 数据足、切分合理还出现这种情况 → 训练集里可能有「难样本」混入，可以排查一下。\n"
                "• 真正的判断标准还是去测评页看回答。"
            ),
            evidence=f"训练 loss 平均 {train_avg:.3f}，验证 loss 平均 {val_avg:.3f}。",
            suggestion="不必慌，先去测评看真实回答。",
            next_step="如果数据少，下次训练前补到 100 条以上，验证集会更可靠。",
            action=DiagnosticAction(label="去测评", action="goto_eval"),
            topic="train_eval",
            rank=60,
        )
    ]


def _check_no_validation_split(exp: Experiment) -> list[DiagnosticCard]:
    """数据 < 30 条没切验证集，给出解释。"""
    if exp.method == "dpo":
        return []
    train = _losses(exp)
    val = _eval_losses(exp)
    if not train or val:
        return []
    if exp.dataset_count <= 0 or exp.dataset_count >= 30:
        return []
    return [
        _card(
            level="ok",
            title="这次没有验证 loss，是数据量决定的",
            observation=f"数据集只有 {exp.dataset_count} 条，平台没有切验证集。",
            interpretation="少于 30 条时，切验证集反而不可靠——所以这次只看训练 loss + 测评回答。",
            mechanism=(
                "SFT 默认会从你的数据里切出 15% 当验证集，专门用来检查泛化情况。"
                "但在数据 < 30 条时，15% 只有 3-4 条样本——这么小的验证集，loss 跳变会非常剧烈，"
                "看起来像问题、但其实只是统计噪声。所以平台直接跳过验证集，"
                "把所有数据都用来训练，效果判断完全交给测评页的真实问题对比。"
            ),
            evidence=f"当前 {exp.dataset_count} 条 < 30 条阈值，未切验证集。",
            suggestion="判断效果就看测评页训练前后对比；想看到验证 loss，下次补到 30 条以上。",
            next_step="去测评页问几个真实问题，看回答前后差异。",
            action=DiagnosticAction(label="去测评", action="goto_eval"),
            topic="eval_loss",
            rank=87,
        )
    ]


def _check_data_oscillation_combo(exp: Experiment) -> list[DiagnosticCard]:
    """数据少 + loss 抖动严重 → 联动诊断。"""
    train = _losses(exp)
    if len(train) < 6 or exp.dataset_count <= 0 or exp.dataset_count >= 20:
        return []
    changes = _direction_changes(train)
    oscillation_rate = changes / max(len(train) - 2, 1)
    avg_loss = sum(train) / len(train)
    step_ratio = _average_abs_step(train) / avg_loss if avg_loss > 0 else 0
    if oscillation_rate <= 0.5 or step_ratio <= 0.04:
        return []
    return [
        _card(
            level="warn",
            title=f"loss 抖动 + 数据只有 {exp.dataset_count} 条，先补数据再说调参",
            observation=f"曲线明显抖动，且数据集只有 {exp.dataset_count} 条。",
            interpretation="这两个信号连起来看，几乎可以确定不是学习率的问题，而是数据太少了。",
            mechanism=(
                "数据少于 20 条时，每个 batch 里的样本几乎就是「整个数据集的不同切片」，"
                "batch 之间的难度差异会被放大。即使学习率合适，曲线也会持续抖动。"
                "这种情况下降学习率不会让曲线平稳，反而可能导致模型学不动。"
            ),
            how_to_tell=(
                "为什么不是学习率：\n"
                "• 数据足够（> 50）但 loss 抖 → 通常是学习率高了。\n"
                "• 数据 < 20 + loss 抖 → 几乎一定是数据不够，先补数据。"
            ),
            evidence=f"数据 {exp.dataset_count} 条，loss 反向变化占 {oscillation_rate * 100:.0f}%。",
            suggestion="优先补数据到 30 条以上，再来看曲线是否平稳。",
            next_step="先补数据再训，不要急着调学习率。",
            action=DiagnosticAction(label="去补充数据", action="goto_data"),
            topic="data",
            rank=40,
        )
    ]


def _check_repetition_strong_drop_combo(
    exp: Experiment,
    dataset_rows: list[dict] | None,
) -> list[DiagnosticCard]:
    """重复多 + loss 降得很彻底 → 警惕背书。"""
    has_repeat, _, _, repetition_rate = _has_data_quality_issue(dataset_rows)
    if not has_repeat:
        return []
    train = _losses(exp)
    drop = _drop_rate(train)
    if drop is None or drop < 0.7:
        return []
    return [
        _card(
            level="warn",
            title=f"重复样本占 {repetition_rate * 100:.0f}% + loss 降得很彻底，警惕背书",
            observation=f"训练数据有 {repetition_rate * 100:.0f}% 重复，整体 loss 降幅 {drop * 100:.0f}%。",
            interpretation="loss 降这么多本来是好事，但叠加高重复率，更像是模型在背训练样本，不是学到了规律。",
            mechanism=(
                "loss 降幅 > 70% 说明模型对训练样本预测得相当准。"
                "如果数据没什么重复，这是真本事；但如果重复占了 20% 以上，"
                "模型只要记住这几个高频问题的标准答案，就能让 loss 大幅下降——"
                "本质上是「考前刷重复题」，换个问法就露馅。"
            ),
            how_to_tell=(
                "怎么验证：\n"
                "• 把训练集里出现过的原题直接问 → 答得很完美。\n"
                "• 把同一问题换种问法（同义词、扩展、补充背景）→ 如果立刻变差，就是背书。"
            ),
            evidence=f"重复率 {repetition_rate * 100:.0f}%，loss 降幅 {drop * 100:.0f}%。",
            suggestion="去测评页用「换问法」的方式问几个问题，看是不是真学到了。",
            next_step="如果发现是背书，回到数据页去重，并补不同问法的样本。",
            action=DiagnosticAction(label="去测评", action="goto_eval"),
            topic="data",
            rank=40,
        )
    ]


def _check_short_answer_plateau_combo(
    exp: Experiment,
    dataset_rows: list[dict] | None,
) -> list[DiagnosticCard]:
    """答案普遍很短 + loss 趋稳 → 平稳不代表学到。"""
    _, short_answer, avg_len, _ = _has_data_quality_issue(dataset_rows)
    if not short_answer:
        return []
    train = _losses(exp)
    drop = _drop_rate(train)
    tail = _tail_drop_rate(train)
    if drop is None or tail is None:
        return []
    _, _, tail_drop = tail
    if drop < 0.15 or abs(tail_drop) > 0.04:
        return []
    return [
        _card(
            level="warn",
            title=f"答案平均 {avg_len:.0f} 字 + loss 已经稳，平稳不代表学到",
            observation=f"答案平均长度 {avg_len:.1f} 字，loss 已经趋于平稳。",
            interpretation="短答案让 loss 很容易看起来「学完了」，但实际能力可能没真的长。",
            mechanism=(
                "loss 衡量的是模型对每个 token 的预测难度。"
                "如果答案只有几个字，模型只需要学会「以什么开头、以什么结尾」就能让 loss 降得很彻底——"
                "看起来曲线很漂亮，但模型其实只学到了表层格式，没学到判断逻辑或专业内容。"
            ),
            how_to_tell=(
                "怎么验证：\n"
                "• 测评时问需要「展开解释」的问题 → 如果还是回简短一句，说明只学了格式。\n"
                "• 真想要它学内容 → 把训练答案补到至少 50 字，包含原因和判断步骤。"
            ),
            evidence=f"答案平均 {avg_len:.1f} 字，整体 loss 降 {drop * 100:.0f}%、末段已稳。",
            suggestion="先去测评看回答深度，再决定要不要补数据。",
            next_step="如果回答仍然很短、缺细节，回到数据页把答案写完整。",
            action=DiagnosticAction(label="去测评", action="goto_eval"),
            topic="data",
            rank=58,
        )
    ]


# --------------------------------------------------------------------------- #
# Original data quality (after-training) — content above
# --------------------------------------------------------------------------- #
