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
    cards += _check_loss_rising(exp)
    cards += _check_classic_overfit(exp)
    cards += _check_memorization_overfit(exp)
    cards += _check_underfit(exp)
    cards += _check_loss_still_dropping(exp)
    cards += _check_loss_plateau(exp)
    cards += _check_loss_oscillation(exp)
    cards += _check_validation_improving(exp)
    cards += _check_validation_noisy(exp)
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
    next_step: str | None = None,
    evidence: str | None = None,
    action: DiagnosticAction | None = None,
    rank: int = 50,
) -> DiagnosticCard:
    return DiagnosticCard(
        level=level,  # type: ignore[arg-type]
        title=title,
        suggestion=suggestion,
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
        suggestion="可以去测评页问几个真实问题，看回答是否更接近你的预期。",
        next_step="先测评 3 到 5 个常见问题，再决定要不要继续补数据或调参数。",
        action=DiagnosticAction(label="去测评", action="goto_eval"),
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
            suggestion="可以先去测评页看回答有没有变化。",
            next_step="如果测评也没有变化，再检查训练日志或重新跑一次。",
            evidence="实验记录里的 loss 序列为空。",
            action=DiagnosticAction(label="去测评", action="goto_eval"),
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
            suggestion="可以补到 20 条以上，再做一次对照实验。",
            next_step="优先补真实问题和你满意的回答，不需要一次补很多。",
            evidence=f"当前数据集 {n} 条。少于 15 条时，训练结果容易受单条样本影响。",
            action=DiagnosticAction(label="去补充数据", action="goto_data"),
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
                rank=70,
            )
        ]

    return [
        _card(
            level="warn",
            title="loss 抖动偏多，学习过程不够稳",
            observation="曲线整体在上下摆动，不是一路平滑下降。",
            interpretation="这通常不是模型完全没学，而是每一步更新有点急。常见原因是学习率偏高、数据量少，或样本风格差异大。",
            suggestion="可以先去测评。如果回答忽好忽坏，再用更低学习率重试。",
            next_step="先测评几个真实问题；如果效果不稳，把学习率降一半再跑一次。",
            evidence=(
                f"loss 曲线有 {changes} 次方向反转，占 {oscillation_rate * 100:.0f}%。"
            ),
            action=DiagnosticAction(
                label="降低学习率重试",
                action="retrain",
                params={"learning_rate": exp.params.learning_rate * 0.5},
            ),
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
                interpretation="常见原因是数据格式不对、答案风格离模型原本能力太远，或选的底座模型不合适。",
                suggestion="可以先检查几条训练数据，再决定要不要换模型。",
                next_step="先看数据列是否放对，再看回答是否包含大量模型没见过的格式。",
                evidence="SFT 的初始 loss 通常在 1 到 5 附近，过高时要先排查数据和模型匹配度。",
                rank=35,
            )
        ]
    if initial < 0.3 and exp.dataset_count > 5:
        return [
            _card(
                level="warn",
                title=f"一开始的 loss 就很低（{initial:.2f}）",
                observation="模型在训练前就很容易预测这些答案。",
                interpretation="这可能说明底座模型已经会这类任务，继续训练带来的变化不会太大。",
                suggestion="可以直接去测评，看回答是否真的有改进。",
                next_step="如果测评前后差不多，下一轮重点换数据，而不是加训练轮次。",
                evidence="初始 loss 极低，通常表示训练样本对模型来说不难。",
                action=DiagnosticAction(label="去测评", action="goto_eval"),
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
                rank=30,
            )
        ]

    return [
        _card(
            level="error",
            title="loss 往上走，训练方向可能不对",
            observation=f"loss 从 {train[0]:.4f} 变到 {train[-1]:.4f}，结束时没有变好。",
            interpretation="这通常表示模型没有沿着数据学习，可能是学习率太高、数据格式有问题，或 DPO 标注方向写反了。",
            suggestion="可以先检查数据，再降低学习率重试。",
            next_step="先抽查 5 条数据；如果数据没问题，把学习率降一半再跑。",
            evidence=f"最终 loss 比初始 loss 高 {((train[-1] / train[0]) - 1) * 100:.1f}%。",
            action=DiagnosticAction(
                label="降低学习率重试",
                action="retrain",
                params={"learning_rate": exp.params.learning_rate * 0.5},
            ),
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
            interpretation="模型可能还在吸收数据里的模式，训练停得有点早。",
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
            interpretation="这通常说明模型已经学到主要模式，继续加轮次未必带来明显收益。",
            suggestion="可以先去测评页看实际回答，不急着继续加训练轮次。",
            next_step="如果测评满意，就保留这次结果；如果还差一点，再考虑补数据。",
            evidence=f"训练 loss 总体下降 {drop * 100:.1f}%，尾段变化已经不大。",
            action=DiagnosticAction(label="去测评", action="goto_eval"),
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
            interpretation="这表示模型对训练集越来越熟，但对没见过的数据反而变差。",
            suggestion=f"可以把训练轮次缩到第 {best_epoch} 轮，或补更多样本。",
            next_step="如果训练前后测评出现固定话术、泛化差，优先减少轮次。",
            evidence=f"训练 loss 降了 {train_drop * 100:.0f}%，验证 loss 在第 {best_epoch} 轮后回升。",
            action=DiagnosticAction(
                label=f"用 {best_epoch} 轮重试",
                action="retrain",
                params={"epochs": best_epoch},
            ),
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
            interpretation="模型可能记住了训练样本，但没有学到可迁移的规律。",
            suggestion="可以增加数据多样性，或减少训练轮次。",
            next_step="补一些不同问法、不同场景的样本，再重新训练。",
            evidence=f"训练 loss 降了 {train_drop * 100:.0f}%，验证 loss 只变化 {val_drop * 100:.1f}%。",
            action=DiagnosticAction(label="去补充数据", action="goto_data"),
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
                rank=60,
            )
        ]

    return [
        _card(
            level="warn",
            title="loss 几乎没降，模型可能没学到多少",
            observation=f"训练了 {epochs} 轮，loss 只从 {train[0]:.4f} 到 {train[-1]:.4f}。",
            interpretation="这通常说明数据里的信号不够清楚，或者这次训练强度不够。",
            suggestion="可以先检查数据质量，再尝试更强一点的训练档位。",
            next_step="先看样本是否太短、太重复；如果数据没问题，再增加轮次。",
            evidence=f"loss 降幅只有 {train_drop * 100:.1f}%。",
            action=DiagnosticAction(
                label="增加轮次重试",
                action="retrain",
                params={"epochs": min(epochs * 2, 30)},
            ),
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
            suggestion="可以去测评页看真实回答是否也变好了。",
            next_step="用几个没放进训练集的问题做对比，重点看回答风格和准确性。",
            evidence=f"训练 loss 降了 {train_drop * 100:.0f}%，验证 loss 降了 {val_drop * 100:.0f}%。",
            action=DiagnosticAction(label="去测评", action="goto_eval"),
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
            suggestion="可以把验证 loss 当参考，重点还是看测评回答。",
            next_step="用训练集之外的问题做对比；如果回答也不稳，再补更多样本。",
            evidence=f"验证 loss 有 {changes} 次方向反转，占 {rate * 100:.0f}%。",
            action=DiagnosticAction(label="去测评", action="goto_eval"),
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
                suggestion="可以把典型回答写得更完整一点。",
                next_step="补上原因、步骤或边界条件，让模型看到更明确的示范。",
                evidence=f"全部 {len(answers)} 条回答的平均长度为 {avg_len:.1f} 字。",
                action=DiagnosticAction(label="去补充数据", action="goto_data"),
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
                suggestion="可以去重，并补一些不同说法的真实问题。",
                next_step="保留最典型的一条，把重复项换成其他用户可能会问的表达。",
                evidence=(
                    f"重复率 {repetition_rate * 100:.0f}%"
                    f"（{len(unique_q)} 个不同问题 / {len(questions)} 条总数据）。"
                ),
                action=DiagnosticAction(label="去整理数据", action="goto_data"),
                rank=50,
            )
        )

    return cards
