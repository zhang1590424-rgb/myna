"""Training diagnostics: rule-based analysis after training completes.

Produces structured DiagnosticCard list from an Experiment's loss curves,
dataset statistics, and training parameters. Multi-dimensional backend analysis
with actionable suggestions.
"""
from __future__ import annotations

from .domain import DiagnosticAction, DiagnosticCard, Experiment


def compute_diagnostics(
    exp: Experiment,
    dataset_rows: list[dict] | None = None,
) -> list[DiagnosticCard]:
    """Run all diagnostic rules and return cards (may be empty if all healthy)."""
    cards: list[DiagnosticCard] = []

    cards += _check_data_quantity(exp)
    cards += _check_loss_oscillation(exp)
    cards += _check_initial_loss_anomaly(exp)
    cards += _check_loss_still_dropping(exp)
    cards += _check_classic_overfit(exp)
    cards += _check_memorization_overfit(exp)
    cards += _check_underfit(exp)
    cards += _check_dpo_reward_margin(exp)
    cards += _check_data_quality(exp, dataset_rows)

    if not cards:
        return [_build_ok_card(exp)]
    return cards


def compute_live_diagnostics(exp: Experiment) -> list[DiagnosticCard]:
    """Compute diagnostics that can fire during training (no dataset read needed).

    Only includes rules that make sense with partial loss data.
    Returns empty list when nothing noteworthy yet.
    """
    cards: list[DiagnosticCard] = []

    cards += _check_data_quantity(exp)
    cards += _check_initial_loss_anomaly(exp)
    cards += _check_loss_oscillation(exp)
    cards += _check_classic_overfit(exp)

    return cards


def preflight_data_check(
    dataset_rows: list[dict],
    method: str = "sft",
) -> list[DiagnosticCard]:
    """Pre-training data quality check. Run before starting training."""
    cards: list[DiagnosticCard] = []

    n = len(dataset_rows)
    if n < 15:
        cards.append(
            DiagnosticCard(
                level="warn",
                title=f"数据仅 {n} 条，模型可能学不到稳定模式",
                suggestion="建议补充到 20 条以上再开始训练。",
                evidence="少于 15 条时模型容易记住而非学到通用规律。",
                action=DiagnosticAction(label="去补充数据", action="goto_data"),
            )
        )

    # Answer length check
    answers = [
        row.get("output") or row.get("answer") or row.get("chosen") or ""
        for row in dataset_rows
    ]
    if answers:
        avg_len = sum(len(a) for a in answers) / len(answers)
        if avg_len < 25:
            cards.append(
                DiagnosticCard(
                    level="warn",
                    title=f"回答平均仅 {avg_len:.0f} 字，信息量可能不足",
                    suggestion="短回答难以让模型学到足够的风格信号，建议丰富回答内容。",
                    evidence=f"全部 {len(answers)} 条回答的平均长度为 {avg_len:.1f} 字。",
                    action=DiagnosticAction(label="去补充数据", action="goto_data"),
                )
            )

    # Repetition check
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
                    DiagnosticCard(
                        level="warn",
                        title=f"数据中有 {dup_count} 条重复问题",
                        suggestion="重复问题会让模型对这些问题过度记忆，建议去重或替换为新场景。",
                        evidence=f"重复率 {repetition_rate * 100:.0f}%（{len(unique_q)} 个不同问题 / {len(questions)} 条总数据）。",
                        action=DiagnosticAction(label="去补充数据", action="goto_data"),
                    )
                )

    return cards


# --------------------------------------------------------------------------- #
# Individual rules
# --------------------------------------------------------------------------- #


def _build_ok_card(exp: Experiment) -> DiagnosticCard:
    """Generate an ok card with a human-readable training summary."""
    train = [v for v in exp.loss if isinstance(v, (int, float)) and v == v]
    parts = []
    if exp.dataset_count > 0:
        parts.append(f"{exp.dataset_count} 条数据")
    parts.append(f"{exp.params.epochs} 轮训练")
    if len(train) >= 2:
        parts.append(f"loss 从 {train[0]:.2f} 降到 {train[-1]:.2f}")

    summary = "、".join(parts) + "。" if parts else ""
    return DiagnosticCard(
        level="ok",
        title="训练指标看起来正常",
        suggestion=f"{summary}去测评页对比实际效果吧。",
        action=DiagnosticAction(label="去测评", action="goto_eval"),
    )


def _check_data_quantity(exp: Experiment) -> list[DiagnosticCard]:
    n = exp.dataset_count
    if n <= 0:
        return []
    if n < 15:
        return [
            DiagnosticCard(
                level="warn",
                title=f"训练数据仅 {n} 条，模型可能学不到稳定的模式",
                suggestion="建议补充到 20 条以上再重新训练。",
                evidence=f"当前数据集 {n} 条。少于 15 条时模型容易记住而非学到通用规律。",
                action=DiagnosticAction(label="去补充数据", action="goto_data"),
            )
        ]
    return []


def _check_loss_oscillation(exp: Experiment) -> list[DiagnosticCard]:
    """Detect high loss oscillation (unstable training)."""
    train = [v for v in exp.loss if isinstance(v, (int, float)) and v == v]
    if len(train) < 5:
        return []

    # Compute direction changes
    direction_changes = 0
    for i in range(2, len(train)):
        prev_dir = train[i - 1] - train[i - 2]
        curr_dir = train[i] - train[i - 1]
        if prev_dir * curr_dir < 0:
            direction_changes += 1

    oscillation_rate = direction_changes / (len(train) - 2)
    if oscillation_rate > 0.6:
        return [
            DiagnosticCard(
                level="warn",
                title="训练 loss 抖动较大，学习过程不够稳定",
                suggestion="可尝试降低学习率，或增加数据量让训练更平稳。",
                evidence=f"loss 曲线有 {direction_changes} 次方向反转（占 {oscillation_rate * 100:.0f}%），正常训练应平稳下降。",
                action=DiagnosticAction(
                    label="用更低学习率重试",
                    action="retrain",
                    params={"learning_rate": exp.params.learning_rate * 0.5},
                ),
            )
        ]
    return []


def _check_initial_loss_anomaly(exp: Experiment) -> list[DiagnosticCard]:
    """Detect abnormally high or low initial loss (data-model mismatch)."""
    train = [v for v in exp.loss if isinstance(v, (int, float)) and v == v]
    if not train:
        return []
    initial = train[0]

    if initial > 8.0:
        return [
            DiagnosticCard(
                level="warn",
                title=f"初始 loss 很高（{initial:.2f}），数据与模型可能不太匹配",
                suggestion="检查数据格式是否符合模型期望，或尝试换一个模型。",
                evidence="正常 SFT 初始 loss 通常在 1~5 之间，过高可能说明数据内容和模型预训练分布差异大。",
            )
        ]
    if initial < 0.3 and exp.dataset_count > 5:
        return [
            DiagnosticCard(
                level="warn",
                title=f"初始 loss 就很低（{initial:.2f}），模型可能已经会了",
                suggestion="说明模型本身就能应对这类数据，训练效果提升可能有限。去测评页确认一下。",
                evidence="初始 loss 极低意味着模型在训练前就能较好预测答案。",
                action=DiagnosticAction(label="去测评", action="goto_eval"),
            )
        ]
    return []


def _check_loss_still_dropping(exp: Experiment) -> list[DiagnosticCard]:
    """Detect if loss is still significantly dropping at the end of training."""
    train = [v for v in exp.loss if isinstance(v, (int, float)) and v == v]
    if len(train) < 3:
        return []

    # Use last 1/3 window average slope instead of just last 2 points
    window_size = max(2, len(train) // 3)
    tail = train[-window_size:]
    if len(tail) < 2:
        return []

    # Average drop per step in the tail window
    avg_first_half = sum(tail[: len(tail) // 2]) / (len(tail) // 2)
    avg_second_half = sum(tail[len(tail) // 2 :]) / (len(tail) - len(tail) // 2)
    if avg_first_half <= 0:
        return []

    drop_rate = (avg_first_half - avg_second_half) / avg_first_half
    epochs = exp.params.epochs

    if drop_rate > 0.05:
        suggested_epochs = min(epochs + 3, 30)
        return [
            DiagnosticCard(
                level="warn",
                title="训练 loss 仍在明显下降，模型可能还没学完",
                suggestion=f"尝试将训练轮次从 {epochs} 增加到 {suggested_epochs}。",
                evidence=f"训练尾段平均 loss 从 {avg_first_half:.4f} 降到 {avg_second_half:.4f}，降幅 {drop_rate * 100:.1f}%。",
                action=DiagnosticAction(
                    label="增加轮次重试",
                    action="retrain",
                    params={"epochs": suggested_epochs},
                ),
            )
        ]
    return []


def _check_classic_overfit(exp: Experiment) -> list[DiagnosticCard]:
    train = [v for v in exp.loss if isinstance(v, (int, float)) and v == v]
    val = [v for v in exp.eval_loss if isinstance(v, (int, float)) and v == v]
    if len(train) < 2 or len(val) < 2:
        return []

    train_first, train_last = train[0], train[-1]
    if train_first <= 0:
        return []
    train_drop = (train_first - train_last) / train_first

    val_min = min(val)
    val_last = val[-1]
    val_rose = val_last > val_min * 1.05  # 降低阈值，5% 即提醒

    if train_drop > 0.1 and val_rose:
        best_epoch = val.index(val_min) + 1
        return [
            DiagnosticCard(
                level="error",
                title="过拟合：模型开始背答案而非学规律",
                suggestion=f"减少训练轮次到第 {best_epoch} 轮，或增加训练数据。",
                evidence=f"训练 loss 降了 {train_drop * 100:.0f}%，但验证 loss 在第 {best_epoch} 轮后回升。",
                action=DiagnosticAction(
                    label=f"用 {best_epoch} 轮重试",
                    action="retrain",
                    params={"epochs": best_epoch},
                ),
            )
        ]
    return []


def _check_memorization_overfit(exp: Experiment) -> list[DiagnosticCard]:
    train = [v for v in exp.loss if isinstance(v, (int, float)) and v == v]
    val = [v for v in exp.eval_loss if isinstance(v, (int, float)) and v == v]
    if len(train) < 2 or len(val) < 2:
        return []

    train_first, train_last = train[0], train[-1]
    if train_first <= 0:
        return []
    train_drop = (train_first - train_last) / train_first

    val_first, val_last = val[0], val[-1]
    if val_first <= 0:
        return []
    val_drop = (val_first - val_last) / val_first

    if train_drop > 0.75 and abs(val_drop) < 0.1:
        return [
            DiagnosticCard(
                level="error",
                title="训练 loss 接近 0 但验证 loss 没变化，模型记住了数据",
                suggestion="增加数据多样性，或减少训练轮次。",
                evidence=f"训练 loss 降了 {train_drop * 100:.0f}%，验证 loss 仅变化 {val_drop * 100:.1f}%。",
                action=DiagnosticAction(label="去补充数据", action="goto_data"),
            )
        ]
    return []


def _check_underfit(exp: Experiment) -> list[DiagnosticCard]:
    train = [v for v in exp.loss if isinstance(v, (int, float)) and v == v]
    if len(train) < 2:
        return []
    epochs = exp.params.epochs
    if epochs < 3:
        return []

    train_first, train_last = train[0], train[-1]
    if train_first <= 0:
        return []
    train_drop = (train_first - train_last) / train_first

    if train_drop < 0.1:
        return [
            DiagnosticCard(
                level="warn",
                title="训练 loss 几乎没下降，模型似乎没有学到有效信息",
                suggestion="检查数据质量，或尝试「精细」档增加训练量。",
                evidence=f"经过 {epochs} 轮训练，loss 仅从 {train_first:.4f} 降到 {train_last:.4f}（降幅 {train_drop * 100:.1f}%）。",
                action=DiagnosticAction(
                    label="用精细档重试",
                    action="retrain",
                    params={"epochs": min(epochs * 2, 30)},
                ),
            )
        ]
    return []


def _check_dpo_reward_margin(exp: Experiment) -> list[DiagnosticCard]:
    """DPO-specific: check if reward margin is improving."""
    if exp.method != "dpo":
        return []
    train = [v for v in exp.loss if isinstance(v, (int, float)) and v == v]
    if len(train) < 3:
        return []

    # For DPO, loss should decrease meaningfully; if it plateaus, preference signal weak
    train_first, train_last = train[0], train[-1]
    if train_first <= 0:
        return []
    train_drop = (train_first - train_last) / train_first

    if train_drop < 0.05 and exp.params.epochs >= 2:
        return [
            DiagnosticCard(
                level="warn",
                title="DPO 训练 loss 几乎不变，偏好信号可能不明显",
                suggestion="检查 chosen 和 rejected 之间是否有足够区分度，或增加数据对数。",
                evidence=f"经过 {exp.params.epochs} 轮，DPO loss 仅从 {train_first:.4f} 变为 {train_last:.4f}。",
                action=DiagnosticAction(label="去补充数据", action="goto_data"),
            )
        ]

    # DPO loss going UP means reward hacking or data issue
    if train_last > train_first * 1.1:
        return [
            DiagnosticCard(
                level="error",
                title="DPO loss 反而上升了，模型没有朝期望方向学习",
                suggestion="检查数据的 chosen/rejected 标注是否正确，或降低学习率重试。",
                evidence=f"DPO loss 从 {train_first:.4f} 升到 {train_last:.4f}。",
                action=DiagnosticAction(
                    label="降低学习率重试",
                    action="retrain",
                    params={"learning_rate": exp.params.learning_rate * 0.5},
                ),
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
        ans = row.get("output") or row.get("answer") or ""
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
            DiagnosticCard(
                level="warn",
                title=f"训练数据回答平均仅 {avg_len:.0f} 字，信息量可能不足",
                suggestion="短回答难以让模型学到足够的风格信号，建议丰富回答内容。",
                evidence=f"全部 {len(answers)} 条回答的平均长度为 {avg_len:.1f} 字。",
                action=DiagnosticAction(label="去补充数据", action="goto_data"),
            )
        )

    if repetition_rate > 0.2:
        dup_count = len(questions) - len(unique_q)
        cards.append(
            DiagnosticCard(
                level="warn",
                title=f"训练数据中有 {dup_count} 条重复问题",
                suggestion="重复问题会让模型对这些问题过度记忆，建议去重或替换为新场景。",
                evidence=f"重复率 {repetition_rate * 100:.0f}%（{len(unique_q)} 个不同问题 / {len(questions)} 条总数据）。",
                action=DiagnosticAction(label="去补充数据", action="goto_data"),
            )
        )

    return cards
