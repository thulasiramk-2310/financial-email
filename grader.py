from __future__ import annotations
from typing import Any
from models import GradeResult, TaskName


def _clamp_score(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _field_accuracy(history: list[dict[str, Any]], key: str) -> float:
    """Proportion of rows where a boolean field is True."""
    if not history:
        return 0.0
    hits = sum(1 for row in history if bool(row.get(key, False)))
    return hits / len(history)


def _fn_penalty(false_negatives: int, total: int, weight: float = 0.25) -> float:
    """
    False negatives (missing high-risk emails) are more dangerous than false
    positives in a financial context, so we apply a configurable penalty.
    """
    if total == 0:
        return 0.0
    return (false_negatives / total) * weight


def _fp_penalty(false_positives: int, total: int, weight: float = 0.10) -> float:
    """
    False positives (over-flagging safe emails) reduce operational efficiency.
    Penalised at a lower rate than false negatives.
    """
    if total == 0:
        return 0.0
    return (false_positives / total) * weight


def grade(task_name: TaskName, history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return GradeResult(
            score=0.0,
            breakdown={"classification": 0.0, "risk": 0.0, "decision": 0.0},
            metrics={
                "false_positives": 0,
                "false_negatives": 0,
                "escalations": 0,
                "total_evaluated": 0,
            },
            feedback="No history provided — nothing to grade.",
        ).model_dump()

    total = len(history)

    # ── Core accuracy metrics ────────────────────────────────────────────────
    classification_acc = _field_accuracy(history, "classification_correct")
    risk_acc = _field_accuracy(history, "risk_correct")
    decision_acc = _field_accuracy(history, "decision_correct")

    # ── Error counts ─────────────────────────────────────────────────────────
    false_positives = sum(
        1
        for row in history
        if row.get("predicted_risk_level") == "high"
        and row.get("effective_risk") in {"low", "medium"}
    )
    false_negatives = sum(
        1
        for row in history
        if row.get("predicted_risk_level") == "low"
        and row.get("effective_risk") == "high"
    )
    escalations = sum(
        1 for row in history if row.get("predicted_decision") == "escalate"
    )
    blocks = sum(
        1 for row in history if row.get("predicted_decision") == "block"
    )

    # ── Task-specific scoring ────────────────────────────────────────────────
    if task_name == "easy":
        # Classification only — penalise false negatives (missed urgent/fraud)
        raw_score = classification_acc
        penalty = _fn_penalty(false_negatives, total, weight=0.20)
        score = _clamp_score(raw_score - penalty)

    elif task_name == "medium":
        # Classification + risk — weighted 40 / 60 (risk matters more here)
        raw_score = 0.4 * classification_acc + 0.6 * risk_acc
        penalty = (
            _fn_penalty(false_negatives, total, weight=0.25)
            + _fp_penalty(false_positives, total, weight=0.10)
        )
        score = _clamp_score(raw_score - penalty)

    elif task_name == "hard":
        # Full pipeline — classification 25%, risk 35%, decision 40%
        raw_score = (
            0.25 * classification_acc
            + 0.35 * risk_acc
            + 0.40 * decision_acc
        )
        penalty = (
            _fn_penalty(false_negatives, total, weight=0.30)
            + _fp_penalty(false_positives, total, weight=0.10)
        )
        score = _clamp_score(raw_score - penalty)

    else:
        raise ValueError(f"Unknown task_name: {task_name!r}")

    # ── Human-readable feedback ──────────────────────────────────────────────
    feedback_parts: list[str] = []

    if classification_acc < 0.6:
        feedback_parts.append(
            "Classification accuracy is low — review spam vs finance vs urgent boundaries."
        )
    if risk_acc < 0.6 and task_name in {"medium", "hard"}:
        feedback_parts.append(
            "Risk scoring is weak — ensure high-risk signals (large amounts, unknown senders) are weighted."
        )
    if decision_acc < 0.6 and task_name == "hard":
        feedback_parts.append(
            "Decision logic needs work — verify block/escalate/review/pass rules align with risk levels."
        )
    if false_negatives > total * 0.2:
        feedback_parts.append(
            f"High false-negative rate ({false_negatives}/{total}): dangerous high-risk emails are being missed."
        )
    if false_positives > total * 0.3:
        feedback_parts.append(
            f"High false-positive rate ({false_positives}/{total}): too many safe emails flagged as high-risk."
        )
    if not feedback_parts:
        feedback_parts.append("Good performance across all metrics.")

    result = GradeResult(
        score=score,
        breakdown={
            "classification": _clamp_score(classification_acc),
            "risk": _clamp_score(risk_acc),
            "decision": _clamp_score(decision_acc),
        },
        metrics={
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives),
            "escalations": int(escalations),
            "blocks": int(blocks),
            "total_evaluated": total,
        },
        feedback=" | ".join(feedback_parts),
    )
    return result.model_dump()