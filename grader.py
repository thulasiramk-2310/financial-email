from __future__ import annotations

from typing import Any

from models import GradeResult, TaskName


def _clamp_score(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _field_accuracy(history: list[dict[str, Any]], key: str) -> float:
    if not history:
        return 0.0
    hits = sum(1 for row in history if bool(row.get(key, False)))
    return hits / len(history)


def grade(task_name: TaskName, history: list[dict[str, Any]]) -> dict[str, Any]:
    classification_acc = _field_accuracy(history, "classification_correct")
    risk_acc = _field_accuracy(history, "risk_correct")
    decision_acc = _field_accuracy(history, "decision_correct")

    if task_name == "easy":
        score = classification_acc
    elif task_name == "medium":
        score = (classification_acc + risk_acc) / 2.0
    elif task_name == "hard":
        score = (classification_acc + risk_acc + decision_acc) / 3.0
    else:
        raise ValueError(f"Unknown task_name: {task_name}")

    false_positives = sum(
        1
        for row in history
        if row.get("predicted_risk_level") == "high" and row.get("effective_risk") in {"low", "medium"}
    )
    false_negatives = sum(
        1
        for row in history
        if row.get("predicted_risk_level") == "low" and row.get("effective_risk") == "high"
    )
    escalations = sum(1 for row in history if row.get("predicted_decision") == "escalate")

    result = GradeResult(
        score=_clamp_score(score),
        breakdown={
            "classification": _clamp_score(classification_acc),
            "risk": _clamp_score(risk_acc),
            "decision": _clamp_score(decision_acc),
        },
        metrics={
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives),
            "escalations": int(escalations),
        },
    )
    return result.model_dump()
