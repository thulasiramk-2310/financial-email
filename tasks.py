from __future__ import annotations

from models import TaskSpec


TASKS: list[TaskSpec] = [
    TaskSpec(
        name="easy",
        difficulty="easy",
        required_fields=["classification"],
        description="Classify each email as spam, finance, or urgent.",
    ),
    TaskSpec(
        name="medium",
        difficulty="medium",
        required_fields=["classification", "risk_level"],
        description="Classify each email and assign a risk level.",
    ),
    TaskSpec(
        name="hard",
        difficulty="hard",
        required_fields=["classification", "risk_level", "decision"],
        description="Run full pipeline: classification, risk assignment, and decision.",
    ),
]


TASK_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    task.name: tuple(task.required_fields) for task in TASKS
}


def get_tasks() -> list[dict]:
    return [task.model_dump() for task in TASKS]
