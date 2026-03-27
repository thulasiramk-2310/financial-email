from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


LabelType = Literal["spam", "finance", "urgent"]
RiskType = Literal["low", "medium", "high"]
DecisionType = Literal["approve", "reject", "escalate"]
ActionType = Literal["classify", "assign_risk", "decide"]
TaskName = Literal["easy", "medium", "hard"]
Difficulty = Literal["easy", "medium", "hard"]


class Email(BaseModel):
    id: int = Field(..., ge=0)
    subject: str = Field(..., min_length=1)
    sender: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    true_label: LabelType
    true_risk: RiskType
    correct_decision: DecisionType


class Action(BaseModel):
    action_type: ActionType
    classification: Optional[str] = None
    risk_level: Optional[str] = None
    decision: Optional[str] = None


class Observation(BaseModel):
    current_email: Optional[Email] = None
    trust_score: float = Field(..., ge=0.0, le=1.0)
    step_count: int = Field(..., ge=0)
    task_name: TaskName


class State(BaseModel):
    emails: list[Email]
    current_index: int = Field(..., ge=0)
    trust_scores: dict[str, float]
    total_reward: float
    done: bool


class TaskSpec(BaseModel):
    name: TaskName
    difficulty: Difficulty
    required_fields: list[str]
    description: str


class GradeResult(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: dict[str, float]
    metrics: dict[str, int]


class StepResult(BaseModel):
    observation: Optional[Observation]
    reward: float
    done: bool
    info: dict[str, Any]
