from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from models import Action, Email, Observation, RiskType, State, TaskName
from tasks import TASK_REQUIRED_FIELDS


class FinancialEmailEnv:
    CLASSIFICATION_REWARD = 0.2
    RISK_REWARD = 0.3
    DECISION_REWARD = 0.5

    CLASSIFICATION_PENALTY = -0.1
    RISK_PENALTY = -0.2
    DECISION_PENALTY = -0.5

    APPROVE_HIGH_RISK_PENALTY = -1.0
    UNNECESSARY_ESCALATION_PENALTY = -0.1
    STEP_LIMIT_PENALTY = -0.2

    TRUST_GAIN_DECISION = 0.08
    TRUST_PENALTY_DECISION = 0.12
    TRUST_GAIN_GENERAL = 0.04
    TRUST_PENALTY_GENERAL = 0.06

    def __init__(self, data_path: Optional[Path] = None, initial_trust_score: float = 0.5) -> None:
        self._data_path = data_path or Path(__file__).with_name("data.json")
        self._initial_trust_score = self._clamp(initial_trust_score)
        self._source_emails = self._load_emails(self._data_path)

        self.active_task: TaskName = "hard"
        self.step_limit: int = len(self._source_emails)

        self.emails: list[Email] = []
        self.current_index: int = 0
        self.trust_scores: dict[str, float] = {}
        self.total_reward: float = 0.0
        self.done: bool = False

    def reset(self, task_name: TaskName = "hard", step_limit: Optional[int] = None) -> Observation:
        if task_name not in TASK_REQUIRED_FIELDS:
            raise ValueError(f"Unknown task_name: {task_name}")
        if not self._source_emails:
            raise ValueError("No emails found in dataset.")

        self.active_task = task_name
        self.emails = list(self._source_emails)
        self.current_index = 0
        self.total_reward = 0.0
        self.done = False
        self.step_limit = step_limit if step_limit is not None else len(self.emails)
        self.trust_scores = {email.sender: self._initial_trust_score for email in self.emails}

        return self._current_observation()

    def step(self, action: Action | dict[str, Any]) -> tuple[Optional[Observation], float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is complete. Call reset() before step().")
        if not self.emails:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        parsed_action = action if isinstance(action, Action) else Action.model_validate(action)
        required_fields = TASK_REQUIRED_FIELDS[self.active_task]
        self._validate_action_for_task(parsed_action, required_fields)

        current_email = self.emails[self.current_index]
        sender = current_email.sender
        trust_before = self.trust_scores.get(sender, self._initial_trust_score)
        effective_risk = self._biased_risk(current_email.true_risk, trust_before)

        field_results: dict[str, bool] = {}
        reward = 0.0

        if "classification" in required_fields:
            classification_correct = parsed_action.classification == current_email.true_label
            field_results["classification"] = classification_correct
            reward += self.CLASSIFICATION_REWARD if classification_correct else self.CLASSIFICATION_PENALTY

        if "risk_level" in required_fields:
            risk_correct = parsed_action.risk_level == effective_risk
            field_results["risk_level"] = risk_correct
            reward += self.RISK_REWARD if risk_correct else self.RISK_PENALTY

        if "decision" in required_fields:
            decision_correct = parsed_action.decision == current_email.correct_decision
            field_results["decision"] = decision_correct
            reward += self.DECISION_REWARD if decision_correct else self.DECISION_PENALTY

        if "decision" in required_fields and parsed_action.decision == "approve" and effective_risk == "high":
            reward += self.APPROVE_HIGH_RISK_PENALTY

        if "decision" in required_fields and parsed_action.decision == "escalate" and effective_risk == "low":
            reward += self.UNNECESSARY_ESCALATION_PENALTY

        if self.current_index >= self.step_limit:
            reward += self.STEP_LIMIT_PENALTY

        trust_after = self._update_trust(trust_before, field_results)
        self.trust_scores[sender] = trust_after

        self.total_reward += reward
        self.current_index += 1
        self.done = self.current_index >= len(self.emails)

        observation = None if self.done else self._current_observation()
        info = {
            "task_name": self.active_task,
            "required_fields": list(required_fields),
            "field_results": field_results,
            "classification_correct": bool(field_results.get("classification", False)),
            "risk_correct": bool(field_results.get("risk_level", False)),
            "decision_correct": bool(field_results.get("decision", False)),
            "predicted_classification": parsed_action.classification,
            "predicted_risk_level": parsed_action.risk_level,
            "predicted_decision": parsed_action.decision,
            "evaluated_email_id": current_email.id,
            "true_label": current_email.true_label,
            "true_risk": current_email.true_risk,
            "effective_risk": effective_risk,
            "correct_decision": current_email.correct_decision,
            "sender": sender,
            "trust_before": trust_before,
            "trust_after": trust_after,
            "current_index": self.current_index,
            "total_reward": self.total_reward,
        }
        return observation, reward, self.done, info

    def state(self) -> dict[str, Any]:
        state = State(
            emails=self.emails,
            current_index=self.current_index,
            trust_scores=dict(self.trust_scores),
            total_reward=self.total_reward,
            done=self.done,
        )
        return state.model_dump()

    def _current_observation(self) -> Observation:
        email = self.emails[self.current_index]
        return Observation(
            current_email=email,
            trust_score=self.trust_scores.get(email.sender, self._initial_trust_score),
            step_count=self.current_index,
            task_name=self.active_task,
        )

    def _update_trust(self, trust_before: float, field_results: dict[str, bool]) -> float:
        if "decision" in field_results:
            if field_results["decision"]:
                return self._clamp(trust_before + self.TRUST_GAIN_DECISION)
            return self._clamp(trust_before - self.TRUST_PENALTY_DECISION)

        if field_results and all(field_results.values()):
            return self._clamp(trust_before + self.TRUST_GAIN_GENERAL)
        return self._clamp(trust_before - self.TRUST_PENALTY_GENERAL)

    @staticmethod
    def _biased_risk(true_risk: RiskType, trust_score: float) -> RiskType:
        if trust_score >= 0.3:
            return true_risk

        if true_risk == "low":
            return "medium"
        if true_risk == "medium":
            return "high"
        return "high"

    @staticmethod
    def _validate_action_for_task(action: Action, required_fields: tuple[str, ...]) -> None:
        if "classification" in required_fields and action.classification is None:
            raise ValueError("classification is required for this task")
        if "risk_level" in required_fields and action.risk_level is None:
            raise ValueError("risk_level is required for this task")
        if "decision" in required_fields and action.decision is None:
            raise ValueError("decision is required for this task")

    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(value, upper))

    @staticmethod
    def _load_emails(path: Path) -> list[Email]:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return [Email(**row) for row in payload.get("emails", [])]
