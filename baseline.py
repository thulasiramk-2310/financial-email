from __future__ import annotations

from models import Action, DecisionType, Observation, RiskType, TaskName
from server.environment import FinancialEmailEnv
from tasks import get_tasks
from grader import grade


KNOWN_DOMAINS = {"company.com", "trustedbank.com", "operations.bank"}


def _is_unknown_sender(sender: str) -> bool:
    if "@" not in sender:
        return True
    domain = sender.rsplit("@", 1)[-1].lower()
    return domain not in KNOWN_DOMAINS


def _rule_based_prediction(observation: Observation) -> tuple[str, RiskType, DecisionType]:
    subject = observation.current_email.subject.lower() if observation.current_email else ""
    sender = observation.current_email.sender if observation.current_email else ""

    if "urgent" in subject:
        classification = "urgent"
        risk: RiskType = "high"
    elif _is_unknown_sender(sender):
        classification = "spam"
        risk = "medium"
    else:
        classification = "finance"
        risk = "low"

    if risk == "high":
        decision: DecisionType = "escalate"
    elif risk == "medium":
        decision = "reject"
    else:
        decision = "approve"

    return classification, risk, decision


def _build_action(task_name: TaskName, observation: Observation) -> Action:
    classification, risk_level, decision = _rule_based_prediction(observation)

    if task_name == "easy":
        return Action(action_type="classify", classification=classification)
    if task_name == "medium":
        return Action(
            action_type="assign_risk",
            classification=classification,
            risk_level=risk_level,
        )
    return Action(
        action_type="decide",
        classification=classification,
        risk_level=risk_level,
        decision=decision,
    )


def run_task(task_name: TaskName) -> dict:
    env = FinancialEmailEnv()
    observation = env.reset(task_name=task_name)
    done = False
    history: list[dict] = []

    while not done:
        action = _build_action(task_name, observation)
        next_observation, _, done, info = env.step(action)

        history.append(
            {
                "classification_correct": bool(info.get("classification_correct", False)),
                "risk_correct": bool(info.get("risk_correct", False)),
                "decision_correct": bool(info.get("decision_correct", False)),
                "predicted_risk_level": info.get("predicted_risk_level"),
                "effective_risk": info.get("effective_risk"),
                "predicted_decision": info.get("predicted_decision"),
            }
        )

        if next_observation is not None:
            observation = next_observation

    return grade(task_name, history)


def run_baseline() -> dict:
    results: dict[str, dict] = {}
    for task in get_tasks():
        task_name = task["name"]
        results[task_name] = run_task(task_name)
    return results


def main() -> None:
    print("Baseline Results")
    print("================")
    results = run_baseline()
    for task_name, result in results.items():
        print(
            f"{task_name}: {result['score']:.3f} | "
            f"breakdown={result['breakdown']} | metrics={result['metrics']}"
        )


if __name__ == "__main__":
    main()
