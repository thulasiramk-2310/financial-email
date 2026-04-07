from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from grader import grade

SPAM_KEYWORDS = {"won", "gift", "prize", "claim", "lottery", "guaranteed", "double", "free", "offer", "selected", "congratulations", "pre-approved", "limited time"}
URGENT_KEYWORDS = {"urgent", "alert", "immediate", "action required", "down", "failing", "unauthorized", "security", "verify", "critical", "breach", "suspend"}
HIGH_RISK_WORDS = {"wire transfer", "overseas", "unauthorized", "ssn", "bank details", "phishing", "fraud", "critical", "production down", "security alert"}
MEDIUM_RISK_WORDS = {"overdue", "due", "compliance", "kyc", "audit", "re-submit", "invoice", "legal action", "approved loan", "application"}

SAMPLE_EMAILS = [
    {"subject": "Wire Transfer of $47,500 Initiated", "sender": "alerts@bank.com", "content": "A wire transfer to an overseas account was initiated. Confirm or deny.", "true_label": "urgent", "true_risk": "high", "correct_decision": "escalate"},
    {"subject": "You won a $1000 gift card!", "sender": "promo@spam.com", "content": "Click here to claim your free reward before it expires!", "true_label": "spam", "true_risk": "high", "correct_decision": "block"},
    {"subject": "Invoice #INV-4521 Payment Due", "sender": "billing@vendor.com", "content": "Please find attached invoice for $3,200. Payment due in 15 days.", "true_label": "finance", "true_risk": "medium", "correct_decision": "review"},
    {"subject": "March Salary Credited", "sender": "hr@company.com", "content": "Your salary of $5,400 has been processed successfully.", "true_label": "finance", "true_risk": "low", "correct_decision": "pass"},
    {"subject": "URGENT: Production DB is down", "sender": "ops@company.com", "content": "All services are failing. Immediate action required.", "true_label": "urgent", "true_risk": "high", "correct_decision": "escalate"},
]


def _text(email: dict) -> str:
    return f"{email.get('subject', '')} {email.get('content', '')} {email.get('sender', '')}".lower()


def classify(email: dict) -> str:
    t = _text(email)
    if any(k in t for k in SPAM_KEYWORDS):
        return "spam"
    if any(k in t for k in URGENT_KEYWORDS):
        return "urgent"
    return "finance"


def assess_risk(email: dict, cls: str) -> str:
    if cls == "spam":
        return "high"
    t = _text(email)
    if any(k in t for k in HIGH_RISK_WORDS):
        return "high"
    if any(k in t for k in MEDIUM_RISK_WORDS):
        return "medium"
    return "low"


def decide(cls: str, risk: str) -> str:
    if cls == "spam":
        return "block"
    if risk == "high":
        return "escalate"
    if risk == "medium":
        return "review"
    return "pass"


def predict(email: dict, task_name: str = "hard") -> dict:
    cls = classify(email)
    risk = assess_risk(email, cls)
    dec = decide(cls, risk)
    result: dict = {"classification": cls}
    if task_name in {"medium", "hard"}:
        result["risk_level"] = risk
    if task_name == "hard":
        result["decision"] = dec
    return result


def run_inference(emails: list[dict], task_name: str = "hard") -> dict:
    history = []
    for email in emails:
        pred = predict(email, task_name)
        history.append(
            {
                "classification_correct": pred.get("classification") == email.get("true_label"),
                "risk_correct": pred.get("risk_level") == email.get("true_risk"),
                "decision_correct": pred.get("decision") == email.get("correct_decision"),
                "predicted_risk_level": pred.get("risk_level"),
                "effective_risk": email.get("true_risk"),
                "predicted_decision": pred.get("decision"),
            }
        )
    return grade(task_name, history)


def _step_reward(email: dict, pred: dict, task_name: str) -> float:
    reward = 0.0
    if pred.get("classification") == email.get("true_label"):
        reward += 0.2
    else:
        reward -= 0.1
    if task_name in {"medium", "hard"}:
        if pred.get("risk_level") == email.get("true_risk"):
            reward += 0.3
        else:
            reward -= 0.2
    if task_name == "hard":
        if pred.get("decision") == email.get("correct_decision"):
            reward += 0.5
        else:
            reward -= 0.5
    return reward


def _normalize_task_name(raw: str) -> str:
    task_name = (raw or "hard").strip().lower()
    if task_name not in {"easy", "medium", "hard"}:
        return "hard"
    return task_name


def _extract_emails(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        candidate = payload.get("emails", payload)
    else:
        candidate = payload
    if isinstance(candidate, list):
        return [row for row in candidate if isinstance(row, dict)]
    return []


def _load_emails_from_input() -> list[dict]:
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            return _extract_emails(json.load(f))

    if not sys.stdin.isatty():
        raw = sys.stdin.read()
        if raw and raw.strip():
            return _extract_emails(json.loads(raw))

    return list(SAMPLE_EMAILS)


if __name__ == "__main__":
    task_name = _normalize_task_name(os.getenv("TASK_NAME", "hard"))
    print(f"[START] task={task_name}", flush=True)

    history: list[dict] = []
    step_count = 0
    score = 0.0

    try:
        try:
            emails = _load_emails_from_input()
        except Exception:
            emails = list(SAMPLE_EMAILS)

        if not emails:
            emails = list(SAMPLE_EMAILS)
        if not emails:
            emails = [{}]

        for step, email in enumerate(emails, start=1):
            step_count = step
            reward = 0.0
            try:
                pred = predict(email, task_name)
                reward = _step_reward(email, pred, task_name)
                history.append(
                    {
                        "classification_correct": pred.get("classification") == email.get("true_label"),
                        "risk_correct": pred.get("risk_level") == email.get("true_risk"),
                        "decision_correct": pred.get("decision") == email.get("correct_decision"),
                        "predicted_risk_level": pred.get("risk_level"),
                        "effective_risk": email.get("true_risk"),
                        "predicted_decision": pred.get("decision"),
                    }
                )
            except Exception:
                history.append(
                    {
                        "classification_correct": False,
                        "risk_correct": False,
                        "decision_correct": False,
                        "predicted_risk_level": None,
                        "effective_risk": email.get("true_risk"),
                        "predicted_decision": None,
                    }
                )
            print(f"[STEP] step={step} reward={reward:.3f}", flush=True)

        try:
            score = float(grade(task_name, history).get("score", 0.0))
        except Exception:
            score = 0.0
    finally:
        if step_count == 0:
            step_count = 1
            print("[STEP] step=1 reward=0.000", flush=True)
        print(f"[END] task={task_name} score={score:.3f} steps={step_count}", flush=True)
