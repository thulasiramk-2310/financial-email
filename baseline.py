from __future__ import annotations

"""
Baseline agent for the Financial Email OpenEnv environment.

Strategy: rule-based keyword heuristics for classification, risk, and decision.
This serves as the lower-bound benchmark that RL / LLM agents should beat.
"""

import os
import json
import requests
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Config — override via env vars or edit defaults below
# ─────────────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
DEFAULT_TASK = os.getenv("TASK_NAME", "hard")          # easy | medium | hard
NUM_EMAILS   = int(os.getenv("NUM_EMAILS", "5"))
EPISODES     = int(os.getenv("EPISODES", "3"))


# ─────────────────────────────────────────────────────────────────────────────
# Keyword heuristics
# ─────────────────────────────────────────────────────────────────────────────
SPAM_KEYWORDS    = {"won", "gift", "prize", "claim", "lottery", "guaranteed",
                     "double", "free", "offer", "selected", "congratulations",
                     "pre-approved", "limited time"}
URGENT_KEYWORDS  = {"urgent", "alert", "immediate", "action required", "down",
                     "failing", "unauthorized", "security", "verify", "critical",
                     "breach", "suspend"}
HIGH_RISK_WORDS  = {"wire transfer", "overseas", "$47,500", "unauthorized",
                     "ssn", "bank details", "phishing", "fraud", "critical",
                     "production down", "security alert", "scam", "won"}
MEDIUM_RISK_WORDS = {"overdue", "due", "compliance", "kyc", "audit", "re-submit",
                      "invoice", "legal action", "approved loan", "application"}


def _text(email: dict[str, Any]) -> str:
    return f"{email['subject']} {email['content']} {email['sender']}".lower()


def classify(email: dict[str, Any]) -> str:
    t = _text(email)
    if any(k in t for k in SPAM_KEYWORDS):
        return "spam"
    if any(k in t for k in URGENT_KEYWORDS):
        return "urgent"
    return "finance"


def assess_risk(email: dict[str, Any], classification: str) -> str:
    if classification == "spam":
        return "high"
    t = _text(email)
    if any(k in t for k in HIGH_RISK_WORDS):
        return "high"
    if any(k in t for k in MEDIUM_RISK_WORDS):
        return "medium"
    return "low"


def decide(classification: str, risk: str) -> str:
    if classification == "spam":
        return "block"
    if risk == "high":
        return "escalate"
    if risk == "medium":
        return "review"
    return "pass"


# ─────────────────────────────────────────────────────────────────────────────
# Environment interaction helpers
# ─────────────────────────────────────────────────────────────────────────────
def api(method: str, path: str, **kwargs) -> dict[str, Any]:
    url = f"{API_BASE_URL}{path}"
    resp = requests.request(method, url, **kwargs)
    resp.raise_for_status()
    return resp.json()


def run_episode(task_name: str = DEFAULT_TASK, num_emails: int = NUM_EMAILS) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"  Episode | task={task_name} | emails={num_emails}")
    print(f"{'='*60}")

    # Reset
    obs = api("POST", "/reset", json={"task_name": task_name, "num_emails": num_emails})
    total_reward = 0.0
    step = 0

    while True:
        email = obs.get("current_email")
        if email is None:
            break

        # Agent decision
        cls   = classify(email)
        risk  = assess_risk(email, cls) if task_name in {"medium", "hard"} else None
        dec   = decide(cls, risk or "low") if task_name == "hard" else None

        action: dict[str, Any] = {"action_type": "classify", "classification": cls}
        if risk:
            action["risk_level"] = risk
            action["action_type"] = "assign_risk"
        if dec:
            action["decision"] = dec
            action["action_type"] = "decide"

        # Step
        result = api("POST", "/step", json={"action": action})
        reward = result["reward"]
        total_reward += reward
        info = result.get("info", {})

        print(
            f"  Step {step+1:02d} | "
            f"cls={'✅' if info.get('classification_correct') else '❌'} "
            f"risk={'✅' if info.get('risk_correct') else ('➖' if risk is None else '❌')} "
            f"dec={'✅' if info.get('decision_correct') else ('➖' if dec is None else '❌')} "
            f"| reward={reward:+.1f} | total={total_reward:.1f}"
        )

        step += 1
        if result["done"]:
            break

        obs = result["observation"]

    # Grade
    grade_result = api("GET", "/grade/current")
    print(f"\n  📊 Grade: score={grade_result['score']:.3f}")
    print(f"  Breakdown: {grade_result['breakdown']}")
    print(f"  Metrics:   {grade_result['metrics']}")
    print(f"  Feedback:  {grade_result['feedback']}")

    return {
        "task": task_name,
        "total_reward": total_reward,
        "steps": step,
        "grade": grade_result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Financial Email Baseline Agent")
    print(f"   API: {API_BASE_URL}")
    print(f"   Task: {DEFAULT_TASK} | Emails per episode: {NUM_EMAILS} | Episodes: {EPISODES}")

    results = []
    for episode in range(EPISODES):
        print(f"\n▶ Episode {episode + 1}/{EPISODES}")
        result = run_episode(task_name=DEFAULT_TASK, num_emails=NUM_EMAILS)
        results.append(result)

    # Summary
    avg_score  = sum(r["grade"]["score"] for r in results) / len(results)
    avg_reward = sum(r["total_reward"] for r in results) / len(results)

    print(f"\n{'='*60}")
    print(f"  ✅ Baseline Summary ({EPISODES} episodes)")
    print(f"  Avg Score:  {avg_score:.3f}")
    print(f"  Avg Reward: {avg_reward:.2f}")
    print(f"{'='*60}\n")

    print(json.dumps({
        "baseline": "rule-based keyword heuristics",
        "task": DEFAULT_TASK,
        "episodes": EPISODES,
        "avg_score": round(avg_score, 4),
        "avg_reward": round(avg_reward, 4),
    }, indent=2))
