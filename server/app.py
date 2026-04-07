from __future__ import annotations

import random
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import (
    Action, Email, Observation, StepResult,
    TaskName,
)
from tasks import TASKS, TASK_REQUIRED_FIELDS
from grader import grade
from baseline import classify as baseline_classify, assess_risk as baseline_assess_risk, decide as baseline_decide


# ─────────────────────────────────────────────────────────────────────────────
# Sample email pool
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_EMAILS: list[dict[str, Any]] = [
    {"subject": "You won a $1000 gift card!", "sender": "promo@spam.com",
     "content": "Click here to claim your reward before it expires!",
     "true_label": "spam", "true_risk": "high", "correct_decision": "block"},

    {"subject": "Invoice #INV-4521 – Payment Due", "sender": "billing@vendor.com",
     "content": "Please find attached invoice for $3,200. Payment due in 15 days.",
     "true_label": "finance", "true_risk": "medium", "correct_decision": "review"},

    {"subject": "URGENT: Production DB is down", "sender": "ops@company.com",
     "content": "All services are failing. Immediate action required.",
     "true_label": "urgent", "true_risk": "high", "correct_decision": "escalate"},

    {"subject": "March Salary Credited", "sender": "hr@company.com",
     "content": "Your salary of $5,400 has been processed successfully.",
     "true_label": "finance", "true_risk": "low", "correct_decision": "pass"},

    {"subject": "Security Alert: Unknown Login", "sender": "security@bank.com",
     "content": "Unauthorized login detected from a new device. Verify now.",
     "true_label": "urgent", "true_risk": "high", "correct_decision": "escalate"},

    {"subject": "Loan Application Approved", "sender": "loans@bank.com",
     "content": "Your loan has been approved. Please review the terms.",
     "true_label": "finance", "true_risk": "medium", "correct_decision": "review"},

    {"subject": "Double your money in 24 hours!", "sender": "invest@scam.net",
     "content": "Guaranteed returns. Send $500 to get started.",
     "true_label": "spam", "true_risk": "high", "correct_decision": "block"},

    {"subject": "Account Statement Ready", "sender": "noreply@bank.com",
     "content": "Your monthly statement is available for download.",
     "true_label": "finance", "true_risk": "low", "correct_decision": "pass"},
]


# ─────────────────────────────────────────────────────────────────────────────
# In-memory environment
# ─────────────────────────────────────────────────────────────────────────────
env_state: dict[str, Any] = {}


def _build_emails(n: int = 5):
    pool = random.sample(SAMPLE_EMAILS, min(n, len(SAMPLE_EMAILS)))
    return [Email(id=i, **e) for i, e in enumerate(pool)]


def _compute_reward(email: Email, action: Action, task_name: TaskName) -> float:
    reward = 0.0

    if action.classification == email.true_label:
        reward += 1.0

    if task_name in {"medium", "hard"} and action.risk_level == email.true_risk:
        reward += 1.0

    if task_name == "hard":
        if action.decision == email.correct_decision:
            reward += 1.0

        if email.true_risk == "high" and action.decision == "pass":
            reward -= 1.5

    return reward


# ─────────────────────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    env_state.clear()
    yield


app = FastAPI(
    title="Financial Email OpenEnv",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_name: TaskName = "hard"
    num_emails: int = 5

    model_config = {"extra": "ignore"}


class StepRequest(BaseModel):
    action: Action


class GradeRequest(BaseModel):
    task_name: TaskName
    history: list[dict[str, Any]]


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running"}


@app.get("/tasks")
def list_tasks():
    return [t.model_dump() for t in TASKS]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = Body(default=None)):
    if req is None:
        req = ResetRequest()

    emails = _build_emails(req.num_emails)

    env_state.clear()
    env_state.update({
        "task_name": req.task_name,
        "emails": emails,
        "current_index": 0,
        "trust_scores": {str(e.id): 1.0 for e in emails},
        "total_reward": 0.0,
        "history": [],
        "done": False,
    })

    first_email = emails[0]

    return Observation(
        current_email=first_email,
        trust_score=1.0,
        step_count=0,
        task_name=req.task_name,
    )


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    if not env_state:
        raise HTTPException(400, "Call /reset first")

    if env_state["done"]:
        raise HTTPException(400, "Episode finished. Reset required")

    task_name = env_state["task_name"]
    emails = env_state["emails"]
    idx = env_state["current_index"]

    current_email = emails[idx]
    action = req.action

    # Validation
    required = TASK_REQUIRED_FIELDS[task_name]
    missing = [f for f in required if getattr(action, f, None) is None]

    if missing:
        raise HTTPException(400, f"Missing required fields: {missing}")

    # Scoring
    classification_correct = action.classification == current_email.true_label
    risk_correct = action.risk_level == current_email.true_risk if task_name != "easy" else None
    decision_correct = action.decision == current_email.correct_decision if task_name == "hard" else None

    reward = _compute_reward(current_email, action, task_name)
    env_state["total_reward"] += reward

    # Trust update
    trust = env_state["trust_scores"][str(current_email.id)]
    trust += 0.05 if classification_correct else -0.05
    trust = max(0.0, min(1.0, trust))
    env_state["trust_scores"][str(current_email.id)] = trust

    # History
    env_state["history"].append({
        "email_id": current_email.id,
        "classification_correct": classification_correct,
        "risk_correct": bool(risk_correct) if risk_correct is not None else False,
        "decision_correct": bool(decision_correct) if decision_correct is not None else False,
    })

    # Move forward
    next_index = idx + 1
    done = next_index >= len(emails)

    env_state["current_index"] = next_index
    env_state["done"] = done

    next_obs = None
    if not done:
        next_email = emails[next_index]
        next_obs = Observation(
            current_email=next_email,
            trust_score=env_state["trust_scores"][str(next_email.id)],
            step_count=next_index,
            task_name=task_name,
        )

    return StepResult(
        observation=next_obs,
        reward=reward,
        done=done,
        info={"total_reward": env_state["total_reward"]},
    )


@app.post("/grade")
def grade_episode(req: GradeRequest):
    if not req.history and env_state.get("history"):
        return grade(req.task_name, env_state["history"])

    return grade(req.task_name, req.history)


@app.get("/grade/current")
def grade_current():
    task_name = env_state.get("task_name", "hard") if env_state else "hard"
    history = env_state.get("history", []) if env_state else []
    return grade(task_name, history)


@app.get("/grader")
def grader_current():
    return grade_current()


@app.get("/baseline")
def baseline_run():
    # Deterministic baseline over current episode emails if available; otherwise fixed sample set.
    task_name: TaskName = env_state.get("task_name", "hard")
    emails: list[Email] = env_state.get("emails") or [Email(id=i, **e) for i, e in enumerate(SAMPLE_EMAILS[:5])]
    history: list[dict[str, Any]] = []

    for email in emails:
        cls = baseline_classify(email.model_dump())
        risk = baseline_assess_risk(email.model_dump(), cls) if task_name in {"medium", "hard"} else None
        dec = baseline_decide(cls, risk or "low") if task_name == "hard" else None
        history.append(
            {
                "classification_correct": cls == email.true_label,
                "risk_correct": risk == email.true_risk,
                "decision_correct": dec == email.correct_decision,
                "predicted_risk_level": risk,
                "effective_risk": email.true_risk,
                "predicted_decision": dec,
            }
        )

    return {
        "task_name": task_name,
        "evaluated": len(history),
        "grade": grade(task_name, history),
    }


@app.get("/state")
def state():
    return env_state


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT (REQUIRED FOR DEPLOYMENT)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )


if __name__ == "__main__":
    main()
