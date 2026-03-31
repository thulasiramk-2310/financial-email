from __future__ import annotations

import random
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── adjust import path when running inside server/ ───────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import (
    Action, Email, Observation, State, StepResult,
    LabelType, RiskType, DecisionType, TaskName,
)
from tasks import TASKS, TASK_REQUIRED_FIELDS
from grader import grade

# ─────────────────────────────────────────────────────────────────────────────
# Sample email pool  (env randomly draws from these each episode)
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

    {"subject": "Loan Application #LN-8823 Approved", "sender": "loans@bank.com",
     "content": "Your loan has been approved. Please review the terms.",
     "true_label": "finance", "true_risk": "medium", "correct_decision": "review"},

    {"subject": "Double your money in 24 hours!", "sender": "invest@scam.net",
     "content": "Guaranteed returns. Send $500 to get started.",
     "true_label": "spam", "true_risk": "high", "correct_decision": "block"},

    {"subject": "Q1 Account Statement Ready", "sender": "noreply@bank.com",
     "content": "Your monthly statement is available for download.",
     "true_label": "finance", "true_risk": "low", "correct_decision": "pass"},

    {"subject": "Wire Transfer of $47,500 Initiated", "sender": "alerts@bank.com",
     "content": "A wire transfer to an overseas account was initiated. Confirm or deny.",
     "true_label": "urgent", "true_risk": "high", "correct_decision": "escalate"},

    {"subject": "KYC Re-submission Required", "sender": "compliance@bank.com",
     "content": "Please re-submit your KYC documents within 7 days to avoid account freeze.",
     "true_label": "finance", "true_risk": "medium", "correct_decision": "review"},
]

# ─────────────────────────────────────────────────────────────────────────────
# In-memory environment state
# ─────────────────────────────────────────────────────────────────────────────
env_state: dict[str, Any] = {}


def _build_emails(n: int = 5) -> list[Email]:
    pool = random.sample(SAMPLE_EMAILS, min(n, len(SAMPLE_EMAILS)))
    return [Email(id=i, **e) for i, e in enumerate(pool)]


def _compute_reward(email: Email, action: Action, task_name: TaskName) -> float:
    reward = 0.0
    if action.classification == email.true_label:
        reward += 1.0
    if task_name in {"medium", "hard"} and action.risk_level == email.true_risk:
        reward += 1.0
    if task_name == "hard" and action.decision == email.correct_decision:
        reward += 1.0
        # Extra penalty for missing high-risk emails
        if email.true_risk == "high" and action.decision == "pass":
            reward -= 1.5
    return reward


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    env_state.clear()
    yield


app = FastAPI(
    title="Financial Email OpenEnv",
    description="RL environment for financial email triage — classify, assess risk, and decide.",
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

    model_config = {"extra": "ignore"}  # ignore unknown fields from checker


class StepRequest(BaseModel):
    action: Action


class GradeRequest(BaseModel):
    task_name: TaskName
    history: list[dict[str, Any]]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "running"}


@app.get("/tasks", tags=["Tasks"])
def list_tasks():
    """List all available tasks with descriptions and examples."""
    return [t.model_dump() for t in TASKS]


@app.post("/reset", response_model=Observation, tags=["Environment"])
def reset(req: Optional[ResetRequest] = Body(default=None)):
    """Reset the environment and return the first observation."""
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
        trust_score=env_state["trust_scores"][str(first_email.id)],
        step_count=0,
        task_name=req.task_name,
    )


@app.post("/step", response_model=StepResult, tags=["Environment"])
def step(req: StepRequest):
    """Take one action on the current email and advance the environment."""
    if not env_state:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    if env_state["done"]:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset to start a new one.")

    task_name: TaskName = env_state["task_name"]
    emails: list[Email] = env_state["emails"]
    idx: int = env_state["current_index"]
    action = req.action

    # Validate required fields per task
    required = TASK_REQUIRED_FIELDS[task_name]
    missing = [f for f in required if getattr(action, f.replace("_level", "").replace("risk", "risk_level"), None) is None]

    current_email = emails[idx]

    # Score the action
    classification_correct = action.classification == current_email.true_label
    risk_correct = action.risk_level == current_email.true_risk if task_name in {"medium", "hard"} else None
    decision_correct = action.decision == current_email.correct_decision if task_name == "hard" else None

    reward = _compute_reward(current_email, action, task_name)
    env_state["total_reward"] += reward

    # Update trust score
    trust_delta = 0.05 if classification_correct else -0.05
    old_trust = env_state["trust_scores"][str(current_email.id)]
    new_trust = max(0.0, min(1.0, old_trust + trust_delta))
    env_state["trust_scores"][str(current_email.id)] = new_trust

    # Record history for grading
    env_state["history"].append({
        "email_id": current_email.id,
        "classification_correct": classification_correct,
        "risk_correct": bool(risk_correct) if risk_correct is not None else False,
        "decision_correct": bool(decision_correct) if decision_correct is not None else False,
        "predicted_risk_level": action.risk_level,
        "effective_risk": current_email.true_risk,
        "predicted_decision": action.decision,
    })

    # Advance
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
        info={
            "classification_correct": classification_correct,
            "risk_correct": risk_correct,
            "decision_correct": decision_correct,
            "total_reward": env_state["total_reward"],
            "email_id": current_email.id,
        },
    )


@app.get("/state", tags=["Environment"])
def get_state():
    """Return the full current environment state."""
    if not env_state:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    return {
        "task_name": env_state["task_name"],
        "current_index": env_state["current_index"],
        "total_emails": len(env_state["emails"]),
        "total_reward": env_state["total_reward"],
        "done": env_state["done"],
        "trust_scores": env_state["trust_scores"],
    }


@app.post("/grade", tags=["Grading"])
def grade_episode(req: GradeRequest):
    """Grade a completed episode history."""
    if not req.history:
        # Use in-memory history if available
        if env_state.get("history"):
            return grade(req.task_name, env_state["history"])
        raise HTTPException(status_code=400, detail="No history provided and no active episode.")
    return grade(req.task_name, req.history)


@app.get("/grade/current", tags=["Grading"])
def grade_current():
    """Grade the current in-progress or completed episode."""
    if not env_state or not env_state.get("history"):
        raise HTTPException(status_code=400, detail="No active episode history to grade.")
    return grade(env_state["task_name"], env_state["history"])
