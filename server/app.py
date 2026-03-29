from __future__ import annotations

from typing import Optional, Union

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from baseline import run_baseline
from grader import grade
from models import Action, Observation, State, StepResult, TaskName
from server.environment import FinancialEmailEnv
from tasks import get_tasks


app = FastAPI(
    title="Financial Email Risk OpenEnv API",
    version="1.0.0",
    description="Task-aware deterministic OpenEnv server for financial email risk handling.",
)


env: Optional[FinancialEmailEnv] = None
history: list[dict] = []
initialized = False


class ResetRequest(BaseModel):
    task_name: TaskName = "hard"
    step_limit: Optional[int] = Field(default=None, ge=1)


class ResetResponse(BaseModel):
    observation: Observation


class StepRequest(BaseModel):
    action: Action


class GraderResponse(BaseModel):
    score: float
    breakdown: dict[str, float]
    metrics: dict[str, int]


class HealthResponse(BaseModel):
    status: str


@app.get("/")
def root():
    return {"status": "running"}


@app.on_event("startup")
def startup():
    global env
    env = FinancialEmailEnv()


def validate_action(action: Action) -> tuple[bool, Optional[str]]:
    if action.action_type == "assign_risk":
        if not action.classification or not action.risk_level:
            return False, "assign_risk requires classification and risk_level"

    if action.action_type == "decide":
        if not all([action.classification, action.risk_level, action.decision]):
            return False, "decide requires classification, risk_level, and decision"

    return True, None


@app.post("/reset", response_model=ResetResponse)
def reset(payload: ResetRequest) -> ResetResponse:
    global initialized

    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    try:
        observation = env.reset(task_name=payload.task_name, step_limit=payload.step_limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    history.clear()
    initialized = True
    return ResetResponse(observation=observation)


@app.post("/step", response_model=StepResult)
def step(payload: Union[StepRequest, Action] = Body(...)) -> StepResult:
    if not initialized:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    action = payload.action if isinstance(payload, StepRequest) else payload

    valid, msg = validate_action(action)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    try:
        observation, reward, done, info = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

    observation_payload = observation.model_dump() if observation is not None else None

    return StepResult(
        observation=observation_payload,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=State)
def state() -> State:
    if not initialized:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    return State.model_validate(env.state())


@app.get("/tasks")
def tasks() -> list[dict]:
    return get_tasks()


@app.get("/grader", response_model=GraderResponse)
def grader() -> GraderResponse:
    if not initialized:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    if env is None:
        raise HTTPException(status_code=500, detail="Environment not initialized")

    result = grade(env.active_task, history)
    return GraderResponse(**result)


@app.get("/baseline")
def baseline() -> dict:
    return run_baseline()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")