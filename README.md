---
title: Financial Email Risk Environment
emoji: 💰
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
---

# Financial Email Risk Environment

Financial Email Risk Environment is a deterministic OpenEnv-compatible FastAPI service for simulating financial email triage.

## Problem

Financial teams receive a mix of legitimate operational messages and potentially malicious requests. Wrong approvals can cause direct losses, while over-escalation slows operations.

This environment benchmarks an agent's ability to:
- classify emails
- assign risk
- decide to approve, reject, or escalate

## Why It Matters

- Reproducible evaluation for risk workflows
- Cost-sensitive reward design (unsafe approvals penalized)
- Trust-aware behavior modeling by sender
- Clean API for agent integration and hackathon demos

## Originality & Design Decisions

This project was implemented as an original environment for the hackathon workflow, with custom task structure, scoring, and baseline behavior.

Design choices made in this repo:
- Three-tier task progression (`easy`, `medium`, `hard`) with increasing action complexity
- Cost-sensitive grader with stronger penalties for dangerous false negatives
- Rule-based baseline policy built specifically for financial email triage signals
- Deterministic API behavior and reproducible grading outputs for fair comparisons

Attribution note:
- General concepts and API patterns follow OpenEnv/FastAPI ecosystem documentation
- Environment logic, task content, and grading rules are authored for this project

## Architecture

- `models.py` - Pydantic request/response/state contracts
- `server/environment.py` - core deterministic environment logic (`reset`, `step`, `state`)
- `server/app.py` - FastAPI endpoints
- `tasks.py` - task definitions and required fields
- `grader.py` - scoring and metrics
- `baseline.py` - rule-based baseline runner
- `openenv.yaml` - OpenEnv metadata and endpoint/schema definitions

## Tasks

- `easy`: `classification`
- `medium`: `classification + risk_level`
- `hard`: `classification + risk_level + decision`

## Reward Logic

Base rewards:
- `+0.2` correct classification
- `+0.3` correct risk
- `+0.5` correct decision

Penalties:
- `-0.1` wrong classification
- `-0.2` wrong risk
- `-0.5` wrong decision
- `-1.0` approve high-risk email
- `-0.1` unnecessary escalation on low-risk email
- `-0.2` step-limit overflow penalty

Trust dynamics:
- Trust increases on correct handling
- Trust decreases on errors
- Very low trust biases effective risk upward

## API Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /grader`
- `GET /baseline`
- `GET /health`

## Run Locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t financial-email-risk-env .
docker run --rm -p 8000:8000 financial-email-risk-env
```

## Usage Flow

1. Reset first (required before `/step`):

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name":"hard"}'
```

2. Step request (both formats are accepted):

Wrapped body:

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "decide",
      "classification": "spam",
      "risk_level": "high",
      "decision": "reject"
    }
  }'
```

Direct action body:

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "decide",
    "classification": "spam",
    "risk_level": "high",
    "decision": "reject"
  }'
```

3. Check score:

```bash
curl http://localhost:8000/grader
```

4. Run baseline across all tasks:

```bash
curl http://localhost:8000/baseline
```

## Grader Output

```json
{
  "score": 0.0,
  "breakdown": {
    "classification": 0.0,
    "risk": 0.0,
    "decision": 0.0
  },
  "metrics": {
    "false_positives": 0,
    "false_negatives": 0,
    "escalations": 0
  }
}
```

## Hugging Face Spaces (Docker)

1. Create a Docker Space.
2. Push this repository.
3. Keep container command: `uvicorn server.app:app --host 0.0.0.0 --port 8000`.
4. Verify `/health`, `/tasks`, `/reset`, `/step`, `/grader`.

## Project Structure

```text
.
+-- models.py
+-- tasks.py
+-- grader.py
+-- baseline.py
+-- openenv.yaml
+-- Dockerfile
+-- README.md
+-- server/
    +-- app.py
    +-- environment.py
    +-- data.json
```
