---
title: AI Software Engineer Environment
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# AI Software Engineer Environment

An RL environment that evaluates whether AI agents can debug and fix real-world Python code. The agent receives broken code, submits fixes, and receives graded feedback across 7 tasks of increasing difficulty — spanning syntax errors, type errors, logic bugs, off-by-one errors, edge case handling, concurrency bugs, and memory efficiency issues.

At the end of a full run, the environment produces a **diagnostic skill report card** showing per-category strengths and weaknesses, and posts the model's score to a **live leaderboard**.

---

## Environment Overview

Real-world software engineers spend a significant portion of their time debugging and fixing broken code. This environment models that task directly — giving agents the same feedback loop a developer gets: submit a fix, see what tests pass or fail, iterate.

The environment is designed to be a genuine diagnostic benchmark, not just a pass/fail scorer. The skill report card tells you *which categories* of bugs a model handles well and which it struggles with — actionable signal for model evaluation and improvement.

---

## Action Space

**`AiSeEnvAction`**

| Field | Type | Description |
|---|---|---|
| `action_type` | string | One of: `fix`, `refactor`, `review` |
| `content` | string | The submitted Python code |

Action type semantics:
- `fix` — submit a corrected version of the code. Costs a step. Full grading applied.
- `refactor` — submit a fix graded on correctness plus code quality metrics (line count, variable count via AST). Costs a step.
- `review` — request grader feedback without committing a fix. Reward = 0, step not consumed. Free peek.

---

## Observation Space

**`AiSeEnvObservation`**

| Field | Type | Description |
|---|---|---|
| `code` | string | The original broken code the agent must fix |
| `task_description` | string | Natural language description of the bug and what to fix |
| `history` | list[string] | Past submissions with grader feedback injected |
| `hint` | string or null | Optional hint surfaced after 2 failed attempts |
| `done` | boolean | Whether the episode has ended |
| `reward` | float | Score for the last action (0.0 – 1.0) |

---

## Tasks

| Task ID | Difficulty | Bug Type | Description |
|---|---|---|---|
| `easy` | Easy | Syntax | Missing colon in function definition |
| `easy_2` | Easy | Type | Function returns `str` instead of `int` |
| `medium` | Medium | Logic | Off-by-one error in average formula (`- 1` subtracted) |
| `medium_2` | Medium | Off-by-one | Loop range excludes last element — max value never checked |
| `hard` | Hard | Edge case | Stock price aggregator crashes on missing keys, string types, zero window |
| `hard_2` | Hard | Concurrency | Counter increment is not thread-safe — agent must add `threading.Lock` |
| `hard_3` | Hard | Performance | Unbounded list accumulation — agent must refactor into a generator |

All tasks include adversarial test cases designed to catch naive fixes that work on simple inputs but fail on edge cases.

---

## Reward Function

Grading uses a 3-pillar scoring system:

**For tasks without a structure rule (easy, medium, hard):**
- Syntax validity: +0.2
- Test case correctness: up to +0.8 (proportional to tests passed)
- Maximum: 1.0

**For tasks with a structure rule (hard_2, hard_3):**
- Syntax validity: +0.2
- Test case correctness: up to +0.6
- Structure rule compliance: +0.2 (uses `threading.Lock` / uses `yield`)
- Maximum: 1.0

**Additional reward shaping:**
- **Efficiency bonus** — solving on step 1 = 1.0, step 2 = 0.95, step 3+ = 0.90
- **Regression penalty** — −0.05 per test that was passing before but now fails
- **Hint** — natural language hint surfaced after 2 failed attempts
- **Code quality bonus** — additional +0.1 when using `refactor` action type

---

## Skill Report Card

After running all 7 tasks, call `env.skill_report(formatted=True)` to get a diagnostic report:

```
====================================================
  AI SOFTWARE ENGINEER — SKILL REPORT CARD
====================================================
  Tasks Attempted : 7
  Tasks Solved    : 6
  Overall Score   : 0.98  [Excellent]
----------------------------------------------------
  Category                        Score  Rating
----------------------------------------------------
  Syntax Errors                   1.00  Excellent
  Type Errors                     1.00  Excellent
  Logic Errors                    1.00  Excellent
  Off-by-One Errors               1.00  Excellent
  Edge Case Handling              0.84  Good
  Concurrency & Thread Safety     1.00  Excellent
  Performance & Memory Efficiency 1.00  Excellent
----------------------------------------------------
  Verdict: Outstanding — this model handles all bug categories with high reliability.
====================================================
```

---

## Baseline Scores

Evaluated using `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Router:

| Task | Score | Solved | Steps |
|---|---|---|---|
| easy | 1.000 | ✅ | 1 |
| easy_2 | 1.000 | ✅ | 1 |
| medium | 1.000 | ✅ | 1 |
| medium_2 | 1.000 | ✅ | 1 |
| hard | 0.840 | ❌ | 5 |
| hard_2 | 1.000 | ✅ | 1 |
| hard_3 | 1.000 | ✅ | 1 |
| **Overall** | **0.977** | **6/7** | — |

---

## Setup & Usage

### Running via HuggingFace Space

The environment is live at:
```
https://huggingface.co/spaces/nagaadityaanand/AI_SE_ENV
```

API endpoints:
- `POST /reset` — reset the environment for a new episode
- `POST /step` — execute an action
- `GET /state` — get current environment state
- `GET /schema` — get action/observation schemas
- `GET /health` — health check
- `GET /docs` — full OpenAPI documentation
- `WS /ws` — WebSocket for persistent low-latency sessions

### Running Locally

```bash
git clone https://huggingface.co/spaces/nagaadityaanand/AI_SE_ENV
cd AI_SE_ENV
uv sync
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Running with Docker

```bash
docker build -t ai-se-env .
docker run -p 8000:8000 ai-se-env
```

### Running the Inference Script

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

---

## Project Structure

```
AI_SE_ENV/
├── inference.py                   # Baseline inference script (run all 7 tasks)
├── models.py                      # AiSeEnvAction, AiSeEnvObservation
├── openenv.yaml                   # OpenEnv manifest
├── pyproject.toml                 # Project metadata and dependencies
├── client.py                      # AiSeEnvEnv client
├── Dockerfile                     # Container image
├── README.md                      # This file
└── server/
    ├── AI_SE_ENV_environment.py   # Core environment logic
    ├── app.py                     # FastAPI application
    ├── skill_report.py            # Diagnostic skill report card
    ├── leaderboard.py             # Live leaderboard
    ├── graders/
    │   └── code_grader.py         # 3-pillar grader with AST analysis
    └── tasks/
        ├── easy.py                # easy_1, easy_2
        ├── medium.py              # medium_1, medium_2
        └── hard.py                # hard_1, hard_2, hard_3
```

---

## Connecting Programmatically

```python
from AI_SE_ENV import AiSeEnvAction, AiSeEnvEnv

env = AiSeEnvEnv(base_url="https://nagaadityaanand-ai-se-env-d917431.hf.space/web")

obs = env.reset(difficulty="hard")
print(obs.observation.task_description)

result = env.step(AiSeEnvAction(
    action_type="fix",
    content="def get_average_price(api_responses, window_size):\n    ..."
))
print(result.reward)
env.close()
```