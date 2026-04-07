"""
inference.py — AI Software Engineer Environment
================================================
Runs all 7 tasks against the configured model and emits
mandatory [START] / [STEP] / [END] log lines per the OpenEnv spec.

stdout: only [START], [STEP], [END] lines — exactly as the spec requires.
stderr: [INFO], [DEBUG], skill report, leaderboard — for human reading only.

Must be placed in the ROOT of the AI_SE_ENV project.

Environment variables:
  HF_TOKEN      Your HuggingFace API token           (required, no default)
  API_BASE_URL  LLM endpoint                         (default: HF router)
  MODEL_NAME    Model identifier                     (default: Qwen2.5-72B)
"""

import os
import sys
from typing import List, Optional

from openai import OpenAI

from models import AiSeEnvAction
from server.AI_SE_ENV_environment import AiSeEnvEnvironment
from server import leaderboard as lb

# ── Environment variables ────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK         = "ai_se_env"
MAX_STEPS         = 5
MAX_TOKENS        = 512
TEMPERATURE       = 0.2
SUCCESS_THRESHOLD = 0.9

ALL_TASKS = [
    "easy",
    "easy_2",
    "medium",
    "medium_2",
    "hard",
    "hard_2",
    "hard_3",
]


# ── Spec-compliant logging — stdout only ─────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    action_inline = action.replace("\n", " ").strip()[:120]
    error_val     = error if error else "null"
    print(
        f"[STEP] step={step} action={action_inline} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.01"
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ── Human-readable logging — stderr only ─────────────────────────────

def info(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ── Code cleaning ────────────────────────────────────────────────────

def clean_code(text: str) -> str:
    if "```" in text:
        lines = text.split("\n")
        return "\n".join(l for l in lines if "```" not in l).strip()
    return text.strip()


# ── Model call ───────────────────────────────────────────────────────

def get_model_response(
    client: OpenAI,
    code: str,
    description: str,
    history: List[str],
    hint: Optional[str] = None,
) -> str:
    history_block = ""
    if history:
        history_block = "\n\nPrevious attempts (with grader feedback):\n"
        for entry in history[-3:]:
            history_block += f"\n{entry}\n"
        history_block += "\nLearn from the feedback above and fix accordingly."

    hint_block = f"\n\nHint from the environment: {hint}" if hint else ""

    prompt = (
        "You are an expert software engineer.\n\n"
        "Fix the following code so it passes all test cases.\n\n"
        "RULES:\n"
        "- Return ONLY valid Python code.\n"
        "- Do NOT use markdown (no ``` fences).\n"
        "- Do NOT explain — just return the fixed code.\n\n"
        f"Task:\n{description}\n\n"
        f"Code to fix:\n{code}"
        f"{history_block}"
        f"{hint_block}\n\n"
        "Return ONLY the corrected code:"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        info(f"[DEBUG] Model call failed: {exc}")
        return code


# ── Single episode runner ────────────────────────────────────────────

def run_episode(client: OpenAI, env: AiSeEnvEnvironment, difficulty: str) -> dict:
    """
    Runs one full episode. Always emits [START], [STEP]s, and [END]
    on stdout even if an exception occurs mid-episode.
    """
    rewards: List[float] = []
    steps_taken = 0
    episode_score = 0.0
    success = False

    log_start(task=difficulty, model=MODEL_NAME)

    try:
        obs = env.reset(difficulty=difficulty)

        for step in range(1, MAX_STEPS + 1):
            raw  = get_model_response(
                client, obs.code, obs.task_description, obs.history, obs.hint
            )
            code = clean_code(raw)
            action = AiSeEnvAction(action_type="fix", content=code)

            try:
                obs    = env.step(action)
                reward = obs.reward
                done   = obs.done
                error  = None
            except Exception as step_exc:
                reward = 0.01
                done   = True
                error  = str(step_exc)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=code, reward=reward, done=done, error=error)

            if done:
                break

    except Exception as episode_exc:
        info(f"[DEBUG] Episode error: {episode_exc}")

    finally:
            try:
                if hasattr(env, "close"):
                    env.close()
            except Exception:
                pass

            episode_score = max(rewards) if rewards else 0.01
            success       = episode_score >= SUCCESS_THRESHOLD
            log_end(success=success, steps=steps_taken, rewards=rewards)

    return {
        "task":    difficulty,
        "score":   episode_score,
        "success": success,
        "steps":   steps_taken,
        "rewards": rewards,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env    = AiSeEnvEnvironment()

    info(f"\n[INFO] Starting benchmark | model={MODEL_NAME} | tasks={len(ALL_TASKS)}\n")

    results = []
    for difficulty in ALL_TASKS:
        summary = run_episode(client, env, difficulty)
        results.append(summary)
        info(
            f"[INFO] Completed {difficulty:<10} | "
            f"score={summary['score']:.3f} | "
            f"solved={'yes' if summary['success'] else 'no '}"
        )

    # ── Skill report → stderr ─────────────────────────────────────
    info("\n" + env.skill_report(formatted=True))

    # ── Leaderboard → stderr ──────────────────────────────────────
    env.submit_to_leaderboard(MODEL_NAME)
    info("\n" + lb.formatted_leaderboard())

    # ── Final summary → stderr ────────────────────────────────────
    total_solved = sum(1 for r in results if r["success"])
    overall      = sum(r["score"] for r in results) / len(results)
    info(
        f"\n[INFO] Benchmark complete | "
        f"solved={total_solved}/{len(ALL_TASKS)} | "
        f"overall_score={overall:.3f}"
    )


if __name__ == "__main__":
    main()