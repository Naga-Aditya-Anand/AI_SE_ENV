# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AI Software Engineer Environment.

Presents broken Python code to an agent and evaluates its ability
to fix it across 7 tasks of increasing difficulty, spanning syntax errors,
logic bugs, type errors, off-by-one errors, edge case handling,
concurrency bugs, and memory efficiency issues.

Tasks:
    easy      — syntax error (missing colon)
    easy_2    — type error (returns str instead of int)
    medium    — logic error (off-by-one in formula)
    medium_2  — off-by-one error in loop range
    hard      — edge case handling (messy API data)
    hard_2    — concurrency bug (non thread-safe counter)
    hard_3    — memory leak (unbounded list vs generator)
"""

from uuid import uuid4
import math

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AiSeEnvAction, AiSeEnvObservation
except ImportError:
    from models import AiSeEnvAction, AiSeEnvObservation

from .graders.code_grader import grade_code
from .skill_report import SkillTracker
from . import leaderboard as lb
from .tasks.easy import EASY_TASK, EASY_TASK_2
from .tasks.medium import MEDIUM_TASK, MEDIUM_TASK_2
from .tasks.hard import HARD_TASK, HARD_TASK_2, HARD_TASK_3


def _strict_score(value, default=0.01) -> float:
    """Clamp score to [0.01, 0.99] and reject non-finite values."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default

    if not math.isfinite(score):
        score = default

    clamped = max(0.01, min(score, 0.99))
    rounded = round(clamped, 4)
    return max(0.01, min(rounded, 0.99))


class AiSeEnvEnvironment(Environment):
    """
    AI Software Engineer environment.

    The agent receives broken Python code and must fix it.
    Three action types are supported:
      fix      — submit a fix (costs a step)
      refactor — fix graded on correctness + code quality (costs a step)
      review   — free feedback peek (reward=0, step not consumed)

    Reward shaping:
      - Partial credit via 3-pillar grader (syntax / tests / structure)
      - Efficiency bonus: step-1 solve = 1.0, step-2 = 0.95, step-3+ = 0.9
      - Regression penalty: -0.05 per previously passing test now failing
      - Hint surfaced after 2 failed attempts
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # All available task keys in difficulty order
    ALL_TASKS = [
        "easy", "easy_2",
        "medium", "medium_2",
        "hard", "hard_2", "hard_3",
    ]

    MAX_STEPS = 5
    _LAST_RESET_DIFFICULTY: str = "easy"

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self._tasks = {
            "easy":     EASY_TASK,
            "easy_2":   EASY_TASK_2,
            "medium":   MEDIUM_TASK,
            "medium_2": MEDIUM_TASK_2,
            "hard":     HARD_TASK,
            "hard_2":   HARD_TASK_2,
            "hard_3":   HARD_TASK_3,
        }

        self._current_task      = None
        self._current_difficulty = None
        self._history           = []
        self._steps             = 0
        self._review_steps      = 0
        self._prev_passed       = set()
        self._episode_best      = _strict_score(0.01)
        self._tracker           = SkillTracker()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, **kwargs) -> AiSeEnvObservation:
        """
        Reset the environment for a new episode.

        Args:
            difficulty: task key — one of ALL_TASKS (default: "easy")
        
        Returns:
            AiSeEnvObservation with initial state
            
        Raises:
            Exception: If difficulty is not found in available tasks
        """
        try:
            difficulty = kwargs.get("difficulty", "easy")
            
            if difficulty not in self._tasks:
                difficulty = "easy"

            self._current_task       = self._tasks[difficulty]
            self._current_difficulty = difficulty
            AiSeEnvEnvironment._LAST_RESET_DIFFICULTY = difficulty
            self._history            = []
            self._steps              = 0
            self._review_steps       = 0
            self._prev_passed        = set()
            self._episode_best       = _strict_score(0.01)
            self._state              = State(episode_id=str(uuid4()), step_count=0)

            observation = AiSeEnvObservation(
                code=self._current_task["code"],
                task_description=self._current_task["description"],
                history=[],
                hint=None,
                done=False,
                reward=_strict_score(0.01),
            )
            
            return observation
            
        except Exception as e:
            import sys
            import traceback
            print(f"[ERROR] Reset failed: {str(e)}", file=sys.stderr)
            print(f"[TRACEBACK] {traceback.format_exc()}", file=sys.stderr)
            raise

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action: AiSeEnvAction) -> AiSeEnvObservation:  # type: ignore[override]
        """Execute one step in the environment."""

        # In stateless HTTP mode, step() may be called on a fresh env instance
        # that hasn't seen reset(). Ensure a task is always initialised.
        if self._current_task is None:
            fallback_difficulty = (
                self._current_difficulty
                or AiSeEnvEnvironment._LAST_RESET_DIFFICULTY
                or "easy"
            )
            self.reset(difficulty=fallback_difficulty)

        self._state.step_count += 1

        # ── REVIEW — free feedback, no step consumed ───────────────
        if action.action_type == "review":
            _, feedback = grade_code(
                self._current_task, action.content, action_type="review"
            )
            self._review_steps += 1
            self._history.append(
                f"[REVIEW #{self._review_steps}]\n"
                f"Code Submitted:\n{action.content}\n\n"
                f"Grader Feedback:\n{feedback}"
            )
            return AiSeEnvObservation(
                code=self._current_task["code"],
                task_description=self._current_task["description"],
                history=self._history,
                hint=None,
                done=False,
                reward=_strict_score(0.01),
            )

        # ── FIX / REFACTOR — consume a step ───────────────────────
        self._steps += 1
        score, feedback = grade_code(
            self._current_task, action.content, action_type=action.action_type
        )

        # Efficiency bonus (ensure score stays strictly between 0 and 1)
        if score >= 0.95:
            if self._steps == 1:
                score = 0.98
            elif self._steps == 2:
                score = min(score, 0.94)
            else:
                score = min(score, 0.89)

        # Regression penalty
        test_cases = self._current_task.get("test_cases", [])
        current_passed = set()
        for i in range(len(test_cases)):
            if f"Test {i+1}" not in feedback:
                current_passed.add(i)

        if self._prev_passed:
            regressions = self._prev_passed - current_passed
            if regressions:
                penalty = len(regressions) * 0.05
                score = score - penalty
                feedback += (
                    f"\n⚠ Regression penalty: -{penalty:.2f} "
                    f"({len(regressions)} previously passing test(s) now fail)"
                )
        self._prev_passed = current_passed

        # Ensure score is strictly between 0 and 1 after all modifications
        score = _strict_score(score)

        # Track best score
        self._episode_best = max(self._episode_best, score)

        # Hint after 2 failed attempts
        hint = None
        if self._steps >= 2 and score < 0.95:
            hint = self._current_task.get("hint")

        # Done condition (success when score >= 0.95)
        done = score >= 0.95 or self._steps >= self.MAX_STEPS

        # Record into skill tracker when episode ends
        if done:
            bug_type = self._current_task.get("bug_type", "logic")
            self._tracker.record(bug_type, self._episode_best)

        # History
        self._history.append(
            f"[{action.action_type.upper()} #{self._steps}]\n"
            f"Code Submitted:\n{action.content}\n\n"
            f"Grader Feedback:\n{feedback}"
        )

        # THE ULTIMATE FAILSAFE: Guarantee the score is strictly between (0, 1)
        safe_score = _strict_score(score)

        return AiSeEnvObservation(
            code=self._current_task["code"],
            task_description=self._current_task["description"],
            history=self._history,
            hint=hint,
            done=done,
            reward=safe_score,
        )

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # skill report + leaderboard (called from inference script)
    # ------------------------------------------------------------------
    def skill_report(self, formatted: bool = False):
        if formatted:
            return self._tracker.formatted_report()
        return self._tracker.report()

    def submit_to_leaderboard(self, model_name: str):
        report  = self._tracker.report()
        summary = report["summary"]
        skill_scores = {
            bug_type: data["avg_score"]
            for bug_type, data in report["skills"].items()
        }
        lb.submit(
            model_name=model_name,
            overall_score=summary["overall_score"],
            tasks_solved=summary["tasks_solved"],
            skill_scores=skill_scores,
        )

    def reset_skill_tracker(self):
        self._tracker.reset_all()
