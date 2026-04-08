import math
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from models import AiSeEnvAction, AiSeEnvObservation
from server import leaderboard as lb
from server.AI_SE_ENV_environment import AiSeEnvEnvironment
from server.app import app
from server.graders.code_grader import grade_code
from server.tasks.easy import EASY_TASK, EASY_TASK_2
from server.tasks.hard import HARD_TASK, HARD_TASK_2, HARD_TASK_3
from server.tasks.medium import MEDIUM_TASK, MEDIUM_TASK_2


ALL_TASKS = [
    ("easy", EASY_TASK),
    ("easy_2", EASY_TASK_2),
    ("medium", MEDIUM_TASK),
    ("medium_2", MEDIUM_TASK_2),
    ("hard", HARD_TASK),
    ("hard_2", HARD_TASK_2),
    ("hard_3", HARD_TASK_3),
]


def assert_open_interval_score(tc: unittest.TestCase, value) -> None:
    tc.assertIsInstance(value, (int, float))
    tc.assertTrue(math.isfinite(float(value)))
    tc.assertGreater(float(value), 0.0)
    tc.assertLess(float(value), 1.0)


class TestScoreRange(unittest.TestCase):
    def test_grader_scores_are_strictly_between_zero_and_one(self) -> None:
        for _, task in ALL_TASKS:
            score_fix, _ = grade_code(task, task["expected_solution"], action_type="fix")
            score_refactor, _ = grade_code(
                task, task["expected_solution"], action_type="refactor"
            )
            assert_open_interval_score(self, score_fix)
            assert_open_interval_score(self, score_refactor)

    def test_grader_handles_invalid_code_with_in_range_score(self) -> None:
        for _, task in ALL_TASKS:
            score, _ = grade_code(task, "def broken(:\n    pass", action_type="fix")
            assert_open_interval_score(self, score)

    def test_environment_rewards_are_strictly_bounded(self) -> None:
        env = AiSeEnvEnvironment()

        reset_obs = env.reset(difficulty="easy")
        assert_open_interval_score(self, reset_obs.reward)

        review_obs = env.step(
            AiSeEnvAction(action_type="review", content=EASY_TASK["code"])
        )
        assert_open_interval_score(self, review_obs.reward)

        fix_obs = env.step(
            AiSeEnvAction(action_type="fix", content=EASY_TASK["expected_solution"])
        )
        assert_open_interval_score(self, fix_obs.reward)

    def test_step_without_reset_does_not_crash_and_keeps_score_in_range(self) -> None:
        env = AiSeEnvEnvironment()
        obs = env.step(
            AiSeEnvAction(action_type="fix", content=EASY_TASK["expected_solution"])
        )
        assert_open_interval_score(self, obs.reward)

    def test_http_reset_then_step_returns_in_range_reward(self) -> None:
        client = TestClient(app)

        for difficulty, task in ALL_TASKS:
            reset_resp = client.post("/reset", json={"difficulty": difficulty})
            self.assertEqual(reset_resp.status_code, 200, reset_resp.text)

            step_resp = client.post(
                "/step",
                json={
                    "action": {
                        "action_type": "fix",
                        "content": task["expected_solution"],
                    }
                },
            )
            self.assertEqual(step_resp.status_code, 200, step_resp.text)
            payload = step_resp.json()
            assert_open_interval_score(self, payload.get("reward"))

    def test_environment_clamps_non_finite_or_extreme_grader_scores(self) -> None:
        env = AiSeEnvEnvironment()
        env.reset(difficulty="easy")

        with patch(
            "server.AI_SE_ENV_environment.grade_code",
            return_value=(float("nan"), "nan score"),
        ):
            obs = env.step(
                AiSeEnvAction(action_type="fix", content=EASY_TASK["expected_solution"])
            )
            assert_open_interval_score(self, obs.reward)

        env.reset(difficulty="easy")
        with patch(
            "server.AI_SE_ENV_environment.grade_code",
            return_value=(-5.0, "negative score"),
        ):
            obs = env.step(
                AiSeEnvAction(action_type="fix", content=EASY_TASK["expected_solution"])
            )
            assert_open_interval_score(self, obs.reward)

        env.reset(difficulty="easy")
        with patch(
            "server.AI_SE_ENV_environment.grade_code",
            return_value=(10.0, "too large"),
        ):
            obs = env.step(
                AiSeEnvAction(action_type="fix", content=EASY_TASK["expected_solution"])
            )
            assert_open_interval_score(self, obs.reward)

    def test_skill_report_scores_remain_in_range(self) -> None:
        env = AiSeEnvEnvironment()
        for difficulty, task in ALL_TASKS:
            env.reset(difficulty=difficulty)
            env.step(AiSeEnvAction(action_type="fix", content=task["expected_solution"]))

        report = env.skill_report()
        assert_open_interval_score(self, report["summary"]["overall_score"])
        for _, data in report["skills"].items():
            assert_open_interval_score(self, data["avg_score"])

    def test_leaderboard_safeguards_score_range(self) -> None:
        lb.reset_leaderboard()
        lb.submit(
            model_name="range-test-model",
            overall_score=1.0,
            tasks_solved=7,
            skill_scores={"syntax": float("inf"), "logic": -10.0},
        )
        entry = lb.get_model_entry("range-test-model")
        self.assertIsNotNone(entry)
        assert_open_interval_score(self, entry["best_score"])
        assert_open_interval_score(self, entry["avg_score"])
        for _, score in entry["skill_scores"].items():
            assert_open_interval_score(self, score)

    def test_observation_schema_rejects_boundary_values(self) -> None:
        with self.assertRaises(ValidationError):
            AiSeEnvObservation(reward=0.0)
        with self.assertRaises(ValidationError):
            AiSeEnvObservation(reward=1.0)


if __name__ == "__main__":
    unittest.main()
