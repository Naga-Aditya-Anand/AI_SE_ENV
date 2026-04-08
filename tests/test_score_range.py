import math
import unittest

from fastapi.testclient import TestClient
from pydantic import ValidationError

from models import AiSeEnvAction, AiSeEnvObservation
from server.AI_SE_ENV_environment import AiSeEnvEnvironment
from server.app import app
from server.graders.code_grader import grade_code
from server.tasks.easy import EASY_TASK, EASY_TASK_2
from server.tasks.medium import MEDIUM_TASK, MEDIUM_TASK_2
from server.tasks.hard import HARD_TASK, HARD_TASK_2, HARD_TASK_3


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
            score_refactor, _ = grade_code(task, task["expected_solution"], action_type="refactor")
            assert_open_interval_score(self, score_fix)
            assert_open_interval_score(self, score_refactor)

    def test_environment_step_without_reset_is_safe(self) -> None:
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

    def test_observation_schema_rejects_boundary_values(self) -> None:
        with self.assertRaises(ValidationError):
            AiSeEnvObservation(reward=0.0)
        with self.assertRaises(ValidationError):
            AiSeEnvObservation(reward=1.0)


if __name__ == "__main__":
    unittest.main()
