# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ai Se Env Environment Client."""

import math
from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AiSeEnvAction, AiSeEnvObservation


class AiSeEnvEnv(
    EnvClient[AiSeEnvAction, AiSeEnvObservation, State]
):
    """
    Client for the Ai Se Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with AiSeEnvEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(difficulty="easy")
        ...     print(result.observation.code)
        ...
        ...     result = client.step(AiSeEnvAction(action_type="fix", content="fixed_code"))
        ...     print(result.observation.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = AiSeEnvEnv.from_docker_image("AI_SE_ENV-env:latest")
        >>> try:
        ...     result = client.reset(difficulty="easy")
        ...     result = client.step(AiSeEnvAction(action_type="fix", content="code"))
        ... finally:
        ...     client.close()
    """

    def _reset_payload(self, **kwargs) -> Dict:
        """
        Convert reset keyword arguments to JSON payload.

        Args:
            **kwargs: Reset parameters (e.g., difficulty="easy")

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return kwargs

    def _step_payload(self, action: AiSeEnvAction) -> Dict:
        """
        Convert AiSeEnvAction to JSON payload for step message.

        Args:
            action: AiSeEnvAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
            "content": action.content,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AiSeEnvObservation]:
        """
        Parse server response into StepResult[AiSeEnvObservation].

        The server may return done/reward either at the top level of the payload
        OR nested inside the "observation" dict (since AiSeEnvObservation carries
        those fields itself). We check obs_data first and fall back to the top
        level, always providing a safe default so we never pass None to a typed
        float/bool field.

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with AiSeEnvObservation
        """
        obs_data = payload.get("observation", {})

        # done and reward live inside AiSeEnvObservation, so the server may
        # serialise them under "observation" rather than at the top level.
        # Check obs_data first, then fall back to the top-level payload key,
        # then fall back to a safe default.
        done   = obs_data.get("done",   payload.get("done",   False))
        reward = obs_data.get("reward", payload.get("reward", 0.01))

        # Guard against the server returning explicit null for reward.
        if reward is None:
            reward = 0.01

        observation = AiSeEnvObservation(
            code=obs_data.get("code", ""),
            task_description=obs_data.get("task_description", ""),
            history=obs_data.get("history", []),
            hint=obs_data.get("hint"),
            done=done,
            reward=self._strict_score(reward),
        )

        return StepResult(
            observation=observation,
            reward=self._strict_score(reward),
            done=done,
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    @staticmethod
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
