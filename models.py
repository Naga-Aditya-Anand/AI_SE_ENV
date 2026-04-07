# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the AI Software Engineer Environment.

The AI_SE_ENV environment presents broken code to an agent
and evaluates its ability to fix it across multiple bug categories.
"""

from typing import List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class AiSeEnvAction(Action):
    """
    Action for the AI Software Engineer environment.

    The agent chooses HOW it wants to interact this step:
      fix      — submit a corrected version of the code (costs a step)
      refactor — submit a fix graded on correctness + code quality (costs a step)
      review   — request grader feedback without consuming a step (reward=0)
    """

    action_type: str = Field(
        default="fix",
        description="One of: 'fix', 'refactor', 'review'",
    )
    content: str = Field(
        ...,
        description="The submitted Python code",
    )


class AiSeEnvObservation(Observation):
    """
    Observation from the AI Software Engineer environment.

    Contains the broken code, task description, history of
    past attempts with grader feedback, and an optional hint.
    """

    code: str = Field(
        default="",
        description="The original broken code the agent must fix",
    )
    task_description: str = Field(
        default="",
        description="Natural language description of the bug and what to fix",
    )
    history: List[str] = Field(
        default_factory=list,
        description="Past action entries with grader feedback injected",
    )
    hint: Optional[str] = Field(
        default=None,
        description="Optional hint surfaced after 2 failed attempts",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended",
    )
    reward: float = Field(
        default=0.01,
        description="Reward for the last action (0.01–1.0)",
    )