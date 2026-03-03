"""Stand skill — static standing pose from STAND_JOINT_POSITIONS."""

from __future__ import annotations

import numpy as np

from bridge.isaaclab.ainex_cfg import STAND_JOINT_POSITIONS
from bridge.isaaclab.joint_map import JOINT_NAMES
from training.rl.skills.base_skill import BaseSkill, SkillParams, SkillStatus


class StandSkill(BaseSkill):
    """Hold the robot at the default standing pose."""

    name = "stand"
    action_dim = 24  # full body
    requires_rl = False

    def __init__(self) -> None:
        self._target = np.array(
            [STAND_JOINT_POSITIONS[n] for n in JOINT_NAMES], dtype=np.float32,
        )

    def reset(self, params: SkillParams | None = None) -> None:
        pass

    def get_action(self, obs: np.ndarray) -> tuple[np.ndarray, SkillStatus]:
        return self._target.copy(), SkillStatus.RUNNING
