"""Turn skill — locomotion policy with yaw-only velocity commands."""

from __future__ import annotations

import numpy as np
import torch

from training.rl.skills.base_skill import BaseSkill, SkillParams, SkillStatus
from training.rl.skills.walk_skill import WalkSkill, NUM_LEG_JOINTS


class TurnSkill(BaseSkill):
    """Turn in place using the locomotion policy with yaw commands.

    Reuses the walk policy but overrides the velocity command to be yaw-only.
    """

    name = "turn"
    action_dim = NUM_LEG_JOINTS  # 12
    requires_rl = True

    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu") -> None:
        self._walk = WalkSkill(checkpoint_path=checkpoint_path, device=device)
        self._params = SkillParams()
        self._step = 0

    def reset(self, params: SkillParams | None = None) -> None:
        self._params = params or SkillParams()
        self._walk.reset(self._params)
        self._step = 0

    def get_action(self, obs: np.ndarray) -> tuple[np.ndarray, SkillStatus]:
        self._step += 1

        if self._params.duration_sec > 0:
            elapsed = self._step * 0.02
            if elapsed >= self._params.duration_sec:
                return np.zeros(self.action_dim, dtype=np.float32), SkillStatus.COMPLETED

        # Modify observation to set vx=0, vy=0, yaw_rate=direction*speed.
        obs_modified = obs.copy()
        # cmd_vel is at indices 9:12 in the 48-dim observation.
        yaw_rate = self._params.direction * self._params.speed * 0.5  # scale to [-0.5, 0.5]
        obs_modified[9] = 0.0   # vx = 0
        obs_modified[10] = 0.0  # vy = 0
        obs_modified[11] = yaw_rate

        return self._walk.get_action(obs_modified)

    def load_checkpoint(self, path: str) -> None:
        self._walk.load_checkpoint(path)
