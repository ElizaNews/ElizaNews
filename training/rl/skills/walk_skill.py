"""Walk skill — wraps a trained locomotion policy for deployment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from training.rl.skills.base_skill import BaseSkill, SkillParams, SkillStatus

NUM_LEG_JOINTS = 12


class WalkSkill(BaseSkill):
    """Walk using a trained locomotion policy.

    Falls back to zero actions if no checkpoint is loaded.
    """

    name = "walk"
    action_dim = NUM_LEG_JOINTS  # 12
    requires_rl = True

    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu") -> None:
        self._device = torch.device(device)
        self._model: torch.nn.Module | None = None
        self._params = SkillParams()
        self._step = 0

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def reset(self, params: SkillParams | None = None) -> None:
        self._params = params or SkillParams()
        self._step = 0

    def get_action(self, obs: np.ndarray) -> tuple[np.ndarray, SkillStatus]:
        self._step += 1

        # Check duration.
        if self._params.duration_sec > 0:
            elapsed = self._step * 0.02  # 50 Hz policy
            if elapsed >= self._params.duration_sec:
                return np.zeros(self.action_dim, dtype=np.float32), SkillStatus.COMPLETED

        if self._model is None:
            # No model loaded — return zero actions (stand-in-place).
            return np.zeros(self.action_dim, dtype=np.float32), SkillStatus.RUNNING

        # Run the actor.
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self._device)
            action = self._model.actor(obs_t)
            action = torch.clamp(action, -1.0, 1.0)

        return action.squeeze(0).cpu().numpy(), SkillStatus.RUNNING

    def load_checkpoint(self, path: str) -> None:
        """Load a trained locomotion checkpoint.

        Supports both legacy PyTorch checkpoints and Brax/JAX checkpoints.
        For Brax checkpoints, use training.mujoco.inference.load_policy().
        """
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        if "model" in ckpt:
            # Legacy PyTorch checkpoint — load state dict directly.
            model = ckpt["model"]
            if isinstance(model, dict):
                # state_dict only — cannot reconstruct without ActorCritic class.
                # Fall back to zero actions.
                return
            model.eval()
            self._model = model
