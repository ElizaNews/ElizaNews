"""RL policy runtime — implements PolicyRuntime for bridge deployment.

Converts bridge telemetry (RobotObservation) into RL observation vectors,
runs the trained policy, and converts actions back to PolicyOutput commands.

Supports both:
1. Brax/JAX checkpoints (from training/mujoco/ pipeline) via BraxWalkSkill
2. Walk-parameter or direct-joint deployment modes

Direct joint mode outputs servo.set commands with per-joint radians→pulse
conversion, bypassing the Hiwonder walking engine entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from training.interfaces import PolicyRuntime, PolicyOutput, PolicyVector, RobotObservation
from training.rl.skills.base_skill import SkillParams, SkillStatus
from training.rl.skills.registry import SkillRegistry
from training.rl.skills.stand_skill import StandSkill
from training.rl.skills.brax_walk_skill import BraxWalkSkill
from training.rl.skills.turn_skill import TurnSkill
from training.rl.skills.wave_skill import WaveSkill
from training.rl.skills.bow_skill import BowSkill
from training.rl.meta.text_encoder import TextEncoder
from training.rl.meta.command_parser import CommandParser
from training.rl.meta.meta_policy import MetaPolicy

NUM_LEG_JOINTS = 12
OBS_DIM = 48


class RLPolicyRuntime(PolicyRuntime):
    """PolicyRuntime that uses trained RL policies and skill dispatch.

    Integrates:
    - BraxWalkSkill (Brax/JAX checkpoint for direct joint control)
    - Command parser (text -> skill selection)
    - Skill registry (skill execution)
    - Meta-policy (learned skill selection, optional)

    Deploy modes:
    - "walk_params": Legacy mode, maps actions to walk.set parameters
    - "direct_joint": New mode, outputs joint_positions dict for servo.set
    """

    def __init__(
        self,
        locomotion_checkpoint: str | None = None,
        meta_checkpoint: str | None = None,
        device: str = "cpu",
        use_meta_policy: bool = False,
        deploy_mode: str = "walk_params",  # "walk_params" or "direct_joint"
    ):
        self.device = device
        self.deploy_mode = deploy_mode

        # Build skill registry with BraxWalkSkill for locomotion.
        self.registry = SkillRegistry()
        self.registry.register(StandSkill())

        self._brax_walk = BraxWalkSkill(checkpoint_path=locomotion_checkpoint)
        self.registry.register(self._brax_walk)

        self.registry.register(TurnSkill(device=device))
        self.registry.register(WaveSkill())
        self.registry.register(BowSkill())

        # Command parser.
        self._encoder = TextEncoder(prefer_transformer=False)
        self._parser = CommandParser(encoder=self._encoder)

        # Meta-policy (optional).
        self._meta: MetaPolicy | None = None
        if use_meta_policy and meta_checkpoint:
            skill_names = self.registry.list_skills()
            self._meta = MetaPolicy(
                skill_names=skill_names,
                text_dim=self._encoder.dim,
                device=device,
            )
            self._meta.load_checkpoint(meta_checkpoint)

        # Active skill state.
        self._active_skill_name: str = "stand"
        self._active_skill = self.registry.get("stand")
        if self._active_skill is not None:
            self._active_skill.reset()

        # Observation buffer for RL.
        self._last_obs = np.zeros(OBS_DIM, dtype=np.float32)
        self._last_action = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)

        # Joint position feedback buffer (for direct_joint mode)
        self._joint_positions = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)

    def infer(self, obs: RobotObservation, z: PolicyVector) -> PolicyOutput:
        """Run one inference step.

        The PolicyVector.values can contain:
        - Empty: continue current skill
        - Text command hash: parse and switch skill
        """
        # Get action from active skill.
        if self._active_skill is not None:
            if isinstance(self._active_skill, BraxWalkSkill):
                # Use structured telemetry for BraxWalkSkill
                action, status = self._active_skill.get_action_from_telemetry(
                    imu_roll=obs.imu_roll,
                    imu_pitch=obs.imu_pitch,
                    joint_positions=self._joint_positions,
                )
            else:
                rl_obs = self._bridge_obs_to_rl(obs)
                action, status = self._active_skill.get_action(rl_obs)

            self._last_action = action[:NUM_LEG_JOINTS] if len(action) >= NUM_LEG_JOINTS else action

            if status == SkillStatus.COMPLETED:
                self._switch_skill("stand")
        else:
            action = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)

        return self._action_to_output(action, obs)

    def update_joint_feedback(self, joint_positions: np.ndarray) -> None:
        """Update joint position feedback from servo telemetry.

        Called by the bridge when servo position data is available.
        """
        self._joint_positions = joint_positions[:NUM_LEG_JOINTS].astype(np.float32)

    def set_velocity_command(self, vx: float, vy: float, vyaw: float) -> None:
        """Set velocity command on the BraxWalkSkill."""
        self._brax_walk.set_command(vx, vy, vyaw)

    def handle_text_command(self, text: str) -> str:
        """Parse a text command and switch to the appropriate skill.

        Returns the selected skill name.
        """
        if self._meta is not None:
            text_emb = self._encoder.encode_single(text)
            state = self._last_obs[:12]
            skill_name, params = self._meta.select_skill(text_emb, state)
        else:
            result = self._parser.parse(text)
            skill_name = result.skill_name
            params = result.params.to_dict()

        skill_params = SkillParams(
            speed=params.get("speed", 1.0),
            direction=params.get("direction", 0.0),
            magnitude=params.get("magnitude", 1.0),
            duration_sec=params.get("duration_sec", 0.0),
        )

        self._switch_skill(skill_name, skill_params)
        return skill_name

    def _switch_skill(self, name: str, params: SkillParams | None = None) -> None:
        """Switch to a new skill."""
        skill = self.registry.get(name)
        if skill is not None:
            self._active_skill_name = name
            self._active_skill = skill
            skill.reset(params)

    def _bridge_obs_to_rl(self, obs: RobotObservation) -> np.ndarray:
        """Convert bridge RobotObservation to 48-dim RL observation.

        This is an approximate mapping since the bridge doesn't provide
        all the state that the RL policy expects. Missing values are
        filled from the last known state.
        """
        rl_obs = self._last_obs.copy()

        # IMU -> approximate gravity vector (indices 6-8).
        rl_obs[6] = np.sin(obs.imu_roll)
        rl_obs[7] = np.sin(obs.imu_pitch)
        rl_obs[8] = np.cos(obs.imu_roll) * np.cos(obs.imu_pitch)

        # Previous actions (indices 36-47).
        rl_obs[36:36 + NUM_LEG_JOINTS] = self._last_action

        self._last_obs = rl_obs
        return rl_obs

    def _action_to_output(self, action: np.ndarray, obs: RobotObservation) -> PolicyOutput:
        """Convert RL action to PolicyOutput for the bridge."""
        if self.deploy_mode == "direct_joint":
            # Direct joint control: action is absolute joint positions (radians).
            # Store in PolicyOutput.action_name as marker, actual joint data
            # is sent via the joint_positions property.
            action_name = ""
            if self._active_skill_name in ("wave", "bow"):
                action_name = self._active_skill_name

            return PolicyOutput(
                walk_x=0.0,
                walk_y=0.0,
                walk_yaw=0.0,
                walk_height=0.036,
                walk_speed=0,
                action_name=action_name,
            )
        else:
            # Legacy walk_params mode.
            walk_x = float(np.clip(np.mean(action[2:4]) * 0.02, -0.05, 0.05))
            walk_y = float(np.clip(np.mean(action[0:2]) * 0.01, -0.03, 0.03))
            walk_yaw = float(np.clip(np.mean(action[6:8]) * 5.0, -10.0, 10.0))

            action_name = ""
            if self._active_skill_name in ("wave", "bow"):
                action_name = self._active_skill_name

            return PolicyOutput(
                walk_x=walk_x,
                walk_y=walk_y,
                walk_yaw=walk_yaw,
                walk_height=0.036,
                walk_speed=2,
                action_name=action_name,
            )

    @property
    def joint_positions(self) -> np.ndarray:
        """Current joint position targets (radians) from the active skill.

        Used in direct_joint mode to get servo.set targets.
        """
        return self._last_action.copy()

    @property
    def active_skill_name(self) -> str:
        return self._active_skill_name
