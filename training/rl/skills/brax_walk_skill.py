"""Walk skill backed by Brax/JAX checkpoint from MuJoCo Playground training.

Loads the v12 (domain-randomized) walking policy via training.mujoco.inference,
maintains a 3-frame observation history, and outputs 12-dim joint targets in
radians. Designed for direct joint control mode (servo.set) rather than the
legacy walk.set parameter mode.

Usage:
    skill = BraxWalkSkill(checkpoint_path="checkpoints/mujoco_locomotion_v12_dr")
    skill.set_command(vx=0.5, vy=0.0, vyaw=0.0)
    action, status = skill.get_action(bridge_obs)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from training.rl.skills.base_skill import BaseSkill, SkillParams, SkillStatus

# Observation layout (matches joystick.py):
# gyro(3) + gravity(3) + command(3) + leg_pos(12) + leg_vel(12) + last_act(12) = 45
SINGLE_OBS_DIM = 45
OBS_HISTORY_SIZE = 3
TOTAL_OBS_DIM = SINGLE_OBS_DIM * OBS_HISTORY_SIZE  # 135

NUM_LEG_JOINTS = 12
ACTION_SCALE = 0.3
CTRL_DT = 0.02  # 50 Hz


class BraxWalkSkill(BaseSkill):
    """Walk using the trained Brax/JAX locomotion policy.

    This skill:
    1. Loads the Brax checkpoint via training.mujoco.inference.load_policy()
    2. Maintains a 3-frame obs history buffer (45 * 3 = 135 dims)
    3. Accepts velocity commands via set_command(vx, vy, vyaw)
    4. Maps bridge telemetry to the 45-dim Brax obs format
    5. Outputs 12-dim joint targets (radians, absolute) for servo.set

    The obs mapping from bridge telemetry:
        gyro(3)     <- IMU angular velocity (if available) or zeros
        gravity(3)  <- computed from IMU roll/pitch
        command(3)  <- set_command(vx, vy, vyaw)
        leg_pos(12) <- servo position feedback - default_pose
        leg_vel(12) <- finite-diff from positions * 0.05
        last_act(12)<- internal buffer
    """

    name = "walk"
    action_dim = NUM_LEG_JOINTS
    requires_rl = True

    def __init__(
        self,
        checkpoint_path: str | None = None,
        default_pose: np.ndarray | None = None,
    ) -> None:
        self._inference_fn = None
        self._config: dict | None = None

        # Velocity command
        self._command = np.zeros(3, dtype=np.float32)

        # State buffers
        self._last_action = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)
        self._obs_history = np.zeros(TOTAL_OBS_DIM, dtype=np.float32)
        self._last_positions = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)

        # Default standing pose (leg joints only, in radians)
        # Bent-knee standing pose matching training
        if default_pose is not None:
            self._default_pose = default_pose[:NUM_LEG_JOINTS].copy()
        else:
            self._default_pose = np.array([
                # Right leg: hip_yaw, hip_roll, hip_pitch, knee, ank_pitch, ank_roll
                0, 0, -0.3, 0.6, -0.3, 0,
                # Left leg: hip_yaw, hip_roll, hip_pitch, knee, ank_pitch, ank_roll
                0, 0, 0.3, -0.6, 0.3, 0,
            ], dtype=np.float32)

        self._params = SkillParams()
        self._step = 0

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path: str) -> None:
        """Load Brax checkpoint via training.mujoco.inference."""
        from training.mujoco.inference import load_policy
        self._inference_fn, self._config = load_policy(path)

    def set_command(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0) -> None:
        """Set velocity command for the walking policy.

        Args:
            vx: Forward velocity (m/s). Training range: [-0.3, 0.8].
            vy: Lateral velocity (m/s). Training range: [-0.4, 0.4].
            vyaw: Yaw rate (rad/s). Training range: [-0.5, 0.5].
        """
        self._command[0] = np.clip(vx, -0.3, 0.8)
        self._command[1] = np.clip(vy, -0.4, 0.4)
        self._command[2] = np.clip(vyaw, -0.5, 0.5)

    def reset(self, params: SkillParams | None = None) -> None:
        self._params = params or SkillParams()
        self._step = 0
        self._last_action = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)
        self._obs_history = np.zeros(TOTAL_OBS_DIM, dtype=np.float32)
        self._last_positions = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)

        # Map skill params to velocity command
        speed = self._params.speed
        direction = self._params.direction
        self._command[0] = float(speed * 0.5)  # speed multiplier -> vx
        self._command[2] = float(direction * 0.3)  # direction -> vyaw

    def get_action(self, obs: np.ndarray) -> tuple[np.ndarray, SkillStatus]:
        """Compute one step of walking policy.

        Args:
            obs: Bridge observation. Can be:
                - Raw bridge obs (arbitrary format) — will use internal state
                - Pre-built 135-dim obs history — passed directly to policy

        Returns:
            (joint_targets, status) where joint_targets are absolute radians
            for the 12 leg joints, suitable for servo.set.
        """
        self._step += 1

        # Check duration
        if self._params.duration_sec > 0:
            elapsed = self._step * CTRL_DT
            if elapsed >= self._params.duration_sec:
                return self._default_pose.copy(), SkillStatus.COMPLETED

        if self._inference_fn is None:
            return self._default_pose.copy(), SkillStatus.RUNNING

        # Build observation if needed
        if obs.shape[0] == TOTAL_OBS_DIM:
            # Already a full 135-dim obs history
            full_obs = obs
        else:
            # Build from whatever we have
            full_obs = self._build_obs_from_bridge(obs)

        # Run policy
        action = self._inference_fn(full_obs)
        action = np.clip(action, -1.0, 1.0)

        # Convert to absolute joint targets
        joint_targets = self._default_pose + action[:NUM_LEG_JOINTS] * ACTION_SCALE

        self._last_action = action[:NUM_LEG_JOINTS].astype(np.float32)
        return joint_targets, SkillStatus.RUNNING

    def get_action_from_telemetry(
        self,
        gyro: np.ndarray | None = None,
        imu_roll: float = 0.0,
        imu_pitch: float = 0.0,
        joint_positions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, SkillStatus]:
        """Compute action from structured bridge telemetry.

        This is the preferred entry point when bridge provides structured data.

        Args:
            gyro: Angular velocity [wx, wy, wz] in body frame. None = zeros.
            imu_roll: Roll angle in radians.
            imu_pitch: Pitch angle in radians.
            joint_positions: Current leg joint positions in radians (12-dim).

        Returns:
            (joint_targets, status) where joint_targets are absolute radians.
        """
        self._step += 1

        if self._params.duration_sec > 0:
            elapsed = self._step * CTRL_DT
            if elapsed >= self._params.duration_sec:
                return self._default_pose.copy(), SkillStatus.COMPLETED

        if self._inference_fn is None:
            return self._default_pose.copy(), SkillStatus.RUNNING

        # Build 45-dim single obs
        if gyro is None:
            gyro = np.zeros(3, dtype=np.float32)

        # Gravity from IMU orientation
        gravity = np.array([
            np.sin(imu_roll),
            np.sin(imu_pitch),
            np.cos(imu_roll) * np.cos(imu_pitch),
        ], dtype=np.float32)

        # Joint positions relative to default
        if joint_positions is not None:
            leg_pos = joint_positions[:NUM_LEG_JOINTS] - self._default_pose
            # Finite difference velocity estimate
            leg_vel = (joint_positions[:NUM_LEG_JOINTS] - self._last_positions) * 0.05
            self._last_positions = joint_positions[:NUM_LEG_JOINTS].copy()
        else:
            leg_pos = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)
            leg_vel = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)

        single_obs = np.concatenate([
            gyro,                   # 3
            gravity,                # 3
            self._command,          # 3
            leg_pos,                # 12
            leg_vel,                # 12
            self._last_action,      # 12
        ]).astype(np.float32)       # 45

        # Stack into history
        full_obs = self._stack_history(single_obs)

        # Run policy
        action = self._inference_fn(full_obs)
        action = np.clip(action, -1.0, 1.0)

        joint_targets = self._default_pose + action[:NUM_LEG_JOINTS] * ACTION_SCALE
        self._last_action = action[:NUM_LEG_JOINTS].astype(np.float32)

        return joint_targets, SkillStatus.RUNNING

    def _build_obs_from_bridge(self, obs: np.ndarray) -> np.ndarray:
        """Build 135-dim obs from partial bridge observation."""
        # Extract what we can from the bridge obs
        gyro = np.zeros(3, dtype=np.float32)
        gravity = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # If obs has IMU data at known positions, extract it
        if obs.shape[0] >= 5:
            # Assume obs has at least [timestamp, battery, imu_roll, imu_pitch, is_walking]
            imu_roll = float(obs[2]) if obs.shape[0] > 2 else 0.0
            imu_pitch = float(obs[3]) if obs.shape[0] > 3 else 0.0
            gravity[0] = np.sin(imu_roll)
            gravity[1] = np.sin(imu_pitch)
            gravity[2] = np.cos(imu_roll) * np.cos(imu_pitch)

        single_obs = np.concatenate([
            gyro,
            gravity,
            self._command,
            np.zeros(NUM_LEG_JOINTS, dtype=np.float32),  # leg_pos (no feedback)
            np.zeros(NUM_LEG_JOINTS, dtype=np.float32),  # leg_vel (no feedback)
            self._last_action,
        ]).astype(np.float32)

        return self._stack_history(single_obs)

    def _stack_history(self, obs: np.ndarray) -> np.ndarray:
        """Push new obs into front of history buffer."""
        self._obs_history = np.roll(self._obs_history, obs.size)
        self._obs_history[:obs.size] = obs
        return self._obs_history.copy()

    @property
    def default_pose(self) -> np.ndarray:
        """Return the default standing pose for leg joints."""
        return self._default_pose.copy()

    @property
    def command(self) -> np.ndarray:
        """Current velocity command [vx, vy, vyaw]."""
        return self._command.copy()

    @property
    def is_loaded(self) -> bool:
        """Whether a checkpoint has been loaded."""
        return self._inference_fn is not None
