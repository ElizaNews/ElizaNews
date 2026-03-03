"""Target-reaching locomotion environment for AiNex.

Walk toward a randomly-placed target. The target is virtual (tracked in info
dict as a JAX array) — no extra bodies in the MJCF, keeping MJX JIT fast.

Extends the Joystick pattern: same robot, same physics, different task.

Usage:
    from training.mujoco.target import TargetReaching, default_config
    env = TargetReaching()

    # Or with training:
    python3 -m training.mujoco.train --target
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from perception.entity_slots.sim_provider import empty_entity_slots, sim_entity_slots_jax
from perception.entity_slots.slot_noise import apply_entity_slot_noise
from training.mujoco import base_env as ainex_base


def default_config() -> config_dict.ConfigDict:
    """Default configuration for AiNex target reaching."""
    return config_dict.create(
        ctrl_dt=0.02,       # 50 Hz control
        sim_dt=0.004,       # 250 Hz physics
        episode_length=1000,
        Kp=21.1,            # Position gain (matching OP3)
        Kd=1.084,           # Velocity damping (matching OP3)
        early_termination=True,
        action_repeat=1,
        action_scale=0.3,
        obs_noise=0.05,
        obs_history_size=3,
        max_foot_height=0.05,
        # Target task parameters
        target_radius_min=0.5,   # metres
        target_radius_max=2.0,   # metres
        target_reached_threshold=0.15,  # metres
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Target-reaching rewards
                target_distance=5.0,      # shaped: closer = better
                target_heading=1.0,       # face the target
                target_reached=50.0,      # bonus on arrival
                target_velocity=2.0,      # move toward target
                # Stability penalties (same as joystick)
                lin_vel_z=-2.0,
                ang_vel_xy=-0.05,
                orientation=-5.0,
                torques=-0.0002,
                action_rate=-0.01,
                termination=-1.0,
                feet_slip=-0.1,
                feet_orientation=-2.0,
                energy=-0.0001,
            ),
        ),
        velocity_kick=[0.5, 3.0],
        kick_durations=[0.05, 0.2],
        kick_wait_times=[1.0, 3.0],
        enable_entity_slots=False,
    )


class TargetReaching(ainex_base.AiNexEnv):
    """Walk toward a randomly-placed target position."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config=config, config_overrides=config_overrides)
        self._init_robot()
        if self._config.enable_entity_slots:
            self._init_entities()

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, target_rng, noise_rng, entity_rng = jax.random.split(rng, 4)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=self._init_q,
            qvel=jp.zeros(self.mjx_model.nv),
        )
        data = mjx.forward(self.mjx_model, data)

        # Sample initial target position
        target_pos = self._sample_target(target_rng, data)
        robot_xy = data.qpos[:2]
        prev_dist = jp.linalg.norm(target_pos - robot_xy)

        info = {
            "rng": rng,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_vel": jp.zeros(self.mjx_model.nv - 6),
            "target_pos": target_pos,       # (2,) x, y in world frame
            "prev_target_dist": prev_dist,   # scalar
            "targets_reached": jp.int32(0),
            "step": 0,
            "motor_targets": jp.zeros(self.mjx_model.nu),
        }

        if self._config.enable_entity_slots:
            scene_rng, slot_rng = jax.random.split(entity_rng)
            scene = self.sample_entity_scene(scene_rng, data.qpos[:2])
            info["entity_positions"] = scene["entity_positions"]
            info["entity_types"] = scene["entity_types"]
            info["entity_sizes"] = scene["entity_sizes"]
            info["entity_mask"] = scene["entity_mask"]

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["targets_reached"] = jp.zeros(())

        obs_history = jp.zeros(self._config.obs_history_size * self._single_obs_size)
        obs = self._get_obs(data, info, obs_history, noise_rng)
        if self._config.enable_entity_slots:
            obs = jp.concatenate([obs, self._compute_entity_slots(data, info, slot_rng)])
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng, target_rng, noise_rng, entity_rng = jax.random.split(state.info["rng"], 4)

        # Apply action
        motor_targets = self._default_pose + action * self._config.action_scale
        motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )

        # Termination
        done = self.get_termination(data)

        # Compute target-relative state
        robot_xy = data.qpos[:2]
        target_pos = state.info["target_pos"]
        target_dist = jp.linalg.norm(target_pos - robot_xy)
        reached = target_dist < self._config.target_reached_threshold

        # Compute rewards
        rewards = self._get_reward(data, action, state.info, done, target_dist)
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, -10.0, 10000.0)

        # Respawn target if reached
        new_target = self._sample_target(target_rng, data)
        target_pos = jp.where(reached, new_target, target_pos)
        new_dist = jp.where(
            reached,
            jp.linalg.norm(new_target - robot_xy),
            target_dist,
        )
        targets_reached = state.info["targets_reached"] + jp.int32(reached)

        # Build observation
        info = {
            "rng": rng,
            "last_act": action,
            "last_last_act": state.info["last_act"],
            "last_vel": data.qvel[6:],
            "target_pos": target_pos,
            "prev_target_dist": new_dist,
            "targets_reached": targets_reached,
            "step": state.info["step"] + 1,
            "motor_targets": motor_targets,
        }

        if self._config.enable_entity_slots:
            # Carry entity scene forward (static within episode)
            info["entity_positions"] = state.info["entity_positions"]
            info["entity_types"] = state.info["entity_types"]
            info["entity_sizes"] = state.info["entity_sizes"]
            info["entity_mask"] = state.info["entity_mask"]

        obs_history = state.obs[:self._config.obs_history_size * self._single_obs_size] if self._config.enable_entity_slots else state.obs
        obs = self._get_obs(data, info, obs_history, noise_rng)
        if self._config.enable_entity_slots:
            obs = jp.concatenate([obs, self._compute_entity_slots(data, info, entity_rng)])

        # Update metrics
        metrics = {}
        for k, v in rewards.items():
            metrics[f"reward/{k}"] = v
        metrics["targets_reached"] = jp.float32(targets_reached)

        done = jp.float32(done)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _sample_target(self, rng: jax.Array, data: mjx.Data) -> jax.Array:
        """Sample a random target position around the robot."""
        rng1, rng2 = jax.random.split(rng)
        angle = jax.random.uniform(rng1, minval=0, maxval=2 * jp.pi)
        radius = jax.random.uniform(
            rng2,
            minval=self._config.target_radius_min,
            maxval=self._config.target_radius_max,
        )
        robot_xy = data.qpos[:2]
        return robot_xy + radius * jp.array([jp.cos(angle), jp.sin(angle)])

    @property
    def _single_obs_size(self) -> int:
        """Single obs: gyro(3) + gravity(3) + target_vec(2) + target_dist(1) +
        target_bearing(1) + joint_pos(24) + last_act(24) = 58."""
        return 3 + 3 + 2 + 1 + 1 + self.mjx_model.nu + self.mjx_model.nu

    def _compute_entity_slots(self, data: mjx.Data, info: dict[str, Any], rng: jax.Array) -> jax.Array:
        """Compute entity slots from randomized scene with DR noise."""
        robot_pos = data.qpos[:3]
        robot_yaw = self.get_yaw(data)
        slots = sim_entity_slots_jax(
            robot_pos, robot_yaw,
            info["entity_positions"],
            info["entity_types"],
            info["entity_sizes"],
            entity_mask=info["entity_mask"],
        )
        return apply_entity_slot_noise(slots, rng)

    def _get_obs(
        self,
        data: mjx.Data,
        info: dict[str, Any],
        obs_history: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        gyro = self.get_gyro(data)
        gravity = self.get_gravity(data)

        # Target in body frame
        robot_xy = data.qpos[:2]
        delta = info["target_pos"] - robot_xy
        yaw = self.get_yaw(data)

        cos_yaw = jp.cos(-yaw)
        sin_yaw = jp.sin(-yaw)
        target_vec = jp.array([
            delta[0] * cos_yaw - delta[1] * sin_yaw,
            delta[0] * sin_yaw + delta[1] * cos_yaw,
        ])

        target_dist = jp.linalg.norm(delta)
        target_angle = jp.arctan2(delta[1], delta[0])
        target_bearing = jp.arctan2(
            jp.sin(target_angle - yaw), jp.cos(target_angle - yaw)
        )

        obs = jp.concatenate([
            gyro,                                # 3
            gravity,                             # 3
            target_vec,                          # 2
            jp.array([target_dist]),              # 1
            jp.array([target_bearing]),           # 1
            data.qpos[7:] - self._default_pose,  # 24
            info["last_act"],                    # 24
        ])  # total = 58

        obs = self.apply_obs_noise(obs, rng)
        return self.stack_obs_history(obs, obs_history)

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
        target_dist: jax.Array,
    ) -> dict[str, jax.Array]:
        delta = info["target_pos"] - data.qpos[:2]
        yaw = self.get_yaw(data)

        # Target bearing
        target_angle = jp.arctan2(delta[1], delta[0])
        target_bearing = jp.arctan2(
            jp.sin(target_angle - yaw), jp.cos(target_angle - yaw)
        )

        # Body-frame velocity toward target
        local_vel = self.get_local_linvel(data)
        vel_toward = (
            local_vel[0] * jp.cos(target_bearing)
            + local_vel[1] * jp.sin(target_bearing)
        )

        return {
            # Target-reaching rewards
            "target_distance": info["prev_target_dist"] - target_dist,
            "target_heading": jp.exp(-target_bearing**2 / 0.5),
            "target_reached": jp.float32(
                target_dist < self._config.target_reached_threshold
            ),
            "target_velocity": jp.clip(vel_toward, 0.0, 1.0),
            # Stability penalties (from base)
            "lin_vel_z": self.cost_lin_vel_z(data),
            "ang_vel_xy": self.cost_ang_vel_xy(data),
            "orientation": self.cost_orientation(data),
            "torques": self.cost_torques(data),
            "action_rate": self.cost_action_rate(
                action, info["last_act"], info["last_last_act"]
            ),
            "termination": self.cost_termination(done, info["step"]),
            "feet_slip": self.cost_feet_slip(data),
            "feet_orientation": self.cost_feet_orientation(data),
            "energy": self.cost_energy(data),
        }
