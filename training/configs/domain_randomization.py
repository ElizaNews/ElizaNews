"""Domain randomization configuration for sim-to-real transfer.

Defines randomization ranges for physics parameters, sensor noise,
actuator properties, and visual appearance. These are applied during
training to improve policy robustness for real-world deployment.

Usage:
    from training.configs.domain_randomization import (
        get_randomization_config,
        apply_randomization_to_env_config,
    )

    dr_cfg = get_randomization_config("moderate")
    env_cfg = apply_randomization_to_env_config(env_cfg, dr_cfg)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from typing import Any


@dataclass
class DomainRandomizationConfig:
    """Full domain randomization configuration."""

    name: str = "none"

    # Ground physics
    friction_range: tuple[float, float] = (1.0, 1.0)

    # Robot physics
    mass_scale_range: tuple[float, float] = (1.0, 1.0)
    joint_stiffness_scale: tuple[float, float] = (1.0, 1.0)
    joint_damping_scale: tuple[float, float] = (1.0, 1.0)

    # Sensor noise
    imu_noise_std: float = 0.0          # radians
    observation_noise_std: float = 0.0   # fraction of range
    battery_noise_mv: float = 0.0        # millivolts

    # Actuator
    action_delay_steps: int = 0          # control steps of delay
    action_noise_std: float = 0.0        # fraction of range

    # External perturbation
    push_force_range: tuple[float, float] = (0.0, 0.0)  # Newtons
    push_interval_steps: int = 100

    # Visual (for camera-based policies)
    brightness_range: tuple[float, float] = (1.0, 1.0)
    contrast_range: tuple[float, float] = (1.0, 1.0)
    camera_noise_std: float = 0.0        # pixel noise
    color_jitter: bool = False

    def summary(self) -> str:
        lines = [f"Domain Randomization: {self.name}"]
        if self.friction_range != (1.0, 1.0):
            lines.append(f"  Friction: {self.friction_range}")
        if self.mass_scale_range != (1.0, 1.0):
            lines.append(f"  Mass scale: {self.mass_scale_range}")
        if self.imu_noise_std > 0:
            lines.append(f"  IMU noise: {self.imu_noise_std:.3f} rad")
        if self.action_delay_steps > 0:
            lines.append(f"  Action delay: {self.action_delay_steps} steps")
        if self.push_force_range[1] > 0:
            lines.append(f"  Push force: {self.push_force_range} N")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

PRESETS: dict[str, DomainRandomizationConfig] = {
    "none": DomainRandomizationConfig(name="none"),

    "light": DomainRandomizationConfig(
        name="light",
        friction_range=(0.8, 1.2),
        mass_scale_range=(0.95, 1.05),
        imu_noise_std=0.02,
        observation_noise_std=0.01,
        action_delay_steps=0,
    ),

    "moderate": DomainRandomizationConfig(
        name="moderate",
        friction_range=(0.5, 1.5),
        mass_scale_range=(0.9, 1.1),
        joint_stiffness_scale=(0.8, 1.2),
        joint_damping_scale=(0.8, 1.2),
        imu_noise_std=0.05,
        observation_noise_std=0.02,
        action_delay_steps=1,
        action_noise_std=0.02,
        push_force_range=(0.0, 2.0),
        push_interval_steps=50,
        brightness_range=(0.8, 1.2),
        contrast_range=(0.9, 1.1),
        camera_noise_std=3.0,
    ),

    "aggressive": DomainRandomizationConfig(
        name="aggressive",
        friction_range=(0.3, 1.8),
        mass_scale_range=(0.85, 1.15),
        joint_stiffness_scale=(0.7, 1.3),
        joint_damping_scale=(0.5, 2.0),
        imu_noise_std=0.08,
        observation_noise_std=0.03,
        action_delay_steps=2,
        action_noise_std=0.03,
        push_force_range=(0.5, 3.0),
        push_interval_steps=30,
        brightness_range=(0.6, 1.4),
        contrast_range=(0.7, 1.3),
        camera_noise_std=5.0,
        color_jitter=True,
    ),

    "real_world": DomainRandomizationConfig(
        name="real_world",
        # Calibrated to real AiNex hardware measurements
        friction_range=(0.4, 1.2),         # tile vs carpet
        mass_scale_range=(0.92, 1.08),     # manufacturing tolerance
        joint_stiffness_scale=(0.8, 1.2),  # servo temperature effects
        joint_damping_scale=(0.6, 1.5),    # wear
        imu_noise_std=0.04,                # measured IMU noise
        observation_noise_std=0.02,
        battery_noise_mv=100.0,            # ADC noise
        action_delay_steps=1,              # USB serial latency ~20ms
        action_noise_std=0.015,
        push_force_range=(0.0, 1.5),       # gentle human touch
        push_interval_steps=100,
        brightness_range=(0.7, 1.3),       # indoor lighting variation
        contrast_range=(0.8, 1.2),
        camera_noise_std=4.0,
    ),
}


def get_randomization_config(preset: str = "moderate") -> DomainRandomizationConfig:
    """Get a domain randomization config by preset name.

    Args:
        preset: One of "none", "light", "moderate", "aggressive", "real_world"

    Returns:
        DomainRandomizationConfig
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
    return PRESETS[preset]


def apply_randomization_to_env_config(
    env_cfg: Any,
    dr_cfg: DomainRandomizationConfig,
) -> Any:
    """Apply domain randomization settings to an environment config.

    For MuJoCo environments, use training.mujoco.domain_randomization instead.
    This function is kept for compatibility with any config object that has
    the expected attributes.
    """
    env_cfg.randomize = dr_cfg.name != "none"
    env_cfg.friction_range = dr_cfg.friction_range
    env_cfg.mass_scale_range = dr_cfg.mass_scale_range
    env_cfg.imu_noise_std = dr_cfg.imu_noise_std
    env_cfg.observation_noise_std = dr_cfg.observation_noise_std
    env_cfg.action_delay_steps = dr_cfg.action_delay_steps
    env_cfg.push_force_range = dr_cfg.push_force_range
    env_cfg.push_interval_steps = dr_cfg.push_interval_steps
    return env_cfg


# ---------------------------------------------------------------------------
# Graduated transfer protocol
# ---------------------------------------------------------------------------

@dataclass
class TransferStage:
    """One stage of the graduated sim-to-real transfer pipeline."""
    name: str
    backend: str              # mock, isaac, ros_sim, ros_real
    max_speed: int = 4        # walk speed cap
    max_stride: float = 0.05  # walk_x/y cap
    max_steps: int = 500      # episode length cap
    max_pitch_deg: float = 45 # fall detection threshold
    description: str = ""


TRANSFER_PIPELINE: list[TransferStage] = [
    TransferStage(
        name="sim_validation",
        backend="isaac",
        description="Validate in IsaacLab with full domain randomization",
    ),
    TransferStage(
        name="bridge_validation",
        backend="mock",
        description="Validate through bridge protocol with mock backend",
    ),
    TransferStage(
        name="gazebo_validation",
        backend="ros_sim",
        description="Validate in Gazebo with ROS-integrated physics",
    ),
    TransferStage(
        name="real_conservative",
        backend="ros_real",
        max_speed=1,
        max_stride=0.02,
        max_steps=50,
        max_pitch_deg=30,
        description="Real robot with tight safety limits",
    ),
    TransferStage(
        name="real_moderate",
        backend="ros_real",
        max_speed=2,
        max_stride=0.03,
        max_steps=100,
        max_pitch_deg=35,
        description="Real robot with moderate limits",
    ),
    TransferStage(
        name="real_full",
        backend="ros_real",
        max_speed=3,
        max_stride=0.05,
        max_steps=500,
        max_pitch_deg=45,
        description="Real robot with standard limits",
    ),
]


def get_transfer_pipeline() -> list[TransferStage]:
    """Get the full graduated transfer pipeline."""
    return list(TRANSFER_PIPELINE)
