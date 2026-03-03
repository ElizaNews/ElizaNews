"""Phase 2 configuration: skill training variants.

Note: Phase 1 locomotion training now uses the MuJoCo Playground pipeline
(training/mujoco/). These configs define skill-specific overrides for
the deployment/skills layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VelocityCommandCfg:
    """Velocity command ranges for skill training."""
    vx_range: tuple[float, float] = (-0.3, 0.8)
    vy_range: tuple[float, float] = (-0.4, 0.4)
    yaw_range: tuple[float, float] = (-0.5, 0.5)


@dataclass
class SkillTrainingCfg:
    """Base config for skill-specific training."""
    skill_name: str = ""
    commands: VelocityCommandCfg = field(default_factory=VelocityCommandCfg)
    episode_length: int = 1000
    action_scale: float = 0.3


@dataclass
class WalkSkillCfg(SkillTrainingCfg):
    """Walk skill — default velocity commands."""
    skill_name: str = "walk"


@dataclass
class TurnSkillCfg(SkillTrainingCfg):
    """Turn skill — zero linear velocity, full yaw range."""
    skill_name: str = "turn"
    commands: VelocityCommandCfg = field(default_factory=lambda: VelocityCommandCfg(
        vx_range=(0.0, 0.0),
        vy_range=(0.0, 0.0),
        yaw_range=(-0.5, 0.5),
    ))
