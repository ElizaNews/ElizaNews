"""Policy evaluation result types.

For MuJoCo-based evaluation, use:
    python3 -m training.mujoco.eval_policy --checkpoint checkpoints/mujoco_locomotion
    python3 -m training.mujoco.inference --checkpoint checkpoints/mujoco_locomotion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """Aggregated evaluation results."""
    task: str = ""
    num_episodes: int = 0
    success_rate: float = 0.0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_length: float = 0.0
    std_length: float = 0.0
    max_imu_pitch: float = 0.0
    max_imu_roll: float = 0.0
    mean_action_smoothness: float = 0.0
    falls: int = 0
    timeouts: int = 0
    successes: int = 0
    per_episode: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Task: {self.task}",
            f"Episodes: {self.num_episodes}",
            f"Success rate: {self.success_rate:.1%}",
            f"Mean reward: {self.mean_reward:.2f} (+/-{self.std_reward:.2f})",
            f"Mean length: {self.mean_length:.0f} (+/-{self.std_length:.0f})",
            f"Falls: {self.falls}, Timeouts: {self.timeouts}, Successes: {self.successes}",
            f"Max IMU pitch: {self.max_imu_pitch:.3f} rad",
            f"Max IMU roll: {self.max_imu_roll:.3f} rad",
            f"Action smoothness: {self.mean_action_smoothness:.4f}",
        ]
        return "\n".join(lines)
