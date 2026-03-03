"""Tests for deployment modules."""

import numpy as np
import pytest

from training.interfaces import RobotObservation, PolicyVector
from training.rl.deploy.rl_policy_runtime import RLPolicyRuntime
from training.rl.skills.base_skill import SkillStatus


def _make_obs(timestamp: float = 0.0) -> RobotObservation:
    return RobotObservation(
        timestamp=timestamp,
        battery_mv=7400,
        imu_roll=0.0,
        imu_pitch=0.0,
        is_walking=False,
    )


class TestRLPolicyRuntime:
    def test_init_no_checkpoint(self):
        runtime = RLPolicyRuntime()
        assert runtime._active_skill_name == "stand"

    def test_infer_returns_policy_output(self):
        runtime = RLPolicyRuntime()
        obs = _make_obs()
        z = PolicyVector(values=())
        output = runtime.infer(obs, z)
        assert hasattr(output, "walk_x")
        assert hasattr(output, "walk_y")
        assert hasattr(output, "walk_yaw")
        assert hasattr(output, "walk_height")
        assert hasattr(output, "walk_speed")
        assert hasattr(output, "action_name")

    def test_handle_text_command_walk(self):
        runtime = RLPolicyRuntime()
        skill = runtime.handle_text_command("walk forward")
        assert skill == "walk"
        assert runtime._active_skill_name == "walk"

    def test_handle_text_command_stop(self):
        runtime = RLPolicyRuntime()
        runtime.handle_text_command("walk forward")
        skill = runtime.handle_text_command("stop")
        assert skill == "stand"

    def test_handle_text_command_wave(self):
        runtime = RLPolicyRuntime()
        skill = runtime.handle_text_command("wave")
        assert skill == "wave"

    def test_handle_text_command_turn(self):
        runtime = RLPolicyRuntime()
        skill = runtime.handle_text_command("turn left")
        assert skill == "turn"

    def test_handle_text_command_bow(self):
        runtime = RLPolicyRuntime()
        skill = runtime.handle_text_command("bow")
        assert skill == "bow"

    def test_multiple_infer_steps(self):
        runtime = RLPolicyRuntime()
        obs = _make_obs()
        z = PolicyVector(values=())
        for _ in range(10):
            output = runtime.infer(obs, z)
            assert output is not None

    def test_skill_registry_populated(self):
        runtime = RLPolicyRuntime()
        skills = runtime.registry.list_skills()
        assert set(skills) == {"stand", "walk", "turn", "wave", "bow"}
