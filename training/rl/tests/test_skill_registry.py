"""Tests for skill library and registry."""

import numpy as np
import pytest

from training.rl.skills.base_skill import SkillParams, SkillStatus
from training.rl.skills.registry import SkillRegistry
from training.rl.skills.stand_skill import StandSkill
from training.rl.skills.wave_skill import WaveSkill
from training.rl.skills.bow_skill import BowSkill
from training.rl.skills.walk_skill import WalkSkill
from training.rl.skills.turn_skill import TurnSkill


class TestStandSkill:
    def test_returns_standing_pose(self):
        skill = StandSkill()
        skill.reset()
        action, status = skill.get_action(np.zeros(48))
        assert action.shape == (24,)
        assert status == SkillStatus.RUNNING

    def test_action_dim(self):
        assert StandSkill.action_dim == 24
        assert StandSkill.requires_rl is False


class TestWalkSkill:
    def test_no_checkpoint_returns_zeros(self):
        skill = WalkSkill()
        skill.reset()
        action, status = skill.get_action(np.zeros(48))
        assert action.shape == (12,)
        assert np.allclose(action, 0.0)
        assert status == SkillStatus.RUNNING

    def test_duration_limit(self):
        skill = WalkSkill()
        skill.reset(SkillParams(duration_sec=0.05))  # completes after step 3 (0.06s > 0.05s)
        obs = np.zeros(48)
        _, status1 = skill.get_action(obs)  # step 1: 0.02s
        assert status1 == SkillStatus.RUNNING
        _, status2 = skill.get_action(obs)  # step 2: 0.04s
        assert status2 == SkillStatus.RUNNING
        _, status3 = skill.get_action(obs)  # step 3: 0.06s >= 0.05s
        assert status3 == SkillStatus.COMPLETED


class TestTurnSkill:
    def test_no_checkpoint_returns_zeros(self):
        skill = TurnSkill()
        skill.reset(SkillParams(direction=-1.0))
        action, status = skill.get_action(np.zeros(48))
        assert action.shape == (12,)
        assert status == SkillStatus.RUNNING


class TestWaveSkill:
    def test_action_shape(self):
        skill = WaveSkill()
        skill.reset()
        action, status = skill.get_action(np.zeros(48))
        assert action.shape == (24,)
        assert status == SkillStatus.RUNNING


class TestBowSkill:
    def test_action_shape(self):
        skill = BowSkill()
        skill.reset()
        action, status = skill.get_action(np.zeros(48))
        assert action.shape == (24,)
        assert status == SkillStatus.RUNNING


class TestSkillRegistry:
    def test_register_and_get(self):
        reg = SkillRegistry()
        reg.register(StandSkill())
        assert reg.get("stand") is not None
        assert reg.get("nonexistent") is None

    def test_list_skills(self):
        reg = SkillRegistry()
        reg.register(StandSkill())
        reg.register(WalkSkill())
        names = reg.list_skills()
        assert "stand" in names
        assert "walk" in names

    def test_alias_lookup(self):
        reg = SkillRegistry()
        reg.register(StandSkill())
        reg.register(WalkSkill())
        # "stop" is a default alias for "stand".
        assert reg.get("stop") is not None
        assert reg.get("stop").name == "stand"
        # "go forward" is a default alias for "walk".
        assert reg.get("go forward") is not None
        assert reg.get("go forward").name == "walk"

    def test_custom_alias(self):
        reg = SkillRegistry()
        reg.register(WaveSkill())
        reg.add_alias("hi there", "wave")
        assert reg.get("hi there") is not None
        assert reg.get("hi there").name == "wave"

    def test_contains(self):
        reg = SkillRegistry()
        reg.register(StandSkill())
        assert "stand" in reg
        assert "stop" in reg  # alias
        assert "fly" not in reg

    def test_len(self):
        reg = SkillRegistry()
        assert len(reg) == 0
        reg.register(StandSkill())
        assert len(reg) == 1
        reg.register(WalkSkill())
        assert len(reg) == 2

    def test_full_registry(self):
        reg = SkillRegistry()
        reg.register(StandSkill())
        reg.register(WalkSkill())
        reg.register(TurnSkill())
        reg.register(WaveSkill())
        reg.register(BowSkill())
        assert len(reg) == 5
        assert set(reg.list_skills()) == {"stand", "walk", "turn", "wave", "bow"}
