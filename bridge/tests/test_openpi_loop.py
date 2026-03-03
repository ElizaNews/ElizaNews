"""Tests for the OpenPI autonomous policy loop."""

from __future__ import annotations

import asyncio
import unittest

import numpy as np

from bridge.openpi_loop import OpenPIPolicyLoop
from bridge.perception import PerceptionAggregator
from bridge.openpi_adapter import AINEX_STATE_DIM, AINEX_ENTITY_SLOT_DIM
from perception.entity_slots.slot_config import TOTAL_ENTITY_DIMS


class TestPolicyLoopInit(unittest.TestCase):
    def test_init_without_perception(self):
        loop = OpenPIPolicyLoop(enable_perception=False)
        assert loop.pipeline is None
        assert loop.aggregator is not None
        assert not loop.is_running

    def test_init_with_perception(self):
        loop = OpenPIPolicyLoop(enable_perception=True)
        assert loop.pipeline is not None
        assert loop.aggregator is not None

    def test_aggregator_accessible(self):
        loop = OpenPIPolicyLoop(enable_perception=False)
        agg = loop.aggregator
        assert isinstance(agg, PerceptionAggregator)


class TestPolicyLoopObservation(unittest.TestCase):
    def test_get_observation_default(self):
        loop = OpenPIPolicyLoop(enable_perception=False)
        obs = loop.get_observation(task="walk forward")
        assert "state" in obs
        assert "prompt" in obs
        assert len(obs["state"]) == AINEX_STATE_DIM
        assert obs["prompt"] == "walk forward"

    def test_observation_entity_slots_zeros_without_pipeline(self):
        """Without perception pipeline, entity slots should be zeros."""
        loop = OpenPIPolicyLoop(enable_perception=False)
        obs = loop.get_observation()
        state = obs["state"]
        # Proprio is first 11, entity slots are 11:163
        entity_part = state[11:]
        assert len(entity_part) == AINEX_ENTITY_SLOT_DIM
        assert all(v == 0.0 for v in entity_part)

    def test_observation_with_telemetry(self):
        loop = OpenPIPolicyLoop(enable_perception=False)
        loop.update_telemetry({
            "battery_mv": 11500,
            "imu_roll": 0.1,
            "is_walking": True,
        })
        obs = loop.get_observation()
        state = obs["state"]
        # is_walking (index 9) should be 1.0
        self.assertAlmostEqual(state[9], 1.0)

    def test_observation_with_entity_slots(self):
        """Entity slots fed via aggregator should appear in observation."""
        loop = OpenPIPolicyLoop(enable_perception=False)
        # Manually set entity slots (as if pipeline was running)
        fake_slots = tuple([0.1] * TOTAL_ENTITY_DIMS)
        loop.aggregator.update_entity_slots(fake_slots)
        obs = loop.get_observation()
        entity_part = obs["state"][11:]
        assert all(abs(v - 0.1) < 0.001 for v in entity_part)


class TestPolicyLoopActions(unittest.TestCase):
    def test_process_named_action(self):
        loop = OpenPIPolicyLoop(enable_perception=False)
        commands = loop.process_action({
            "walk_x": 0.02,
            "walk_y": 0.0,
            "walk_yaw": 0.0,
            "walk_speed": 2,
            "walk_height": 0.036,
        })
        assert len(commands) >= 1
        assert commands[0]["command"] == "walk.set"
        self.assertAlmostEqual(commands[0]["payload"]["x"], 0.02)

    def test_process_vector_action(self):
        loop = OpenPIPolicyLoop(enable_perception=False)
        commands = loop.process_action({
            "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        })
        assert len(commands) >= 1
        assert commands[0]["command"] == "walk.set"


class TestPolicyLoopRun(unittest.TestCase):
    def test_run_loop_with_mock_policy(self):
        """Policy loop should run for specified steps with mock callbacks."""
        loop = OpenPIPolicyLoop(enable_perception=False, hz=100.0)
        commands_received = []

        async def mock_policy(obs):
            return {"walk_x": 0.01, "walk_y": 0.0, "walk_yaw": 0.0,
                    "walk_speed": 2, "walk_height": 0.036}

        async def mock_send(cmd):
            commands_received.append(cmd)

        async def _run():
            result = await loop.run_loop(
                task="test",
                max_steps=5,
                send_command_fn=mock_send,
                query_policy_fn=mock_policy,
            )
            return result

        result = asyncio.run(_run())
        assert result["status"] == "max_steps"
        assert result["steps"] == 5
        assert len(commands_received) >= 5  # At least 1 command per step

    def test_stop_loop_early(self):
        """Calling stop() should terminate the loop."""
        loop = OpenPIPolicyLoop(enable_perception=False, hz=100.0)

        async def mock_policy(obs):
            return {"walk_x": 0.0, "walk_y": 0.0, "walk_yaw": 0.0,
                    "walk_speed": 2, "walk_height": 0.036}

        async def _run():
            async def _stop_after_2():
                while loop.step_count < 2:
                    await asyncio.sleep(0.001)
                loop.stop()

            task = asyncio.create_task(
                loop.run_loop(task="test", max_steps=1000,
                              query_policy_fn=mock_policy)
            )
            stopper = asyncio.create_task(_stop_after_2())
            result = await task
            return result

        result = asyncio.run(_run())
        assert result["status"] == "stopped"
        assert result["steps"] >= 2

    def test_policy_error_continues(self):
        """Policy query errors should not crash the loop."""
        loop = OpenPIPolicyLoop(enable_perception=False, hz=100.0)
        call_count = 0

        async def flaky_policy(obs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("server down")
            return {"walk_x": 0.0, "walk_y": 0.0, "walk_yaw": 0.0,
                    "walk_speed": 2, "walk_height": 0.036}

        async def _run():
            return await loop.run_loop(
                task="test", max_steps=5,
                query_policy_fn=flaky_policy,
            )

        result = asyncio.run(_run())
        # Should finish despite errors on first 2 queries
        assert result["steps"] == 5


class TestPolicyLoopPerceptionWiring(unittest.TestCase):
    def test_connect_aggregator_wired(self):
        """When perception is enabled, pipeline should be connected to aggregator."""
        loop = OpenPIPolicyLoop(enable_perception=True)
        pipeline = loop.pipeline
        assert pipeline is not None
        # The connect_aggregator adds a callback
        assert len(pipeline._callbacks) >= 1


if __name__ == "__main__":
    unittest.main()
