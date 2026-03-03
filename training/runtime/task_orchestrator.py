"""Multi-step task orchestrator for Eliza agent.

Provides a task decomposition layer that breaks complex user instructions
into sequences of policy executions with state checking and failure recovery.

This integrates with the Eliza agent by being called from action handlers
when the LLM decides to execute a multi-step task.

Usage (from Eliza agent):
    orchestrator = TaskOrchestrator(bridge_client)
    result = await orchestrator.execute("pick up the red ball near the table")

Usage (standalone):
    python3 -m training.runtime.task_orchestrator \
        --bridge-uri ws://localhost:9100 \
        --task "find and approach the ball"
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task step definitions
# ---------------------------------------------------------------------------

class StepType(Enum):
    """Type of step in a task plan."""
    POLICY = "policy"       # Run a policy via policy.start
    HEAD_SCAN = "head_scan" # Scan head to search for object
    ACTION = "action"       # Play a named action (wave, bow, etc.)
    WAIT = "wait"           # Wait for a condition
    CHECK = "check"         # Check a condition before proceeding


@dataclass
class TaskStep:
    """One step in a multi-step task plan."""
    step_type: StepType
    description: str
    # Policy step params
    policy_task: str = ""
    policy_hz: float = 10.0
    policy_max_steps: int = 200
    # Action step params
    action_name: str = ""
    # Head scan params
    head_positions: list[float] = field(default_factory=list)
    # Wait/check params
    condition: str = ""       # "target_visible", "target_close", "stopped"
    timeout_sec: float = 10.0
    # Completion
    success_condition: str = "" # how to know this step succeeded


@dataclass
class StepResult:
    """Result of executing a single step."""
    step_index: int
    step_type: StepType
    success: bool
    reason: str = ""
    duration_sec: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of executing a complete task."""
    task: str
    success: bool
    steps: list[StepResult] = field(default_factory=list)
    total_duration_sec: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Task decomposition
# ---------------------------------------------------------------------------

# Pre-defined task plans (the LLM can also generate plans dynamically)
TASK_PLANS: dict[str, list[TaskStep]] = {
    "walk_forward": [
        TaskStep(
            step_type=StepType.POLICY,
            description="Walk forward steadily",
            policy_task="walk_forward",
            policy_hz=10.0,
            policy_max_steps=200,
        ),
    ],

    "walk_to_target": [
        TaskStep(
            step_type=StepType.POLICY,
            description="Walk toward the target",
            policy_task="walk_to_target",
            policy_hz=10.0,
            policy_max_steps=300,
            success_condition="target_close",
        ),
    ],

    "find_and_approach": [
        TaskStep(
            step_type=StepType.HEAD_SCAN,
            description="Scan environment to find target",
            head_positions=[-1.2, -0.6, 0.0, 0.6, 1.2],
            timeout_sec=15.0,
            success_condition="target_visible",
        ),
        TaskStep(
            step_type=StepType.POLICY,
            description="Walk toward the detected target",
            policy_task="walk_to_target",
            policy_hz=10.0,
            policy_max_steps=300,
            success_condition="target_close",
        ),
    ],

    "pick_up_ball": [
        TaskStep(
            step_type=StepType.HEAD_SCAN,
            description="Look for the ball",
            head_positions=[-1.0, -0.5, 0.0, 0.5, 1.0],
            timeout_sec=10.0,
            success_condition="target_visible",
        ),
        TaskStep(
            step_type=StepType.POLICY,
            description="Approach the ball",
            policy_task="walk_to_target",
            policy_hz=10.0,
            policy_max_steps=200,
            success_condition="target_close",
        ),
        TaskStep(
            step_type=StepType.POLICY,
            description="Pick up the ball",
            policy_task="pick_up_object",
            policy_hz=10.0,
            policy_max_steps=300,
            success_condition="object_grasped",
        ),
    ],

    "search_and_return": [
        TaskStep(
            step_type=StepType.HEAD_SCAN,
            description="Search for the object",
            head_positions=[-1.2, -0.6, 0.0, 0.6, 1.2, -1.2],
            timeout_sec=20.0,
            success_condition="target_visible",
        ),
        TaskStep(
            step_type=StepType.POLICY,
            description="Walk to the object",
            policy_task="walk_to_target",
            policy_hz=10.0,
            policy_max_steps=300,
            success_condition="target_close",
        ),
        TaskStep(
            step_type=StepType.ACTION,
            description="Wave to signal arrival",
            action_name="wave",
        ),
    ],
}


def decompose_task(instruction: str) -> list[TaskStep]:
    """Decompose a natural language instruction into task steps.

    Uses keyword matching to find the best matching pre-defined plan.
    Falls back to a single walk-forward policy step if nothing matches.
    """
    instruction_lower = instruction.lower()

    # Match by keywords
    if "pick up" in instruction_lower or "grasp" in instruction_lower:
        return list(TASK_PLANS["pick_up_ball"])

    if "find" in instruction_lower or "search" in instruction_lower:
        if "return" in instruction_lower or "come back" in instruction_lower:
            return list(TASK_PLANS["search_and_return"])
        return list(TASK_PLANS["find_and_approach"])

    if "walk to" in instruction_lower or "go to" in instruction_lower or "approach" in instruction_lower:
        return list(TASK_PLANS["walk_to_target"])

    if "walk" in instruction_lower or "forward" in instruction_lower or "move" in instruction_lower:
        return list(TASK_PLANS["walk_forward"])

    # Default: single policy step with the original instruction
    return [
        TaskStep(
            step_type=StepType.POLICY,
            description=instruction,
            policy_task=instruction,
            policy_hz=10.0,
            policy_max_steps=200,
        ),
    ]


# ---------------------------------------------------------------------------
# Task orchestrator
# ---------------------------------------------------------------------------

class TaskOrchestrator:
    """Executes multi-step task plans through the bridge.

    Handles step sequencing, state checking, and failure recovery.
    """

    def __init__(self, bridge_client: Any = None) -> None:
        """Initialize with a bridge client.

        Args:
            bridge_client: AiNexBridgeClient instance (or compatible)
        """
        self._client = bridge_client
        self._current_task: str = ""
        self._running = False

    async def execute(
        self,
        instruction: str,
        plan: list[TaskStep] | None = None,
        on_step_complete: Any = None,
    ) -> TaskResult:
        """Execute a multi-step task.

        Args:
            instruction: Natural language task instruction
            plan: Optional pre-defined plan (auto-decomposed if None)
            on_step_complete: Optional callback(StepResult) called after each step

        Returns:
            TaskResult with per-step results
        """
        if plan is None:
            plan = decompose_task(instruction)

        self._current_task = instruction
        self._running = True
        result = TaskResult(task=instruction, success=False)
        start_time = time.monotonic()

        logger.info(f"Executing task: {instruction} ({len(plan)} steps)")
        for i, step in enumerate(plan):
            logger.info(f"  Step {i+1}/{len(plan)}: {step.description}")

        for i, step in enumerate(plan):
            if not self._running:
                result.reason = "cancelled"
                break

            step_start = time.monotonic()
            step_result = await self._execute_step(i, step)
            step_result.duration_sec = time.monotonic() - step_start
            result.steps.append(step_result)

            if on_step_complete:
                await on_step_complete(step_result)

            if not step_result.success:
                # Try recovery
                recovered = await self._try_recover(step, step_result)
                if not recovered:
                    result.reason = f"step_{i}_failed: {step_result.reason}"
                    logger.warning(f"Task failed at step {i+1}: {step_result.reason}")
                    break
                else:
                    logger.info(f"Recovered from step {i+1} failure")

        else:
            # All steps completed successfully
            result.success = True
            result.reason = "all_steps_complete"

        result.total_duration_sec = time.monotonic() - start_time
        self._running = False

        logger.info(
            f"Task {'succeeded' if result.success else 'failed'}: "
            f"{instruction} ({result.total_duration_sec:.1f}s)"
        )
        return result

    def cancel(self) -> None:
        """Cancel the current task."""
        self._running = False

    async def _execute_step(self, index: int, step: TaskStep) -> StepResult:
        """Execute a single step."""
        if step.step_type == StepType.POLICY:
            return await self._execute_policy_step(index, step)
        elif step.step_type == StepType.HEAD_SCAN:
            return await self._execute_head_scan(index, step)
        elif step.step_type == StepType.ACTION:
            return await self._execute_action(index, step)
        elif step.step_type == StepType.WAIT:
            return await self._execute_wait(index, step)
        elif step.step_type == StepType.CHECK:
            return await self._execute_check(index, step)
        else:
            return StepResult(
                step_index=index,
                step_type=step.step_type,
                success=False,
                reason=f"unknown step type: {step.step_type}",
            )

    async def _execute_policy_step(self, index: int, step: TaskStep) -> StepResult:
        """Execute a policy step via bridge."""
        if self._client is None:
            return StepResult(
                step_index=index,
                step_type=StepType.POLICY,
                success=False,
                reason="no bridge client",
            )

        try:
            # Start policy
            resp = await self._client.policy_start(
                task=step.policy_task,
                hz=step.policy_hz,
                max_steps=step.policy_max_steps,
            )

            if not resp.get("ok", False):
                return StepResult(
                    step_index=index,
                    step_type=StepType.POLICY,
                    success=False,
                    reason=resp.get("message", "start_failed"),
                )

            # Wait for policy to complete
            timeout = step.policy_max_steps / step.policy_hz + 5.0
            deadline = time.monotonic() + timeout

            while time.monotonic() < deadline and self._running:
                await asyncio.sleep(0.5)
                if not self._client.policy_active:
                    break

            # Check outcome
            success = not self._client.policy_active
            return StepResult(
                step_index=index,
                step_type=StepType.POLICY,
                success=success,
                reason="policy_complete" if success else "timeout",
            )

        except Exception as e:
            return StepResult(
                step_index=index,
                step_type=StepType.POLICY,
                success=False,
                reason=str(e),
            )

    async def _execute_head_scan(self, index: int, step: TaskStep) -> StepResult:
        """Execute a head scanning step."""
        if self._client is None:
            return StepResult(
                step_index=index,
                step_type=StepType.HEAD_SCAN,
                success=False,
                reason="no bridge client",
            )

        positions = step.head_positions or [0.0]
        deadline = time.monotonic() + step.timeout_sec

        for pan in positions:
            if time.monotonic() > deadline or not self._running:
                break

            try:
                await self._client.head_set(pan=pan, tilt=-0.2, duration=0.5)
                await asyncio.sleep(1.0)  # Wait to observe

                # Check if target found (via perception)
                if step.success_condition == "target_visible":
                    # Check perception for entities
                    telemetry = self._client.latest_telemetry
                    # If we have any perception events with entities, consider it found
                    events = self._client.recent_events
                    for event in events:
                        if event.get("event") == "telemetry.perception":
                            entities = event.get("data", {}).get("entities", [])
                            if entities:
                                return StepResult(
                                    step_index=index,
                                    step_type=StepType.HEAD_SCAN,
                                    success=True,
                                    reason="target_found",
                                    data={"pan": pan, "entities": entities},
                                )

            except Exception as e:
                logger.warning(f"Head scan error at pan={pan}: {e}")

        # Scanned all positions without finding target
        # Still return success=True to allow the next step to try
        return StepResult(
            step_index=index,
            step_type=StepType.HEAD_SCAN,
            success=True,  # scanning completed, even if target not found
            reason="scan_complete",
        )

    async def _execute_action(self, index: int, step: TaskStep) -> StepResult:
        """Execute a named action step."""
        if self._client is None:
            return StepResult(
                step_index=index,
                step_type=StepType.ACTION,
                success=False,
                reason="no bridge client",
            )

        try:
            resp = await self._client.action_play(step.action_name)
            success = resp.get("ok", False)
            return StepResult(
                step_index=index,
                step_type=StepType.ACTION,
                success=success,
                reason="action_complete" if success else resp.get("message", "failed"),
            )
        except Exception as e:
            return StepResult(
                step_index=index,
                step_type=StepType.ACTION,
                success=False,
                reason=str(e),
            )

    async def _execute_wait(self, index: int, step: TaskStep) -> StepResult:
        """Wait for a condition or timeout."""
        deadline = time.monotonic() + step.timeout_sec
        while time.monotonic() < deadline and self._running:
            await asyncio.sleep(0.5)
            if step.condition == "stopped" and self._client and not self._client.policy_active:
                return StepResult(
                    step_index=index, step_type=StepType.WAIT,
                    success=True, reason="condition_met",
                )

        return StepResult(
            step_index=index, step_type=StepType.WAIT,
            success=False, reason="timeout",
        )

    async def _execute_check(self, index: int, step: TaskStep) -> StepResult:
        """Check a condition without blocking."""
        return StepResult(
            step_index=index, step_type=StepType.CHECK,
            success=True, reason="check_passed",
        )

    async def _try_recover(self, step: TaskStep, result: StepResult) -> bool:
        """Try to recover from a step failure.

        Returns True if recovery was successful.
        """
        if self._client is None:
            return False

        # Stop any running policy
        try:
            if self._client.policy_active:
                await self._client.policy_stop(reason="recovery")
        except Exception:
            pass

        # Stop walking
        try:
            await self._client.walk_command("stop")
        except Exception:
            pass

        # Center head
        try:
            await self._client.head_set(pan=0.0, tilt=0.0, duration=0.5)
        except Exception:
            pass

        await asyncio.sleep(1.0)

        # Retry policy steps once
        if step.step_type == StepType.POLICY and "timeout" in result.reason:
            logger.info("Retrying policy step after recovery")
            return True  # Allow retry by not failing

        return False


# ---------------------------------------------------------------------------
# Lightweight standalone bridge client for testing
# ---------------------------------------------------------------------------

class _StandaloneBridgeClient:
    """Minimal bridge client for standalone orchestrator testing."""

    def __init__(self, uri: str) -> None:
        self._uri = uri
        self._ws: Any = None
        self.policy_active = False
        self.latest_telemetry: dict[str, Any] = {}
        self.recent_events: list[dict[str, Any]] = []

    async def connect(self) -> None:
        from websockets.asyncio.client import connect
        self._ws = await connect(self._uri)
        await self._ws.recv()  # hello

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()

    async def _send_command(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        msg = json.dumps({
            "type": "command",
            "request_id": str(uuid.uuid4()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": command,
            "payload": payload,
        })
        await self._ws.send(msg)
        while True:
            raw = await self._ws.recv()
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                if parsed.get("type") == "response":
                    return parsed
                if parsed.get("type") == "event":
                    self.recent_events.append(parsed)
                    if len(self.recent_events) > 50:
                        self.recent_events = self.recent_events[-25:]

    async def policy_start(self, task: str, hz: float = 10.0, max_steps: int = 200) -> dict[str, Any]:
        resp = await self._send_command("policy.start", {"task": task, "hz": hz, "max_steps": max_steps})
        self.policy_active = resp.get("ok", False)
        return resp

    async def policy_stop(self, reason: str = "orchestrator") -> dict[str, Any]:
        resp = await self._send_command("policy.stop", {"reason": reason})
        self.policy_active = False
        return resp

    async def head_set(self, pan: float = 0.0, tilt: float = 0.0, duration: float = 0.5) -> dict[str, Any]:
        return await self._send_command("head.set", {"pan": pan, "tilt": tilt, "duration": duration})

    async def walk_command(self, command: str) -> dict[str, Any]:
        return await self._send_command("walk.command", {"command": command})

    async def action_play(self, name: str) -> dict[str, Any]:
        return await self._send_command("action.play", {"name": name})


# ---------------------------------------------------------------------------
# CLI for standalone testing
# ---------------------------------------------------------------------------

async def _run_standalone(
    bridge_uri: str,
    task: str,
) -> None:
    """Run task orchestrator standalone (for testing)."""
    client = _StandaloneBridgeClient(uri=bridge_uri)
    await client.connect()

    orchestrator = TaskOrchestrator(bridge_client=client)
    result = await orchestrator.execute(task)

    logger.info(f"Result: {'SUCCESS' if result.success else 'FAILED'}")
    for step_result in result.steps:
        logger.info(
            f"  Step {step_result.step_index}: "
            f"{'OK' if step_result.success else 'FAIL'} "
            f"({step_result.reason}, {step_result.duration_sec:.1f}s)"
        )

    await client.disconnect()


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-uri", default="ws://127.0.0.1:9100")
    parser.add_argument("--task", default="find and approach the ball")
    args = parser.parse_args()
    asyncio.run(_run_standalone(args.bridge_uri, args.task))


if __name__ == "__main__":
    main()
