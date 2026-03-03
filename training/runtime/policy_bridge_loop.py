"""Run a policy runtime loop using the policy lifecycle protocol.

Uses policy.start/policy.tick/policy.stop commands with full safety gating,
instead of sending raw walk.set commands directly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from dataclasses import dataclass

from websockets.asyncio.client import connect

from bridge.protocol import utc_now_iso
from training.interfaces import PolicyOutput, PolicyRuntime, PolicyVector, RobotObservation

logger = logging.getLogger(__name__)


@dataclass
class ConstantForwardPolicy(PolicyRuntime):
    """Simple baseline policy used until learned model export is wired."""

    def infer(self, obs: RobotObservation, z: PolicyVector) -> PolicyOutput:
        _ = obs
        _ = z
        return PolicyOutput(
            walk_x=0.01,
            walk_y=0.0,
            walk_yaw=0.0,
            walk_height=0.036,
            walk_speed=2,
            action_name="",
        )


def _command_envelope(command: str, payload: dict[str, object]) -> str:
    return json.dumps({
        "type": "command",
        "request_id": str(uuid.uuid4()),
        "timestamp": utc_now_iso(),
        "command": command,
        "payload": payload,
    })


async def _wait_for_response(ws: object) -> dict[str, object]:
    """Wait for a response envelope, skipping events."""
    while True:
        raw = await ws.recv()
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed.get("type") == "response":
            return parsed


async def _collect_telemetry(ws: object) -> dict[str, object]:
    """Collect the latest telemetry event, skipping other messages."""
    while True:
        raw = await ws.recv()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            continue
        if parsed.get("type") == "event" and parsed.get("event") == "telemetry.basic":
            return parsed.get("data", {})
        # Also accept response messages (just skip them)
        if parsed.get("type") == "response":
            continue
        # Accept policy events
        if parsed.get("type") == "event":
            event_name = parsed.get("event", "")
            if event_name == "safety.policy_guard":
                logger.warning(f"Safety guard: {parsed.get('data', {}).get('reason', 'unknown')}")
                return {}  # Bail and let caller handle
            if event_name == "policy.status":
                state = parsed.get("data", {}).get("state", "")
                if state == "idle":
                    logger.info(f"Policy stopped: {parsed.get('data', {}).get('reason', 'unknown')}")
                    return {}
            continue


def _telemetry_to_observation(data: dict[str, object]) -> RobotObservation:
    return RobotObservation(
        timestamp=asyncio.get_running_loop().time(),
        battery_mv=int(data.get("battery_mv", 0)),
        imu_roll=float(data.get("imu_roll", 0.0)),
        imu_pitch=float(data.get("imu_pitch", 0.0)),
        is_walking=bool(data.get("is_walking", False)),
    )


async def run_policy_loop(
    uri: str,
    policy: PolicyRuntime,
    task: str = "baseline_forward",
    hz: float = 5.0,
    max_steps: int = 50,
) -> None:
    """Run a policy loop using the policy lifecycle protocol.

    Args:
        uri: Bridge websocket URI (e.g., ws://127.0.0.1:9100)
        policy: PolicyRuntime instance to use for inference
        task: Task description for policy.start
        hz: Target tick rate in Hz
        max_steps: Maximum number of policy steps
    """
    interval = 1.0 / hz
    step = 0

    async with connect(uri) as ws:
        # Receive session.hello
        hello = await ws.recv()
        logger.info(f"Connected: {json.loads(hello).get('event', 'unknown')}")

        # Start policy via lifecycle command
        await ws.send(_command_envelope("policy.start", {
            "task": task,
            "hz": hz,
            "max_steps": max_steps,
        }))

        # Wait for start response
        resp = await _wait_for_response(ws)
        if not resp.get("ok"):
            logger.error(f"Failed to start policy: {resp.get('message', 'unknown')}")
            return

        logger.info(f"Policy started: task={task}, hz={hz}, max_steps={max_steps}")

        try:
            while step < max_steps:
                # Collect telemetry
                data = await asyncio.wait_for(_collect_telemetry(ws), timeout=3.0)
                if not data:
                    logger.warning("Empty telemetry, checking policy status")
                    await ws.send(_command_envelope("policy.status", {}))
                    status_resp = await _wait_for_response(ws)
                    if not status_resp.get("data", {}).get("active", False):
                        logger.info("Policy no longer active, exiting loop")
                        break
                    continue

                # Build observation and run inference
                obs = _telemetry_to_observation(data)
                z = PolicyVector(values=(0.0,))
                output = policy.infer(obs, z)

                # Send policy tick with action
                action_payload = {
                    "walk_x": output.walk_x,
                    "walk_y": output.walk_y,
                    "walk_yaw": output.walk_yaw,
                    "walk_height": output.walk_height,
                    "walk_speed": output.walk_speed,
                }
                await ws.send(_command_envelope("policy.tick", {
                    "action": action_payload,
                }))

                # Wait for tick response
                tick_resp = await _wait_for_response(ws)
                if not tick_resp.get("ok"):
                    logger.warning(f"Tick failed at step {step}: {tick_resp.get('message', 'unknown')}")
                    break

                step += 1
                await asyncio.sleep(interval)

        except asyncio.TimeoutError:
            logger.warning("Telemetry timeout, stopping policy")
        except KeyboardInterrupt:
            logger.info("Interrupted, stopping policy")
        finally:
            # Stop policy cleanly
            await ws.send(_command_envelope("policy.stop", {
                "reason": "loop_complete" if step >= max_steps else "loop_exit",
            }))
            stop_resp = await _wait_for_response(ws)
            logger.info(
                f"Policy stopped after {step} steps: "
                f"{'success' if stop_resp.get('ok') else stop_resp.get('message', 'error')}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy loop using lifecycle protocol")
    parser.add_argument("--uri", type=str, default="ws://127.0.0.1:9100")
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--task", type=str, default="baseline_forward")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    policy = ConstantForwardPolicy()
    asyncio.run(run_policy_loop(
        uri=args.uri,
        policy=policy,
        task=args.task,
        hz=args.hz,
        max_steps=args.steps,
    ))


if __name__ == "__main__":
    main()
