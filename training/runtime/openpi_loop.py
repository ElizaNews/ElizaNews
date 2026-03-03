"""OpenPI policy loop runner for AiNex.

Connects to both the bridge websocket server and an OpenPI inference server,
runs the full observation -> inference -> tick cycle with the custom AiNex
adapter for observation/action translation.

Usage:
    python -m training.runtime.openpi_loop \
        --bridge-uri ws://localhost:9100 \
        --openpi-url http://localhost:8000 \
        --task "walk to the red cup" \
        --hz 5 --max-steps 500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from typing import Any

from websockets.asyncio.client import connect

from bridge.openpi_adapter import (
    build_observation,
    decode_action,
    default_perception,
    observation_to_dict,
)
from bridge.perception import PerceptionAggregator
from bridge.protocol import utc_now_iso
from training.interfaces import AinexPerceptionObservation, PolicyState, PolicyTransitionRecord

logger = logging.getLogger(__name__)


def _command_envelope(command: str, payload: dict[str, Any]) -> str:
    return json.dumps({
        "type": "command",
        "request_id": str(uuid.uuid4()),
        "timestamp": utc_now_iso(),
        "command": command,
        "payload": payload,
    })


class OpenPIInferenceClient:
    """Client for the OpenPI inference server.

    Supports two modes:
    - HTTP POST to a REST endpoint (default, works with openpi serve)
    - WebSocket streaming (for lower latency)

    Falls back to a dummy pass-through if no server is available.
    """

    def __init__(self, url: str = "", timeout: float = 2.0) -> None:
        self._url = url
        self._timeout = timeout
        self._session: Any = None

    async def connect(self) -> None:
        if not self._url:
            logger.warning("No OpenPI server URL configured, using passthrough mode")
            return
        try:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            )
            logger.info(f"OpenPI client ready: {self._url}")
        except ImportError:
            logger.warning("aiohttp not installed, using urllib fallback")

    async def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Send observation to OpenPI and return action dict."""
        if not self._url:
            # Passthrough: return zero action (safe idle)
            return {"action": [0.0] * 7, "confidence": 0.5}

        if self._session is not None:
            try:
                async with self._session.post(
                    f"{self._url}/infer",
                    json=observation,
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        logger.warning(f"OpenPI inference returned {resp.status}")
                        return {"action": [0.0] * 7, "confidence": 0.0}
            except Exception as e:
                logger.warning(f"OpenPI inference failed: {e}")
                return {"action": [0.0] * 7, "confidence": 0.0}

        # Fallback: synchronous HTTP (no aiohttp)
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self._url}/infer",
                data=json.dumps(observation).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:
            logger.warning(f"OpenPI inference fallback failed: {e}")
            return {"action": [0.0] * 7, "confidence": 0.0}

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


async def _wait_for_response(ws: Any) -> dict[str, Any]:
    """Wait for a response, consuming events along the way."""
    while True:
        raw = await ws.recv()
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed.get("type") == "response":
            return parsed


async def run_openpi_loop(
    bridge_uri: str = "ws://127.0.0.1:9100",
    openpi_url: str = "",
    task: str = "idle",
    hz: float = 5.0,
    max_steps: int = 500,
    confidence_threshold: float = 0.1,
    trace_path: str = "",
) -> list[PolicyTransitionRecord]:
    """Run the full OpenPI policy loop.

    Args:
        bridge_uri: Bridge websocket URI
        openpi_url: OpenPI inference server URL (empty = passthrough mode)
        task: Task description / language instruction
        hz: Target tick rate
        max_steps: Maximum steps before auto-stop
        confidence_threshold: Stop policy if confidence drops below this
        trace_path: Optional JSONL path for per-tick trace logging

    Returns:
        List of policy transition records for the session.
    """
    interval = 1.0 / hz
    transitions: list[PolicyTransitionRecord] = []
    perception = PerceptionAggregator()
    openpi = OpenPIInferenceClient(url=openpi_url, timeout=2.0)
    await openpi.connect()

    trace_file = None
    if trace_path:
        trace_file = open(trace_path, "a")

    step = 0
    policy_state = PolicyState.IDLE

    def _record_transition(from_state: PolicyState, to_state: PolicyState, reason: str) -> None:
        nonlocal policy_state
        record = PolicyTransitionRecord(
            timestamp=time.monotonic(),
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            task=task,
            step=step,
        )
        transitions.append(record)
        policy_state = to_state
        logger.info(f"Policy transition: {from_state.value} -> {to_state.value} ({reason})")

    try:
        async with connect(bridge_uri) as ws:
            # Receive hello
            hello_raw = await ws.recv()
            logger.info("Connected to bridge")

            # Start policy
            _record_transition(PolicyState.IDLE, PolicyState.STARTING, "loop_start")
            await ws.send(_command_envelope("policy.start", {
                "task": task,
                "hz": hz,
                "max_steps": max_steps,
            }))

            resp = await _wait_for_response(ws)
            if not resp.get("ok"):
                _record_transition(PolicyState.STARTING, PolicyState.FAILED, resp.get("message", "start_failed"))
                return transitions

            _record_transition(PolicyState.STARTING, PolicyState.RUNNING, "started")

            while step < max_steps:
                tick_start = time.monotonic()

                # Collect telemetry (non-blocking, use latest available)
                telemetry_data: dict[str, Any] = {}
                try:
                    # Drain all pending messages to get latest telemetry
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.05)
                        msg = json.loads(raw)
                        if msg.get("type") == "event" and msg.get("event") == "telemetry.basic":
                            telemetry_data = msg.get("data", {})
                        elif msg.get("type") == "event":
                            event_name = msg.get("event", "")
                            if event_name == "safety.policy_guard":
                                reason = msg.get("data", {}).get("reason", "safety_guard")
                                _record_transition(PolicyState.RUNNING, PolicyState.FAILED, reason)
                                return transitions
                            if event_name == "policy.status":
                                if msg.get("data", {}).get("state") == "idle":
                                    _record_transition(PolicyState.RUNNING, PolicyState.IDLE, msg.get("data", {}).get("reason", "server_stopped"))
                                    return transitions
                except asyncio.TimeoutError:
                    pass  # No more messages, proceed with what we have

                # Update perception from telemetry
                if telemetry_data:
                    perception.update_telemetry(telemetry_data)

                # Build observation
                perception_snap = perception.snapshot(
                    language_instruction=task,
                )
                obs = build_observation(perception_snap)
                obs_dict = observation_to_dict(obs)

                # Inference
                inference_start = time.monotonic()
                action_raw = await openpi.infer(obs_dict)
                inference_ms = (time.monotonic() - inference_start) * 1000

                # Decode action
                action = decode_action(action_raw)

                # Check confidence threshold
                if action.confidence < confidence_threshold:
                    logger.warning(f"Low confidence {action.confidence:.3f} < {confidence_threshold}, stopping")
                    _record_transition(PolicyState.RUNNING, PolicyState.STOPPING, "low_confidence")
                    break

                # Send policy tick
                await ws.send(_command_envelope("policy.tick", {
                    "action": {
                        "walk_x": action.walk_x,
                        "walk_y": action.walk_y,
                        "walk_yaw": action.walk_yaw,
                        "walk_height": action.walk_height,
                        "walk_speed": action.walk_speed,
                        "head_pan": action.head_pan,
                        "head_tilt": action.head_tilt,
                    },
                }))

                # Wait for tick response
                tick_resp = await _wait_for_response(ws)
                tick_ms = (time.monotonic() - tick_start) * 1000

                if not tick_resp.get("ok"):
                    msg = tick_resp.get("message", "tick_failed")
                    logger.warning(f"Tick failed at step {step}: {msg}")
                    _record_transition(PolicyState.RUNNING, PolicyState.FAILED, msg)
                    break

                # Trace logging
                if trace_file is not None:
                    trace_record = {
                        "step": step,
                        "timestamp": utc_now_iso(),
                        "observation_summary": {
                            "state_dim": len(obs.state),
                            "prompt": obs.prompt[:100],
                            "has_image": bool(obs.image),
                        },
                        "action_summary": {
                            "walk_x": action.walk_x,
                            "walk_y": action.walk_y,
                            "walk_yaw": action.walk_yaw,
                            "confidence": action.confidence,
                        },
                        "inference_ms": round(inference_ms, 2),
                        "tick_ms": round(tick_ms, 2),
                        "clamped": tick_resp.get("data", {}).get("clamped", {}),
                    }
                    trace_file.write(json.dumps(trace_record) + "\n")

                step += 1
                # Sleep to maintain target Hz
                elapsed = time.monotonic() - tick_start
                sleep_time = max(0.0, interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            # Stop policy cleanly
            if policy_state == PolicyState.RUNNING:
                _record_transition(PolicyState.RUNNING, PolicyState.STOPPING, "loop_complete")

            await ws.send(_command_envelope("policy.stop", {
                "reason": "loop_complete" if step >= max_steps else "loop_exit",
            }))
            stop_resp = await _wait_for_response(ws)
            _record_transition(
                PolicyState.STOPPING, PolicyState.IDLE,
                f"stopped_after_{step}_steps"
            )

    except Exception as e:
        logger.error(f"OpenPI loop error: {e}")
        _record_transition(policy_state, PolicyState.FAILED, str(e))
    finally:
        await openpi.close()
        if trace_file is not None:
            trace_file.close()

    return transitions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenPI policy loop for AiNex")
    parser.add_argument("--bridge-uri", type=str, default="ws://127.0.0.1:9100",
                        help="Bridge websocket URI")
    parser.add_argument("--openpi-url", type=str, default="",
                        help="OpenPI inference server URL (empty = passthrough)")
    parser.add_argument("--task", type=str, default="walk forward",
                        help="Task description / language instruction")
    parser.add_argument("--hz", type=float, default=5.0,
                        help="Target tick rate")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum policy steps")
    parser.add_argument("--confidence-threshold", type=float, default=0.1,
                        help="Minimum confidence to continue policy")
    parser.add_argument("--trace-path", type=str, default="",
                        help="JSONL path for per-tick trace logging")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parse_args()
    transitions = asyncio.run(run_openpi_loop(
        bridge_uri=args.bridge_uri,
        openpi_url=args.openpi_url,
        task=args.task,
        hz=args.hz,
        max_steps=args.max_steps,
        confidence_threshold=args.confidence_threshold,
        trace_path=args.trace_path,
    ))
    logger.info(f"Session complete: {len(transitions)} transitions")
    for t in transitions:
        logger.info(f"  {t.from_state.value} -> {t.to_state.value}: {t.reason}")


if __name__ == "__main__":
    main()
