"""Deploy a trained policy through the bridge server.

Bridges the gap between a trained checkpoint and live robot execution.
Loads a policy, connects to the bridge websocket, and runs the policy
loop using the existing policy lifecycle protocol.

This is the sim-to-real deployment path:
    trained checkpoint → this script → bridge server → backend (mock/ros/isaac)

Usage:
    # Against mock backend (testing)
    python3 -m training.runtime.deploy_to_bridge \
        --checkpoint checkpoints/walk_stable/best.pt \
        --bridge-uri ws://localhost:9100 \
        --task "walk forward" \
        --hz 10 --max-steps 200

    # Against real robot
    python3 -m training.runtime.deploy_to_bridge \
        --checkpoint checkpoints/walk_stable/best.pt \
        --bridge-uri ws://localhost:9100 \
        --task "walk forward" \
        --hz 5 --max-steps 50 --max-speed 1 --max-stride 0.02
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from websockets.asyncio.client import connect

from bridge.openpi_adapter import (
    AINEX_ACTION_DIM,
    AINEX_STATE_DIM,
    _WALK_X_RANGE,
    _WALK_Y_RANGE,
    _clamp,
    _denorm,
    _norm,
)
from bridge.protocol import utc_now_iso

logger = logging.getLogger(__name__)


def _command_envelope(command: str, payload: dict[str, Any]) -> str:
    return json.dumps({
        "type": "command",
        "request_id": str(uuid.uuid4()),
        "timestamp": utc_now_iso(),
        "command": command,
        "payload": payload,
    })


async def _wait_for_response(ws: Any) -> dict[str, Any]:
    while True:
        raw = await ws.recv()
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed.get("type") == "response":
            return parsed


async def deploy_policy(
    checkpoint_path: str,
    bridge_uri: str = "ws://127.0.0.1:9100",
    task: str = "walk forward",
    hz: float = 10.0,
    max_steps: int = 200,
    max_speed: int = 4,
    max_stride: float = 0.05,
    device: str = "cpu",
) -> dict[str, Any]:
    """Deploy a trained policy through the bridge.

    Args:
        checkpoint_path: Path to policy checkpoint
        bridge_uri: Bridge websocket URI
        task: Task description
        hz: Control rate
        max_steps: Maximum policy steps
        max_speed: Speed cap (1-4) for safety
        max_stride: Stride cap (meters) for safety
        device: PyTorch device

    Returns:
        Session summary dict
    """
    # Try Brax checkpoint first, fall back to PyTorch
    ckpt_path = Path(checkpoint_path)
    if (ckpt_path / "config.json").exists():
        from training.mujoco.inference import load_policy
        inference_fn, _ = load_policy(str(ckpt_path))

        class _BraxPolicy:
            def __init__(self, fn):
                self._fn = fn
                self.obs_dim = AINEX_STATE_DIM
            def get_action(self, obs):
                return self._fn(obs)
            def get_deterministic_action(self, obs_t):
                import torch as _torch
                obs_np = obs_t.squeeze(0).cpu().numpy()
                action_np = self._fn(obs_np)
                return _torch.from_numpy(action_np).unsqueeze(0)

        policy = _BraxPolicy(inference_fn)
        has_torch = True
    else:
        import torch
        has_torch = True
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        if hasattr(ckpt, "eval"):
            policy = ckpt
        elif isinstance(ckpt, dict) and "model" in ckpt and not isinstance(ckpt["model"], dict):
            policy = ckpt["model"]
            policy.eval()
        else:
            raise ValueError(f"Cannot load policy from {checkpoint_path}")

    logger.info(f"Loaded policy from {checkpoint_path}")

    # Detect policy input dimension
    policy_obs_dim = getattr(policy, "obs_dim", AINEX_STATE_DIM)
    logger.info(f"Policy obs_dim: {policy_obs_dim}")

    has_torch = False
    try:
        import torch
        has_torch = True
    except ImportError:
        pass

    interval = 1.0 / hz
    step = 0
    total_reward_proxy = 0.0  # rough proxy from telemetry

    async with connect(bridge_uri) as ws:
        # Receive hello
        hello = await ws.recv()
        logger.info("Connected to bridge")

        # Start policy
        await ws.send(_command_envelope("policy.start", {
            "task": task,
            "hz": hz,
            "max_steps": max_steps,
        }))

        resp = await _wait_for_response(ws)
        if not resp.get("ok"):
            logger.error(f"Failed to start policy: {resp.get('message')}")
            return {"ok": False, "reason": resp.get("message", "start_failed")}

        logger.info(f"Policy started: task={task}, hz={hz}")

        try:
            while step < max_steps:
                tick_start = time.monotonic()

                # Drain telemetry
                telemetry: dict[str, Any] = {}
                try:
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.05)
                        msg = json.loads(raw)
                        if msg.get("type") == "event":
                            if msg.get("event") == "telemetry.basic":
                                telemetry = msg.get("data", {})
                            elif msg.get("event") == "policy.status":
                                if msg.get("data", {}).get("state") == "idle":
                                    logger.info("Policy stopped by server")
                                    return {"ok": True, "steps": step, "reason": "server_stopped"}
                except asyncio.TimeoutError:
                    pass

                # Build observation from telemetry
                obs = _telemetry_to_obs(telemetry, obs_dim=policy_obs_dim)

                # Get action from policy
                if has_torch:
                    import torch
                    obs_t = torch.from_numpy(obs).unsqueeze(0).float()
                    with torch.no_grad():
                        action = policy.get_deterministic_action(obs_t)
                        action_np = action.squeeze(0).cpu().numpy()
                elif hasattr(policy, "get_action"):
                    action_np = policy.get_action(obs)
                else:
                    action_np = np.zeros(AINEX_ACTION_DIM, dtype=np.float32)

                action_np = np.clip(action_np, -1.0, 1.0)

                # Denormalize and apply safety caps
                walk_x = _clamp(
                    _denorm(float(action_np[0]), -_WALK_X_RANGE, _WALK_X_RANGE),
                    -max_stride, max_stride,
                )
                walk_y = _clamp(
                    _denorm(float(action_np[1]), -_WALK_Y_RANGE, _WALK_Y_RANGE),
                    -max_stride, max_stride,
                )
                walk_yaw = _denorm(float(action_np[2]), -10.0, 10.0)
                walk_height = _denorm(float(action_np[3]), 0.015, 0.06)
                walk_speed = min(max_speed, max(1, int(round(_denorm(float(action_np[4]), 1.0, 4.0)))))
                head_pan = _denorm(float(action_np[5]), -1.5, 1.5)
                head_tilt = _denorm(float(action_np[6]), -1.0, 1.0)

                # Send policy tick
                await ws.send(_command_envelope("policy.tick", {
                    "action": {
                        "walk_x": walk_x,
                        "walk_y": walk_y,
                        "walk_yaw": walk_yaw,
                        "walk_height": walk_height,
                        "walk_speed": walk_speed,
                        "head_pan": head_pan,
                        "head_tilt": head_tilt,
                    },
                }))

                tick_resp = await _wait_for_response(ws)
                if not tick_resp.get("ok"):
                    logger.warning(f"Tick failed: {tick_resp.get('message')}")
                    break

                step += 1

                # Maintain tick rate
                elapsed = time.monotonic() - tick_start
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            await ws.send(_command_envelope("policy.stop", {"reason": "deploy_complete"}))
            await _wait_for_response(ws)
            logger.info(f"Policy stopped after {step} steps")

    return {"ok": True, "steps": step, "reason": "complete"}


def _telemetry_to_obs(telemetry: dict[str, Any], obs_dim: int = AINEX_STATE_DIM) -> np.ndarray:
    """Convert bridge telemetry to normalized observation vector.

    Args:
        telemetry: Telemetry dict from bridge
        obs_dim: Expected observation dimension (11 for proprio-only, 16 for target-aware)
    """
    obs = np.zeros(obs_dim, dtype=np.float32)
    if not telemetry:
        return obs

    obs[0] = _norm(float(telemetry.get("walk_x", 0.0)), -0.05, 0.05)
    obs[1] = _norm(float(telemetry.get("walk_y", 0.0)), -0.05, 0.05)
    obs[2] = _norm(float(telemetry.get("walk_yaw", 0.0)), -10.0, 10.0)
    obs[3] = _norm(float(telemetry.get("walk_height", 0.036)), 0.015, 0.06)
    obs[4] = _norm(float(telemetry.get("walk_speed", 2)), 1.0, 4.0)
    obs[5] = _norm(float(telemetry.get("head_pan", 0.0)), -1.5, 1.5)
    obs[6] = _norm(float(telemetry.get("head_tilt", 0.0)), -1.0, 1.0)
    obs[7] = _norm(float(telemetry.get("imu_roll", 0.0)), -3.14159, 3.14159)
    obs[8] = _norm(float(telemetry.get("imu_pitch", 0.0)), -3.14159, 3.14159)
    obs[9] = 1.0 if telemetry.get("is_walking", False) else -1.0
    obs[10] = _norm(float(telemetry.get("battery_mv", 12000)), 10400.0, 12600.0)

    # Target observations (dims 11-15) from perception data if available
    if obs_dim > AINEX_STATE_DIM:
        target = telemetry.get("target", {})
        obs[11] = _norm(float(target.get("distance", 2.0)), 0.0, 5.0)
        obs[12] = _norm(float(target.get("angle", 0.0)), -3.14159, 3.14159)
        obs[13] = 1.0 if target.get("visible", False) else -1.0
        obs[14] = _norm(float(target.get("pixel_x", 0.5)), 0.0, 1.0)
        obs[15] = _norm(float(target.get("pixel_y", 0.5)), 0.0, 1.0)

    return obs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy trained policy to bridge")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--bridge-uri", type=str, default="ws://127.0.0.1:9100")
    parser.add_argument("--task", type=str, default="walk forward")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-speed", type=int, default=4)
    parser.add_argument("--max-stride", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = _parse_args()
    result = asyncio.run(deploy_policy(
        checkpoint_path=args.checkpoint,
        bridge_uri=args.bridge_uri,
        task=args.task,
        hz=args.hz,
        max_steps=args.max_steps,
        max_speed=args.max_speed,
        max_stride=args.max_stride,
        device=args.device,
    ))
    logger.info(f"Deployment result: {result}")


if __name__ == "__main__":
    main()
