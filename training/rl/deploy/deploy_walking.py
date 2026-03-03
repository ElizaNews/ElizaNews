"""Deploy trained walking policy to the real AiNex robot via WebSocket bridge.

Runs the BraxWalkSkill at configurable frequency, converts joint targets
(radians) to servo pulse commands, and sends them through the bridge's
servo.set interface. Includes safety features for first-time deployment:
gradual ramp-up, joint velocity limiting, fall detection, and duration limits.

Architecture:
    deploy_walking.py (50Hz loop)
        → BraxWalkSkill.get_action_from_telemetry()
        → 12-dim joint targets (radians)
        → joint_name → servo_id + radians → pulse conversion
        → WebSocket → bridge server → ROS backend → bus servos

Usage:
    # Dry run (log commands, don't send):
    python3 -m training.rl.deploy.deploy_walking --dry-run

    # Test with mock backend (bridge must be running with --backend mock):
    python3 -m training.rl.deploy.deploy_walking --bridge ws://localhost:9100

    # Real robot (conservative: 20Hz, 30s, slow ramp):
    python3 -m training.rl.deploy.deploy_walking --hz 20 --duration 30 --ramp-seconds 5

    # Full speed (after validation):
    python3 -m training.rl.deploy.deploy_walking --hz 50 --duration 60

    # Stand only (test servo communication):
    python3 -m training.rl.deploy.deploy_walking --stand-only --duration 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

from training.rl.skills.brax_walk_skill import BraxWalkSkill, NUM_LEG_JOINTS
from training.mujoco.ainex_constants import LEG_JOINT_NAMES
from bridge.isaaclab.joint_map import (
    joint_name_to_servo_id,
    radians_to_pulse,
    pulse_to_radians,
)


# Safety limits
MAX_JOINT_VELOCITY = 2.0  # rad/s — max change per second
MAX_JOINT_DELTA = 0.1     # rad — max single-step change (at 50Hz = 5 rad/s)
FALL_PITCH_THRESHOLD = 0.8  # rad — ~46 deg, robot is falling
FALL_ROLL_THRESHOLD = 0.8   # rad


class DeployWalking:
    """Deploys the trained walking policy to the real robot."""

    def __init__(
        self,
        checkpoint: str = "checkpoints/mujoco_locomotion_v12_dr",
        hz: float = 50.0,
        duration: float = 30.0,
        ramp_seconds: float = 3.0,
        dry_run: bool = False,
        stand_only: bool = False,
        vx: float = 0.3,
        vy: float = 0.0,
        vyaw: float = 0.0,
        max_joint_delta: float = MAX_JOINT_DELTA,
    ):
        self.hz = hz
        self.dt = 1.0 / hz
        self.duration = duration
        self.ramp_seconds = ramp_seconds
        self.dry_run = dry_run
        self.stand_only = stand_only
        self.max_joint_delta = max_joint_delta

        # Load policy
        print(f"Loading walking policy from {checkpoint}...")
        self.skill = BraxWalkSkill(checkpoint_path=checkpoint)
        self.skill.set_command(vx, vy, vyaw)
        print(f"Policy loaded. Command: vx={vx}, vy={vy}, vyaw={vyaw}")

        # State
        self._last_targets = self.skill.default_pose.copy()
        self._imu_roll = 0.0
        self._imu_pitch = 0.0
        self._joint_feedback: np.ndarray | None = None
        self._step = 0
        self._stopped = False
        self._telemetry_received = False

    def joint_targets_to_servo_commands(
        self, targets: np.ndarray
    ) -> list[dict[str, int]]:
        """Convert 12-dim joint targets (radians) to servo commands.

        Returns list of {id: servo_id, position: pulse} dicts.
        """
        commands = []
        for i, name in enumerate(LEG_JOINT_NAMES):
            rad = float(targets[i])
            servo_id = joint_name_to_servo_id(name)
            pulse = radians_to_pulse(rad, servo_id)
            commands.append({"id": servo_id, "position": pulse})
        return commands

    def safety_clamp(
        self, targets: np.ndarray, ramp_factor: float
    ) -> np.ndarray:
        """Apply safety limits to joint targets.

        - Limits per-step change (joint velocity limiting)
        - Applies ramp factor (0→1 over ramp_seconds)
        - Blends with default pose during ramp
        """
        default = self.skill.default_pose

        # Blend with default pose during ramp-up
        blended = default + (targets - default) * ramp_factor

        # Limit per-step delta
        delta = blended - self._last_targets
        max_delta = self.max_joint_delta
        clamped_delta = np.clip(delta, -max_delta, max_delta)
        result = self._last_targets + clamped_delta

        self._last_targets = result.copy()
        return result

    def check_fall(self) -> bool:
        """Check if robot is falling based on IMU."""
        if abs(self._imu_pitch) > FALL_PITCH_THRESHOLD:
            print(f"FALL DETECTED: pitch={self._imu_pitch:.3f} rad")
            return True
        if abs(self._imu_roll) > FALL_ROLL_THRESHOLD:
            print(f"FALL DETECTED: roll={self._imu_roll:.3f} rad")
            return True
        return False

    def policy_step(self, ramp_factor: float) -> list[dict[str, int]]:
        """Run one policy step, return servo commands."""
        self._step += 1

        if self.stand_only:
            targets = self.skill.default_pose.copy()
        else:
            targets, _ = self.skill.get_action_from_telemetry(
                imu_roll=self._imu_roll,
                imu_pitch=self._imu_pitch,
                joint_positions=self._joint_feedback,
            )

        # Apply safety
        safe_targets = self.safety_clamp(targets, ramp_factor)
        return self.joint_targets_to_servo_commands(safe_targets)

    async def run_with_bridge(self, bridge_url: str) -> None:
        """Run the deploy loop via WebSocket bridge."""
        import websockets

        print(f"Connecting to bridge at {bridge_url}...")
        async with websockets.connect(bridge_url) as ws:
            print("Connected to bridge.")

            # Start policy mode
            # Bridge validates hz in 1-30 range; we run our own timing loop
            # so we just need to activate policy mode at a nominal rate.
            bridge_hz = min(self.hz, 30.0)
            await self._send_command(ws, "policy.start", {
                "task": "deploy_walking",
                "hz": bridge_hz,
            })
            resp = await self._recv_response(ws)
            if not resp.get("ok"):
                print(f"Failed to start policy: {resp.get('message')}")
                return

            print(f"Policy mode started on bridge.")
            print(f"Running at {self.hz}Hz for {self.duration}s "
                  f"(ramp: {self.ramp_seconds}s)")
            if self.dry_run:
                print("DRY RUN — commands will be logged but not sent to servos")
            print()

            try:
                await self._control_loop(ws)
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
            finally:
                # Stop policy mode and return to standing
                print("Stopping policy mode...")
                await self._send_standing_pose(ws)
                await asyncio.sleep(0.5)
                await self._send_command(ws, "policy.stop", {})
                try:
                    await self._recv_response(ws)
                except Exception:
                    pass
                print("Policy stopped.")

    async def _control_loop(self, ws) -> None:
        """Main control loop at target Hz."""
        start_time = time.monotonic()
        last_status_time = start_time

        while not self._stopped:
            t0 = time.monotonic()
            elapsed = t0 - start_time

            # Check duration
            if elapsed >= self.duration:
                print(f"\nDuration limit reached ({self.duration}s).")
                break

            # Ramp factor: 0 → 1 over ramp_seconds
            ramp = min(1.0, elapsed / self.ramp_seconds) if self.ramp_seconds > 0 else 1.0

            # Check for fall
            if self.check_fall():
                print("Emergency stop: fall detected!")
                break

            # Run policy step
            servo_cmds = self.policy_step(ramp)

            # Send to bridge
            if not self.dry_run:
                action_payload = {
                    "joint_positions": {
                        LEG_JOINT_NAMES[i]: float(self._last_targets[i])
                        for i in range(NUM_LEG_JOINTS)
                    },
                    "duration": int(self.dt * 1000),  # ms
                }
                await self._send_command(ws, "policy.tick", action_payload)
                resp = await self._recv_response(ws)

                # Parse telemetry from response events
                await self._process_events(ws)

            # Status print every 2 seconds
            if t0 - last_status_time >= 2.0:
                self._print_status(elapsed, ramp, servo_cmds)
                last_status_time = t0

            # Maintain target frequency
            step_time = time.monotonic() - t0
            sleep_time = self.dt - step_time
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def _send_standing_pose(self, ws) -> None:
        """Send standing pose to the robot."""
        default = self.skill.default_pose
        action_payload = {
            "joint_positions": {
                LEG_JOINT_NAMES[i]: float(default[i])
                for i in range(NUM_LEG_JOINTS)
            },
            "duration": 500,  # slow return to standing
        }
        await self._send_command(ws, "policy.tick", action_payload)
        try:
            await self._recv_response(ws)
        except Exception:
            pass

    async def _send_command(self, ws, command: str, payload: dict) -> None:
        """Send a command envelope via WebSocket."""
        msg = {
            "type": "command",
            "request_id": f"deploy-{self._step}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "command": command,
            "payload": payload,
        }
        await ws.send(json.dumps(msg))

    async def _recv_response(self, ws) -> dict:
        """Receive response from WebSocket, with timeout.

        Skips event messages (which lack request_id) and returns the
        first actual response envelope.
        """
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                remaining = max(0.01, deadline - time.monotonic())
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                data = json.loads(raw)
                # Process events but keep waiting for the response
                if "event" in data:
                    if "data" in data:
                        self._update_telemetry(data["data"])
                    continue
                # This is a response (has request_id)
                if "data" in data:
                    self._update_telemetry(data["data"])
                return data
            except asyncio.TimeoutError:
                break
        return {"ok": False, "message": "timeout"}

    async def _process_events(self, ws) -> None:
        """Try to receive any pending event messages (non-blocking)."""
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.001)
                data = json.loads(raw)
                if data.get("event") == "telemetry.basic":
                    self._update_telemetry(data.get("data", {}))
        except (asyncio.TimeoutError, Exception):
            pass

    def _update_telemetry(self, data: dict) -> None:
        """Update internal state from telemetry data."""
        if "imu_roll" in data:
            self._imu_roll = float(data["imu_roll"])
        if "imu_pitch" in data:
            self._imu_pitch = float(data["imu_pitch"])
        if "joint_positions" in data and isinstance(data["joint_positions"], dict):
            jp = data["joint_positions"]
            feedback = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)
            for i, name in enumerate(LEG_JOINT_NAMES):
                if name in jp:
                    feedback[i] = float(jp[name])
            self._joint_feedback = feedback
            self._telemetry_received = True

    def _print_status(
        self, elapsed: float, ramp: float, servo_cmds: list[dict]
    ) -> None:
        """Print status line."""
        mode = "STAND" if self.stand_only else "WALK"
        dry = " [DRY]" if self.dry_run else ""
        fb = "YES" if self._joint_feedback is not None else "NO"

        # Show a few representative joint pulse values
        sample_joints = ["r_hip_pitch", "r_knee", "l_hip_pitch", "l_knee"]
        pulse_str = ""
        for cmd in servo_cmds:
            sid = cmd["id"]
            for name in sample_joints:
                if joint_name_to_servo_id(name) == sid:
                    pulse_str += f" {name[-8:]}={cmd['position']}"
                    break

        print(
            f"  [{mode}{dry}] t={elapsed:.1f}s ramp={ramp:.2f} "
            f"step={self._step} imu=({self._imu_roll:.3f},{self._imu_pitch:.3f}) "
            f"fb={fb}{pulse_str}"
        )

    def run_dry(self) -> None:
        """Run in dry mode — no bridge connection, just log commands."""
        print(f"DRY RUN: Simulating {self.duration}s at {self.hz}Hz")
        print(f"Ramp: {self.ramp_seconds}s, Stand only: {self.stand_only}")
        print()

        start = time.monotonic()
        while time.monotonic() - start < self.duration:
            elapsed = time.monotonic() - start
            ramp = min(1.0, elapsed / self.ramp_seconds) if self.ramp_seconds > 0 else 1.0

            servo_cmds = self.policy_step(ramp)

            if self._step % int(self.hz * 2) == 1:  # every 2 seconds
                self._print_status(elapsed, ramp, servo_cmds)
                # Print full joint table occasionally
                if self._step % int(self.hz * 10) == 1:
                    print("    Joint targets (radians):")
                    for i, name in enumerate(LEG_JOINT_NAMES):
                        print(f"      {name:15s}: {self._last_targets[i]:+.4f} rad "
                              f"→ pulse {servo_cmds[i]['position']}")

            time.sleep(self.dt)

        print(f"\nDry run complete. {self._step} steps.")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy walking policy to real AiNex robot"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/mujoco_locomotion_v12_dr",
        help="Path to Brax checkpoint directory",
    )
    parser.add_argument(
        "--bridge", type=str, default="ws://localhost:9100",
        help="Bridge WebSocket URL",
    )
    parser.add_argument(
        "--hz", type=float, default=20.0,
        help="Control frequency in Hz (default: 20, max: 50)",
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Max duration in seconds",
    )
    parser.add_argument(
        "--ramp-seconds", type=float, default=3.0,
        help="Seconds to ramp from standing to full policy (0=instant)",
    )
    parser.add_argument(
        "--vx", type=float, default=0.3,
        help="Forward velocity command (m/s)",
    )
    parser.add_argument(
        "--vy", type=float, default=0.0,
        help="Lateral velocity command (m/s)",
    )
    parser.add_argument(
        "--vyaw", type=float, default=0.0,
        help="Yaw rate command (rad/s)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without bridge — log commands only",
    )
    parser.add_argument(
        "--stand-only", action="store_true",
        help="Only send standing pose (test servo communication)",
    )
    parser.add_argument(
        "--max-delta", type=float, default=MAX_JOINT_DELTA,
        help="Max joint position change per step (radians)",
    )

    args = parser.parse_args()

    # Safety check
    if args.hz > 50:
        print("WARNING: Hz > 50 exceeds policy training frequency. Clamping to 50.")
        args.hz = 50.0

    deployer = DeployWalking(
        checkpoint=args.checkpoint,
        hz=args.hz,
        duration=args.duration,
        ramp_seconds=args.ramp_seconds,
        dry_run=args.dry_run,
        stand_only=args.stand_only,
        vx=args.vx,
        vy=args.vy,
        vyaw=args.vyaw,
        max_joint_delta=args.max_delta,
    )

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nCtrl+C received, stopping...")
        deployer._stopped = True

    signal.signal(signal.SIGINT, signal_handler)

    if args.dry_run:
        deployer.run_dry()
    else:
        print(f"\n{'='*60}")
        print(f"DEPLOYING WALKING POLICY TO REAL ROBOT")
        print(f"{'='*60}")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Bridge:     {args.bridge}")
        print(f"  Frequency:  {args.hz} Hz")
        print(f"  Duration:   {args.duration}s")
        print(f"  Ramp:       {args.ramp_seconds}s")
        print(f"  Command:    vx={args.vx} vy={args.vy} vyaw={args.vyaw}")
        print(f"  Max delta:  {args.max_delta} rad/step")
        print(f"{'='*60}")
        print()

        asyncio.run(deployer.run_with_bridge(args.bridge))


if __name__ == "__main__":
    main()
