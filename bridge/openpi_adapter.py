"""Custom AiNex OpenPI observation/action adapter.

Translates between AiNex perception/telemetry and the OpenPI policy server
wire format.  Enforces schema validation and deterministic defaulting for
missing signals so the policy always receives a well-formed observation and
the robot always receives a bounded command.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict
from typing import Any

from training.interfaces import (
    AinexPerceptionObservation,
    OpenPIActionChunk,
    OpenPIObservationPayload,
    TrackedEntity,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalization constants (maps raw AiNex values to [-1, 1] or [0, 1])
# ---------------------------------------------------------------------------
_WALK_X_RANGE = 0.05       # ±0.05 m/step
_WALK_Y_RANGE = 0.05
_WALK_YAW_RANGE = 10.0     # ±10 deg/step
_WALK_HEIGHT_MIN = 0.015
_WALK_HEIGHT_MAX = 0.06
_WALK_SPEED_MIN = 1
_WALK_SPEED_MAX = 4
_HEAD_PAN_RANGE = 1.5      # ±1.5 rad
_HEAD_TILT_RANGE = 1.0     # ±1.0 rad
_IMU_RANGE = math.pi       # ±π rad
_BATTERY_MIN = 10400       # mV
_BATTERY_MAX = 12600

# State vector dimension for the AiNex proprioception
AINEX_PROPRIO_DIM = 11
# Entity slot dimensions (8 slots x 19 dims)
AINEX_ENTITY_SLOT_DIM = 152
# Full state dimension (proprioception + entity slots)
AINEX_STATE_DIM = AINEX_PROPRIO_DIM + AINEX_ENTITY_SLOT_DIM  # 163

# Action vector dimension from OpenPI
AINEX_ACTION_DIM = 7  # walk_x, walk_y, walk_yaw, walk_height, walk_speed, head_pan, head_tilt


def _norm(value: float, lo: float, hi: float) -> float:
    """Normalize value from [lo, hi] to [-1, 1]."""
    if hi == lo:
        return 0.0
    return 2.0 * (value - lo) / (hi - lo) - 1.0


def _denorm(value: float, lo: float, hi: float) -> float:
    """Denormalize from [-1, 1] to [lo, hi]."""
    return lo + (value + 1.0) * 0.5 * (hi - lo)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def build_observation(
    perception: AinexPerceptionObservation,
) -> OpenPIObservationPayload:
    """Convert an AiNex perception snapshot into an OpenPI observation payload."""

    proprio = (
        _norm(perception.walk_x, -_WALK_X_RANGE, _WALK_X_RANGE),
        _norm(perception.walk_y, -_WALK_Y_RANGE, _WALK_Y_RANGE),
        _norm(perception.walk_yaw, -_WALK_YAW_RANGE, _WALK_YAW_RANGE),
        _norm(perception.walk_height, _WALK_HEIGHT_MIN, _WALK_HEIGHT_MAX),
        _norm(float(perception.walk_speed), float(_WALK_SPEED_MIN), float(_WALK_SPEED_MAX)),
        _norm(perception.head_pan, -_HEAD_PAN_RANGE, _HEAD_PAN_RANGE),
        _norm(perception.head_tilt, -_HEAD_TILT_RANGE, _HEAD_TILT_RANGE),
        _norm(perception.imu_roll, -_IMU_RANGE, _IMU_RANGE),
        _norm(perception.imu_pitch, -_IMU_RANGE, _IMU_RANGE),
        1.0 if perception.is_walking else -1.0,
        _norm(float(perception.battery_mv), float(_BATTERY_MIN), float(_BATTERY_MAX)),
    )

    # Entity slots (already normalized to [-1, 1] by slot encoder)
    if perception.entity_slots and len(perception.entity_slots) == AINEX_ENTITY_SLOT_DIM:
        entity_slots = tuple(perception.entity_slots)
    else:
        if perception.entity_slots and len(perception.entity_slots) != AINEX_ENTITY_SLOT_DIM:
            logger.warning(
                "entity_slots has %d dims, expected %d; using zeros",
                len(perception.entity_slots), AINEX_ENTITY_SLOT_DIM,
            )
        entity_slots = (0.0,) * AINEX_ENTITY_SLOT_DIM

    state = proprio + entity_slots

    metadata: dict[str, Any] = {
        "timestamp": perception.timestamp,
        "battery_mv": perception.battery_mv,
    }
    if perception.tracked_entities:
        metadata["entities"] = [
            {
                "id": e.entity_id,
                "label": e.label,
                "confidence": e.confidence,
                "xyz": [e.x, e.y, e.z],
            }
            for e in perception.tracked_entities
        ]

    return OpenPIObservationPayload(
        state=state,
        prompt=perception.language_instruction,
        image=perception.camera_frame,
        metadata=metadata,
    )


def observation_to_dict(obs: OpenPIObservationPayload) -> dict[str, Any]:
    """Serialize an observation payload to a dict suitable for the OpenPI client."""
    d: dict[str, Any] = {
        "state": list(obs.state),
        "prompt": obs.prompt,
    }
    if obs.image:
        d["image"] = obs.image
    if obs.metadata:
        d["metadata"] = obs.metadata
    return d


# ---------------------------------------------------------------------------
# Action decoder
# ---------------------------------------------------------------------------

def decode_action(raw: dict[str, Any]) -> OpenPIActionChunk:
    """Decode an OpenPI action response into an AiNex action chunk.

    Accepts either:
    - A dict with an ``action`` key containing a list of floats (raw vector)
    - A dict with explicit named fields (walk_x, walk_y, etc.)
    """
    confidence = float(raw.get("confidence", 1.0))

    action_vector = raw.get("action")
    if isinstance(action_vector, (list, tuple)) and len(action_vector) >= AINEX_ACTION_DIM:
        # Raw action vector from policy: decode positionally
        av = [float(v) for v in action_vector]
        walk_x = _clamp(_denorm(av[0], -_WALK_X_RANGE, _WALK_X_RANGE), -_WALK_X_RANGE, _WALK_X_RANGE)
        walk_y = _clamp(_denorm(av[1], -_WALK_Y_RANGE, _WALK_Y_RANGE), -_WALK_Y_RANGE, _WALK_Y_RANGE)
        walk_yaw = _clamp(_denorm(av[2], -_WALK_YAW_RANGE, _WALK_YAW_RANGE), -_WALK_YAW_RANGE, _WALK_YAW_RANGE)
        walk_height = _clamp(_denorm(av[3], _WALK_HEIGHT_MIN, _WALK_HEIGHT_MAX), _WALK_HEIGHT_MIN, _WALK_HEIGHT_MAX)
        walk_speed = int(round(_clamp(_denorm(av[4], float(_WALK_SPEED_MIN), float(_WALK_SPEED_MAX)), float(_WALK_SPEED_MIN), float(_WALK_SPEED_MAX))))
        head_pan = _clamp(_denorm(av[5], -_HEAD_PAN_RANGE, _HEAD_PAN_RANGE), -_HEAD_PAN_RANGE, _HEAD_PAN_RANGE)
        head_tilt = _clamp(_denorm(av[6], -_HEAD_TILT_RANGE, _HEAD_TILT_RANGE), -_HEAD_TILT_RANGE, _HEAD_TILT_RANGE)

        return OpenPIActionChunk(
            raw_action=tuple(av),
            walk_x=walk_x,
            walk_y=walk_y,
            walk_yaw=walk_yaw,
            walk_height=walk_height,
            walk_speed=walk_speed,
            head_pan=head_pan,
            head_tilt=head_tilt,
            confidence=confidence,
        )

    # Named-field format (e.g., from a structured policy)
    return OpenPIActionChunk(
        walk_x=_clamp(float(raw.get("walk_x", 0.0)), -_WALK_X_RANGE, _WALK_X_RANGE),
        walk_y=_clamp(float(raw.get("walk_y", 0.0)), -_WALK_Y_RANGE, _WALK_Y_RANGE),
        walk_yaw=_clamp(float(raw.get("walk_yaw", 0.0)), -_WALK_YAW_RANGE, _WALK_YAW_RANGE),
        walk_height=_clamp(float(raw.get("walk_height", 0.036)), _WALK_HEIGHT_MIN, _WALK_HEIGHT_MAX),
        walk_speed=int(round(_clamp(float(raw.get("walk_speed", 2)), float(_WALK_SPEED_MIN), float(_WALK_SPEED_MAX)))),
        head_pan=_clamp(float(raw.get("head_pan", 0.0)), -_HEAD_PAN_RANGE, _HEAD_PAN_RANGE),
        head_tilt=_clamp(float(raw.get("head_tilt", 0.0)), -_HEAD_TILT_RANGE, _HEAD_TILT_RANGE),
        action_name=str(raw.get("action_name", "")),
        confidence=confidence,
    )


def action_to_bridge_commands(action: OpenPIActionChunk) -> list[dict[str, Any]]:
    """Convert an OpenPI action chunk into a list of bridge command payloads.

    Returns one or more command dicts ready to be sent as CommandEnvelope payloads.
    """
    commands: list[dict[str, Any]] = []

    # Walk set command
    commands.append({
        "command": "walk.set",
        "payload": {
            "speed": action.walk_speed,
            "height": action.walk_height,
            "x": action.walk_x,
            "y": action.walk_y,
            "yaw": action.walk_yaw,
        },
    })

    # Head set command (only if non-zero)
    if action.head_pan != 0.0 or action.head_tilt != 0.0:
        commands.append({
            "command": "head.set",
            "payload": {
                "pan": action.head_pan,
                "tilt": action.head_tilt,
                "duration": 0.1,  # Fast tracking during policy mode
            },
        })

    # Named action (only if specified)
    if action.action_name:
        commands.append({
            "command": "action.play",
            "payload": {
                "name": action.action_name,
            },
        })

    return commands


# ---------------------------------------------------------------------------
# Default / fallback observation (for when perception is unavailable)
# ---------------------------------------------------------------------------

def default_perception() -> AinexPerceptionObservation:
    """Return a safe default perception with all values at neutral."""
    return AinexPerceptionObservation(
        timestamp=0.0,
        battery_mv=12000,
        imu_roll=0.0,
        imu_pitch=0.0,
        is_walking=False,
        walk_x=0.0,
        walk_y=0.0,
        walk_yaw=0.0,
        walk_height=0.036,
        walk_speed=2,
        head_pan=0.0,
        head_tilt=0.0,
    )
