"""Global perception configuration with YAML loading support."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

from perception.calibration import CameraIntrinsics

_ROOT = Path(__file__).resolve().parent

# Derive camera defaults from the single source of truth
_DEFAULT_INTRINSICS = CameraIntrinsics()


@dataclass
class CameraConfig:
    """Camera parameters."""
    width: int = _DEFAULT_INTRINSICS.width
    height: int = _DEFAULT_INTRINSICS.height
    fps: int = 30
    device: int = 0
    fx: float = _DEFAULT_INTRINSICS.fx
    fy: float = _DEFAULT_INTRINSICS.fy
    cx: float = _DEFAULT_INTRINSICS.cx
    cy: float = _DEFAULT_INTRINSICS.cy
    dist_coeffs: tuple[float, ...] = _DEFAULT_INTRINSICS.dist_coeffs


@dataclass
class DetectorConfig:
    """Detector thresholds."""
    face_confidence: float = 0.5
    face_recognition_threshold: float = 0.4
    object_confidence: float = 0.5
    skeleton_confidence: float = 0.3
    depth_enabled: bool = True


@dataclass
class EntitySlotConfig:
    """Entity slot encoding parameters."""
    num_slots: int = 8
    slot_dim: int = 19
    max_distance: float = 5.0
    max_velocity: float = 2.0
    max_size: float = 2.0
    recency_horizon: float = 5.0


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    entity_slots: EntitySlotConfig = field(default_factory=EntitySlotConfig)
    stale_timeout_sec: float = 5.0
    data_dir: Path = field(default_factory=lambda: _ROOT / "data")


def load_config(path: Path | None = None) -> PipelineConfig:
    """Load pipeline config from YAML file, falling back to defaults."""
    if path is None or not path.exists():
        return PipelineConfig()
    try:
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except ImportError:
        return PipelineConfig()

    cfg = PipelineConfig()
    if "camera" in raw:
        cam = raw["camera"]
        cfg.camera = CameraConfig(
            width=cam.get("width", 640),
            height=cam.get("height", 480),
            fps=cam.get("fps", 30),
            device=cam.get("device", 0),
            fx=cam.get("fx", 533.0),
            fy=cam.get("fy", 533.0),
            cx=cam.get("cx", 320.0),
            cy=cam.get("cy", 240.0),
            dist_coeffs=tuple(cam.get("dist_coeffs", [0.0] * 5)),
        )
    if "detector" in raw:
        det = raw["detector"]
        valid_fields = {f.name for f in dataclasses.fields(DetectorConfig)}
        cfg.detector = DetectorConfig(**{
            k: det[k] for k in det if k in valid_fields
        })
    if "entity_slots" in raw:
        es = raw["entity_slots"]
        valid_fields = {f.name for f in dataclasses.fields(EntitySlotConfig)}
        cfg.entity_slots = EntitySlotConfig(**{
            k: es[k] for k in es if k in valid_fields
        })
    if "stale_timeout_sec" in raw:
        cfg.stale_timeout_sec = float(raw["stale_timeout_sec"])
    return cfg
