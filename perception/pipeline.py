"""Main perception pipeline orchestrator.

Runs all detectors on each frame, updates the world model,
and produces entity slots for the RL policy.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from perception.calibration import CameraIntrinsics
from perception.config import PipelineConfig
from perception.detectors.depth_estimator import DepthEstimator, DepthResult
from perception.detectors.face_detector import FaceDetector
from perception.detectors.face_recognizer import FaceRecognizer
from perception.detectors.face_tracker import FaceTracker
from perception.detectors.object_detector import ObjectDetector
from perception.detectors.object_tracker import ObjectTracker
from perception.detectors.skeleton_estimator import SkeletonEstimator
from perception.entity_slots.slot_encoder import encode_entity_slots
from perception.frame_source import FrameSource
from perception.world_model.world_state import WorldState

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Output of a single pipeline step."""
    entity_slots: np.ndarray          # (152,) flat
    entities: list[Any]               # PersistentEntity list
    depth: DepthResult | None = None
    frame_timestamp: float = 0.0
    processing_ms: float = 0.0


class PerceptionPipeline:
    """Main perception pipeline: frame → detections → world model → entity slots.

    Initializes all detectors and runs them per-frame. Non-available detectors
    are silently skipped.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._intrinsics = CameraIntrinsics(
            fx=self._config.camera.fx,
            fy=self._config.camera.fy,
            cx=self._config.camera.cx,
            cy=self._config.camera.cy,
            dist_coeffs=self._config.camera.dist_coeffs,
            width=self._config.camera.width,
            height=self._config.camera.height,
        )
        # Detectors
        self._face_detector = FaceDetector(
            confidence_threshold=self._config.detector.face_confidence,
        )
        self._face_recognizer = FaceRecognizer(
            recognition_threshold=self._config.detector.face_recognition_threshold,
            gallery_dir=self._config.data_dir / "face_gallery",
        )
        self._face_tracker = FaceTracker()
        self._object_detector = ObjectDetector(
            confidence_threshold=self._config.detector.object_confidence,
        )
        # Share YOLO model between detector and tracker to avoid double GPU memory
        self._object_tracker = ObjectTracker(
            confidence_threshold=self._config.detector.object_confidence,
            model=self._object_detector._model if self._object_detector.is_available else None,
        )
        self._skeleton_estimator = SkeletonEstimator(
            confidence_threshold=self._config.detector.skeleton_confidence,
        )
        self._depth_estimator = DepthEstimator() if self._config.detector.depth_enabled else None

        # World model
        self._world = WorldState(
            intrinsics=self._intrinsics,
            stale_timeout_sec=self._config.stale_timeout_sec,
        )

        # Callbacks
        self._callbacks: list[Callable[[PipelineResult], None]] = []

    def add_callback(self, callback: Callable[[PipelineResult], None]) -> None:
        """Register a callback invoked after each pipeline step."""
        self._callbacks.append(callback)

    def connect_aggregator(self, aggregator: Any) -> None:
        """Connect pipeline output to a PerceptionAggregator.

        Registers a callback that feeds entity slots and tracked entities
        into the aggregator on every frame, closing the bridge integration gap.
        """
        def _feed_aggregator(result: PipelineResult) -> None:
            # Update entity slots
            aggregator.update_entity_slots(tuple(result.entity_slots.tolist()))
            # Also update entities batch for the scene_summary / Eliza path
            entities_batch = []
            for e in result.entities:
                entities_batch.append({
                    "entity_id": e.entity_id,
                    "label": e.label,
                    "confidence": e.confidence,
                    "x": float(e.position[0]),
                    "y": float(e.position[1]),
                    "z": float(e.position[2]),
                    "source": e.source,
                })
            if entities_batch:
                aggregator.update_entities_batch(entities_batch)

        self._callbacks.append(_feed_aggregator)

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        """Run full pipeline on a single BGR frame."""
        t0 = time.monotonic()

        # Depth estimation
        depth = None
        if self._depth_estimator is not None:
            depth = self._depth_estimator.estimate(frame)

        # Face detection → recognition → tracking
        face_dets = self._face_detector.detect(frame)
        identity_ids = []
        for det in face_dets:
            if det.embedding is not None:
                identity_id, _ = self._face_recognizer.recognize(det)
                identity_ids.append(identity_id)
            else:
                identity_ids.append("")
        face_tracks = self._face_tracker.update(face_dets, identity_ids)
        self._world.update_from_faces(face_tracks, depth)

        # Object detection + tracking
        if self._object_tracker.is_available:
            tracked_objs = self._object_tracker.track(frame)
            self._world.update_from_objects(tracked_objs, depth)
        elif self._object_detector.is_available:
            obj_dets = self._object_detector.detect(frame)
            self._world.update_from_objects(obj_dets, depth)

        # Skeleton estimation
        skeletons = self._skeleton_estimator.estimate(frame)
        if skeletons:
            self._world.update_from_skeletons(skeletons, depth)

        # Prune stale
        self._world.prune_stale()

        # Encode entity slots
        entity_slots = encode_entity_slots(self._world.entity_list)

        t1 = time.monotonic()
        result = PipelineResult(
            entity_slots=entity_slots,
            entities=self._world.entity_list,
            depth=depth,
            frame_timestamp=t0,
            processing_ms=(t1 - t0) * 1000,
        )

        for cb in self._callbacks:
            try:
                cb(result)
            except Exception as e:
                logger.warning("Pipeline callback error: %s", e)

        return result

    def run(self, source: FrameSource) -> None:
        """Run pipeline continuously on a frame source."""
        with source:
            for frame in source:
                self.process_frame(frame)

    @property
    def world_state(self) -> WorldState:
        return self._world

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics
