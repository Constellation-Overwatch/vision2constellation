#!/usr/bin/env python3
"""Live deployment with idempotent object IDs and CAS entity-state merges."""

from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deploy import (
    vision_service,
    vision_service_live,
    vision_service_live_cas,
    vision_service_live_single_writer,
    vision_service_nkey_file,
)
from src.config.models import DetectionMode, get_model_config
from src.services.communication.publisher import ConstellationPublisher
from src.services.communication.service import OverwatchCommunication
from src.services.detection.factory import DetectorFactory
from src.services.tracking_id import TrackingIDService


class IdempotentObjectEntityKV(
    vision_service_live_cas.RevisionSafeEntityKV
):
    """Replace vision objects by track ID while preserving telemetry scopes."""

    LEGACY_EVENT_OBJECT_KEYS = {
        "bbox",
        "confidence",
        "cx",
        "cy",
        "dx",
        "dy",
        "first_seen",
        "frame_count",
        "label",
        "last_seen",
    }

    def __init__(self, raw_kv: Any, owner: OverwatchCommunication):
        super().__init__(raw_kv, owner)
        self._filtered_count = 0

    def _merge(
        self,
        latest: dict[str, Any],
        proposed: dict[str, Any],
    ) -> dict[str, Any]:
        merged = super()._merge(latest, proposed)
        detections = merged.get("detections")
        if not isinstance(detections, dict):
            return merged

        detections = copy.deepcopy(detections)
        objects = detections.get("objects")
        if not isinstance(objects, dict):
            return merged

        proposed_objects = self._objects(proposed)
        current_owned = {
            str(track_id)
            for track_id in self._owner._last_published_objects
            if str(track_id) in proposed_objects
        }

        filtered: dict[str, Any] = {}
        removed = 0
        for track_id, obj in objects.items():
            if not isinstance(obj, dict):
                filtered[track_id] = obj
                continue

            is_current_writer = (
                obj.get("source") == self.WRITER
                and obj.get("writer_id") == self._writer_id
            )
            is_legacy_event_copy = (
                obj.get("source") is None
                and self.LEGACY_EVENT_OBJECT_KEYS.issubset(obj.keys())
            )

            if is_current_writer and track_id not in current_owned:
                removed += 1
                continue
            if is_legacy_event_copy:
                removed += 1
                continue
            filtered[track_id] = obj

        detections["objects"] = filtered
        merged["detections"] = detections

        if removed:
            self._filtered_count += removed
            if self._filtered_count == removed or self._filtered_count % 250 == 0:
                print(
                    "Removed stale/duplicate detection materializations "
                    f"(cumulative={self._filtered_count})"
                )
        return merged


_UPSTREAM_CREATE_DETECTOR = DetectorFactory.create_detector
_UPSTREAM_PUBLISH_STATE = OverwatchCommunication.publish_state_to_kv
_UPSTREAM_PUBLISH_DETECTION = (
    OverwatchCommunication.publish_detection_event
)
_UPSTREAM_BUILD_DETECTION = ConstellationPublisher.build_detection


def _create_stable_detector(detection_mode, args):
    if detection_mode == DetectionMode.YOLOE_C4ISR:
        return vision_service_live_single_writer.StableC4ISRThreatDetector(
            args,
            get_model_config(detection_mode),
        )
    return _UPSTREAM_CREATE_DETECTOR(detection_mode, args)


def _build_idempotent_detection(
    self: ConstellationPublisher,
    detection_data: dict[str, Any],
) -> dict[str, Any]:
    message = _UPSTREAM_BUILD_DETECTION(self, detection_data)
    track_id = str(detection_data.get("track_id", "")).strip()
    if track_id:
        # Pulsar consumers can upsert without knowing the nested envelope.
        message["track_id"] = track_id
        message["object_id"] = track_id
        message["idempotency_key"] = (
            f"{self.entity_id}:detection:{track_id}"
        )
    return message


async def _publish_detection_once_per_track(
    self: OverwatchCommunication,
    detection_data: dict[str, Any],
) -> None:
    track_id = str(detection_data.get("track_id", "")).strip()
    if not track_id:
        return

    published_ids = getattr(self, "_published_track_ids", None)
    if published_ids is None:
        published_ids = set()
        self._published_track_ids = published_ids
    if track_id in published_ids:
        return

    await _UPSTREAM_PUBLISH_DETECTION(self, detection_data)
    published_ids.add(track_id)


async def _publish_active_objects(
    self: OverwatchCommunication,
    tracking_state: Any,
    analytics: dict[str, Any],
) -> None:
    persistent = {}
    if hasattr(tracking_state, "get_persistent_objects"):
        persistent = tracking_state.get_persistent_objects(min_frames=3)

    active_ids = {
        str(track_id)
        for track_id, obj in persistent.items()
        if isinstance(obj, dict) and obj.get("is_active", False)
    }
    for track_id in list(self._last_published_objects):
        if track_id not in active_ids:
            self._last_published_objects.pop(track_id, None)

    await _UPSTREAM_PUBLISH_STATE(self, tracking_state, analytics)


async def _setup_idempotent_kv(
    self: OverwatchCommunication,
) -> None:
    await vision_service_live_cas._UPSTREAM_SETUP_KV(self)
    if self.kv is not None:
        self.kv = IdempotentObjectEntityKV(self.kv, self)
        print(
            "Enabled idempotent object KV merges "
            "(CAS + replace-by-track_id)"
        )


async def main() -> None:
    vision_service_live_cas._set_entity_output_path()
    vision_service.HeadlessRTSPVideoService.open_video_stream = (
        vision_service_live._open_rtsp_persistently
    )
    vision_service.RTSPPublisher = (
        vision_service_live_cas.BackoffRTSPPublisher
    )
    DetectorFactory.create_detector = staticmethod(
        _create_stable_detector
    )
    TrackingIDService.cleanup_stale_ids = (
        vision_service_live_single_writer._cleanup_ids_with_grace
    )
    ConstellationPublisher.build_detection = (
        _build_idempotent_detection
    )
    OverwatchCommunication._connect_nats = (
        vision_service_nkey_file._connect_with_trimmed_seed
    )
    OverwatchCommunication._setup_kv_store = _setup_idempotent_kv
    OverwatchCommunication.publish_detection_event = (
        _publish_detection_once_per_track
    )
    OverwatchCommunication.publish_state_to_kv = (
        _publish_active_objects
    )
    OverwatchCommunication.publish_threat_intelligence = (
        vision_service_live_single_writer._publish_threat_intelligence_once
    )
    await vision_service.main()


if __name__ == "__main__":
    asyncio.run(main())
