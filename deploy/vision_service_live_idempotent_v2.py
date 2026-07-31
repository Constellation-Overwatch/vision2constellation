#!/usr/bin/env python3
"""Idempotent live runtime with normalized Pulsar detection scopes."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deploy import (
    vision_service,
    vision_service_live,
    vision_service_live_cas,
    vision_service_live_idempotent,
    vision_service_live_single_writer,
    vision_service_nkey_file,
)
from src.services.communication.publisher import ConstellationPublisher
from src.services.communication.service import OverwatchCommunication
from src.services.detection.factory import DetectorFactory
from src.services.tracking_id import TrackingIDService


async def _publish_active_objects_normalized(
    self: OverwatchCommunication,
    tracking_state: Any,
    analytics: dict[str, Any],
) -> None:
    entity_state = await self._get_entity_state()
    detections = entity_state.get("detections")
    if not isinstance(detections, dict):
        detections = {}
        entity_state["detections"] = detections
    if not isinstance(detections.get("objects"), dict):
        detections["objects"] = {}
    self._entity_state_cache = entity_state

    await vision_service_live_idempotent._publish_active_objects(
        self,
        tracking_state,
        analytics,
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
        vision_service_live_idempotent._create_stable_detector
    )
    TrackingIDService.cleanup_stale_ids = (
        vision_service_live_single_writer._cleanup_ids_with_grace
    )
    ConstellationPublisher.build_detection = (
        vision_service_live_idempotent._build_idempotent_detection
    )
    OverwatchCommunication._connect_nats = (
        vision_service_nkey_file._connect_with_trimmed_seed
    )
    OverwatchCommunication._setup_kv_store = (
        vision_service_live_idempotent._setup_idempotent_kv
    )
    OverwatchCommunication.publish_detection_event = (
        vision_service_live_idempotent._publish_detection_once_per_track
    )
    OverwatchCommunication.publish_state_to_kv = (
        _publish_active_objects_normalized
    )
    OverwatchCommunication.publish_threat_intelligence = (
        vision_service_live_single_writer._publish_threat_intelligence_once
    )
    await vision_service.main()


if __name__ == "__main__":
    asyncio.run(main())
