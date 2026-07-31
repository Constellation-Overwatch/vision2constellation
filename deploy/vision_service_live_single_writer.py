#!/usr/bin/env python3
"""Live deployment with Pulsar as the sole detection-KV materializer."""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deploy import (
    vision_service,
    vision_service_live,
    vision_service_live_cas,
    vision_service_nkey_file,
)
from src.config.models import DetectionMode
from src.config.threats import ALL_CLASSES, CLASS_TO_THREAT_LEVEL
from src.services.communication.service import OverwatchCommunication
from src.services.detection.factory import DetectorFactory
from src.services.detection.yoloe_c4isr import C4ISRThreatDetector
from src.services.tracking_id import TrackingIDService


class StableC4ISRThreatDetector(C4ISRThreatDetector):
    """Use spatial fallback IDs without treating frame indexes as tracks."""

    def process_frame(
        self,
        frame: Any,
        frame_timestamp: str,
        frame_count: int,
    ):
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
            persist=True,
            tracker="bytetrack.yaml",
        )

        result = results[0]
        detections = []
        current_cuids: set[str] = set()

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            if result.boxes.id is not None:
                native_ids = result.boxes.id.int().cpu().tolist()
            else:
                native_ids = [None] * len(boxes)
                print(
                    f"Frame {frame_count}: tracker IDs unavailable for "
                    f"{len(boxes)} detections; using spatial identity fallback"
                )

            for box, conf, cls_id, native_id in zip(
                boxes,
                confidences,
                class_ids,
                native_ids,
            ):
                x1, y1, x2, y2 = box
                class_name = (
                    ALL_CLASSES[cls_id]
                    if cls_id < len(ALL_CLASSES)
                    else f"class_{cls_id}"
                )
                threat_level = CLASS_TO_THREAT_LEVEL.get(
                    class_name, "NORMAL"
                )
                bbox = {
                    "x_min": float(x1 / frame.shape[1]),
                    "y_min": float(y1 / frame.shape[0]),
                    "x_max": float(x2 / frame.shape[1]),
                    "y_max": float(y2 / frame.shape[0]),
                }

                cuid = self.tracking_id_service.get_stable_cuid(
                    bbox=bbox,
                    label=class_name,
                    confidence=float(conf),
                    native_id=native_id,
                    model_type=self.model_type,
                )

                # A spatial fallback must not assign one identity to two
                # simultaneous detections in the same frame.
                if cuid in current_cuids:
                    cuid = self.tracking_id_service.cuid_generator()
                    spatial_hash = (
                        self.tracking_id_service._calculate_spatial_hash(
                            bbox, class_name
                        )
                    )
                    self.tracking_id_service.spatial_mapping[
                        spatial_hash
                    ] = cuid
                    self.tracking_id_service.cuid_to_spatial[
                        cuid
                    ] = spatial_hash
                    self.tracking_id_service._update_object_history(
                        cuid,
                        bbox,
                        class_name,
                        float(conf),
                    )

                current_cuids.add(cuid)
                suspicious_indicators = (
                    self._calculate_suspicious_indicators(
                        class_name,
                        conf,
                        threat_level,
                    )
                )
                detections.append(
                    self.tracking_id_service.format_detection_payload(
                        track_id=cuid,
                        label=class_name,
                        confidence=float(conf),
                        bbox=bbox,
                        timestamp=frame_timestamp,
                        model_type=self.model_type,
                        native_id=native_id,
                        threat_level=threat_level,
                        suspicious_indicators=suspicious_indicators,
                    )
                )

        return detections, self._visualize_c4isr_detections(
            frame, detections
        )


class PulsarMaterializedEntityKV(
    vision_service_live_cas.RevisionSafeEntityKV
):
    """Keep Pulsar-owned detections and CAS-merge vision C4ISR analytics."""

    def _merge(
        self,
        latest: dict[str, Any],
        proposed: dict[str, Any],
    ) -> dict[str, Any]:
        merged = super()._merge(latest, proposed)

        detections = merged.get("detections")
        if isinstance(detections, dict):
            detections = copy.deepcopy(detections)
            objects = detections.get("objects")
            if isinstance(objects, dict):
                detections["objects"] = {
                    track_id: obj
                    for track_id, obj in objects.items()
                    if not (
                        isinstance(obj, dict)
                        and obj.get("source") == self.WRITER
                    )
                }
            merged["detections"] = detections
        self._owned_ids.clear()

        proposed_analytics = proposed.get("analytics")
        c4isr_summary = None
        if isinstance(proposed_analytics, dict):
            c4isr_summary = proposed_analytics.get("c4isr_summary")

        latest_analytics = latest.get("analytics")
        if not isinstance(latest_analytics, dict):
            latest_analytics = {}
        else:
            latest_analytics = copy.deepcopy(latest_analytics)

        if isinstance(c4isr_summary, dict):
            latest_analytics[self.WRITER] = {
                "c4isr_summary": copy.deepcopy(c4isr_summary)
            }
        merged["analytics"] = latest_analytics
        return merged


_UPSTREAM_CREATE_DETECTOR = DetectorFactory.create_detector
_UPSTREAM_CLEANUP_IDS = TrackingIDService.cleanup_stale_ids


def _create_stable_detector(detection_mode, args):
    if detection_mode == DetectionMode.YOLOE_C4ISR:
        from src.config.models import get_model_config

        return StableC4ISRThreatDetector(
            args,
            get_model_config(detection_mode),
        )
    return _UPSTREAM_CREATE_DETECTOR(detection_mode, args)


def _cleanup_ids_with_grace(
    self: TrackingIDService,
    active_ids: set[str],
) -> None:
    """Retain tracker mappings across brief missed detections."""
    grace_frames = 150
    misses = getattr(self, "_inactive_miss_counts", {})
    known_ids = (
        set(self.object_history)
        | set(self.cuid_to_spatial)
        | set(self.id_mapping.values())
    )

    for cuid in known_ids:
        if cuid in active_ids:
            misses[cuid] = 0
        else:
            misses[cuid] = misses.get(cuid, 0) + 1

    retained_ids = set(active_ids)
    retained_ids.update(
        cuid
        for cuid, missed_frames in misses.items()
        if missed_frames <= grace_frames
    )
    _UPSTREAM_CLEANUP_IDS(self, retained_ids)
    self._inactive_miss_counts = {
        cuid: missed_frames
        for cuid, missed_frames in misses.items()
        if cuid in retained_ids
    }


async def _setup_pulsar_materialized_kv(
    self: OverwatchCommunication,
) -> None:
    await vision_service_live_cas._UPSTREAM_SETUP_KV(self)
    if self.kv is not None:
        self.kv = PulsarMaterializedEntityKV(self.kv, self)
        print(
            "Enabled revision-safe KV: Pulsar owns detections; "
            "vision owns C4ISR analytics"
        )


async def _publish_detection_state_via_events(
    self: OverwatchCommunication,
    tracking_state: Any,
    analytics: dict[str, Any],
) -> None:
    """Detection objects are materialized from JetStream events by Pulsar."""
    return None


async def _publish_threat_intelligence_once(
    self: OverwatchCommunication,
    tracking_state: Any,
) -> None:
    if (
        not self.kv
        or not hasattr(tracking_state, "threat_alerts")
        or not await self._is_connected()
    ):
        return

    try:
        analytics = tracking_state.get_analytics()
        timestamp = datetime.now(timezone.utc).isoformat()
        state = await self._get_entity_state()
        state["c4isr"] = {
            "timestamp": timestamp,
            "mission": "C4ISR",
            "threat_intelligence": {
                "threat_summary": {
                    "total_threats": analytics.get(
                        "active_threat_count", 0
                    ),
                    "threat_distribution": analytics.get(
                        "threat_distribution", {}
                    ),
                    "alert_level": (
                        "HIGH"
                        if analytics.get(
                            "threat_distribution", {}
                        ).get("HIGH_THREAT", 0)
                        > 0
                        else "NORMAL"
                    ),
                },
                "threat_alerts": analytics.get("threat_alerts", []),
            },
        }
        if not isinstance(state.get("analytics"), dict):
            state["analytics"] = {}
        state["analytics"]["c4isr_summary"] = {
            "timestamp": timestamp,
            **analytics,
        }
        state["updated_at"] = timestamp
        await self.kv.put(self.entity_id, json.dumps(state).encode())
        self._entity_state_cache = state
    except Exception as error:
        if not self._is_reconnecting:
            print(
                "Error publishing threat intelligence to KV: "
                f"{error}"
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
    TrackingIDService.cleanup_stale_ids = _cleanup_ids_with_grace
    OverwatchCommunication._connect_nats = (
        vision_service_nkey_file._connect_with_trimmed_seed
    )
    OverwatchCommunication._setup_kv_store = (
        _setup_pulsar_materialized_kv
    )
    OverwatchCommunication.publish_state_to_kv = (
        _publish_detection_state_via_events
    )
    OverwatchCommunication.publish_threat_intelligence = (
        _publish_threat_intelligence_once
    )
    await vision_service.main()


if __name__ == "__main__":
    asyncio.run(main())
