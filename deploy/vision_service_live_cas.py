#!/usr/bin/env python3
"""Live deployment with revision-safe, writer-scoped entity KV updates."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

from nats.js.errors import KeyNotFoundError, KeyWrongLastSequenceError

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deploy import vision_service, vision_service_live, vision_service_nkey_file
from src.services.communication.service import OverwatchCommunication
from src.services.video.publisher import RTSPPublisher


class RevisionSafeEntityKV:
    """Merge only vision-owned scopes into the latest entity document."""

    MAX_CAS_ATTEMPTS = 25
    WRITER = "vision2constellation"

    def __init__(self, raw_kv: Any, owner: OverwatchCommunication):
        self._raw = raw_kv
        self._owner = owner
        self._lock = asyncio.Lock()
        self._owned_ids: set[str] = set()
        self._writer_id = str(
            (owner.device_fingerprint or {}).get("device_id", self.WRITER)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)

    @staticmethod
    def _document(value: bytes) -> dict[str, Any]:
        decoded = json.loads(value.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("entity KV value must be a JSON object")
        return decoded

    @staticmethod
    def _objects(document: dict[str, Any]) -> dict[str, Any]:
        detections = document.get("detections")
        if not isinstance(detections, dict):
            return {}
        objects = detections.get("objects")
        return objects if isinstance(objects, dict) else {}

    def _merge(
        self,
        latest: dict[str, Any],
        proposed: dict[str, Any],
    ) -> dict[str, Any]:
        if latest:
            merged = copy.deepcopy(latest)
        else:
            # Bootstrap only. In normal operation Pulsar has already created the key.
            merged = copy.deepcopy(proposed)

        latest_detections = merged.get("detections")
        if not isinstance(latest_detections, dict):
            latest_detections = {}
        else:
            latest_detections = copy.deepcopy(latest_detections)

        latest_objects = latest_detections.get("objects")
        if not isinstance(latest_objects, dict):
            latest_objects = {}
        else:
            latest_objects = copy.deepcopy(latest_objects)

        proposed_objects = self._objects(proposed)
        tracked_ids = {
            str(track_id)
            for track_id in self._owner._last_published_objects.keys()
        }
        current_owned = tracked_ids.intersection(
            str(track_id) for track_id in proposed_objects.keys()
        )

        for track_id in current_owned:
            obj = proposed_objects.get(track_id)
            if not isinstance(obj, dict):
                continue
            owned_obj = copy.deepcopy(obj)
            owned_obj["source"] = self.WRITER
            owned_obj["writer_id"] = self._writer_id
            latest_objects[track_id] = owned_obj

        for track_id in self._owned_ids - current_owned:
            existing = latest_objects.get(track_id)
            if (
                isinstance(existing, dict)
                and existing.get("source") == self.WRITER
                and existing.get("writer_id") == self._writer_id
            ):
                latest_objects.pop(track_id, None)

        self._owned_ids = current_owned
        latest_detections["objects"] = latest_objects
        proposed_detections = proposed.get("detections")
        if isinstance(proposed_detections, dict):
            vision_timestamp = proposed_detections.get("timestamp")
            if vision_timestamp:
                latest_detections["vision_timestamp"] = vision_timestamp
            if "timestamp" not in latest_detections and vision_timestamp:
                latest_detections["timestamp"] = vision_timestamp
        merged["detections"] = latest_detections

        proposed_analytics = proposed.get("analytics")
        if isinstance(proposed_analytics, dict):
            latest_analytics = merged.get("analytics")
            if not isinstance(latest_analytics, dict):
                latest_analytics = {}
            else:
                latest_analytics = copy.deepcopy(latest_analytics)
            latest_analytics[self.WRITER] = copy.deepcopy(proposed_analytics)
            merged["analytics"] = latest_analytics

        proposed_c4isr = proposed.get("c4isr")
        if isinstance(proposed_c4isr, dict):
            merged["c4isr"] = copy.deepcopy(proposed_c4isr)

        merged["vision_updated_at"] = proposed.get(
            "updated_at", datetime.now(timezone.utc).isoformat()
        )
        return merged

    async def put(
        self,
        key: str,
        value: bytes,
        validate_keys: bool = True,
    ) -> int:
        if key != self._owner.entity_id:
            return await self._raw.put(
                key, value, validate_keys=validate_keys
            )

        proposed = self._document(value)
        async with self._lock:
            for attempt in range(1, self.MAX_CAS_ATTEMPTS + 1):
                try:
                    entry = await self._raw.get(
                        key, validate_keys=validate_keys
                    )
                except KeyNotFoundError:
                    merged = self._merge({}, proposed)
                    try:
                        return await self._raw.create(
                            key,
                            json.dumps(merged, separators=(",", ":")).encode(),
                            validate_keys=validate_keys,
                        )
                    except KeyWrongLastSequenceError:
                        await asyncio.sleep(0)
                        continue

                merged = self._merge(self._document(entry.value), proposed)
                encoded = json.dumps(
                    merged, separators=(",", ":")
                ).encode("utf-8")
                try:
                    return await self._raw.update(
                        key,
                        encoded,
                        last=entry.revision,
                        validate_keys=validate_keys,
                    )
                except KeyWrongLastSequenceError:
                    if attempt == self.MAX_CAS_ATTEMPTS:
                        raise
                    await asyncio.sleep(0)

        raise RuntimeError("entity KV compare-and-set attempts exhausted")


class BackoffRTSPPublisher(RTSPPublisher):
    """Avoid rapid FFmpeg respawns when MediaMTX rejects the publish path."""

    MIN_RECONNECT_DELAY = 30.0

    def _write_frame(self, frame: Any) -> None:
        process = self._process
        if process is not None and process.poll() is not None:
            delay = max(self.reconnect_delay, self.MIN_RECONNECT_DELAY)
            print(
                "RTSP output process exited; "
                f"retrying publisher in {delay:g}s"
            )
            self.reconnects += 1
            self.frames_dropped += 1
            self._retry_after = time.monotonic() + delay
            self._stop_process()
            return
        super()._write_frame(frame)


_UPSTREAM_SETUP_KV = OverwatchCommunication._setup_kv_store


async def _setup_revision_safe_kv(self: OverwatchCommunication) -> None:
    await _UPSTREAM_SETUP_KV(self)
    if self.kv is not None:
        self.kv = RevisionSafeEntityKV(self.kv, self)
        print("Enabled revision-safe writer-scoped entity KV updates")


def _set_entity_output_path() -> None:
    entity_id = os.environ["CONSTELLATION_ENTITY_ID"].strip()
    output_url = os.environ["VISION_OUTPUT_RTSP_URL"]
    parsed = urlsplit(output_url)
    entity_path = f"/{quote(entity_id, safe='')}/pulsar"
    os.environ["VISION_OUTPUT_RTSP_URL"] = urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            entity_path,
            parsed.query,
            parsed.fragment,
        )
    )

    reconnect_delay = float(
        os.getenv("VISION_OUTPUT_RECONNECT_DELAY", "30")
    )
    os.environ["VISION_OUTPUT_RECONNECT_DELAY"] = str(
        max(reconnect_delay, BackoffRTSPPublisher.MIN_RECONNECT_DELAY)
    )


async def main() -> None:
    _set_entity_output_path()
    vision_service.HeadlessRTSPVideoService.open_video_stream = (
        vision_service_live._open_rtsp_persistently
    )
    vision_service.RTSPPublisher = BackoffRTSPPublisher
    OverwatchCommunication._connect_nats = (
        vision_service_nkey_file._connect_with_trimmed_seed
    )
    OverwatchCommunication._setup_kv_store = _setup_revision_safe_kv
    await vision_service.main()


if __name__ == "__main__":
    asyncio.run(main())
