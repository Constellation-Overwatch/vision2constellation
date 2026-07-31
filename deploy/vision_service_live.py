#!/usr/bin/env python3
"""Production adapter with persistent RTSP acquisition and NKey auth."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deploy import vision_service, vision_service_nkey_file
from src.services.communication.service import OverwatchCommunication


def _open_rtsp_persistently(self) -> bool:
    """Remain alive while waiting for a decodable live-source keyframe."""
    self.video_source, self.source_type, self.selected_device = (
        self.determine_video_source()
    )
    if self.source_type != "rtsp":
        raise RuntimeError("The production deployment requires an RTSP input")

    attempt = 0
    while not vision_service.upstream.is_shutdown_requested():
        attempt += 1
        self.cap = self._open_stream_with_retries()
        if self.cap is not None and self.cap.isOpened():
            self._apply_stream_optimizations()
            if attempt > 1:
                print(f"RTSP startup recovered on attempt {attempt}")
            return True

        print("RTSP source has no decodable frame; retrying in 3 seconds")
        time.sleep(3)

    return False


async def main() -> None:
    vision_service.HeadlessRTSPVideoService.open_video_stream = (
        _open_rtsp_persistently
    )
    OverwatchCommunication._connect_nats = (
        vision_service_nkey_file._connect_with_trimmed_seed
    )
    await vision_service.main()


if __name__ == "__main__":
    asyncio.run(main())
