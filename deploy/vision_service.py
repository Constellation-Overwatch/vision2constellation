#!/usr/bin/env python3
"""Headless C4ISR service adapter with annotated RTSP output."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import src.overwatch as upstream
from src.services.video.publisher import RTSPPublisher
from src.services.video.service import VideoService
from src.utils.args import parse_arguments, validate_arguments
from src.utils.urls import redact_stream_url


class HeadlessRTSPVideoService(VideoService):
    """RTSP-only video service that never creates a desktop window."""

    def _setup_rtsp_source(self):
        print("\n=== RTSP Stream Mode (Headless) ===")
        print(f"Connecting to: {redact_stream_url(self.args.rtsp)}")
        print(
            "Resilience: "
            f"max_failures={self.max_consecutive_failures}, "
            f"reconnect_delay={self.reconnect_delay}s"
        )
        print("===================================\n")
        return self.args.rtsp, "rtsp", None

    def open_video_stream(self) -> bool:
        self.video_source, self.source_type, self.selected_device = (
            self.determine_video_source()
        )
        if self.source_type != "rtsp":
            raise RuntimeError("The headless deployment requires an RTSP input")

        self.cap = self._open_stream_with_retries()
        if self.cap is None or not self.cap.isOpened():
            print(
                "Error: Could not open RTSP source: "
                f"{redact_stream_url(self.video_source)}"
            )
            return False

        self._apply_stream_optimizations()
        return True

    def setup_display_window(self, camera_name: str, mode_name: str) -> None:
        print("OpenCV display disabled (headless service mode)")

    def display_frame(self, frame) -> bool:
        return False

    def cleanup(self) -> None:
        if self.cap:
            self.cap.release()
        print("Headless video resources cleaned up")


class VisionServiceOrchestrator(upstream.OverwatchOrchestrator):
    """Add a bounded RTSP output worker without forking upstream detection logic."""

    def __init__(self, output_url: str):
        super().__init__()
        self.output_url = output_url
        self.rtsp_publisher = None

    async def initialize(self, args) -> None:
        await super().initialize(args)

        self.rtsp_publisher = RTSPPublisher(
            url=self.output_url,
            fps=float(os.getenv("VISION_OUTPUT_FPS", "15")),
            bitrate=os.getenv("VISION_OUTPUT_BITRATE", "4M"),
            encoder=os.getenv("VISION_OUTPUT_ENCODER", "h264_nvenc"),
            reconnect_delay=float(
                os.getenv("VISION_OUTPUT_RECONNECT_DELAY", "3")
            ),
        )
        self.rtsp_publisher.start()

        add_status_overlay = self.detector.add_status_overlay

        def publish_annotated_frame(frame, device_id, stats):
            annotated = add_status_overlay(frame, device_id, stats)
            self.rtsp_publisher.publish_frame(annotated)
            return annotated

        self.detector.add_status_overlay = publish_annotated_frame

    async def _print_final_stats(
        self,
        frame_count: int,
        total_detections: int,
        total_kv_updates: int,
    ) -> None:
        await super()._print_final_stats(
            frame_count,
            total_detections,
            total_kv_updates,
        )
        if self.rtsp_publisher:
            stats = self.rtsp_publisher.get_stats()
            print("\n=== RTSP Output Statistics ===")
            print(f"Endpoint: {stats['endpoint']}")
            print(f"Frames published: {stats['frames_published']}")
            print(f"Frames dropped: {stats['frames_dropped']}")
            print(f"Reconnects: {stats['reconnects']}")
            print("=" * 30)

    async def cleanup(self) -> None:
        publisher = self.rtsp_publisher
        self.rtsp_publisher = None
        if publisher:
            publisher.stop()
        await super().cleanup()


def build_upstream_args():
    """Use upstream parsing without placing secret RTSP URLs in process argv."""
    input_url = os.environ["VISION_INPUT_RTSP_URL"]
    model = os.getenv("VISION_MODEL", "yoloe_c4isr")
    confidence = os.getenv("VISION_CONFIDENCE", "0.25")

    original_argv = sys.argv
    try:
        sys.argv = [
            original_argv[0],
            "--model",
            model,
            "--rtsp",
            input_url,
            "--conf",
            confidence,
        ]
        return validate_arguments(parse_arguments())
    finally:
        sys.argv = original_argv


async def main() -> None:
    upstream.setup_logging()
    args = build_upstream_args()

    # The upstream orchestrator resolves this module global at initialization.
    upstream.VideoService = HeadlessRTSPVideoService

    orchestrator = VisionServiceOrchestrator(
        output_url=os.environ["VISION_OUTPUT_RTSP_URL"]
    )
    upstream.setup_signal_handlers(orchestrator.cleanup)

    try:
        await orchestrator.initialize(args)
        await orchestrator.run_detection_loop(args)
    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
