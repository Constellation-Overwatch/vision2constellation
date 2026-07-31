"""Focused tests for the RTSP output worker."""

import io

from src.services.video.publisher import RTSPPublisher, build_ffmpeg_command
from src.utils.urls import redact_stream_url, stream_identity


def test_redact_stream_url_removes_credentials_and_query():
    url = "rtsp://user:secret@example.test:8554/live/stream?token=hidden"
    assert redact_stream_url(url) == "rtsp://example.test:8554/live/stream"


def test_stream_identity_ignores_credentials():
    first = "rtsp://reader:one@example.test:8554/live/stream"
    second = "rtsp://publisher:two@example.test:8554/live/stream"
    assert stream_identity(first) == stream_identity(second)


def test_build_ffmpeg_command_uses_h264_and_rtsp_tcp():
    command = build_ffmpeg_command(
        "rtsp://example.test:8554/vision/c4threatisr",
        1280,
        720,
        15,
        "4M",
        "h264_nvenc",
    )
    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-rtsp_transport") + 1] == "tcp"
    assert "yuv420p" in command


class _FakeProcess:
    def __init__(self):
        self.stdin = io.BytesIO()
        self.terminated = False

    def poll(self):
        return None if not self.terminated else 0

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


def test_publisher_command_is_created_lazily():
    calls = []

    def factory(command, **kwargs):
        calls.append(command)
        return _FakeProcess()

    publisher = RTSPPublisher(
        "rtsp://example.test:8554/vision/c4threatisr",
        process_factory=factory,
    )
    assert calls == []
    publisher._start_process(640, 480)
    assert len(calls) == 1
    publisher.stop()
