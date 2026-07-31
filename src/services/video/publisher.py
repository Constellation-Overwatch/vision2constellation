"""Non-blocking RTSP publisher backed by an FFmpeg subprocess."""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from argparse import Namespace
from typing import Any, Callable, Optional

from ...utils.urls import redact_stream_url


def build_ffmpeg_command(
    url: str,
    width: int,
    height: int,
    fps: float,
    bitrate: str,
    encoder: str,
) -> list[str]:
    """Build an FFmpeg command that publishes raw BGR frames over RTSP/TCP."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:g}",
        "-i",
        "pipe:0",
        "-an",
    ]

    if encoder == "h264_nvenc":
        command.extend([
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",
            "-tune",
            "ll",
            "-b:v",
            bitrate,
        ])
    elif encoder == "libx264":
        command.extend([
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-b:v",
            bitrate,
        ])
    else:
        raise ValueError(f"Unsupported H.264 encoder: {encoder}")

    gop_size = max(2, round(fps * 2))
    command.extend([
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(gop_size),
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        "-muxdelay",
        "0.1",
        url,
    ])
    return command


class RTSPPublisher:
    """Publish annotated frames without allowing network stalls to block inference."""

    _STOP = object()

    def __init__(
        self,
        url: str,
        fps: float = 15.0,
        bitrate: str = "4M",
        encoder: str = "h264_nvenc",
        reconnect_delay: float = 3.0,
        queue_size: int = 2,
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ):
        self.url = url
        self.fps = fps
        self.bitrate = bitrate
        self.encoder = encoder
        self.reconnect_delay = reconnect_delay
        self.process_factory = process_factory

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None
        self._dimensions: Optional[tuple[int, int]] = None
        self._retry_after = 0.0
        self._last_enqueue_time = 0.0
        self._lock = threading.Lock()

        self.frames_queued = 0
        self.frames_published = 0
        self.frames_dropped = 0
        self.reconnects = 0

    @classmethod
    def from_args(cls, args: Namespace) -> "RTSPPublisher":
        """Create a publisher from validated CLI arguments."""
        return cls(
            url=args.rtsp_output,
            fps=args.output_fps,
            bitrate=args.output_bitrate,
            encoder=args.output_encoder,
            reconnect_delay=args.output_reconnect_delay,
        )

    def start(self) -> None:
        """Start the worker; the FFmpeg process is created lazily on first frame."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="rtsp-publisher",
            daemon=True,
        )
        self._thread.start()
        print(
            "RTSP output enabled: "
            f"{redact_stream_url(self.url)} "
            f"({self.encoder}, {self.fps:g} FPS, {self.bitrate}, TCP)"
        )

    def publish_frame(self, frame: Any) -> bool:
        """Queue the newest frame, dropping stale work when the queue is full."""
        if frame is None or self._stop_event.is_set():
            return False

        now = time.monotonic()
        min_interval = 1.0 / self.fps
        with self._lock:
            if now - self._last_enqueue_time < min_interval:
                self.frames_dropped += 1
                return False
            self._last_enqueue_time = now

        frame_copy = frame.copy()
        try:
            self._queue.put_nowait(frame_copy)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self.frames_dropped += 1
            self._queue.put_nowait(frame_copy)

        self.frames_queued += 1
        return True

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if item is self._STOP:
                    return
                self._write_frame(item)
            finally:
                self._queue.task_done()

    def _write_frame(self, frame: Any) -> None:
        if time.monotonic() < self._retry_after:
            self.frames_dropped += 1
            return

        height, width = frame.shape[:2]
        dimensions = (width, height)

        if self._process and (
            self._process.poll() is not None or self._dimensions != dimensions
        ):
            self._stop_process()

        if self._process is None:
            self._start_process(width, height)

        try:
            if self._process is None or self._process.stdin is None:
                raise BrokenPipeError("FFmpeg stdin is unavailable")
            self._process.stdin.write(frame.tobytes())
            self.frames_published += 1
        except (BrokenPipeError, OSError, ValueError) as error:
            print(
                "RTSP output disconnected "
                f"({type(error).__name__}); retrying in {self.reconnect_delay:g}s"
            )
            self.reconnects += 1
            self.frames_dropped += 1
            self._retry_after = time.monotonic() + self.reconnect_delay
            self._stop_process()

    def _start_process(self, width: int, height: int) -> None:
        command = build_ffmpeg_command(
            self.url,
            width,
            height,
            self.fps,
            self.bitrate,
            self.encoder,
        )
        try:
            self._process = self.process_factory(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            )
            self._dimensions = (width, height)
        except (OSError, ValueError) as error:
            print(
                "Could not start FFmpeg RTSP output "
                f"({type(error).__name__}); retrying in {self.reconnect_delay:g}s"
            )
            self.reconnects += 1
            self._retry_after = time.monotonic() + self.reconnect_delay
            self._process = None
            self._dimensions = None

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        self._dimensions = None
        if process is None:
            return

        if process.stdin:
            try:
                process.stdin.close()
            except OSError:
                pass

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def get_stats(self) -> dict[str, int | str]:
        """Return a snapshot suitable for shutdown reporting."""
        return {
            "endpoint": redact_stream_url(self.url),
            "frames_queued": self.frames_queued,
            "frames_published": self.frames_published,
            "frames_dropped": self.frames_dropped,
            "reconnects": self.reconnects,
        }

    def stop(self) -> None:
        """Stop the publisher and release FFmpeg without blocking indefinitely."""
        self._stop_event.set()
        self._stop_process()
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self._queue.put_nowait(self._STOP)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        print("RTSP output resources cleaned up")
