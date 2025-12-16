"""H.264 video encoder for WebRTC-compatible streaming via FFmpeg.

Optimized MPEG-TS output for NATS JetStream → WebRTC pipeline:
- libx264 baseline profile (universal browser support)
- ultrafast preset + zerolatency tune (minimal latency)
- UDP-friendly chunk sizes (7 × 188 = 1316 bytes)
- Continuous keyframe-aware streaming

Bandwidth: ~150-300 KB/s vs ~1.5-2.5 MB/s for MJPEG (~90% savings)
"""

import subprocess
import threading
import queue
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import time


@dataclass
class EncodedChunk:
    """MPEG-TS chunk with metadata."""
    data: bytes
    sequence: int
    is_keyframe: bool
    pts: int  # Presentation timestamp


class H264Encoder:
    """
    Streaming H.264 encoder using persistent FFmpeg subprocess.

    Designed for real-time OpenCV → NATS → WebRTC pipeline.
    """

    # MPEG-TS packet size
    TS_PACKET_SIZE = 188
    # Optimal chunk: 7 packets fits in UDP datagram
    CHUNK_PACKETS = 7
    CHUNK_SIZE = TS_PACKET_SIZE * CHUNK_PACKETS  # 1316 bytes

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 15,
        bitrate: str = "1500k",
        gop_size: int = 30,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.gop_size = gop_size

        self._process: Optional[subprocess.Popen] = None
        self._output_queue: queue.Queue[EncodedChunk] = queue.Queue(maxsize=120)
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False

        # Stats
        self._frame_count = 0
        self._chunk_sequence = 0
        self._bytes_encoded = 0
        self._start_time: Optional[float] = None
        self._input_resolution: Tuple[int, int] = (0, 0)

        # Keyframe tracking
        self._last_keyframe_sequence = 0

    def start(self, input_width: int, input_height: int) -> bool:
        """Start FFmpeg encoder process with optimized settings."""
        if self._running:
            return True

        self._input_resolution = (input_width, input_height)

        # Build optimized FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            # Input: raw BGR24 frames from OpenCV
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{input_width}x{input_height}",
            "-r", str(self.fps),
            "-i", "pipe:0",
            # Video filter: scale to output resolution
            "-vf", f"scale={self.width}:{self.height}:flags=fast_bilinear",
            # H.264 encoding - WebRTC optimized
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-level", "3.1",
            "-pix_fmt", "yuv420p",
            # Bitrate control
            "-b:v", self.bitrate,
            "-maxrate", self.bitrate,
            "-bufsize", f"{int(self.bitrate.replace('k', '000')) // 2}",
            # GOP structure - keyframe every N frames
            "-g", str(self.gop_size),
            "-keyint_min", str(self.gop_size),
            "-sc_threshold", "0",
            # Disable B-frames for lower latency
            "-bf", "0",
            # MPEG-TS muxer settings
            "-f", "mpegts",
            "-mpegts_flags", "resend_headers",
            "-muxrate", "0",  # Variable bitrate
            "-pcr_period", "20",
            # Output to stdout
            "pipe:1",
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            self._running = True
            self._start_time = time.monotonic()
            self._frame_count = 0
            self._chunk_sequence = 0
            self._bytes_encoded = 0

            # Start background reader thread
            self._reader_thread = threading.Thread(
                target=self._read_output_loop,
                daemon=True,
                name="H264EncoderReader"
            )
            self._reader_thread.start()

            return True

        except FileNotFoundError:
            print("Error: FFmpeg not found. Install: brew install ffmpeg")
            return False
        except Exception as e:
            print(f"Error starting H.264 encoder: {e}")
            return False

    def _detect_keyframe_in_chunk(self, chunk_data: bytes) -> bool:
        """
        Detect if chunk contains H.264 IDR frame by parsing MPEG-TS packets.

        Returns True if any TS packet in the chunk contains an IDR NAL unit (type 5).
        """
        # Parse 188-byte TS packets in the chunk
        for i in range(0, len(chunk_data), self.TS_PACKET_SIZE):
            packet = chunk_data[i:i + self.TS_PACKET_SIZE]
            if len(packet) < self.TS_PACKET_SIZE:
                continue

            # Check sync byte (0x47)
            if packet[0] != 0x47:
                continue

            # Parse adaptation field and payload
            adaptation_field_control = (packet[3] >> 4) & 0x03
            has_payload = adaptation_field_control in (0x01, 0x03)

            if not has_payload:
                continue

            # Calculate payload offset
            offset = 4
            if adaptation_field_control in (0x02, 0x03):  # Has adaptation field
                adaptation_length = packet[4]
                offset += 1 + adaptation_length

            if offset >= len(packet):
                continue

            # Search for H.264 NAL start codes (0x00 0x00 0x01 or 0x00 0x00 0x00 0x01)
            payload = packet[offset:]
            for j in range(len(payload) - 4):
                if payload[j:j+3] == b'\x00\x00\x01' or payload[j:j+4] == b'\x00\x00\x00\x01':
                    # Found NAL start code, check NAL unit type
                    nal_offset = j + 3 if payload[j:j+3] == b'\x00\x00\x01' else j + 4
                    if nal_offset < len(payload):
                        nal_unit_type = payload[nal_offset] & 0x1F
                        if nal_unit_type == 5:  # IDR frame
                            return True

        return False

    def _read_output_loop(self) -> None:
        """Background thread: read MPEG-TS chunks from FFmpeg."""
        while self._running and self._process and self._process.stdout:
            try:
                chunk_data = self._process.stdout.read(self.CHUNK_SIZE)
                if not chunk_data:
                    if self._process.poll() is not None:
                        break
                    continue

                self._chunk_sequence += 1
                self._bytes_encoded += len(chunk_data)

                # Detect keyframe by parsing MPEG-TS/H.264 NAL units
                is_keyframe = self._detect_keyframe_in_chunk(chunk_data)

                chunk = EncodedChunk(
                    data=chunk_data,
                    sequence=self._chunk_sequence,
                    is_keyframe=is_keyframe,
                    pts=int(self._chunk_sequence * (90000 / self.fps)),  # Use sequence for PTS
                )

                # Non-blocking put with overflow handling
                try:
                    self._output_queue.put_nowait(chunk)
                except queue.Full:
                    # Drop oldest chunk (real-time priority)
                    try:
                        self._output_queue.get_nowait()
                        self._output_queue.put_nowait(chunk)
                    except queue.Empty:
                        pass

            except Exception:
                if self._running:
                    continue
                break

    def encode_frame(self, frame: np.ndarray) -> Tuple[List[EncodedChunk], Dict[str, Any]]:
        """
        Feed frame to encoder and collect available MPEG-TS chunks.

        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            (list of EncodedChunks, metadata dict)
        """
        h, w = frame.shape[:2]

        # Start or restart encoder if needed
        if not self._running or (w, h) != self._input_resolution:
            if self._running:
                self.stop()
            if not self.start(w, h):
                return [], {"error": "Failed to start encoder"}

        metadata = {
            "width": self.width,
            "height": self.height,
            "original_width": w,
            "original_height": h,
            "format": "h264",
            "container": "mpegts",
            "profile": "baseline",
            "bitrate": self.bitrate,
            "fps": self.fps,
            "gop_size": self.gop_size,
        }

        try:
            # Write frame to FFmpeg stdin
            frame_bytes = frame.tobytes()
            self._process.stdin.write(frame_bytes)
            self._process.stdin.flush()
            self._frame_count += 1

            # Collect all available chunks
            chunks: List[EncodedChunk] = []
            while True:
                try:
                    chunks.append(self._output_queue.get_nowait())
                except queue.Empty:
                    break

            metadata["frame_number"] = self._frame_count
            metadata["chunks_produced"] = len(chunks)
            metadata["total_bytes"] = sum(len(c.data) for c in chunks)

            return chunks, metadata

        except BrokenPipeError:
            self._running = False
            return [], {"error": "Encoder pipe broken"}
        except Exception as e:
            return [], {"error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get encoder statistics."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        return {
            "frames_encoded": self._frame_count,
            "chunks_produced": self._chunk_sequence,
            "bytes_encoded": self._bytes_encoded,
            "elapsed_seconds": round(elapsed, 2),
            "avg_fps": round(self._frame_count / elapsed, 1) if elapsed > 0 else 0,
            "avg_bitrate_kbps": round((self._bytes_encoded * 8 / 1000) / elapsed, 1) if elapsed > 0 else 0,
            "resolution": f"{self.width}x{self.height}",
            "input_resolution": f"{self._input_resolution[0]}x{self._input_resolution[1]}",
            "running": self._running,
            "queue_size": self._output_queue.qsize(),
        }

    def stop(self) -> None:
        """Stop encoder process gracefully."""
        self._running = False

        if self._process:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=1)
            except Exception:
                pass
            self._process = None

        # Drain remaining chunks
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                break

    def __del__(self):
        self.stop()
