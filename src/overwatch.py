"""
Constellation Overwatch ISR Detection System

Minimal orchestrator that coordinates all detection services.
Defaults to YOLOE C4ISR threat detection mode.
"""

# CRITICAL: Load .env BEFORE any imports that read environment variables
import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not available, skip

# CRITICAL: Set FFmpeg logging level BEFORE cv2 import to suppress RTSP warnings
# These must be set before OpenCV initializes FFmpeg
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # AV_LOG_QUIET
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from .config.models import DetectionMode
from .config.defaults import DEFAULT_CONFIG
from .utils.frame_encoder import calculate_frame_interval
from .utils.args import parse_arguments, validate_arguments
from .utils.device import get_device_fingerprint
from .utils.signals import setup_signal_handlers, is_shutdown_requested
from .utils.logging import setup_logging
from .services.detection import DetectorFactory
from .services.tracking.service import TrackingService
from .services.communication.service import OverwatchCommunication
from .services.video.service import VideoService

class OverwatchOrchestrator:
    """Main orchestrator for the Overwatch detection system."""

    def __init__(self):
        self.detector = None
        self.tracking_service = None
        self.communication = None
        self.video_service = None
        self.device_fingerprint = None
        self.detection_mode = None
        self.last_published_state = {}  # Track last published state per object ID

        # Configurable thresholds via environment variables
        self.movement_threshold = float(os.getenv('SIGINT_MOVEMENT_THRESHOLD', '0.05'))  # Default: 5% bbox movement
        self.confidence_threshold = float(os.getenv('SIGINT_CONFIDENCE_THRESHOLD', '0.1'))  # Default: 10% confidence change

        # Frame streaming configuration
        self.frame_stream_config = DEFAULT_CONFIG["frame_stream"]
        self.frame_interval = calculate_frame_interval(self.frame_stream_config["target_fps"])
        self.last_frame_publish_time = 0.0
    
    async def initialize(self, args) -> None:
        """Initialize all services."""
        # Parse detection mode
        self.detection_mode = DetectionMode(args.model)
        print(f"Detection Mode: {self.detection_mode.value.upper()}")
        
        # Setup video service first to determine device
        self.video_service = VideoService(args)
        if not self.video_service.open_video_stream():
            raise RuntimeError("Failed to open video stream")
        
        # Create detector
        self.detector = DetectorFactory.create_detector(self.detection_mode, args)
        await self.detector.load_model()
        
        # Generate device fingerprint
        selected_device = self.video_service.get_selected_device()
        self.device_fingerprint = get_device_fingerprint(
            "", "", selected_device, self.detector.get_model_info()
        )
        
        # Initialize communication
        self.communication = OverwatchCommunication()
        await self.communication.initialize(
            self.device_fingerprint,
            self.detection_mode.value,
            cli_org_id=args.org_id,
            cli_entity_id=args.entity_id
        )
        
        # Initialize tracking service
        self.tracking_service = TrackingService(self.detection_mode)
        
        # Setup video display
        camera_name = self.device_fingerprint["camera"]["name"]
        mode_name = self.detection_mode.value.replace('_', ' ').title()
        self.video_service.setup_display_window(camera_name, mode_name)

    def _should_publish_detection(self, detection_id: Any, detection: dict) -> bool:
        """
        Determine if detection should be published based on state changes.
        Only publish on:
        - First appearance (new object)
        - Significant movement (bbox change > threshold)
        - Confidence change > threshold
        - Label or threat level change
        """
        # New object - always publish
        if detection_id not in self.last_published_state:
            return True

        prev_state = self.last_published_state[detection_id]

        # Extract threat level from metadata or top-level (support both formats)
        curr_threat_level = detection.get("threat_level") or detection.get("metadata", {}).get("threat_level")
        prev_threat_level = prev_state.get("threat_level")

        # Check for label or threat level change
        if (detection.get("label") != prev_state.get("label") or
            curr_threat_level != prev_threat_level):
            return True

        # Check for significant confidence change
        conf_diff = abs(detection.get("confidence", 0) - prev_state.get("confidence", 0))
        if conf_diff > self.confidence_threshold:
            return True

        # Check for significant movement (bbox change)
        curr_bbox = detection.get("bbox", {})
        prev_bbox = prev_state.get("bbox", {})

        if curr_bbox and prev_bbox:
            # Calculate bbox center movement
            curr_center_x = (curr_bbox.get("x_min", 0) + curr_bbox.get("x_max", 0)) / 2
            curr_center_y = (curr_bbox.get("y_min", 0) + curr_bbox.get("y_max", 0)) / 2
            prev_center_x = (prev_bbox.get("x_min", 0) + prev_bbox.get("x_max", 0)) / 2
            prev_center_y = (prev_bbox.get("y_min", 0) + prev_bbox.get("y_max", 0)) / 2

            movement = ((curr_center_x - prev_center_x)**2 + (curr_center_y - prev_center_y)**2)**0.5

            if movement > self.movement_threshold:
                return True

        # No significant change - don't publish
        return False

    async def run_detection_loop(self, args) -> None:
        """Main detection loop."""
        frame_count = 0
        total_detections = 0
        total_kv_updates = 0

        print("Press 'q' to quit the stream.")
        print(f"Publishing to KV store key: {self.communication.entity_id} (consolidated EntityState)")
        print(f"Minimum frames for persistence: {args.min_frames}")
        print(f"\nSmart Publishing Thresholds:")
        print(f"  Movement threshold: {self.movement_threshold*100:.1f}% (SIGINT_MOVEMENT_THRESHOLD)")
        print(f"  Confidence threshold: {self.confidence_threshold*100:.1f}% (SIGINT_CONFIDENCE_THRESHOLD)")

        # Frame streaming info
        if self.frame_stream_config["enabled"]:
            codec = self.frame_stream_config.get("codec", "h264").upper()
            print(f"\nFrame Streaming: ENABLED ({codec})")
            print(f"  Target FPS: {self.frame_stream_config['target_fps']}")
            if codec == "H264":
                print(f"  Resolution: {self.frame_stream_config.get('h264_width', 1280)}x{self.frame_stream_config.get('h264_height', 720)}")
                print(f"  Bitrate: {self.frame_stream_config.get('h264_bitrate', '1500k')}")
                print(f"  Container: MPEG-TS (WebRTC compatible)")
            else:
                print(f"  JPEG Quality: {self.frame_stream_config['jpeg_quality']}")
                print(f"  Max Dimension: {self.frame_stream_config['max_dimension']}px")
            print(f"  Include Detections: {self.frame_stream_config['include_detections']}")
        else:
            print(f"\nFrame Streaming: DISABLED (set ENABLE_FRAME_STREAMING=true to enable)")
        print()
        
        try:
            while True:
                # Read frame (video service handles RTSP errors internally)
                ret, frame = self.video_service.read_frame()
                if not ret or frame is None:
                    # For RTSP, this means reconnection failed after max retries
                    if self.video_service.source_type in ["rtsp", "http"]:
                        print(f"Stream ended after {self.video_service.total_reconnects} reconnection attempts.")
                    else:
                        print("Error: Failed to capture frame.")
                    break

                frame_count += 1
                frame_timestamp = datetime.now(timezone.utc).isoformat()
                self.tracking_service.update_frame_count(frame_count)
                
                # Process frame
                detections, processed_frame = self.detector.process_frame(
                    frame, frame_timestamp, frame_count
                )
                
                total_detections += len(detections)
                current_ids = set()
                
                # Update tracking and publish detections
                for detection in detections:
                    detection_id = detection.get("track_id", f"{frame_count}_{len(current_ids)}")
                    current_ids.add(detection_id)

                    # Update tracking state
                    # Extract extra kwargs from metadata and other fields
                    extra_kwargs = {}
                    if "metadata" in detection:
                        extra_kwargs.update(detection["metadata"])
                    # Also include any top-level extra fields
                    for k, v in detection.items():
                        if k not in ('track_id', 'label', 'confidence', 'bbox', 'timestamp', 'model_type', 'metadata'):
                            extra_kwargs[k] = v

                    self.tracking_service.update_detection(
                        detection_id,
                        detection["label"],
                        detection["confidence"],
                        detection["bbox"],
                        detection["timestamp"],
                        **extra_kwargs
                    )

                    # Smart publishing: Only publish on significant state changes
                    if self._should_publish_detection(detection_id, detection):
                        await self.communication.publish_detection_event(detection)
                        # Update last published state (extract threat_level from metadata if nested)
                        threat_level = detection.get("threat_level") or detection.get("metadata", {}).get("threat_level")
                        self.last_published_state[detection_id] = {
                            "label": detection.get("label"),
                            "confidence": detection.get("confidence"),
                            "bbox": detection.get("bbox"),
                            "threat_level": threat_level
                        }
                
                # Mark inactive objects and clean up disappeared ones
                self.tracking_service.mark_inactive(current_ids)

                # Clean up state for objects that are no longer tracked
                disappeared_ids = set(self.last_published_state.keys()) - current_ids
                for disappeared_id in disappeared_ids:
                    del self.last_published_state[disappeared_id]

                # Clean up tracking ID service mappings to prevent memory leak
                self.detector.tracking_id_service.cleanup_stale_ids(current_ids)

                # Publish state to KV store for persistent objects
                persistent_objects = self.tracking_service.get_persistent_objects(args.min_frames)
                if persistent_objects:
                    analytics = self.tracking_service.get_analytics()
                    await self.communication.publish_state_to_kv(
                        self.tracking_service.state, analytics
                    )
                    total_kv_updates += 1
                    
                    # Special handling for C4ISR threat intelligence
                    if self.detection_mode == DetectionMode.YOLOE_C4ISR:
                        await self.communication.publish_threat_intelligence(
                            self.tracking_service.state
                        )
                
                # Add status overlay
                analytics = self.tracking_service.get_analytics()
                stats = {
                    'active_count': analytics.get('active_objects_count', 0),
                    'total_unique': analytics.get('total_unique_objects', 0)
                }

                processed_frame = self.detector.add_status_overlay(
                    processed_frame, self.device_fingerprint['device_id'], stats
                )

                # Stream frame if enabled and interval elapsed
                if self.frame_stream_config["enabled"]:
                    current_time = time.monotonic()
                    if current_time - self.last_frame_publish_time >= self.frame_interval:
                        # Choose frame to stream (with or without detections overlay)
                        frame_to_stream = processed_frame if self.frame_stream_config["include_detections"] else frame

                        # Publish frame (H.264/MPEG-TS or JPEG based on config)
                        await self.communication.publish_frame(
                            frame=frame_to_stream,
                            frame_number=frame_count,
                            timestamp=frame_timestamp,
                            detection_count=len(detections)
                        )

                        self.last_frame_publish_time = current_time

                # Display frame and check for quit
                if self.video_service.display_frame(processed_frame):
                    break
                
                # Check for shutdown signal
                if is_shutdown_requested():
                    break
        
        finally:
            await self._print_final_stats(frame_count, total_detections, total_kv_updates)
    
    async def _print_final_stats(self, frame_count: int, total_detections: int,
                                total_kv_updates: int) -> None:
        """Print final statistics."""
        print(f"\n=== Final Detection Statistics ===")
        print(f"Total frames processed: {frame_count}")
        print(f"Total detections: {total_detections}")

        analytics = self.tracking_service.get_analytics()
        print(f"Total unique objects: {analytics.get('total_unique_objects', 0)}")
        print(f"Total KV updates: {total_kv_updates}")

        if hasattr(self.tracking_service.state, 'threat_alerts'):
            print(f"Total threat alerts: {len(self.tracking_service.state.threat_alerts)}")

        # Frame streaming stats
        if self.communication:
            frame_stats = self.communication.get_frame_stream_stats()
            if frame_stats["enabled"]:
                codec = frame_stats.get('codec', 'unknown').upper()
                print(f"Frames streamed: {frame_stats['frames_published']} ({codec})")
                print(f"  Chunks published: {frame_stats.get('chunks_published', 0)}")
                if "h264" in frame_stats:
                    h264 = frame_stats["h264"]
                    mb_encoded = h264.get("bytes_encoded", 0) / (1024 * 1024)
                    print(f"  H.264 encoded: {mb_encoded:.2f} MB @ {h264.get('avg_bitrate_kbps', 0):.0f} kbps")
                    print(f"  Encoder FPS: {h264.get('avg_fps', 0):.1f}")

                # Video transmission format summary for backend verification
                print(f"\n--- VIDEO TRANSMISSION FORMAT ---")
                print(f"Subject: {frame_stats.get('video_subject')}")
                if codec == "H264":
                    print(f"Content-Type: video/mp2t")
                    print(f"Container: MPEG-TS")
                    print(f"Codec: H.264 (libx264)")
                    print(f"Profile: baseline")
                    print(f"Preset: ultrafast")
                    print(f"Tune: zerolatency")
                    print(f"Resolution: {h264.get('resolution', 'N/A')}")
                    print(f"Chunk size: 1316 bytes (7 x 188 TS packets)")
                    print(f"Headers: X-Frame-Type (IDR|P), X-Sequence, X-PTS")
                else:
                    print(f"Content-Type: image/jpeg")
                    print(f"Format: JPEG frames")
                    print(f"Headers: X-Sequence, Frame-Number")
                print(f"---------------------------------")

        print("=" * 35)
    
    async def cleanup(self) -> None:
        """Cleanup all services."""
        if self.communication:
            final_analytics = None
            if self.tracking_service:
                final_analytics = self.tracking_service.get_analytics()
            await self.communication.cleanup(final_analytics)
        
        if self.video_service:
            self.video_service.cleanup()

async def main():
    """Main entry point."""
    # Setup
    setup_logging()
    args = validate_arguments(parse_arguments())
    
    # Handle special modes that don't need full initialization
    if hasattr(args, 'list_models') and args.list_models:
        from .services.detection.factory import DetectorFactory
        DetectorFactory.list_modes()
        return
    
    # Create orchestrator
    orchestrator = OverwatchOrchestrator()
    
    # Setup signal handlers
    setup_signal_handlers(orchestrator.cleanup)
    
    try:
        # Initialize and run
        await orchestrator.initialize(args)
        await orchestrator.run_detection_loop(args)
    finally:
        await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())