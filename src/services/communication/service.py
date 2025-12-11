"""Communication service for NATS/JetStream and KV store operations."""

import json
import nats
import numpy as np
from nats.js.api import KeyValueConfig
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from ...config.defaults import DEFAULT_CONFIG
from ...utils.constellation import get_constellation_ids
from ...utils.h264_encoder import H264Encoder
from .publisher import ConstellationPublisher

class OverwatchCommunication:
    """Service for managing NATS/JetStream communications."""

    def __init__(self):
        self.nc: Optional[nats.NATS] = None
        self.js = None
        self.kv = None
        self.organization_id: Optional[str] = None
        self.entity_id: Optional[str] = None
        self.subject: Optional[str] = None
        self.stream_name: Optional[str] = None
        self.device_fingerprint: Optional[Dict] = None
        self.publisher: Optional[ConstellationPublisher] = None
        self._entity_state_cache: Optional[Dict] = None  # Cache for entity state
        self.detection_mode: Optional[str] = None
        
        # Idempotency tracking for detection events
        self._published_detection_hashes: set = set()
        self._state_update_lock: bool = False

        # Configuration
        self.nats_config = DEFAULT_CONFIG["nats"]
        self.frame_stream_config = DEFAULT_CONFIG["frame_stream"]

        # Video frame streaming
        self.video_subject: Optional[str] = None
        self.frame_stream_enabled = self.frame_stream_config["enabled"]
        self._frame_count = 0
        self._chunk_sequence = 0  # MPEG-TS chunk sequence number
        self._h264_encoder: Optional[H264Encoder] = None
        self._codec = self.frame_stream_config.get("codec", "h264")

    async def initialize(
        self,
        device_fingerprint: Dict[str, Any],
        detection_mode: str = "detection",
        cli_org_id: Optional[str] = None,
        cli_entity_id: Optional[str] = None
    ) -> None:
        """Initialize NATS connection and setup streams."""
        self.device_fingerprint = device_fingerprint
        self.detection_mode = detection_mode

        # Get constellation identifiers (CLI args override env vars)
        self.organization_id, self.entity_id = get_constellation_ids(cli_org_id, cli_entity_id)

        # Initialize publisher abstraction with identity context
        self.publisher = ConstellationPublisher(
            organization_id=self.organization_id,
            entity_id=self.entity_id,
            device_id=device_fingerprint['device_id']
        )

        # Construct subject and stream names
        self.subject = f"{self.nats_config['subject_root']}.{self.organization_id}.{self.entity_id}"
        self.stream_name = self.nats_config["stream_name"]

        # Construct video subject if frame streaming is enabled
        # Subject format: constellation.video.{entity_id} (per spec)
        if self.frame_stream_enabled:
            self.video_subject = f"{self.frame_stream_config['subject_root']}.{self.entity_id}"

            # Initialize H.264 encoder if using h264 codec
            if self._codec == "h264":
                self._h264_encoder = H264Encoder(
                    width=self.frame_stream_config.get("h264_width", 1280),
                    height=self.frame_stream_config.get("h264_height", 720),
                    fps=self.frame_stream_config.get("target_fps", 15),
                    bitrate=self.frame_stream_config.get("h264_bitrate", "1500k"),
                    gop_size=self.frame_stream_config.get("h264_gop_size", 30),
                )

        print(f"Configured NATS subject: {self.subject}")
        print(f"Configured stream name: {self.stream_name}")
        print(f"Configured KV store: {self.nats_config['kv_store_name']}")
        if self.frame_stream_enabled:
            print(f"Configured video subject: {self.video_subject}")
            print(f"Configured video stream: {self.frame_stream_config['stream_name']}")
            print(f"Configured video codec: {self._codec.upper()}")
            if self._codec == "h264":
                print(f"  H.264: {self.frame_stream_config.get('h264_width')}x{self.frame_stream_config.get('h264_height')} @ {self.frame_stream_config.get('h264_bitrate')}")
        print()
        
        # Connect to NATS
        await self._connect_nats()
        await self._setup_jetstream()
        await self._setup_kv_store()
        await self._publish_bootsequence()
    
    async def _connect_nats(self) -> None:
        """Connect to NATS server with optional token authentication."""
        print(f"Attempting to connect to NATS at: {self.nats_config['url']}")

        connect_opts = {"servers": [self.nats_config["url"]]}

        # Token-based authentication
        if self.nats_config.get("auth_token"):
            connect_opts["token"] = self.nats_config["auth_token"]
            print("Using token-based authentication")

        self.nc = await nats.connect(**connect_opts)
        print("Connected to NATS server")
    
    async def _setup_jetstream(self) -> None:
        """Setup JetStream context."""
        self.js = self.nc.jetstream()
        
        # Verify stream exists
        try:
            stream_info = await self.js.stream_info(self.stream_name)
            print(f"Connected to JetStream stream: {self.stream_name}")
            print(f"Stream subjects: {stream_info.config.subjects}")
        except Exception as e:
            print(f"Warning: Stream {self.stream_name} not found.")
            print(f"Error: {e}")
    
    async def _setup_kv_store(self) -> None:
        """Setup Key-Value store."""
        kv_store_name = self.nats_config["kv_store_name"]
        
        try:
            self.kv = await self.js.create_key_value(config=KeyValueConfig(
                bucket=kv_store_name,
                description="Constellation global state for object tracking and threat intelligence",
                history=10,          # Keep last 10 revisions for debugging/rollback
                ttl=86400,          # 24 hours (increased from 1 hour for operational visibility)
                max_value_size=1048576  # 1MB max size for large detection batches
            ))
            print(f"Created/connected to KV store: {kv_store_name}")
        except Exception as e:
            try:
                self.kv = await self.js.key_value(kv_store_name)
                print(f"Connected to existing KV store: {kv_store_name}")
            except Exception as e2:
                print(f"Error accessing KV store: {e2}")
                print("Continuing without KV store")
    
    async def _publish_bootsequence(self) -> None:
        """Publish bootsequence event using publisher abstraction."""
        bootsequence_message = self.publisher.build_bootsequence(
            fingerprint=self.device_fingerprint,
            message=f"Overwatch ISR component initialized: {self.device_fingerprint['component']['type']}"
        )
        
        try:
            ack = await self.js.publish(
                self.subject,
                json.dumps(bootsequence_message).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Event-Type": "bootsequence"
                }
            )
            print(f"Published bootsequence event to JetStream")
            print(f"  Stream: {ack.stream}, Seq: {ack.seq}")
        except Exception as e:
            print(f"Error publishing bootsequence: {e}")
    
    async def _get_entity_state(self) -> Dict[str, Any]:
        """
        Get current EntityState from KV or return base structure.
        Uses in-memory cache to avoid excessive KV reads.
        """
        if self._entity_state_cache:
            return self._entity_state_cache

        try:
            # Try to get existing state from KV
            entry = await self.kv.get(self.entity_id)
            if entry and entry.value:
                self._entity_state_cache = json.loads(entry.value.decode())
                return self._entity_state_cache
        except Exception:
            pass  # State doesn't exist yet, will create new

        # Return base EntityState structure
        self._entity_state_cache = {
            "entity_id": self.entity_id,
            "org_id": self.organization_id,
            "device_id": self.device_fingerprint['device_id'],
            "entity_type": "isr_sensor",
            "status": "active",
            "is_live": True,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "mission": {
                "mode": self.detection_mode or "detection",
                "status": "operational",
                "started_at": datetime.now(timezone.utc).isoformat()
            },
            "detections": {},
            "analytics": {},
            "c4isr": {}
        }
        return self._entity_state_cache

    async def _update_entity_state(self, subsignal: str, data: Dict[str, Any]) -> None:
        """
        Update a specific subsignal scope within EntityState and persist to KV.

        Args:
            subsignal: The scope to update ('detections', 'analytics', 'c4isr')
            data: The data to merge into that scope
        """
        try:
            # Get current state
            entity_state = await self._get_entity_state()

            # Update the specific subsignal scope
            entity_state[subsignal] = data
            entity_state["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Write consolidated state to single KV key
            await self.kv.put(
                self.entity_id,
                json.dumps(entity_state).encode()
            )

            # Update cache
            self._entity_state_cache = entity_state

        except Exception as e:
            print(f"Error updating entity state [{subsignal}]: {e}")

    def _calculate_detection_hash(self, detection_data: Dict[str, Any]) -> str:
        """
        Calculate a hash for detection data to prevent duplicate publishing.
        
        Args:
            detection_data: Detection data dictionary
            
        Returns:
            str: Hash representing the detection
        """
        # Create a signature from key detection properties
        track_id = detection_data.get("track_id", "")
        label = detection_data.get("label", "")
        confidence = round(detection_data.get("confidence", 0), 3)  # Round to avoid float precision issues
        
        bbox = detection_data.get("bbox", {})
        bbox_signature = f"{round(bbox.get('x_min', 0), 3)},{round(bbox.get('y_min', 0), 3)},{round(bbox.get('x_max', 0), 3)},{round(bbox.get('y_max', 0), 3)}"
        
        threat_level = detection_data.get("threat_level") or detection_data.get("metadata", {}).get("threat_level", "")
        
        # Create detection signature
        signature = f"{track_id}:{label}:{confidence}:{bbox_signature}:{threat_level}"
        
        # Return hash
        import hashlib
        return hashlib.sha256(signature.encode()).hexdigest()[:16]

    async def publish_detection_event(self, detection_data: Dict[str, Any]) -> None:
        """Publish detection event to JetStream using publisher abstraction with idempotency."""
        if not self.js:
            return

        try:
            # Calculate detection hash for idempotency
            detection_hash = self._calculate_detection_hash(detection_data)
            
            # Skip if already published
            if detection_hash in self._published_detection_hashes:
                return
            
            message = self.publisher.build_detection(detection_data)
            
            headers = {
                "Content-Type": "application/json",
                "Event-Type": "detection",
                "Device-ID": self.device_fingerprint['device_id'],
                "Detection-Hash": detection_hash  # Include hash in headers for debugging
            }
            
            # Add threat level header for C4ISR mode
            threat_level = detection_data.get("threat_level") or detection_data.get("metadata", {}).get("threat_level")
            if threat_level:
                headers["Threat-Level"] = threat_level
                headers["Label"] = detection_data.get("label", "unknown")
            
            await self.js.publish(
                self.subject,
                json.dumps(message).encode(),
                headers=headers
            )
            
            # Track published hash
            self._published_detection_hashes.add(detection_hash)
            
            # Limit hash cache size to prevent memory growth
            if len(self._published_detection_hashes) > 1000:
                # Remove oldest 200 entries (FIFO)
                hashes_to_remove = list(self._published_detection_hashes)[:200]
                for h in hashes_to_remove:
                    self._published_detection_hashes.discard(h)
                    
        except Exception as e:
            print(f"Error publishing detection event: {e}")
    
    async def publish_state_to_kv(self, tracking_state: Any, analytics: Dict[str, Any]) -> None:
        """
        Publish tracking state to consolidated EntityState in KV store.
        Updates both 'detections' and 'analytics' subsignals.
        """
        if not self.kv or not self.entity_id or self._state_update_lock:
            return

        # Prevent concurrent state updates
        self._state_update_lock = True
        
        try:
            # Prepare detections data
            detections_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "objects": {}
            }

            # Add tracking objects if available
            if hasattr(tracking_state, 'get_persistent_objects'):
                persistent_objects = tracking_state.get_persistent_objects(min_frames=3)
                detections_data["objects"] = {
                    str(tid): {
                        "track_id": obj.get("track_id", obj.get("segment_id", tid)),
                        "label": obj.get("label", "segment"),
                        "category": obj.get("category"),  # RT-DETR category (person, vehicle, etc.)
                        "priority": obj.get("priority"),  # RT-DETR priority level
                        "first_seen": obj["first_seen"],
                        "last_seen": obj["last_seen"],
                        "frame_count": obj["frame_count"],
                        "avg_confidence": obj.get("avg_confidence", 0),
                        "is_active": obj["is_active"],
                        "threat_level": obj.get("threat_level"),  # C4ISR threat level
                        "suspicious_indicators": obj.get("suspicious_indicators", []),
                        "area": obj.get("area"),
                        "current_bbox": obj.get("bbox_history", [])[-1] if obj.get("bbox_history") else obj.get("bbox")
                    }
                    for tid, obj in persistent_objects.items()
                }

            # Prepare analytics data
            analytics_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": analytics
            }

            # Update EntityState with both subsignals
            await self._update_entity_state("detections", detections_data)
            await self._update_entity_state("analytics", analytics_data)

        except Exception as e:
            print(f"Error publishing state to KV: {e}")
        finally:
            self._state_update_lock = False
    
    async def publish_threat_intelligence(self, tracking_state: Any) -> None:
        """
        Publish C4ISR threat intelligence to consolidated EntityState.
        Updates 'c4isr' subsignal and enriches 'analytics' with C4ISR summary.
        """
        if not self.kv or not hasattr(tracking_state, 'threat_alerts'):
            return

        try:
            analytics = tracking_state.get_analytics()

            # Prepare C4ISR threat intelligence data
            c4isr_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mission": "C4ISR",
                "threat_intelligence": {
                    "threat_summary": {
                        "total_threats": analytics.get("active_threat_count", 0),
                        "threat_distribution": analytics.get("threat_distribution", {}),
                        "alert_level": "HIGH" if analytics.get("threat_distribution", {}).get("HIGH_THREAT", 0) > 0 else "NORMAL"
                    },
                    "threat_alerts": analytics.get("threat_alerts", [])
                }
            }

            # Update C4ISR subsignal in EntityState
            await self._update_entity_state("c4isr", c4isr_data)

            # Also update analytics.c4isr_summary for backward compatibility
            entity_state = await self._get_entity_state()
            if "analytics" not in entity_state:
                entity_state["analytics"] = {}
            entity_state["analytics"]["c4isr_summary"] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **analytics
            }
            entity_state["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Write updated state
            await self.kv.put(self.entity_id, json.dumps(entity_state).encode())
            self._entity_state_cache = entity_state

        except Exception as e:
            print(f"Error publishing threat intelligence to KV: {e}")

    async def publish_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: str,
        detection_count: int = 0
    ) -> bool:
        """
        Publish video frame to JetStream using configured codec.

        H.264/MPEG-TS (default): WebRTC-compatible, ~90% bandwidth savings
        JPEG (fallback): Legacy format for compatibility

        Args:
            frame: BGR numpy array from OpenCV
            frame_number: Frame sequence number
            timestamp: ISO 8601 timestamp
            detection_count: Number of detections in frame

        Returns:
            True if published successfully, False otherwise
        """
        if not self.js or not self.frame_stream_enabled or not self.video_subject:
            return False

        try:
            if self._codec == "h264" and self._h264_encoder:
                # H.264/MPEG-TS encoding (WebRTC optimized)
                encoded_bytes, metadata = self._h264_encoder.encode_frame(frame)

                if not encoded_bytes:
                    return False  # No output ready yet (buffering)

                self._frame_count += 1
                self._chunk_sequence += 1

                # Detect keyframe (IDR) by checking for H.264 NAL unit type 5
                # In MPEG-TS, keyframes typically have larger size due to I-frame data
                is_keyframe = self._frame_count == 1 or (self._frame_count % self.frame_stream_config.get("h264_gop_size", 30)) == 1
                frame_type = "IDR" if is_keyframe else "P"

                headers = {
                    "Content-Type": "video/mp2t",
                    "Event-Type": "video_frame",
                    "Codec": "h264",
                    "Container": "mpegts",
                    "X-Frame-Type": frame_type,
                    "X-Sequence": str(self._chunk_sequence),
                    "Frame-Number": str(frame_number),
                    "Timestamp": timestamp,
                    "Width": str(metadata.get("width", 0)),
                    "Height": str(metadata.get("height", 0)),
                    "Original-Width": str(metadata.get("original_width", 0)),
                    "Original-Height": str(metadata.get("original_height", 0)),
                    "Detection-Count": str(detection_count),
                    "Device-ID": self.device_fingerprint['device_id'],
                    "Org-ID": self.organization_id,
                    "Entity-ID": self.entity_id,
                    "Size-Bytes": str(len(encoded_bytes)),
                    "Bitrate": str(metadata.get("bitrate", "1500k")),
                }

                await self.js.publish(
                    self.video_subject,
                    encoded_bytes,
                    headers=headers
                )
            else:
                # MPEG-TS JPEG (optimized)
                import cv2
                import subprocess
                h, w = frame.shape[:2]
                max_dim = self.frame_stream_config.get("max_dimension", 1280)
                quality = self.frame_stream_config.get("jpeg_quality", 75)

                # Scale if needed
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                # Encode JPEG with OpenCV
                _, jpeg_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                
                # Wrap JPEG in MPEG-TS using FFmpeg
                try:
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "image2pipe", "-vcodec", "mjpeg", 
                        "-r", str(self.frame_stream_config.get("target_fps", 15)),
                        "-i", "-",
                        "-c:v", "copy", 
                        "-muxrate", "2000k",
                        "-pat_period", "1", 
                        "-sdt_period", "1",
                        "-f", "mpegts", "-"
                    ]
                    
                    result = subprocess.run(
                        ffmpeg_cmd,
                        input=jpeg_bytes.tobytes(),
                        capture_output=True,
                        timeout=1.0
                    )
                    
                    if result.returncode == 0:
                        mpegts_bytes = result.stdout
                    else:
                        # Fallback to raw JPEG
                        mpegts_bytes = jpeg_bytes.tobytes()
                        
                except Exception:
                    # Fallback to raw JPEG
                    mpegts_bytes = jpeg_bytes.tobytes()

                self._frame_count += 1

                headers = {
                    "Content-Type": "video/mp2t",
                    "Event-Type": "video_frame", 
                    "Codec": "mjpeg",
                    "Container": "mpegts",
                    "Frame-Number": str(frame_number),
                    "Timestamp": timestamp,
                    "Width": str(frame.shape[1]),
                    "Height": str(frame.shape[0]),
                    "Original-Width": str(w),
                    "Original-Height": str(h),
                    "Detection-Count": str(detection_count),
                    "Device-ID": self.device_fingerprint['device_id'],
                    "Org-ID": self.organization_id,
                    "Entity-ID": self.entity_id,
                    "Size-Bytes": str(len(mpegts_bytes)),
                    "Quality": str(quality),
                }

                await self.js.publish(
                    self.video_subject,
                    mpegts_bytes,
                    headers=headers
                )

            return True

        except Exception as e:
            print(f"Error publishing frame: {e}")
            return False

    def get_frame_stream_stats(self) -> Dict[str, Any]:
        """Get frame streaming statistics."""
        stats = {
            "enabled": self.frame_stream_enabled,
            "codec": self._codec,
            "frames_published": self._frame_count,
            "video_subject": self.video_subject,
            "target_fps": self.frame_stream_config.get("target_fps", 15),
        }

        # Add H.264 encoder stats
        if self._h264_encoder:
            encoder_stats = self._h264_encoder.get_stats()
            stats["h264"] = {
                "bytes_encoded": encoder_stats.get("bytes_encoded", 0),
                "avg_bitrate_kbps": round(encoder_stats.get("avg_bitrate_kbps", 0), 1),
                "resolution": encoder_stats.get("resolution", ""),
            }

        return stats

    async def cleanup(self, final_analytics: Optional[Dict] = None) -> None:
        """Clean up connections and publish shutdown event using publisher abstraction."""
        # Stop H.264 encoder if running
        if self._h264_encoder:
            self._h264_encoder.stop()
            self._h264_encoder = None

        if self.js and self.device_fingerprint:
            shutdown_message = self.publisher.build_shutdown(
                message="Overwatch ISR component shutting down gracefully",
                final_analytics=final_analytics
            )
            
            try:
                ack = await self.js.publish(
                    self.subject,
                    json.dumps(shutdown_message).encode(),
                    headers={
                        "Content-Type": "application/json",
                        "Event-Type": "shutdown"
                    }
                )
                print(f"Published shutdown event to JetStream (Seq: {ack.seq})")
            except Exception as e:
                print(f"Error publishing shutdown event: {e}")
        
        if self.nc:
            await self.nc.drain()
            await self.nc.close()
            print("NATS connection closed")