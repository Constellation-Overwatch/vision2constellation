"""Communication service for NATS/JetStream and KV store operations."""

import json
import nats
import numpy as np
from collections import deque
from nats.js.api import KeyValueConfig
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from ...config.defaults import DEFAULT_CONFIG
from ...utils.constellation import get_constellation_ids
from ...utils.h264_encoder import H264Encoder, EncodedChunk
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
        
        # Idempotency tracking for detection events (FIFO with automatic eviction)
        self._published_detection_hashes: deque = deque(maxlen=1000)
        self._state_update_lock: bool = False
        
        # KV store object tracking for incremental updates
        self._last_published_objects: Dict[str, Dict] = {}
        self._kv_cleanup_counter: int = 0

        # Connection state tracking
        self._is_reconnecting: bool = False
        self._reconnection_in_progress: bool = False

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
                print(f"  GOP: {self.frame_stream_config.get('h264_gop_size')} frames")
            else:
                print(f"  JPEG Quality: {self.frame_stream_config.get('jpeg_quality')}")
        print()
        
        # Connect to NATS
        await self._connect_nats()
        await self._setup_jetstream()
        await self._setup_kv_store()
        await self._publish_bootsequence()
    
    async def _connect_nats(self) -> None:
        """Connect to NATS server with reconnection handling and optional token authentication."""
        print(f"Attempting to connect to NATS at: {self.nats_config['url']}")

        connect_opts = {
            "servers": [self.nats_config["url"]],
            # Reconnection settings
            "allow_reconnect": True,
            "reconnect_time_wait": self.nats_config["reconnect_time_wait"],
            "max_reconnect_attempts": self.nats_config["max_reconnect_attempts"],
            "connect_timeout": self.nats_config["connect_timeout"],
            "ping_interval": self.nats_config["ping_interval"],
            "max_outstanding_pings": self.nats_config["max_outstanding_pings"],
            # Connection callbacks
            "disconnected_cb": self._on_disconnected,
            "reconnected_cb": self._on_reconnected,
            "error_cb": self._on_error,
            "closed_cb": self._on_closed,
        }

        # Token-based authentication
        if self.nats_config.get("auth_token"):
            connect_opts["token"] = self.nats_config["auth_token"]
            print("Using token-based authentication")

        self.nc = await nats.connect(**connect_opts)
        print("Connected to NATS server")

    async def _on_disconnected(self):
        """Handle NATS disconnection."""
        self._is_reconnecting = True
        print("NATS connection lost - attempting reconnection...")

    async def _on_reconnected(self):
        """Handle NATS reconnection."""
        if self._reconnection_in_progress:
            return  # Prevent multiple simultaneous reconnection attempts
        
        self._reconnection_in_progress = True
        print("NATS connection restored!")
        
        # Re-initialize JetStream and KV after reconnection
        try:
            self.js = self.nc.jetstream()
            await self._setup_kv_store()
            print("JetStream and KV store re-initialized after reconnection")
            
            # Clear reconnection flags
            self._is_reconnecting = False
            self._reconnection_in_progress = False
            
            # Clear entity state cache to force refresh
            self._entity_state_cache = None
            
        except Exception as e:
            print(f"Error re-initializing after reconnection: {e}")
            self._reconnection_in_progress = False

    async def _on_error(self, error):
        """Handle NATS errors."""
        # Don't spam error logs during reconnection attempts
        if "Connect call failed" not in str(error) or not self._is_reconnecting:
            print(f"NATS error: {error}")

    async def _on_closed(self):
        """Handle NATS connection closed."""
        self._is_reconnecting = True
        print("NATS connection closed")
    
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

    async def _is_connected(self) -> bool:
        """Check if NATS connection is healthy and not in reconnection state."""
        return self.nc and self.nc.is_connected and not self._is_reconnecting

    async def publish_detection_event(self, detection_data: Dict[str, Any]) -> None:
        """Publish detection event to JetStream using publisher abstraction with idempotency."""
        if not self.js or not await self._is_connected():
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
            
            # Track published hash (automatic FIFO eviction when maxlen exceeded)
            self._published_detection_hashes.append(detection_hash)
                    
        except Exception as e:
            # Only log errors if not during reconnection to avoid spam
            if not self._is_reconnecting:
                print(f"Error publishing detection event: {e}")
    
    def _should_publish_object_to_kv(self, obj: Dict[str, Any]) -> bool:
        """
        Determine if an object should be published to KV store based on quality thresholds.
        
        Args:
            obj: Object data dictionary
            
        Returns:
            bool: True if object meets publishing criteria
        """
        kv_confidence_threshold = DEFAULT_CONFIG["detection"]["kv_confidence_threshold"]
        
        # Check confidence threshold
        avg_confidence = obj.get("avg_confidence", 0)
        if avg_confidence < kv_confidence_threshold:
            return False
            
        # Require minimum frame persistence for stability
        frame_count = obj.get("frame_count", 0)
        if frame_count < 5:  # More frames required for KV than general tracking
            return False
            
        # Must be currently active
        if not obj.get("is_active", False):
            return False
            
        return True

    def _has_object_changed(self, track_id: str, current_obj: Dict[str, Any]) -> bool:
        """
        Check if an object has significantly changed since last KV publish.
        
        Args:
            track_id: Unique track identifier
            current_obj: Current object data
            
        Returns:
            bool: True if object has changed enough to warrant KV update
        """
        if track_id not in self._last_published_objects:
            return True  # New object
            
        last_obj = self._last_published_objects[track_id]
        
        # Check confidence change (>10% change)
        confidence_diff = abs(current_obj.get("avg_confidence", 0) - last_obj.get("avg_confidence", 0))
        if confidence_diff > 0.1:
            return True
            
        # Check frame count change (new activity)
        frame_diff = current_obj.get("frame_count", 0) - last_obj.get("frame_count", 0)
        if frame_diff > 5:  # Significant new activity
            return True
            
        # Check bbox movement (>5% change in position)
        current_bbox = current_obj.get("current_bbox", {})
        last_bbox = last_obj.get("current_bbox", {})
        if current_bbox and last_bbox:
            for coord in ["x_min", "y_min", "x_max", "y_max"]:
                diff = abs(current_bbox.get(coord, 0) - last_bbox.get(coord, 0))
                if diff > 0.05:  # 5% movement threshold
                    return True
                    
        # Check threat level change
        if current_obj.get("threat_level") != last_obj.get("threat_level"):
            return True
            
        return False

    async def publish_state_to_kv(self, tracking_state: Any, analytics: Dict[str, Any]) -> None:
        """
        Publish tracking state to consolidated EntityState in KV store with incremental updates.
        Updates both 'detections' and 'analytics' subsignals, only publishing changed/new objects.
        """
        if not self.kv or not self.entity_id or self._state_update_lock or not await self._is_connected():
            return

        # Prevent concurrent state updates
        self._state_update_lock = True
        
        try:
            # Get current persistent objects
            persistent_objects = {}
            if hasattr(tracking_state, 'get_persistent_objects'):
                persistent_objects = tracking_state.get_persistent_objects(min_frames=3)
            
            # Filter objects that meet KV publishing criteria and have changed
            objects_to_publish = {}
            current_active_tracks = set()
            
            for tid, obj in persistent_objects.items():
                track_id = str(tid)
                current_active_tracks.add(track_id)
                
                # Apply quality filters
                if not self._should_publish_object_to_kv(obj):
                    continue
                    
                # Check if object has changed enough to warrant update
                if not self._has_object_changed(track_id, obj):
                    continue
                
                # Format object for KV store
                formatted_obj = {
                    "track_id": obj.get("track_id", obj.get("segment_id", tid)),
                    "label": obj.get("label", "segment"),
                    "category": obj.get("category"),
                    "priority": obj.get("priority"),
                    "first_seen": obj["first_seen"],
                    "last_seen": obj["last_seen"],
                    "frame_count": obj["frame_count"],
                    "avg_confidence": obj.get("avg_confidence", 0),
                    "is_active": obj["is_active"],
                    "threat_level": obj.get("threat_level"),
                    "suspicious_indicators": obj.get("suspicious_indicators", []),
                    "area": obj.get("area"),
                    "current_bbox": obj.get("bbox_history", [])[-1] if obj.get("bbox_history") else obj.get("bbox")
                }
                
                objects_to_publish[track_id] = formatted_obj
                # Update our tracking cache
                self._last_published_objects[track_id] = formatted_obj.copy()
            
            # Periodic cleanup of stale objects from tracking cache
            self._kv_cleanup_counter += 1
            if self._kv_cleanup_counter >= 50:  # Every 50 updates
                await self._cleanup_stale_kv_objects(current_active_tracks)
                self._kv_cleanup_counter = 0
            
            # Only update KV if we have changes to publish
            if objects_to_publish:
                # Get current entity state
                entity_state = await self._get_entity_state()
                
                # Update detections incrementally
                if "detections" not in entity_state:
                    entity_state["detections"] = {"timestamp": "", "objects": {}}
                
                # Merge new/updated objects with existing ones
                entity_state["detections"]["timestamp"] = datetime.now(timezone.utc).isoformat()
                entity_state["detections"]["objects"].update(objects_to_publish)
                
                # Update analytics
                entity_state["analytics"] = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "summary": analytics
                }
                
                entity_state["updated_at"] = datetime.now(timezone.utc).isoformat()
                
                # Write consolidated state to KV
                await self.kv.put(
                    self.entity_id,
                    json.dumps(entity_state).encode()
                )
                
                # Update cache
                self._entity_state_cache = entity_state
                
                print(f"Published {len(objects_to_publish)} changed objects to KV store")

        except Exception as e:
            # Only log errors if not during reconnection to avoid spam
            if not self._is_reconnecting:
                print(f"Error publishing state to KV: {e}")
        finally:
            self._state_update_lock = False

    async def _cleanup_stale_kv_objects(self, current_active_tracks: set) -> None:
        """
        Remove stale/inactive objects from KV store and internal tracking.
        
        Args:
            current_active_tracks: Set of currently active track IDs
        """
        try:
            entity_state = await self._get_entity_state()
            
            if "detections" not in entity_state or "objects" not in entity_state["detections"]:
                return
                
            objects_dict = entity_state["detections"]["objects"]
            stale_tracks = []
            
            # Identify objects to remove (not in current active tracks)
            for track_id in list(objects_dict.keys()):
                if track_id not in current_active_tracks:
                    stale_tracks.append(track_id)
            
            # Remove stale objects from KV store
            if stale_tracks:
                for track_id in stale_tracks:
                    objects_dict.pop(track_id, None)
                    self._last_published_objects.pop(track_id, None)
                
                # Update KV store
                entity_state["detections"]["timestamp"] = datetime.now(timezone.utc).isoformat()
                entity_state["updated_at"] = datetime.now(timezone.utc).isoformat()
                
                await self.kv.put(
                    self.entity_id,
                    json.dumps(entity_state).encode()
                )
                
                self._entity_state_cache = entity_state
                print(f"Cleaned up {len(stale_tracks)} stale objects from KV store")
                
        except Exception as e:
            print(f"Error during KV cleanup: {e}")
    
    async def publish_threat_intelligence(self, tracking_state: Any) -> None:
        """
        Publish C4ISR threat intelligence to consolidated EntityState.
        Updates 'c4isr' subsignal and enriches 'analytics' with C4ISR summary.
        """
        if not self.kv or not hasattr(tracking_state, 'threat_alerts') or not await self._is_connected():
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
            # Only log errors if not during reconnection to avoid spam
            if not self._is_reconnecting:
                print(f"Error publishing threat intelligence to KV: {e}")

    async def publish_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: str,
        detection_count: int = 0
    ) -> bool:
        """
        Publish video frame to JetStream.

        H.264/MPEG-TS mode (codec=h264):
        - Each chunk is 1316 bytes (7 MPEG-TS packets, fits UDP datagram)
        - Chunks published individually for real-time streaming
        - Keyframe markers for seek/reconnection support
        - ~90% bandwidth savings vs JPEG

        JPEG mode (codec=jpeg):
        - Simple JPEG frames for compatibility
        - Higher bandwidth but simpler receiver

        Args:
            frame: BGR numpy array from OpenCV
            frame_number: Frame sequence number
            timestamp: ISO 8601 timestamp
            detection_count: Number of detections in frame

        Returns:
            True if published successfully, False otherwise
        """
        if not self.js or not self.frame_stream_enabled or not self.video_subject or not await self._is_connected():
            return False

        try:
            if self._codec == "h264" and self._h264_encoder:
                # H.264/MPEG-TS encoding (WebRTC optimized)
                chunks, metadata = self._h264_encoder.encode_frame(frame)

                if not chunks:
                    return False  # No output ready (encoder buffering)

                self._frame_count += 1

                # Publish each chunk individually for real-time streaming
                for chunk in chunks:
                    self._chunk_sequence += 1

                    headers = {
                        "Content-Type": "video/mp2t",
                        "Codec": "h264",
                        "X-Frame-Type": "IDR" if chunk.is_keyframe else "P",
                        "X-Sequence": str(chunk.sequence),
                        "X-PTS": str(chunk.pts),
                        "Frame-Number": str(frame_number),
                        "Timestamp": timestamp,
                        "Width": str(metadata.get("width", 0)),
                        "Height": str(metadata.get("height", 0)),
                        "Detection-Count": str(detection_count),
                        "Device-ID": self.device_fingerprint['device_id'],
                        "Entity-ID": self.entity_id,
                    }

                    await self.js.publish(
                        self.video_subject,
                        chunk.data,
                        headers=headers
                    )
            else:
                # JPEG fallback (simpler, higher bandwidth)
                import cv2
                h, w = frame.shape[:2]
                max_dim = self.frame_stream_config.get("max_dimension", 1280)
                quality = self.frame_stream_config.get("jpeg_quality", 75)

                # Scale if needed
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    h, w = frame.shape[:2]

                _, jpeg_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                jpeg_bytes = jpeg_buffer.tobytes()

                self._frame_count += 1
                self._chunk_sequence += 1

                headers = {
                    "Content-Type": "image/jpeg",
                    "Codec": "jpeg",
                    "X-Sequence": str(self._chunk_sequence),
                    "Frame-Number": str(frame_number),
                    "Timestamp": timestamp,
                    "Width": str(w),
                    "Height": str(h),
                    "Detection-Count": str(detection_count),
                    "Device-ID": self.device_fingerprint['device_id'],
                    "Entity-ID": self.entity_id,
                    "Size-Bytes": str(len(jpeg_bytes)),
                }

                await self.js.publish(
                    self.video_subject,
                    jpeg_bytes,
                    headers=headers
                )

            return True

        except Exception as e:
            # Only log errors if not during reconnection to avoid spam
            if not self._is_reconnecting:
                print(f"Error publishing frame: {e}")
            return False

    def get_frame_stream_stats(self) -> Dict[str, Any]:
        """Get frame streaming statistics."""
        stats = {
            "enabled": self.frame_stream_enabled,
            "codec": self._codec,
            "frames_published": self._frame_count,
            "chunks_published": self._chunk_sequence,
            "video_subject": self.video_subject,
            "target_fps": self.frame_stream_config.get("target_fps", 15),
        }

        # Add H.264 encoder stats
        if self._h264_encoder:
            encoder_stats = self._h264_encoder.get_stats()
            stats["h264"] = {
                "frames_encoded": encoder_stats.get("frames_encoded", 0),
                "chunks_produced": encoder_stats.get("chunks_produced", 0),
                "bytes_encoded": encoder_stats.get("bytes_encoded", 0),
                "avg_fps": encoder_stats.get("avg_fps", 0),
                "avg_bitrate_kbps": round(encoder_stats.get("avg_bitrate_kbps", 0), 1),
                "resolution": encoder_stats.get("resolution", ""),
                "queue_size": encoder_stats.get("queue_size", 0),
            }

        return stats

    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status."""
        return {
            "connected": self.nc and self.nc.is_connected,
            "is_reconnecting": self._is_reconnecting,
            "reconnection_in_progress": self._reconnection_in_progress,
            "jetstream_available": self.js is not None,
            "kv_available": self.kv is not None,
            "server_url": self.nats_config["url"],
            "subject": self.subject,
        }

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