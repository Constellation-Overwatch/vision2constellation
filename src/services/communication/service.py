"""Communication service for NATS/JetStream and KV store operations."""

import json
import nats
from nats.js.api import KeyValueConfig
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from ...config.defaults import DEFAULT_CONFIG
from ...utils.constellation import get_constellation_ids
from .publisher import ConstellationPublisher, build_kv_key

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

        # Configuration
        self.nats_config = DEFAULT_CONFIG["nats"]
        self.frame_stream_config = DEFAULT_CONFIG["frame_stream"]

        # Video frame streaming
        self.video_subject: Optional[str] = None
        self.frame_stream_enabled = self.frame_stream_config["enabled"]
        self._frame_count = 0

    async def initialize(self, device_fingerprint: Dict[str, Any], detection_mode: str = "detection") -> None:
        """Initialize NATS connection and setup streams."""
        self.device_fingerprint = device_fingerprint
        self.detection_mode = detection_mode
        
        # Get constellation identifiers
        self.organization_id, self.entity_id = get_constellation_ids()

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
        if self.frame_stream_enabled:
            self.video_subject = f"{self.frame_stream_config['subject_root']}.{self.organization_id}.{self.entity_id}"

        print(f"Configured NATS subject: {self.subject}")
        print(f"Configured stream name: {self.stream_name}")
        print(f"Configured KV store: {self.nats_config['kv_store_name']}")
        if self.frame_stream_enabled:
            print(f"Configured video subject: {self.video_subject}")
            print(f"Configured video stream: {self.frame_stream_config['stream_name']}")
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

    async def publish_detection_event(self, detection_data: Dict[str, Any]) -> None:
        """Publish detection event to JetStream using publisher abstraction."""
        if not self.js:
            return

        try:
            message = self.publisher.build_detection(detection_data)
            
            headers = {
                "Content-Type": "application/json",
                "Event-Type": "detection",
                "Device-ID": self.device_fingerprint['device_id']
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
        except Exception as e:
            print(f"Error publishing detection event: {e}")
    
    async def publish_state_to_kv(self, tracking_state: Any, analytics: Dict[str, Any]) -> None:
        """
        Publish tracking state to consolidated EntityState in KV store.
        Updates both 'detections' and 'analytics' subsignals.
        """
        if not self.kv or not self.entity_id:
            return

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
        frame_bytes: bytes,
        frame_number: int,
        timestamp: str,
        metadata: Dict[str, Any],
        detection_count: int = 0
    ) -> bool:
        """
        Publish video frame to JetStream.

        Args:
            frame_bytes: JPEG-encoded frame bytes
            frame_number: Frame sequence number
            timestamp: ISO 8601 timestamp
            metadata: Frame metadata (width, height, etc.)
            detection_count: Number of detections in frame

        Returns:
            True if published successfully, False otherwise
        """
        if not self.js or not self.frame_stream_enabled or not self.video_subject:
            return False

        try:
            self._frame_count += 1

            headers = {
                "Content-Type": "image/jpeg",
                "Event-Type": "video_frame",
                "Frame-Number": str(frame_number),
                "Timestamp": timestamp,
                "Width": str(metadata.get("width", 0)),
                "Height": str(metadata.get("height", 0)),
                "Original-Width": str(metadata.get("original_width", metadata.get("width", 0))),
                "Original-Height": str(metadata.get("original_height", metadata.get("height", 0))),
                "Detection-Count": str(detection_count),
                "Device-ID": self.device_fingerprint['device_id'],
                "Org-ID": self.organization_id,
                "Entity-ID": self.entity_id,
                "Size-Bytes": str(metadata.get("size_bytes", len(frame_bytes))),
                "Quality": str(metadata.get("quality", 75)),
            }

            await self.js.publish(
                self.video_subject,
                frame_bytes,
                headers=headers
            )
            return True

        except Exception as e:
            print(f"Error publishing frame: {e}")
            return False

    def get_frame_stream_stats(self) -> Dict[str, Any]:
        """Get frame streaming statistics."""
        return {
            "enabled": self.frame_stream_enabled,
            "frames_published": self._frame_count,
            "video_subject": self.video_subject,
            "target_fps": self.frame_stream_config.get("target_fps", 15),
        }

    async def cleanup(self, final_analytics: Optional[Dict] = None) -> None:
        """Clean up connections and publish shutdown event using publisher abstraction."""
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