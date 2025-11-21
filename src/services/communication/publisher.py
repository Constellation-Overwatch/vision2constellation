"""
Constellation Publisher Abstraction

Provides a transport-agnostic payload builder that ensures all messages
comply with Constellation Overwatch payload requirements.

This abstraction allows swapping underlying transports (NATS, Kafka, MQTT, HTTP)
without changing payload structure or business logic.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class ConstellationPublisher:
    """
    Base publisher ensuring Overwatch payload compliance.

    Guarantees all messages include required identity fields:
    - organization_id
    - entity_id
    - device_id
    - timestamp

    Usage:
        publisher = ConstellationPublisher(org_id, entity_id, device_id)
        bootseq = publisher.build_bootsequence(fingerprint, "System initialized")
        detection = publisher.build_detection(detection_data)
        shutdown = publisher.build_shutdown("Graceful shutdown", analytics)
    """

    def __init__(self, organization_id: str, entity_id: str, device_id: str):
        """
        Initialize publisher with identity context.

        Args:
            organization_id: Organization UUID from .env
            entity_id: Entity UUID from .env
            device_id: Hardware fingerprint hash
        """
        self.organization_id = organization_id
        self.entity_id = entity_id
        self.device_id = device_id

    def _enrich_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich payload with required identity fields.

        Ensures every message has org_id, entity_id, device_id at the top level.
        This is critical for Overwatch entity registry and KV store operations.

        Args:
            payload: Base payload dictionary

        Returns:
            Enriched payload with identity fields
        """
        return {
            "organization_id": self.organization_id,
            "entity_id": self.entity_id,
            "device_id": self.device_id,
            **payload
        }

    def build_bootsequence(
        self,
        fingerprint: Dict[str, Any],
        message: str
    ) -> Dict[str, Any]:
        """
        Build Overwatch-compliant bootsequence payload.

        Bootsequence registers an entity with Overwatch when it comes online.
        Entity must send bootsequence before sending telemetry or detections.

        Args:
            fingerprint: Device fingerprint with component, camera, platform metadata
            message: Human-readable initialization message

        Returns:
            Complete bootsequence payload

        Example:
            payload = publisher.build_bootsequence(
                fingerprint=device_fingerprint,
                message="Overwatch ISR component initialized"
            )
        """
        return self._enrich_payload({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "bootsequence",
            "message": message,
            "source": fingerprint
        })

    def build_detection(
        self,
        detection_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build Overwatch-compliant detection event payload.

        Detection events report object detection/tracking results.
        Entity must be registered (via bootsequence) before sending detections.

        Args:
            detection_data: Detection with label, confidence, bbox, track_id, etc.

        Returns:
            Complete detection event payload

        Example:
            payload = publisher.build_detection({
                "label": "person",
                "confidence": 0.95,
                "bbox": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.9, "y_max": 0.8},
                "track_id": "abc123",
                "timestamp": "2025-11-21T14:00:00Z"
            })
        """
        return self._enrich_payload({
            "timestamp": detection_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "event_type": "detection",
            "detection": detection_data
        })

    def build_shutdown(
        self,
        message: str,
        final_analytics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build Overwatch-compliant shutdown event payload.

        Shutdown deregisters an entity when it goes offline.
        After shutdown, entity must send new bootsequence to re-register.

        Args:
            message: Human-readable shutdown message
            final_analytics: Optional analytics summary (frame counts, detections, etc.)

        Returns:
            Complete shutdown event payload

        Example:
            payload = publisher.build_shutdown(
                message="Overwatch ISR component shutting down gracefully",
                final_analytics={"total_frames": 1250, "total_objects": 42}
            )
        """
        return self._enrich_payload({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "shutdown",
            "message": message,
            "final_analytics": final_analytics
        })

    def build_telemetry(
        self,
        message_type: str,
        system_id: int,
        component_id: int,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build MAVLink telemetry payload.

        Telemetry reports real-time vehicle state (position, attitude, status).
        Entity must be registered before sending telemetry.

        Args:
            message_type: MAVLink message type (HEARTBEAT, GPS_RAW_INT, etc.)
            system_id: MAVLink system ID
            component_id: MAVLink component ID
            data: MAVLink message data

        Returns:
            Complete telemetry payload

        Example:
            payload = publisher.build_telemetry(
                message_type="HEARTBEAT",
                system_id=1,
                component_id=1,
                data={"custom_mode": 4, "base_mode": 128}
            )
        """
        return self._enrich_payload({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_type": message_type,
            "system_id": system_id,
            "component_id": component_id,
            "data": data
        })


class TransportAdapter(ABC):
    """
    Abstract transport adapter interface.

    Allows swapping underlying transports (NATS, Kafka, MQTT, HTTP)
    without changing business logic or payload structure.

    Implementations should handle:
    - Connection lifecycle
    - Message serialization
    - Error handling and retries
    - Transport-specific headers/metadata
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to transport."""
        pass

    @abstractmethod
    async def publish_event(
        self,
        subject: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Publish event to transport.

        Args:
            subject: Target subject/topic/stream
            payload: Message payload (pre-built by ConstellationPublisher)
            headers: Optional transport-specific headers
        """
        pass

    @abstractmethod
    async def publish_to_kv(
        self,
        key: str,
        value: Dict[str, Any]
    ) -> None:
        """
        Publish to key-value store.

        Args:
            key: Storage key (must be KV-compliant: alphanumeric, hyphens, underscores)
            value: Value to store
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connection to transport."""
        pass


def validate_entity_id(entity_id: str) -> bool:
    """
    Validate entity_id format per Overwatch requirements.

    Valid characters: a-z, A-Z, 0-9, hyphens, underscores (except leading)
    Invalid: dots, asterisks, spaces, special characters

    Args:
        entity_id: Entity identifier to validate

    Returns:
        True if valid, False otherwise

    Example:
        validate_entity_id("1048bff5-5b97-4fa8")  # True
        validate_entity_id("device.123")          # False (contains dot)
        validate_entity_id("sensor*01")           # False (contains asterisk)
    """
    import re
    pattern = r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$'
    return bool(re.match(pattern, entity_id))


def build_kv_key(entity_id: str, *segments: str) -> str:
    """
    Build NATS KV hierarchical key using dot notation.

    Note: Dots are VALID in KV keys. The Overwatch restriction on dots
    applies to entity_id VALUES only, not KV key paths.

    Args:
        entity_id: Base entity identifier (must be dot-free)
        *segments: Additional key segments

    Returns:
        Hierarchical key string

    Example:
        build_kv_key("abc123", "detections", "objects")
        # Returns: "abc123.detections.objects"

        build_kv_key("abc123", "analytics", "summary")
        # Returns: "abc123.analytics.summary"
    """
    parts = [entity_id] + list(segments)
    return ".".join(parts)
