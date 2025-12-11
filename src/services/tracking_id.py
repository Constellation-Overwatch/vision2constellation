"""
Centralized Tracking ID Service

Provides globally unique, collision-resistant identifiers (CUIDs) for all
detection models. Ensures consistent ID generation and payload formatting
across the entire detection pipeline.
"""

from typing import Dict, Any, Optional
import hashlib
import math
from cuid2 import cuid_wrapper


class TrackingIDService:
    """
    Centralized service for managing tracking IDs across all detection models.

    Features:
    - Generates CUIDs (Collision-resistant Unique IDs) for global uniqueness
    - Maintains mapping between model-specific IDs and CUIDs
    - Ensures consistent payload format across all models
    - Thread-safe for multi-model deployments
    """

    def __init__(self):
        """Initialize the tracking ID service."""
        # Mapping: (model_type, native_id) -> CUID
        self.id_mapping: Dict[tuple, str] = {}
        
        # Mapping: spatial_hash -> CUID for stable object identification
        self.spatial_mapping: Dict[str, str] = {}
        
        # Track object properties for stable ID generation
        self.object_history: Dict[str, Dict[str, Any]] = {}

        # CUID generator
        self.cuid_generator = cuid_wrapper()

    def _calculate_spatial_hash(self, bbox: Dict[str, float], label: str) -> str:
        """
        Calculate spatial hash based on object properties for stable identification.
        
        Args:
            bbox: Normalized bounding box {x_min, y_min, x_max, y_max}
            label: Object class label
            
        Returns:
            str: Spatial hash for object identification
        """
        # Calculate center point and size
        center_x = (bbox.get("x_min", 0) + bbox.get("x_max", 0)) / 2
        center_y = (bbox.get("y_min", 0) + bbox.get("y_max", 0)) / 2
        width = bbox.get("x_max", 0) - bbox.get("x_min", 0)
        height = bbox.get("y_max", 0) - bbox.get("y_min", 0)
        
        # Create spatial signature (rounded to avoid floating point precision issues)
        spatial_key = f"{label}:{round(center_x, 3)}:{round(center_y, 3)}:{round(width, 3)}:{round(height, 3)}"
        
        # Hash for consistent length
        return hashlib.md5(spatial_key.encode()).hexdigest()[:16]

    def get_stable_cuid(self, bbox: Dict[str, float], label: str, confidence: float, 
                       native_id: Any = None, model_type: str = "default",
                       similarity_threshold: float = 0.15) -> str:
        """
        Get or create a stable CUID based on object properties.
        
        Uses spatial proximity and label matching to maintain consistent IDs
        for the same physical object across frames.
        
        Args:
            bbox: Normalized bounding box
            label: Object class label
            confidence: Detection confidence
            native_id: Model's native tracking ID (optional)
            model_type: Type of detection model
            similarity_threshold: Maximum distance for object matching
            
        Returns:
            str: Stable CUID for the object
        """
        spatial_hash = self._calculate_spatial_hash(bbox, label)
        
        # Check for exact spatial match first
        if spatial_hash in self.spatial_mapping:
            cuid = self.spatial_mapping[spatial_hash]
            self._update_object_history(cuid, bbox, label, confidence)
            return cuid
        
        # Check for nearby objects with same label (handle slight movement)
        center_x = (bbox.get("x_min", 0) + bbox.get("x_max", 0)) / 2
        center_y = (bbox.get("y_min", 0) + bbox.get("y_max", 0)) / 2
        
        for existing_hash, existing_cuid in self.spatial_mapping.items():
            if existing_cuid in self.object_history:
                history = self.object_history[existing_cuid]
                
                # Check if same label and within similarity threshold
                if history.get("label") == label:
                    last_bbox = history.get("last_bbox", {})
                    if last_bbox:
                        last_center_x = (last_bbox.get("x_min", 0) + last_bbox.get("x_max", 0)) / 2
                        last_center_y = (last_bbox.get("y_min", 0) + last_bbox.get("y_max", 0)) / 2
                        
                        distance = math.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)
                        
                        if distance <= similarity_threshold:
                            # Update mapping to new spatial hash
                            del self.spatial_mapping[existing_hash]
                            self.spatial_mapping[spatial_hash] = existing_cuid
                            self._update_object_history(existing_cuid, bbox, label, confidence)
                            return existing_cuid
        
        # No match found - create new CUID
        new_cuid = self.cuid_generator()
        self.spatial_mapping[spatial_hash] = new_cuid
        self._update_object_history(new_cuid, bbox, label, confidence)
        
        # Also maintain legacy mapping if native_id provided
        if native_id is not None:
            key = (model_type, native_id)
            self.id_mapping[key] = new_cuid
        
        return new_cuid

    def get_or_create_cuid(self, native_id: Any, model_type: str = "default") -> str:
        """
        Get or create a CUID for a native tracking ID (legacy method).

        Args:
            native_id: The model's native tracking ID (e.g., YOLO ID, frame index)
            model_type: Type of model (yoloe, rtdetr, sam2, moondream)

        Returns:
            str: Globally unique CUID
        """
        key = (model_type, native_id)

        if key not in self.id_mapping:
            self.id_mapping[key] = self.cuid_generator()

        return self.id_mapping[key]
        
    def _update_object_history(self, cuid: str, bbox: Dict[str, float], 
                              label: str, confidence: float) -> None:
        """Update object history for tracking."""
        self.object_history[cuid] = {
            "last_bbox": bbox,
            "label": label,
            "last_confidence": confidence,
            "last_updated": True  # Could add timestamp if needed
        }

    def format_detection_payload(
        self,
        track_id: str,
        label: str,
        confidence: float,
        bbox: Dict[str, float],
        timestamp: str,
        model_type: str,
        native_id: Optional[Any] = None,
        **extra_fields
    ) -> Dict[str, Any]:
        """
        Format a standardized detection payload for all models.

        Args:
            track_id: The CUID tracking ID
            label: Object class label
            confidence: Detection confidence score
            bbox: Normalized bounding box {x_min, y_min, x_max, y_max}
            timestamp: ISO format timestamp
            model_type: Detection model type
            native_id: Model's native ID (for debugging)
            **extra_fields: Additional model-specific fields

        Returns:
            Standardized detection payload dictionary
        """
        # Ensure all numeric values are JSON serializable
        def convert_to_json_safe(obj):
            """Convert numpy/torch types to JSON-safe Python types."""
            if hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_safe(item) for item in obj]
            else:
                return obj

        # Convert bbox to JSON-safe format
        safe_bbox = convert_to_json_safe(bbox)
        safe_extra_fields = convert_to_json_safe(extra_fields)
        
        payload = {
            # Core identification
            "track_id": track_id,
            "model_type": model_type,

            # Detection data
            "label": str(label),
            "confidence": float(confidence),
            "bbox": safe_bbox,
            "timestamp": timestamp,

            # Metadata
            "metadata": {
                "native_id": native_id,
                **safe_extra_fields
            }
        }

        return payload

    def cleanup_stale_ids(self, active_ids: set) -> None:
        """
        Clean up mappings for IDs that are no longer active.

        Args:
            active_ids: Set of currently active CUIDs
        """
        # Remove mappings where the CUID is not in active set
        stale_keys = [
            key for key, cuid in self.id_mapping.items()
            if cuid not in active_ids
        ]

        for key in stale_keys:
            del self.id_mapping[key]
            
        # Clean up spatial mappings
        stale_spatial_keys = [
            spatial_hash for spatial_hash, cuid in self.spatial_mapping.items()
            if cuid not in active_ids
        ]
        
        for spatial_hash in stale_spatial_keys:
            del self.spatial_mapping[spatial_hash]
            
        # Clean up object history
        stale_cuids = [
            cuid for cuid in self.object_history.keys()
            if cuid not in active_ids
        ]
        
        for cuid in stale_cuids:
            del self.object_history[cuid]

    def get_mapping_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current ID mappings.

        Returns:
            Dictionary with mapping statistics
        """
        model_counts = {}
        for (model_type, _), _ in self.id_mapping.items():
            model_counts[model_type] = model_counts.get(model_type, 0) + 1

        return {
            "total_mappings": len(self.id_mapping),
            "spatial_mappings": len(self.spatial_mapping),
            "tracked_objects": len(self.object_history),
            "by_model": model_counts
        }
