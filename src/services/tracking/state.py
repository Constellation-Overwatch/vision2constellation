"""Tracking state management for different detection modes."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Any, Set, List, Optional, Tuple
from collections import defaultdict
import math

class BaseTrackingState(ABC):
    """Base class for tracking state management."""
    
    def __init__(self):
        self.total_frames_processed = 0
        
    @abstractmethod
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics summary."""
        pass
    
    @abstractmethod
    def mark_inactive(self, current_ids: Set) -> None:
        """Mark objects not seen in current frame as inactive."""
        pass

class TrackingState(BaseTrackingState):
    """Standard object tracking state for YOLOE and RTDETR."""

    def __init__(self):
        super().__init__()
        self.tracked_objects = {}  # track_id -> object metadata
        self.total_unique_objects = 0
        self.active_track_ids = set()
        
        # Object deduplication settings
        self.iou_threshold = 0.7  # Intersection over Union threshold for merging
        self.confidence_diff_threshold = 0.3  # Max confidence difference for merging

    def update_object(self, track_id: Any, label: str, confidence: float,
                     bbox: Dict[str, float], frame_timestamp: str,
                     category: str = None, priority: str = None, **kwargs) -> None:
        """Update or create tracked object state."""
        if track_id not in self.tracked_objects:
            # New object detected
            self.total_unique_objects += 1
            self.tracked_objects[track_id] = {
                "track_id": track_id,
                "label": label,
                "category": category or "object",
                "priority": priority or "normal",
                "first_seen": frame_timestamp,
                "last_seen": frame_timestamp,
                "frame_count": 1,
                "total_confidence": confidence,
                "avg_confidence": confidence,
                "bbox_history": [bbox],
                "is_active": True
            }
        else:
            # Update existing object
            obj = self.tracked_objects[track_id]
            obj["last_seen"] = frame_timestamp
            obj["frame_count"] += 1
            obj["total_confidence"] += confidence
            obj["avg_confidence"] = obj["total_confidence"] / obj["frame_count"]
            obj["bbox_history"].append(bbox)
            obj["is_active"] = True
            # Update category/priority if provided
            if category:
                obj["category"] = category
            if priority:
                obj["priority"] = priority

            # Keep only last 30 frames of bbox history
            if len(obj["bbox_history"]) > 30:
                obj["bbox_history"] = obj["bbox_history"][-30:]

        self.active_track_ids.add(track_id)

    def _calculate_iou(self, bbox1: Dict[str, float], bbox2: Dict[str, float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        # Extract coordinates
        x1_min, y1_min = bbox1["x_min"], bbox1["y_min"]
        x1_max, y1_max = bbox1["x_max"], bbox1["y_max"]
        x2_min, y2_min = bbox2["x_min"], bbox2["y_min"]
        x2_max, y2_max = bbox2["x_max"], bbox2["y_max"]
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
            
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def _find_similar_object(self, label: str, confidence: float, bbox: Dict[str, float]) -> Optional[Any]:
        """
        Find existing object that is similar enough to merge with new detection.
        
        Args:
            label: Object label/class
            confidence: Detection confidence
            bbox: Bounding box coordinates
            
        Returns:
            track_id of similar object if found, None otherwise
        """
        best_match_id = None
        best_iou = 0.0
        
        for track_id, obj in self.tracked_objects.items():
            # Only consider objects with same label and currently active
            if obj["label"] != label or not obj["is_active"]:
                continue
                
            # Check confidence similarity
            conf_diff = abs(obj["avg_confidence"] - confidence)
            if conf_diff > self.confidence_diff_threshold:
                continue
                
            # Check bbox overlap (using most recent bbox)
            if obj["bbox_history"]:
                recent_bbox = obj["bbox_history"][-1]
                iou = self._calculate_iou(bbox, recent_bbox)
                
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match_id = track_id
        
        return best_match_id

    def update_object(self, track_id: Any, label: str, confidence: float,
                     bbox: Dict[str, float], frame_timestamp: str,
                     category: str = None, priority: str = None, **kwargs) -> Any:
        """
        Update or create tracked object state with deduplication.
        
        Returns:
            The final track_id used (may be different from input if merged)
        """
        # For new objects, check if we should merge with existing similar object
        if track_id not in self.tracked_objects:
            similar_id = self._find_similar_object(label, confidence, bbox)
            if similar_id is not None:
                # Merge with existing object instead of creating new one
                track_id = similar_id
                
        # Continue with normal update logic
        return self._update_object_internal(track_id, label, confidence, bbox, 
                                          frame_timestamp, category, priority, **kwargs)

    def _update_object_internal(self, track_id: Any, label: str, confidence: float,
                               bbox: Dict[str, float], frame_timestamp: str,
                               category: str = None, priority: str = None, **kwargs) -> Any:
        """Internal method to update or create tracked object."""
        if track_id not in self.tracked_objects:
            # New object detected
            self.total_unique_objects += 1
            self.tracked_objects[track_id] = {
                "track_id": track_id,
                "label": label,
                "category": category or "object",
                "priority": priority or "normal",
                "first_seen": frame_timestamp,
                "last_seen": frame_timestamp,
                "frame_count": 1,
                "total_confidence": confidence,
                "avg_confidence": confidence,
                "bbox_history": [bbox],
                "is_active": True
            }
        else:
            # Update existing object
            obj = self.tracked_objects[track_id]
            obj["last_seen"] = frame_timestamp
            obj["frame_count"] += 1
            obj["total_confidence"] += confidence
            obj["avg_confidence"] = obj["total_confidence"] / obj["frame_count"]
            obj["bbox_history"].append(bbox)
            obj["is_active"] = True
            # Update category/priority if provided
            if category:
                obj["category"] = category
            if priority:
                obj["priority"] = priority

            # Keep only last 30 frames of bbox history
            if len(obj["bbox_history"]) > 30:
                obj["bbox_history"] = obj["bbox_history"][-30:]

        self.active_track_ids.add(track_id)
        return track_id

    def mark_inactive(self, current_track_ids: Set) -> None:
        """Mark objects that weren't seen in this frame as inactive."""
        for track_id in list(self.tracked_objects.keys()):
            if track_id not in current_track_ids:
                self.tracked_objects[track_id]["is_active"] = False
                if track_id in self.active_track_ids:
                    self.active_track_ids.remove(track_id)

    def get_persistent_objects(self, min_frames: int = 3) -> Dict[Any, Dict]:
        """Get objects tracked for at least min_frames."""
        return {
            tid: obj for tid, obj in self.tracked_objects.items()
            if obj["frame_count"] >= min_frames
        }

    def get_analytics(self) -> Dict[str, Any]:
        """Get tracking analytics summary."""
        active_objects = [obj for obj in self.tracked_objects.values() if obj["is_active"]]

        # Count by label and category
        label_counts = defaultdict(int)
        category_counts = defaultdict(int)
        for obj in active_objects:
            label_counts[obj["label"]] += 1
            category_counts[obj.get("category", "object")] += 1

        return {
            "total_unique_objects": self.total_unique_objects,
            "total_frames_processed": self.total_frames_processed,
            "active_objects_count": len(active_objects),
            "tracked_objects_count": len(self.tracked_objects),
            "label_distribution": dict(label_counts),
            "category_distribution": dict(category_counts),
            "active_track_ids": list(self.active_track_ids)
        }

class C4ISRTrackingState(BaseTrackingState):
    """C4ISR threat intelligence tracking state."""
    
    def __init__(self):
        super().__init__()
        self.tracked_objects = {}  # track_id -> object metadata
        self.total_unique_objects = 0
        self.active_track_ids = set()
        
        # C4ISR threat analytics
        self.threat_alerts = []  # List of threat alerts
        self.threat_summary = {
            "HIGH_THREAT": 0,
            "MEDIUM_THREAT": 0,
            "LOW_THREAT": 0,
            "NORMAL": 0
        }

    def update_object(self, track_id: Any, label: str, confidence: float,
                     bbox: Dict[str, float], frame_timestamp: str, 
                     threat_level: str) -> None:
        """Update tracked object with threat intelligence."""
        if track_id not in self.tracked_objects:
            # New object detected
            self.total_unique_objects += 1

            # Create threat alert for high/medium threats
            if threat_level in ["HIGH_THREAT", "MEDIUM_THREAT"]:
                alert = {
                    "alert_id": f"{track_id}_{frame_timestamp}",
                    "track_id": track_id,
                    "label": label,
                    "threat_level": threat_level,
                    "confidence": confidence,
                    "first_detected": frame_timestamp,
                    "bbox": bbox,
                    "status": "active"
                }
                self.threat_alerts.append(alert)

            self.tracked_objects[track_id] = {
                "track_id": track_id,
                "label": label,
                "threat_level": threat_level,
                "first_seen": frame_timestamp,
                "last_seen": frame_timestamp,
                "frame_count": 1,
                "total_confidence": confidence,
                "avg_confidence": confidence,
                "max_confidence": confidence,
                "bbox_history": [bbox],
                "is_active": True,
                "suspicious_indicators": self._calculate_suspicious_indicators(
                    label, confidence, threat_level
                )
            }
        else:
            # Update existing object
            obj = self.tracked_objects[track_id]
            obj["last_seen"] = frame_timestamp
            obj["frame_count"] += 1
            obj["total_confidence"] += confidence
            obj["avg_confidence"] = obj["total_confidence"] / obj["frame_count"]
            obj["max_confidence"] = max(obj["max_confidence"], confidence)
            obj["bbox_history"].append(bbox)
            obj["is_active"] = True
            obj["suspicious_indicators"] = self._calculate_suspicious_indicators(
                label, confidence, threat_level
            )

            # Keep only last 30 frames of bbox history
            if len(obj["bbox_history"]) > 30:
                obj["bbox_history"] = obj["bbox_history"][-30:]

        self.active_track_ids.add(track_id)

    def _calculate_suspicious_indicators(self, label: str, confidence: float, 
                                       threat_level: str) -> List[str]:
        """Calculate suspicious indicators for threat assessment."""
        indicators = []

        # High confidence threats are more suspicious
        if threat_level == "HIGH_THREAT" and confidence > 0.7:
            indicators.append("high_confidence_weapon_detection")

        # Medium confidence threats warrant investigation
        if threat_level == "MEDIUM_THREAT" and confidence > 0.5:
            indicators.append("suspicious_object_detected")

        # Low confidence high threats are uncertain
        if threat_level == "HIGH_THREAT" and confidence < 0.5:
            indicators.append("uncertain_threat_requires_validation")

        return indicators

    def mark_inactive(self, current_track_ids: Set) -> None:
        """Mark objects that weren't seen in this frame as inactive."""
        for track_id in list(self.tracked_objects.keys()):
            if track_id not in current_track_ids:
                self.tracked_objects[track_id]["is_active"] = False
                if track_id in self.active_track_ids:
                    self.active_track_ids.remove(track_id)

    def get_persistent_objects(self, min_frames: int = 1) -> Dict[Any, Dict]:
        """Get objects tracked for at least min_frames."""
        return {
            tid: obj for tid, obj in self.tracked_objects.items()
            if obj["frame_count"] >= min_frames
        }

    def get_analytics(self) -> Dict[str, Any]:
        """Get C4ISR threat analytics."""
        active_objects = [obj for obj in self.tracked_objects.values() if obj["is_active"]]

        # Count by label and threat level
        label_counts = defaultdict(int)
        threat_counts = defaultdict(int)

        for obj in active_objects:
            label_counts[obj["label"]] += 1
            threat_counts[obj["threat_level"]] += 1

        # Get active threats (HIGH and MEDIUM only)
        active_threats = [
            obj for obj in active_objects
            if obj["threat_level"] in ["HIGH_THREAT", "MEDIUM_THREAT"]
        ]

        return {
            "total_unique_objects": self.total_unique_objects,
            "total_frames_processed": self.total_frames_processed,
            "active_objects_count": len(active_objects),
            "tracked_objects_count": len(self.tracked_objects),
            "label_distribution": dict(label_counts),
            "threat_distribution": dict(threat_counts),
            "active_threat_count": len(active_threats),
            "active_track_ids": list(self.active_track_ids),
            "threat_alerts": self.threat_alerts[-10:]  # Last 10 alerts
        }

class SegmentationState(BaseTrackingState):
    """Segmentation tracking state for SAM2."""
    
    def __init__(self):
        super().__init__()
        self.segmented_objects = {}  # segment_id -> object metadata
        self.total_unique_segments = 0
        self.active_segment_ids = set()

    def update_segment(self, segment_id: Any, mask: Any, bbox: Dict[str, float],
                      area: int, confidence: float, frame_timestamp: str) -> None:
        """Update or create segmented object state."""
        if segment_id not in self.segmented_objects:
            # New segment detected
            self.total_unique_segments += 1
            self.segmented_objects[segment_id] = {
                "segment_id": segment_id,
                "first_seen": frame_timestamp,
                "last_seen": frame_timestamp,
                "frame_count": 1,
                "total_confidence": confidence,
                "avg_confidence": confidence,
                "area": area,
                "bbox": bbox,
                "is_active": True
            }
        else:
            # Update existing segment
            obj = self.segmented_objects[segment_id]
            obj["last_seen"] = frame_timestamp
            obj["frame_count"] += 1
            obj["total_confidence"] += confidence
            obj["avg_confidence"] = obj["total_confidence"] / obj["frame_count"]
            obj["area"] = area
            obj["bbox"] = bbox
            obj["is_active"] = True

        self.active_segment_ids.add(segment_id)

    def mark_inactive(self, current_segment_ids: Set) -> None:
        """Mark segments that weren't seen in this frame as inactive."""
        for segment_id in list(self.segmented_objects.keys()):
            if segment_id not in current_segment_ids:
                self.segmented_objects[segment_id]["is_active"] = False
                if segment_id in self.active_segment_ids:
                    self.active_segment_ids.remove(segment_id)

    def get_persistent_segments(self, min_frames: int = 3) -> Dict[Any, Dict]:
        """Get segments tracked for at least min_frames."""
        return {
            sid: obj for sid, obj in self.segmented_objects.items()
            if obj["frame_count"] >= min_frames
        }

    def get_analytics(self) -> Dict[str, Any]:
        """Get segmentation analytics summary."""
        active_segments = [obj for obj in self.segmented_objects.values() if obj["is_active"]]

        return {
            "total_unique_segments": self.total_unique_segments,
            "total_frames_processed": self.total_frames_processed,
            "active_segments_count": len(active_segments),
            "tracked_segments_count": len(self.segmented_objects),
            "active_segment_ids": list(self.active_segment_ids)
        }