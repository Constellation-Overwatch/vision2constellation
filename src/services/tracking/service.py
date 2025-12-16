"""Tracking service for managing object state across detection modes."""

from typing import Dict, Any, Optional
from ...config.models import DetectionMode
from .state import TrackingState, C4ISRTrackingState, SegmentationState, BaseTrackingState

class TrackingService:
    """Service for managing tracking state across different detection modes."""
    
    def __init__(self, detection_mode: DetectionMode):
        self.detection_mode = detection_mode
        self.state = self._create_tracking_state(detection_mode)
        
    def _create_tracking_state(self, mode: DetectionMode) -> BaseTrackingState:
        """Create appropriate tracking state for detection mode."""
        if mode == DetectionMode.YOLOE_C4ISR:
            return C4ISRTrackingState()
        elif mode == DetectionMode.SAM2:
            return SegmentationState()
        else:
            return TrackingState()
    
    def update_frame_count(self, frame_count: int) -> None:
        """Update total frames processed."""
        self.state.total_frames_processed = frame_count
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics from current tracking state."""
        return self.state.get_analytics()
    
    def mark_inactive(self, current_ids: set) -> None:
        """Mark objects not seen in current frame as inactive."""
        self.state.mark_inactive(current_ids)
    
    def get_persistent_objects(self, min_frames: int) -> Dict[Any, Dict]:
        """Get persistently tracked objects."""
        if isinstance(self.state, SegmentationState):
            return self.state.get_persistent_segments(min_frames)
        elif hasattr(self.state, 'get_persistent_objects'):
            return self.state.get_persistent_objects(min_frames)
        else:
            return {}
    
    # Mode-specific update methods
    def update_detection(self, detection_id: Any, label: str, confidence: float,
                        bbox: Dict[str, float], frame_timestamp: str, 
                        **kwargs) -> Any:
        """Update detection in tracking state. Returns the final track_id used."""
        if isinstance(self.state, C4ISRTrackingState):
            threat_level = kwargs.get('threat_level', 'NORMAL')
            return self.state.update_object(detection_id, label, confidence, bbox, 
                                          frame_timestamp, threat_level)
        elif isinstance(self.state, SegmentationState):
            mask = kwargs.get('mask')
            area = kwargs.get('area', 0)
            return self.state.update_segment(detection_id, mask, bbox, area, 
                                           confidence, frame_timestamp)
        elif isinstance(self.state, TrackingState):
            return self.state.update_object(detection_id, label, confidence, bbox, 
                                          frame_timestamp, **kwargs)
        return detection_id
    
    def get_threat_alerts(self) -> list:
        """Get threat alerts (C4ISR mode only)."""
        if isinstance(self.state, C4ISRTrackingState):
            return self.state.threat_alerts
        return []