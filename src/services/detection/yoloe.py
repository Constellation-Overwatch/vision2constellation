"""YOLOE object tracking implementation."""

import numpy as np
from typing import List, Dict, Any, Tuple, Set

from .base import BaseDetector, load_ultralytics_model

class YOLOEDetector(BaseDetector):
    """YOLOE object detection with tracking capabilities."""
    
    def __init__(self, args, model_config):
        super().__init__(args, model_config)
        self.tracker = getattr(args, 'tracker', 'botsort.yaml')
        self.colors = self._generate_colors()
        
    def _generate_colors(self) -> List[Tuple[int, int, int]]:
        """Generate colors for COCO classes."""
        np.random.seed(42)
        return [(int(c[0]), int(c[1]), int(c[2])) 
                for c in np.random.randint(0, 255, size=(80, 3))]
    
    async def load_model(self) -> None:
        """Load YOLOE model with tracking capabilities."""
        from ultralytics import YOLOE

        print("Loading YOLOE model...")
        self.model = load_ultralytics_model(
            YOLOE,
            self.model_config.model_file,
            "YOLOE"
        )
        print(f"✓ YOLOE model loaded successfully with {self.tracker} tracker")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Tracker: {self.tracker.replace('.yaml', '').upper()}")
        print()
    
    def process_frame(self, frame: Any, frame_timestamp: str, 
                     frame_count: int) -> Tuple[List[Dict[str, Any]], Any]:
        """Process frame with YOLOE tracking."""
        # Run YOLOE tracking (using .track() instead of .predict())
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
            persist=True,
            tracker=self.tracker
        )
        
        result = results[0]
        h, w = frame.shape[:2]
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Get tracking IDs (this is the key difference from RT-DETR)
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
            else:
                # No tracking IDs available (shouldn't happen with .track())
                track_ids = list(range(len(boxes)))
                print(f"⚠️ Frame {frame_count}: No tracking IDs available for {len(boxes)} detections!")
            
            # Process each tracked detection
            for box, conf, cls_id, yolo_track_id in zip(boxes, confidences, class_ids, track_ids):
                x1, y1, x2, y2 = box

                # Get class name from COCO classes
                class_name = result.names[cls_id]

                # Normalize bbox coordinates
                bbox = {
                    "x_min": float(x1 / w),
                    "y_min": float(y1 / h),
                    "x_max": float(x2 / w),
                    "y_max": float(y2 / h)
                }

                # Get or create stable CUID using spatial properties
                cuid = self.tracking_id_service.get_stable_cuid(
                    bbox=bbox,
                    label=class_name,
                    confidence=float(conf),
                    native_id=yolo_track_id,
                    model_type=self.model_type
                )

                # Create standardized detection payload
                detection = self.tracking_id_service.format_detection_payload(
                    track_id=cuid,
                    label=class_name,
                    confidence=float(conf),
                    bbox=bbox,
                    timestamp=frame_timestamp,
                    model_type=self.model_type,
                    native_id=yolo_track_id,
                    class_id=int(cls_id)
                )

                detections.append(detection)
        
        # Visualize detections with tracking info
        frame = self._visualize_tracked_detections(frame, detections, frame_count)
        
        return detections, frame
    
    def _visualize_tracked_detections(self, frame: Any, detections: List[Dict[str, Any]], 
                                    frame_count: int) -> Any:
        """Visualize YOLOE detections with tracking information."""
        import cv2
        h, w = frame.shape[:2]
        
        for detection in detections:
            cls_id = detection["class_id"]
            color = self.colors[cls_id % len(self.colors)]
            
            bbox = detection["bbox"]
            x1, y1 = int(bbox["x_min"] * w), int(bbox["y_min"] * h)
            x2, y2 = int(bbox["x_max"] * w), int(bbox["y_max"] * h)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with tracking ID
            track_id = detection["track_id"]
            label_text = f"ID:{track_id} {detection['label']} {detection['confidence']:.2f}"
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(frame, label_text, (x1, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def get_detection_color(self, detection: Dict[str, Any]) -> Tuple[int, int, int]:
        """Get color for detection based on class."""
        cls_id = detection.get("class_id", hash(detection["label"]) % len(self.colors))
        return self.colors[cls_id % len(self.colors)]
    
    def format_label_text(self, detection: Dict[str, Any], 
                         additional_info: str = "") -> str:
        """Format label text with tracking ID."""
        track_id = detection.get("track_id", "?")
        label = detection.get('label', 'unknown')
        confidence = detection.get('confidence', 0.0)
        base_text = f"ID:{track_id} {label} {confidence:.2f}"
        
        if additional_info:
            return f"{base_text} {additional_info}"
        return base_text