"""RT-DETR object detection implementation."""

import os
import shutil
import numpy as np
from typing import List, Dict, Any, Tuple

from .base import BaseDetector

# Detection category colors (aligned with C4ISR styling)
DETECTION_CATEGORIES = {
    "vehicle": {"color": (255, 165, 0), "priority": "high"},      # Orange
    "person": {"color": (0, 255, 255), "priority": "high"},       # Cyan
    "animal": {"color": (0, 255, 0), "priority": "normal"},       # Green
    "object": {"color": (128, 128, 128), "priority": "normal"},   # Gray
}

# COCO class to category mapping
COCO_CATEGORIES = {
    "person": "person",
    "bicycle": "vehicle", "car": "vehicle", "motorcycle": "vehicle",
    "airplane": "vehicle", "bus": "vehicle", "train": "vehicle",
    "truck": "vehicle", "boat": "vehicle",
    "bird": "animal", "cat": "animal", "dog": "animal", "horse": "animal",
    "sheep": "animal", "cow": "animal", "elephant": "animal", "bear": "animal",
    "zebra": "animal", "giraffe": "animal",
}


class RTDETRDetector(BaseDetector):
    """RT-DETR real-time object detection with tracking."""

    def __init__(self, args, model_config):
        super().__init__(args, model_config)
        self.colors = self._generate_colors()

    def _generate_colors(self) -> List[Tuple[int, int, int]]:
        """Generate colors for COCO classes."""
        np.random.seed(42)
        return [(int(c[0]), int(c[1]), int(c[2]))
                for c in np.random.randint(0, 255, size=(80, 3))]

    def _get_category(self, class_name: str) -> str:
        """Get category for a class name."""
        return COCO_CATEGORIES.get(class_name.lower(), "object")
    
    async def load_model(self) -> None:
        """Load RT-DETR model."""
        from ultralytics import RTDETR
        
        print("Loading RT-DETR model...")
        
        # Setup model path
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        models_dir = os.path.join(script_dir, "models")
        model_path = os.path.join(models_dir, self.model_config.model_file)
        
        os.makedirs(models_dir, exist_ok=True)
        
        # Download model if needed
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Downloading RT-DETR-l model...")
            temp_model = RTDETR(self.model_config.model_file)
            default_model_path = os.path.expanduser(f"~/.ultralytics/weights/{self.model_config.model_file}")
            if os.path.exists(default_model_path):
                shutil.copy(default_model_path, model_path)
                print(f"Model saved to: {model_path}")
        
        # Load model
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            self.model = RTDETR(model_path)
            print(f"✓ RT-DETR model loaded successfully")
            print(f"  Confidence threshold: {self.confidence_threshold}")
            print()
        else:
            print(f"Error: Could not load RT-DETR model")
            raise RuntimeError("Failed to load RT-DETR model")
    
    def process_frame(self, frame: Any, frame_timestamp: str,
                     frame_count: int) -> Tuple[List[Dict[str, Any]], Any]:
        """Process frame with RT-DETR detection and tracking."""
        # Run RT-DETR with tracking enabled for persistent IDs (like C4ISR)
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
            persist=True,  # Maintain tracking IDs across frames
            tracker="bytetrack.yaml"  # Use ByteTrack for robust tracking
        )

        result = results[0]
        h, w = frame.shape[:2]
        detections = []
        current_track_ids = set()

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            # Get persistent tracking IDs
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
            else:
                # Fallback to index-based IDs if tracking fails
                track_ids = list(range(len(boxes)))

            # Process each tracked detection
            for box, conf, cls_id, native_track_id in zip(boxes, confidences, class_ids, track_ids):
                x1, y1, x2, y2 = box

                # Get class name from COCO classes
                class_name = result.names[cls_id]

                # Get category for this class
                category = self._get_category(class_name)

                # Get or create CUID using centralized service
                cuid = self.tracking_id_service.get_or_create_cuid(
                    native_id=native_track_id,
                    model_type=self.model_type
                )
                current_track_ids.add(cuid)

                # Normalize bbox coordinates
                bbox = {
                    "x_min": float(x1 / w),
                    "y_min": float(y1 / h),
                    "x_max": float(x2 / w),
                    "y_max": float(y2 / h)
                }

                # Create standardized detection payload (aligned with C4ISR format)
                detection = self.tracking_id_service.format_detection_payload(
                    track_id=cuid,
                    label=class_name,
                    confidence=float(conf),
                    bbox=bbox,
                    timestamp=frame_timestamp,
                    model_type=self.model_type,
                    native_id=native_track_id,
                    # Category metadata (similar to threat_level in C4ISR)
                    category=category,
                    priority=DETECTION_CATEGORIES.get(category, {}).get("priority", "normal")
                )

                detections.append(detection)

        # Visualize detections with category styling
        frame = self._visualize_detections(frame, detections)

        return detections, frame
    
    def _visualize_detections(self, frame: Any, detections: List[Dict[str, Any]]) -> Any:
        """Visualize RT-DETR detections with category styling."""
        import cv2
        h, w = frame.shape[:2]

        # Count detections by category
        category_counts = {"person": 0, "vehicle": 0, "animal": 0, "object": 0}

        for detection in detections:
            # Get category color
            category = detection.get("metadata", {}).get("category", "object")
            if category in category_counts:
                category_counts[category] += 1
            color = DETECTION_CATEGORIES.get(category, DETECTION_CATEGORIES["object"])["color"]

            bbox = detection["bbox"]
            x1, y1 = int(bbox["x_min"] * w), int(bbox["y_min"] * h)
            x2, y2 = int(bbox["x_max"] * w), int(bbox["y_max"] * h)

            # Draw main bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw corner markers for professional look (like C4ISR)
            corner_length = 15
            corner_thickness = 3

            # Top-left corner
            cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)

            # Top-right corner
            cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)

            # Bottom-left corner
            cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)

            # Bottom-right corner
            cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)

            # Draw label with category
            label_text = f"[{category.upper()}] {detection['label']} {detection['confidence']:.2f}"
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 15

            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

            # Draw label background
            padding = 3
            cv2.rectangle(
                frame,
                (x1, text_y - text_height - padding),
                (x1 + text_width + padding * 2, text_y + padding),
                color,
                -1
            )

            # Draw label text
            cv2.putText(frame, label_text, (x1 + padding, text_y),
                       font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Add status overlay
        self._add_status_overlay(frame, category_counts)

        return frame

    def _add_status_overlay(self, frame: Any, category_counts: Dict[str, int]) -> None:
        """Add RT-DETR status overlay."""
        import cv2
        h, w = frame.shape[:2]

        # Status overlay background
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

        # Detection status
        total = sum(category_counts.values())
        if total > 0:
            status_color = (0, 255, 255)  # Cyan
            status_text = f"RT-DETR ACTIVE | {total} objects"
        else:
            status_color = (0, 255, 0)  # Green
            status_text = "RT-DETR SCANNING"

        cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Category counts
        counts_text = f"Person: {category_counts['person']} | Vehicle: {category_counts['vehicle']} | Animal: {category_counts['animal']} | Other: {category_counts['object']}"
        cv2.putText(frame, counts_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def get_detection_color(self, detection: Dict[str, Any]) -> Tuple[int, int, int]:
        """Get color for detection based on category."""
        category = detection.get("metadata", {}).get("category", "object")
        return DETECTION_CATEGORIES.get(category, DETECTION_CATEGORIES["object"])["color"]