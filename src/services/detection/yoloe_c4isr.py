"""C4ISR Threat Detection using YOLOE with threat classification."""

import os
import shutil
from typing import List, Dict, Any, Tuple

from .base import BaseDetector, load_ultralytics_model, get_models_dir
from ...config.threats import ALL_CLASSES, CLASS_TO_THREAT_LEVEL, THREAT_CATEGORIES, add_custom_threat_class

class C4ISRThreatDetector(BaseDetector):
    """YOLOE detector with C4ISR threat classification."""

    def __init__(self, args, model_config):
        super().__init__(args, model_config)

        # Setup threat classes
        self._setup_threat_classes()
        
    def _setup_threat_classes(self):
        """Setup threat classification classes."""
        # Add custom threats if provided
        if self.args.custom_threats:
            for threat_class in self.args.custom_threats:
                add_custom_threat_class(threat_class, "MEDIUM_THREAT")
                print(f"Added custom threat class: {threat_class}")

    async def load_model(self) -> None:
        """Load YOLOE model with C4ISR threat prompts."""
        from ultralytics import YOLOE

        print("="*70)
        print("C4ISR THREAT DETECTION INITIALIZATION")
        print("="*70)
        print("Loading YOLOE model with open-vocabulary threat prompts...\n")

        # Load model using shared utility
        self.model = load_ultralytics_model(
            YOLOE,
            self.model_config.model_file,
            "YOLOE"
        )
        print(f"✓ YOLOE model loaded successfully")

        # Setup MobileClip text encoder
        models_dir = get_models_dir()
        await self._setup_mobileclip(models_dir)

        # Configure text prompts (this may trigger MobileClip download)
        print(f"Setting text prompts for YOLOE...")
        text_embeddings = self.model.get_text_pe(ALL_CLASSES)
        self.model.set_classes(ALL_CLASSES, text_embeddings)
        print(f"✓ Text prompts configured for {len(ALL_CLASSES)} classes")

        # Move MobileClip to ./models/ if it was downloaded to CWD during get_text_pe()
        await self._cleanup_mobileclip_download(models_dir)
        
        # Print threat categories
        print(f"\nThreat Categories:")
        for threat_level, config in THREAT_CATEGORIES.items():
            print(f"  {threat_level}: {len(config['classes'])} classes")
            print(f"    Examples: {', '.join(config['classes'][:3])}")
        
        print(f"\nConfidence threshold: {self.confidence_threshold}")
        print("="*70)
        print()
    
    async def _setup_mobileclip(self, models_dir: str):
        """Setup MobileClip text encoder."""
        mobileclip_filename = "mobileclip_blt.ts"
        mobileclip_local = os.path.join(models_dir, mobileclip_filename)
        mobileclip_cache = os.path.expanduser(f"~/.ultralytics/weights/{mobileclip_filename}")
        mobileclip_cwd = os.path.join(os.getcwd(), mobileclip_filename)

        # Ensure cache directory exists
        os.makedirs(os.path.dirname(mobileclip_cache), exist_ok=True)

        # Check if MobileClip was downloaded to CWD and move it to ./models/
        if os.path.exists(mobileclip_cwd) and not os.path.exists(mobileclip_local):
            shutil.move(mobileclip_cwd, mobileclip_local)
            print(f"✓ MobileClip moved to {mobileclip_local}")

        # Handle MobileClip placement (sync between local and cache)
        if os.path.exists(mobileclip_local) and not os.path.exists(mobileclip_cache):
            shutil.copy(mobileclip_local, mobileclip_cache)
            print(f"✓ MobileClip synced to {mobileclip_cache}")
        elif os.path.exists(mobileclip_cache) and not os.path.exists(mobileclip_local):
            shutil.copy(mobileclip_cache, mobileclip_local)
            print(f"✓ MobileClip synced to {mobileclip_local}")
        elif not os.path.exists(mobileclip_local) and not os.path.exists(mobileclip_cache):
            print(f"MobileClip will download on first use (will be moved to {mobileclip_local})")

    async def _cleanup_mobileclip_download(self, models_dir: str):
        """Move MobileClip from CWD to ./models/ if it was downloaded there."""
        mobileclip_filename = "mobileclip_blt.ts"
        mobileclip_local = os.path.join(models_dir, mobileclip_filename)
        mobileclip_cwd = os.path.join(os.getcwd(), mobileclip_filename)

        if os.path.exists(mobileclip_cwd) and not os.path.exists(mobileclip_local):
            shutil.move(mobileclip_cwd, mobileclip_local)
            print(f"✓ MobileClip moved to {mobileclip_local}")

    def process_frame(self, frame: Any, frame_timestamp: str,
                     frame_count: int) -> Tuple[List[Dict[str, Any]], Any]:
        """Process frame with YOLOE C4ISR threat detection."""
        # Run YOLOE with tracking enabled for persistent IDs
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
                print(f"⚠️ Frame {frame_count}: No tracking IDs available for {len(boxes)} detections!")

            # Process each tracked detection
            for box, conf, cls_id, yolo_track_id in zip(boxes, confidences, class_ids, track_ids):
                x1, y1, x2, y2 = box

                # Get class name
                class_name = ALL_CLASSES[cls_id] if cls_id < len(ALL_CLASSES) else f"class_{cls_id}"

                # Determine threat level
                threat_level = CLASS_TO_THREAT_LEVEL.get(class_name, "NORMAL")

                # Normalize bbox for stable ID generation
                bbox = {
                    "x_min": x1 / frame.shape[1],
                    "y_min": y1 / frame.shape[0], 
                    "x_max": x2 / frame.shape[1],
                    "y_max": y2 / frame.shape[0]
                }

                # Get or create stable CUID using spatial properties
                cuid = self.tracking_id_service.get_stable_cuid(
                    bbox=bbox,
                    label=class_name,
                    confidence=float(conf),
                    native_id=yolo_track_id,
                    model_type=self.model_type
                )
                current_track_ids.add(cuid)

                # Calculate suspicious indicators
                suspicious_indicators = self._calculate_suspicious_indicators(
                    class_name, conf, threat_level
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
                    # C4ISR-specific fields
                    threat_level=threat_level,
                    suspicious_indicators=suspicious_indicators
                )
                
                detections.append(detection)
        
        # Visualize detections with C4ISR styling
        frame = self._visualize_c4isr_detections(frame, detections)
        
        return detections, frame
    
    def _calculate_suspicious_indicators(self, label: str, confidence: float, 
                                       threat_level: str) -> List[str]:
        """Calculate suspicious indicators for threat assessment."""
        indicators = []
        
        if threat_level == "HIGH_THREAT" and confidence > 0.7:
            indicators.append("high_confidence_weapon_detection")
        elif threat_level == "MEDIUM_THREAT" and confidence > 0.5:
            indicators.append("suspicious_object_detected")
        elif threat_level == "HIGH_THREAT" and confidence < 0.5:
            indicators.append("uncertain_threat_requires_validation")
        
        return indicators
    
    def _visualize_c4isr_detections(self, frame: Any, detections: List[Dict[str, Any]]) -> Any:
        """Visualize detections with C4ISR threat styling."""
        import cv2
        h, w = frame.shape[:2]

        # Count threats for alert status
        threat_counts = {"HIGH_THREAT": 0, "MEDIUM_THREAT": 0}

        for detection in detections:
            threat_level = detection["metadata"]["threat_level"]
            if threat_level in threat_counts:
                threat_counts[threat_level] += 1

            # Get threat color and draw enhanced bounding box
            color = THREAT_CATEGORIES[threat_level]["color"]
            
            bbox = detection["bbox"]
            x1, y1 = int(bbox["x_min"] * w), int(bbox["y_min"] * h)
            x2, y2 = int(bbox["x_max"] * w), int(bbox["y_max"] * h)
            
            # Draw main bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw corner markers for professional look
            corner_length = 20
            corner_thickness = 4
            
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
            
            # Draw label with threat level
            threat_label = threat_level.replace('_', ' ')
            label_text = f"[{threat_label}] {detection['label']} {detection['confidence']:.2f}"
            
            # Position label
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 15
            
            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            
            # Draw label background
            padding = 5
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
        
        # Add threat status overlay
        self._add_threat_status_overlay(frame, threat_counts)
        
        return frame
    
    def _add_threat_status_overlay(self, frame: Any, threat_counts: Dict[str, int]) -> None:
        """Add C4ISR threat status overlay."""
        import cv2
        h, w = frame.shape[:2]
        
        # Determine alert level
        if threat_counts["HIGH_THREAT"] > 0:
            alert_color = (0, 0, 255)  # Red
            alert_text = "⚠ HIGH THREAT ALERT"
        elif threat_counts["MEDIUM_THREAT"] > 0:
            alert_color = (0, 165, 255)  # Orange
            alert_text = "⚠ MEDIUM THREAT"
        else:
            alert_color = (0, 255, 0)  # Green
            alert_text = "✓ NORMAL"
        
        # Status overlay background
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        
        # Alert status
        cv2.putText(frame, alert_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, alert_color, 2)
        
        # Threat counts
        status_text = f"HIGH: {threat_counts['HIGH_THREAT']} | MED: {threat_counts['MEDIUM_THREAT']}"
        cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def get_detection_color(self, detection: Dict[str, Any]) -> Tuple[int, int, int]:
        """Get color for threat level."""
        threat_level = detection.get("metadata", {}).get("threat_level", "NORMAL")
        return THREAT_CATEGORIES[threat_level]["color"]