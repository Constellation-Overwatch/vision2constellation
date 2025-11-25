"""Frame encoding utilities for video streaming."""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional


def encode_frame(
    frame: np.ndarray,
    jpeg_quality: int = 75,
    max_dimension: int = 1280,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Encode a frame for JetStream publishing.

    Args:
        frame: BGR numpy array from OpenCV
        jpeg_quality: JPEG compression quality (1-100)
        max_dimension: Maximum width or height (scales down if larger)

    Returns:
        Tuple of (jpeg_bytes, metadata_dict)
    """
    h, w = frame.shape[:2]
    original_h, original_w = h, w
    scaled = False

    # Scale down if needed for bandwidth efficiency
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w
        scaled = True

    # Encode to JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    success, jpeg_buffer = cv2.imencode('.jpg', frame, encode_params)

    if not success:
        raise RuntimeError("Failed to encode frame to JPEG")

    jpeg_bytes = jpeg_buffer.tobytes()

    metadata = {
        "width": w,
        "height": h,
        "original_width": original_w,
        "original_height": original_h,
        "scaled": scaled,
        "format": "jpeg",
        "quality": jpeg_quality,
        "size_bytes": len(jpeg_bytes),
    }

    return jpeg_bytes, metadata


def calculate_frame_interval(target_fps: int) -> float:
    """Calculate the interval between frames for target FPS."""
    return 1.0 / target_fps if target_fps > 0 else 0.0
