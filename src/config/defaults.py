"""Default configuration values."""

import os

# NATS/JetStream configuration
NATS_CONFIG = {
    "url": os.getenv("NATS_URL", "nats://localhost:4222"),
    "subject_root": os.getenv("NATS_SUBJECT_ROOT", "constellation.events.isr"),
    "stream_name": os.getenv("NATS_STREAM_NAME", "CONSTELLATION_EVENTS"),
    "kv_store_name": os.getenv("NATS_KV_STORE_NAME", "CONSTELLATION_GLOBAL_STATE")
}

# Video configuration
VIDEO_CONFIG = {
    "default_camera_index": 0,
    "buffer_size": 1,
    "target_fps": 60,
    "optimization_codec": "MJPG",
    "window_width": 1280,
    "window_height": 720,
    "screen_width": 2560,
    "screen_height": 1440
}

# Detection configuration  
DETECTION_CONFIG = {
    "confidence_threshold": 0.25,
    "min_frames_tracking": 3,
    "min_frames_c4isr": 1,  # Immediate threat alerts
    "imgsz": 1024,  # For SAM2
    "tracker": "botsort.yaml"
}

# OpenCV configuration
OPENCV_CONFIG = {
    "log_level": "SILENT",
    "video_io_debug": "0"
}

# Environment setup
def setup_opencv_environment():
    """Setup OpenCV environment variables."""
    os.environ['OPENCV_LOG_LEVEL'] = OPENCV_CONFIG["log_level"]
    os.environ['OPENCV_VIDEOIO_DEBUG'] = OPENCV_CONFIG["video_io_debug"]

# Consolidated default configuration
DEFAULT_CONFIG = {
    "nats": NATS_CONFIG,
    "video": VIDEO_CONFIG,
    "detection": DETECTION_CONFIG,
    "opencv": OPENCV_CONFIG
}

# Model directories
MODEL_DIRS = {
    "models_dir": "models",
    "cache_dir": os.path.expanduser("~/.ultralytics/weights/")
}