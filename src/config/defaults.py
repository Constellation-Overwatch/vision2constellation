"""Default configuration values."""

import os

# NATS/JetStream configuration
NATS_CONFIG = {
    "url": os.getenv("NATS_URL", "nats://localhost:4222"),
    "auth_token": os.getenv("NATS_AUTH_TOKEN"),
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

# Video frame streaming configuration
FRAME_STREAM_CONFIG = {
    "enabled": os.getenv("ENABLE_FRAME_STREAMING", "false").lower() == "true",
    "codec": os.getenv("FRAME_CODEC", "jpeg"),  # "jpeg" (default) or "h264"
    "target_fps": int(os.getenv("FRAME_TARGET_FPS", "15")),
    "include_detections": os.getenv("FRAME_INCLUDE_DETECTIONS", "true").lower() == "true",
    "stream_name": os.getenv("NATS_VIDEO_STREAM_NAME", "CONSTELLATION_VIDEO_FRAMES"),
    "subject_root": os.getenv("NATS_VIDEO_SUBJECT_ROOT", "constellation.video"),
    # H.264/MPEG-TS settings (WebRTC optimized - default)
    "h264_width": int(os.getenv("FRAME_H264_WIDTH", "1280")),
    "h264_height": int(os.getenv("FRAME_H264_HEIGHT", "720")),
    "h264_bitrate": os.getenv("FRAME_H264_BITRATE", "1500k"),
    "h264_gop_size": int(os.getenv("FRAME_H264_GOP_SIZE", "30")),
    # JPEG fallback settings (legacy)
    "jpeg_quality": int(os.getenv("FRAME_JPEG_QUALITY", "75")),
    "max_dimension": int(os.getenv("FRAME_MAX_DIMENSION", "1280")),
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
    "opencv": OPENCV_CONFIG,
    "frame_stream": FRAME_STREAM_CONFIG
}

# Model directories
MODEL_DIRS = {
    "models_dir": "models",
    "cache_dir": os.path.expanduser("~/.ultralytics/weights/")
}