"""Logging utilities."""

import os
import cv2
from ..config.defaults import setup_opencv_environment

def setup_logging():
    """Setup logging configuration for the application."""
    # Setup OpenCV environment to suppress logging
    setup_opencv_environment()

    # Suppress FFmpeg/libav warnings (h264 decode errors, RTSP packet warnings)
    # These are common with network streams and handled by our resilience layer
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # AV_LOG_QUIET
    os.environ["AV_LOG_FORCE_NOCOLOR"] = "1"

    # Alternative: set via FFmpeg capture options (backup)
    # loglevel=quiet suppresses all ffmpeg output
    if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
        current = os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
        if "loglevel" not in current:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = current + "|loglevel;quiet"
    else:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"

    # Set OpenCV log level
    cv2.setLogLevel(0)

    print("Logging configured: OpenCV and FFmpeg messages suppressed")

def enable_verbose_logging():
    """Enable verbose logging for debugging."""
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "32"  # AV_LOG_INFO
    cv2.setLogLevel(3)  # Show errors
    print("Verbose logging enabled")