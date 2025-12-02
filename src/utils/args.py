"""Argument parsing utilities."""

import argparse
from argparse import Namespace
from ..config.models import DetectionMode

def parse_arguments() -> Namespace:
    """Parse command line arguments with all detection modes and options."""
    parser = argparse.ArgumentParser(
        description='Constellation Overwatch ISR Detection System'
    )
    
    # Model selection
    parser.add_argument(
        '--model', 
        type=str,
        choices=[mode.value for mode in DetectionMode],
        default=DetectionMode.YOLOE_C4ISR.value,
        help='Detection model to use (default: yoloe_c4isr for C4ISR threat detection)'
    )
    
    # Information options
    info_group = parser.add_mutually_exclusive_group()
    info_group.add_argument(
        '--list-models',
        action='store_true', 
        help='List available detection models and exit'
    )
    info_group.add_argument(
        '--list-devices', 
        action='store_true',
        help='List available video devices and exit'
    )
    
    # Video source options  
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        '--camera', 
        type=int, 
        default=None,
        help='Camera device index (e.g., 0, 1, 2)'
    )
    source_group.add_argument(
        '--device', 
        type=str, 
        default=None,
        help='Device path (e.g., /dev/video4)'
    )
    source_group.add_argument(
        '--rtsp',
        type=str,
        default=None,
        help='RTSP URL (e.g., rtsp://192.168.50.2:8554/live/stream)'
    )
    source_group.add_argument(
        '--rtsp-discover',
        action='store_true',
        help='Auto-discover RTSP streams on the local network'
    )
    source_group.add_argument(
        '--http', 
        type=str, 
        default=None,
        help='HTTP stream URL (e.g., http://192.168.1.100:8080/stream)'
    )
    
    # Legacy RTSP options (for backward compatibility)
    parser.add_argument(
        '--rtsp-ip', 
        type=str, 
        default=None,
        help='RTSP stream IP address (legacy)'
    )
    parser.add_argument(
        '--rtsp-port', 
        type=int, 
        default=8554,
        help='RTSP stream port (default: 8554)'
    )
    parser.add_argument(
        '--rtsp-path', 
        type=str, 
        default='/live/stream',
        help='RTSP stream path (default: /live/stream)'
    )
    
    # Additional camera options
    parser.add_argument(
        '--skip-native',
        action='store_true',
        help='Skip built-in/native cameras during auto-detection'
    )
    parser.add_argument(
        '--auto',
        action='store_true',
        help='Auto-select first discovered stream (use with --rtsp-discover)'
    )
    
    # Detection parameters
    parser.add_argument(
        '--conf', 
        type=float, 
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--min-frames', 
        type=int, 
        default=None,  # Will be set based on model
        help='Minimum frames to track before publishing (model-dependent default)'
    )
    
    # Model-specific parameters
    
    # YOLOE tracking options
    parser.add_argument(
        '--tracker', 
        type=str,
        default='botsort.yaml',
        choices=['botsort.yaml', 'bytetrack.yaml'],
        help='Tracker to use for YOLOE (default: botsort.yaml)'
    )
    
    # SAM2 options
    parser.add_argument(
        '--imgsz', 
        type=int, 
        default=1024,
        help='Input image size for SAM2 (default: 1024)'
    )
    
    # C4ISR options
    parser.add_argument(
        '--custom-threats', 
        type=str, 
        nargs='+', 
        default=None,
        help='Additional threat classes to detect for C4ISR mode'
    )
    
    # Moondream options
    parser.add_argument(
        '--prompt',
        type=str,
        default="Objects",
        help='Detection prompt for Moondream (default: "Objects")'
    )
    parser.add_argument(
        '--max-objects',
        type=int,
        default=50,
        help='Maximum objects to detect for Moondream (default: 50)'
    )

    # Constellation identity options
    parser.add_argument(
        '--entity-id',
        type=str,
        default=None,
        help='Constellation Entity ID (overrides CONSTELLATION_ENTITY_ID env var)'
    )
    parser.add_argument(
        '--org-id',
        type=str,
        default=None,
        help='Constellation Organization ID (overrides CONSTELLATION_ORG_ID env var)'
    )

    return parser.parse_args()

def validate_arguments(args: Namespace) -> Namespace:
    """Validate and adjust arguments based on model selection."""
    detection_mode = DetectionMode(args.model)
    
    # Set model-specific defaults if not provided
    if args.min_frames is None:
        if detection_mode == DetectionMode.YOLOE_C4ISR:
            args.min_frames = 1  # Immediate threat alerts
        elif detection_mode in [DetectionMode.YOLOE, DetectionMode.SAM2]:
            args.min_frames = 3  # Reduce noise for tracking/segmentation
        else:
            args.min_frames = 1  # Immediate detection
    
    # Build legacy RTSP URL if using legacy options
    if args.rtsp_ip:
        args.rtsp = f"rtsp://{args.rtsp_ip}:{args.rtsp_port}{args.rtsp_path}"
    
    return args