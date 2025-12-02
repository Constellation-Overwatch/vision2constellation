"""Video service for camera capture and display management."""

import cv2
import platform
import subprocess
import json
import sys
import time
import os
from typing import Dict, Any, Optional, Tuple, List
from argparse import Namespace

from ...config.defaults import DEFAULT_CONFIG
from ...utils.device import enumerate_video_devices, print_device_list
from ...utils.rtsp_discovery import discover_rtsp_streams, select_stream

class VideoService:
    """Service for managing video capture and display."""

    def __init__(self, args: Namespace):
        self.args = args
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_source = None
        self.source_type = None
        self.selected_device = None
        self.window_title = None

        # Configuration
        self.video_config = DEFAULT_CONFIG["video"]

        # RTSP resilience settings
        self.max_consecutive_failures = int(os.getenv("RTSP_MAX_FAILURES", "30"))
        self.reconnect_delay = float(os.getenv("RTSP_RECONNECT_DELAY", "1.0"))
        self.consecutive_failures = 0
        self.last_good_frame = None
        self.total_reconnects = 0
    
    def determine_video_source(self) -> Tuple[Any, str, Optional[Dict]]:
        """Determine video source from arguments."""
        if self.args.list_devices:
            devices = enumerate_video_devices()
            print_device_list(devices)
            sys.exit(0)
        
        # Direct camera index
        if self.args.camera is not None:
            return self._setup_camera_source()
        
        # Device path
        elif self.args.device:
            return self._setup_device_source()
        
        # RTSP URL
        elif self.args.rtsp:
            return self._setup_rtsp_source()

        # RTSP Discovery
        elif getattr(self.args, 'rtsp_discover', False):
            return self._setup_rtsp_discovery()
        
        # HTTP stream
        elif self.args.http:
            return self._setup_http_source()
        
        # Auto-detect
        else:
            return self._auto_detect_source()
    
    def _setup_camera_source(self) -> Tuple[int, str, None]:
        """Setup direct camera index source."""
        print(f"\n=== Camera Mode ===")
        print(f"Using camera index: {self.args.camera}")
        print("===================\n")
        return self.args.camera, "camera", None
    
    def _setup_device_source(self) -> Tuple[str, str, None]:
        """Setup device path source."""
        print(f"\n=== Device Mode ===") 
        print(f"Using device: {self.args.device}")
        print("===================\n")
        return self.args.device, "device", None
    
    def _setup_rtsp_source(self) -> Tuple[str, str, None]:
        """Setup RTSP stream source with concurrent access optimizations."""
        print(f"\n=== RTSP Stream Mode ===")
        print(f"Connecting to: {self.args.rtsp}")
        print(f"Resilience: max_failures={self.max_consecutive_failures}, reconnect_delay={self.reconnect_delay}s")
        print("========================\n")
        return self.args.rtsp, "rtsp", None

    def _setup_rtsp_discovery(self) -> Tuple[str, str, None]:
        """Discover and setup RTSP stream source."""
        auto_select = getattr(self.args, 'auto', False)

        # Discover streams on the network
        streams = discover_rtsp_streams(validate=True)

        if not streams:
            print("\nNo RTSP streams found on the network.")
            print("Try specifying a stream directly with --rtsp <url>")
            sys.exit(1)

        # Select stream (auto or interactive)
        selected_url = select_stream(streams, auto=auto_select)

        if not selected_url:
            print("\nNo stream selected. Exiting.")
            sys.exit(0)

        print(f"\n=== RTSP Stream Mode (Discovered) ===")
        print(f"Connecting to: {selected_url}")
        print("=====================================\n")
        return selected_url, "rtsp", None
    
    def _setup_http_source(self) -> Tuple[str, str, None]:
        """Setup HTTP stream source."""
        print(f"\n=== HTTP Stream Mode ===")
        print(f"Connecting to: {self.args.http}")
        print("========================\n")
        return self.args.http, "http", None
    
    def _auto_detect_source(self) -> Tuple[Any, str, Optional[Dict]]:
        """Auto-detect video source."""
        print("\n=== Auto-detecting video source ===")
        devices = enumerate_video_devices()
        
        # Filter out native cameras if requested
        if self.args.skip_native and devices:
            non_native = [d for d in devices if not d.get('is_native', False)]
            if non_native:
                devices = non_native
                print("Skipping native/built-in cameras...")
            else:
                print("\nError: --skip-native specified but no external cameras found!")
                self._print_available_devices(devices)
                print("\nPlease:")
                print("  1. Connect an external camera/capture device")
                print("  2. Run with --list-devices to see available devices")
                print("  3. Or remove --skip-native to use built-in camera")
                sys.exit(1)
        
        if devices:
            selected_device = devices[0]
            video_source = selected_device.get('index', selected_device.get('path', 0))
            source_type = "camera"
            print(f"Found {len(devices)} device(s)")
            print(f"Selected: {selected_device.get('name', 'Unknown')}")
            print(f"  Index: {selected_device.get('index', selected_device.get('path', 'N/A'))}")
            print(f"  Resolution: {selected_device.get('resolution', 'N/A')}")
            print(f"  FPS: {selected_device.get('fps', 'N/A')}")
            print("===================================\n")
            return video_source, source_type, selected_device
        else:
            if self.args.skip_native:
                print("\nError: No cameras detected!")
                print("  1. Connect a camera/capture device") 
                print("  2. Run with --list-devices to see available devices")
                print("===================================\n")
                sys.exit(1)
            else:
                # Fallback to default camera
                print(f"No devices detected, trying default camera (index 0)")
                print("===================================\n")
                return 0, "camera", None
    
    def _print_available_devices(self, devices: List[Dict]) -> None:
        """Print available devices for error messages."""
        print("\nAvailable devices:")
        for dev in devices:
            native_status = f" (native: {dev.get('is_native', False)})"
            print(f"  - {dev.get('name', 'Unknown')}{native_status}")
    
    def open_video_stream(self) -> bool:
        """Open video stream based on determined source."""
        self.video_source, self.source_type, self.selected_device = self.determine_video_source()
        
        # Open video capture with platform-specific optimizations
        if (self.source_type == "camera" and isinstance(self.video_source, int) and 
            platform.system() == 'Darwin'):
            # macOS: use AVFoundation explicitly
            self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_AVFOUNDATION)
            print(f"Opening camera index {self.video_source} with AVFoundation backend...")
        elif self.source_type in ["rtsp", "http"]:
            # For RTSP/HTTP streams, try FFmpeg backend explicitly with options
            self.cap = self._open_stream_with_retries()
            if self.cap is None:
                return False
        else:
            self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video source: {self.video_source}")
            print(f"Source type: {self.source_type}")
            self._print_troubleshooting()
            return False
        
        # Apply optimizations
        self._apply_stream_optimizations()
        
        # Verify opened device
        if self.source_type == "camera" and isinstance(self.video_source, int):
            self._verify_camera_device()
        
        return True
    
    def _open_stream_with_retries(self) -> Optional[cv2.VideoCapture]:
        """Open RTSP/HTTP stream with backend selection and retry logic."""
        # For RTSP, construct URL with FFmpeg options for concurrent access
        stream_url = self.video_source

        if self.source_type == "rtsp":
            # FFmpeg options for robust RTSP streaming over network:
            # - rtsp_transport=tcp: Use TCP instead of UDP to avoid packet loss
            # - buffer_size=1024000: Larger buffer for network jitter (1MB)
            # - max_delay=500000: Max demuxing delay in microseconds
            # - stimeout=5000000: Socket timeout 5 seconds
            # - reorder_queue_size=500: Allow packet reordering for network delays
            # - fflags=discardcorrupt: Discard corrupted frames instead of erroring
            # - err_detect=ignore_err: Continue on decode errors
            # - loglevel=quiet: Suppress FFmpeg warnings
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|"
                "buffer_size;1024000|"
                "max_delay;500000|"
                "stimeout;5000000|"
                "reorder_queue_size;500|"
                "fflags;discardcorrupt|"
                "err_detect;ignore_err|"
                "loglevel;quiet"
            )
            print("Applied FFmpeg RTSP options (TCP, 1MB buffer, error tolerance)")

        # Try different backends in order of preference
        backends = [
            (cv2.CAP_FFMPEG, "FFMPEG"),
            (cv2.CAP_ANY, "AUTO"),
        ]

        for backend_id, backend_name in backends:
            print(f"Attempting to open stream with {backend_name} backend...")

            cap = cv2.VideoCapture(stream_url, backend_id)

            # Configure stream options
            if self.source_type == "rtsp":
                # RTSP-specific options for network stability
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)  # Larger buffer for network jitter
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)  # 15 second timeout
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)  # 10 second read timeout

            # Try to read a test frame to verify the stream works
            if cap.isOpened():
                print(f"Stream opened with {backend_name}, testing...")
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ Successfully connected with {backend_name} backend")
                    self.last_good_frame = frame.copy()
                    return cap
                else:
                    print(f"✗ Stream opened but failed to read frame with {backend_name}")
                    cap.release()
            else:
                print(f"✗ Failed to open with {backend_name} backend")

        print(f"\n✗ Failed to open stream with any backend")
        self._print_troubleshooting()
        return None
    
    def _apply_stream_optimizations(self) -> None:
        """Apply source-specific optimizations."""
        if self.source_type in ["rtsp", "http"]:
            # Low-latency streaming
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.video_config["buffer_size"])
        elif (self.source_type == "camera" and isinstance(self.video_source, int) and 
              self.video_source > 0):
            # External capture devices
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.video_config["buffer_size"])
            self.cap.set(cv2.CAP_PROP_FPS, self.video_config["target_fps"])
            self.cap.set(cv2.CAP_PROP_FOURCC, 
                        cv2.VideoWriter_fourcc(*self.video_config["optimization_codec"]))
            
            print(f"Applied optimizations for external capture device")
            print(f"  Buffer: Minimal for low latency")
            print(f"  Target FPS: {self.video_config['target_fps']}\n")
    
    def _verify_camera_device(self) -> None:
        """Verify the correct camera was opened."""
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        actual_backend = self.cap.getBackendName()
        
        print(f"\n=== Video Stream Verification ===")
        print(f"Requested index: {self.video_source}")
        print(f"Resolution: {actual_width}x{actual_height}")
        print(f"FPS: {actual_fps}")
        print(f"Backend: {actual_backend}")
        
        # Try to get actual camera name on macOS
        actual_camera_name = "Unknown"
        if platform.system() == 'Darwin':
            try:
                result = subprocess.run(
                    ['system_profiler', 'SPCameraDataType', '-json'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    if ('SPCameraDataType' in data and 
                        len(data['SPCameraDataType']) > self.video_source):
                        actual_camera_name = data['SPCameraDataType'][self.video_source].get('_name', 'Unknown')
            except:
                pass
        
        print(f"Actual camera: {actual_camera_name}")
        
        # Warn if wrong camera
        if (self.selected_device and actual_camera_name != "Unknown"):
            expected_name = self.selected_device.get('name', '')
            if expected_name and expected_name != actual_camera_name:
                print(f"\n⚠️  WARNING: Requested '{expected_name}' but opened '{actual_camera_name}'!")
                print(f"⚠️  OpenCV may have opened the wrong camera index!")
                print(f"⚠️  Try using --camera {self.video_source} explicitly or check connections.")
        
        print("=================================\n")
    
    def _print_troubleshooting(self) -> None:
        """Print troubleshooting information."""
        if self.source_type == "rtsp":
            print("\nTroubleshooting RTSP:")
            print("  1. Verify the RTSP server is running")
            print("  2. Check network connectivity")
            print("  3. Confirm the RTSP URL is correct")
        elif self.source_type == "camera":
            print("\nTroubleshooting Camera:")
            print("  1. Check if camera is connected")
            print("  2. Run with --list-devices to see available devices")
            print("  3. Try a different camera index")
    
    def setup_display_window(self, camera_name: str, mode_name: str) -> None:
        """Setup OpenCV display window."""
        self.window_title = f'Constellation Overwatch - {mode_name} - {camera_name}'
        
        # Create resizable window
        cv2.namedWindow(self.window_title, 
                       cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        
        # Set window size
        window_width = self.video_config["window_width"]
        window_height = self.video_config["window_height"]
        cv2.resizeWindow(self.window_title, window_width, window_height)
        
        # Set fixed position to ensure it's always visible and easy to drag
        # Using a safe offset from top-left rather than centering based on potentially incorrect screen dimensions
        x_pos = 100
        y_pos = 100
        cv2.moveWindow(self.window_title, x_pos, y_pos)
        
        print(f"\nOpenCV Window Setup:")
        print(f"  Title: '{self.window_title}'")
        print(f"  Size: {window_width}x{window_height}")
        print(f"  Position: ({x_pos}, {y_pos})")
        print(f"  The window is draggable and resizable\n")
    
    def read_frame(self) -> Tuple[bool, Any]:
        """Read frame from video capture with error recovery for RTSP streams."""
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()

        # For RTSP/HTTP streams, handle frame errors gracefully
        if self.source_type in ["rtsp", "http"]:
            if ret and frame is not None:
                # Success - reset failure counter and cache frame
                self.consecutive_failures = 0
                self.last_good_frame = frame.copy()
                return True, frame
            else:
                # Frame read failed
                self.consecutive_failures += 1

                # Check if we should attempt reconnection
                if self.consecutive_failures >= self.max_consecutive_failures:
                    print(f"\n⚠️  {self.consecutive_failures} consecutive frame failures - attempting reconnect...")
                    if self._reconnect_stream():
                        self.consecutive_failures = 0
                        # Try reading again after reconnect
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            self.last_good_frame = frame.copy()
                            return True, frame

                    # Reconnect failed - signal to stop
                    print(f"✗ Reconnection failed after {self.total_reconnects} attempts")
                    return False, None

                # Return last good frame to maintain continuity (skip corrupted frame)
                if self.last_good_frame is not None:
                    return True, self.last_good_frame

                # No cached frame available
                return False, None

        # For camera sources, use standard behavior
        return ret, frame

    def _reconnect_stream(self) -> bool:
        """Attempt to reconnect to RTSP/HTTP stream."""
        self.total_reconnects += 1
        print(f"Reconnection attempt #{self.total_reconnects}...")

        # Release current capture
        if self.cap:
            self.cap.release()
            self.cap = None

        # Brief delay before reconnecting
        time.sleep(self.reconnect_delay)

        # Attempt to reopen
        self.cap = self._open_stream_with_retries()

        if self.cap and self.cap.isOpened():
            print(f"✓ Successfully reconnected to stream")
            return True

        print(f"✗ Failed to reconnect")
        return False
    
    def display_frame(self, frame: Any) -> bool:
        """Display frame and check for quit key."""
        if self.window_title:
            cv2.imshow(self.window_title, frame)
            return cv2.waitKey(1) & 0xFF == ord('q')
        return False
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Get frame height and width."""
        if self.cap is None:
            return 0, 0
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return height, width
    
    def get_selected_device(self) -> Optional[Dict]:
        """Get selected device information."""
        return self.selected_device
    
    def cleanup(self) -> None:
        """Clean up video resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Video resources cleaned up")