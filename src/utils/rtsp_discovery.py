"""RTSP stream auto-discovery utilities."""

import socket
import subprocess
import concurrent.futures
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RTSPStream:
    """Discovered RTSP stream."""
    ip: str
    port: int
    path: str
    url: str
    responsive: bool = False
    validated: bool = False

    def __str__(self) -> str:
        status = "✓" if self.validated else ("?" if self.responsive else "✗")
        return f"{status} {self.url}"


# Common RTSP ports and paths
RTSP_PORTS = [554, 8554, 5000, 8080, 4747, 1935]
RTSP_PATHS = [
    "/live/stream",
    "/stream",
    "/video",
    "/h264_ulaw.sdp",
    "/h264.sdp",
    "/cam/realmonitor",
    "/",
    "/1",
    "/stream1",
    "/live",
    "/media/video1",
]


def get_local_subnet() -> Optional[str]:
    """Get the local subnet (e.g., 192.168.50) from the default interface."""
    try:
        # Create a socket to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        # Connect to a public IP (doesn't actually send data)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        # Return subnet (first 3 octets)
        parts = local_ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.{parts[2]}"
    except Exception:
        pass
    return None


def check_port(ip: str, port: int, timeout: float = 0.5) -> bool:
    """Check if a port is open on a given IP."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def scan_host(ip: str, ports: List[int] = None, timeout: float = 0.3) -> List[Tuple[str, int]]:
    """Scan a single host for open RTSP ports."""
    if ports is None:
        ports = RTSP_PORTS

    open_ports = []
    for port in ports:
        if check_port(ip, port, timeout):
            open_ports.append((ip, port))
    return open_ports


def scan_subnet(
    subnet: str = None,
    ports: List[int] = None,
    timeout: float = 0.3,
    max_workers: int = 50,
    progress_callback=None
) -> List[Tuple[str, int]]:
    """Scan entire subnet for open RTSP ports."""
    if subnet is None:
        subnet = get_local_subnet()
        if subnet is None:
            print("Could not determine local subnet")
            return []

    if ports is None:
        ports = RTSP_PORTS

    print(f"\nScanning subnet {subnet}.0/24 for RTSP streams...")
    print(f"Checking ports: {ports}")

    # Generate all IPs in subnet
    ips = [f"{subnet}.{i}" for i in range(1, 255)]

    open_endpoints = []
    scanned = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scan_host, ip, ports, timeout): ip for ip in ips}

        for future in concurrent.futures.as_completed(futures):
            scanned += 1
            if progress_callback:
                progress_callback(scanned, 254)

            try:
                results = future.result()
                if results:
                    open_endpoints.extend(results)
                    for ip, port in results:
                        print(f"  Found open port: {ip}:{port}")
            except Exception:
                pass

    return open_endpoints


def validate_rtsp_stream(url: str, timeout: int = 5) -> bool:
    """Validate if an RTSP URL is actually serving a stream using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-rtsp_transport", "tcp",
                "-i", url,
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0"
            ],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        # Check if we got video stream info
        return "video" in result.stdout.lower()
    except subprocess.TimeoutExpired:
        return False
    except FileNotFoundError:
        # ffprobe not installed, skip validation
        return False
    except Exception:
        return False


def discover_rtsp_streams(
    subnet: str = None,
    ports: List[int] = None,
    paths: List[str] = None,
    validate: bool = True,
    timeout: float = 0.3,
    max_workers: int = 50
) -> List[RTSPStream]:
    """
    Discover RTSP streams on the local network.

    Args:
        subnet: Subnet to scan (e.g., "192.168.50"). Auto-detected if None.
        ports: List of ports to check. Uses RTSP_PORTS if None.
        paths: List of RTSP paths to try. Uses RTSP_PATHS if None.
        validate: Whether to validate streams with ffprobe.
        timeout: Socket timeout for port scanning.
        max_workers: Max concurrent threads for scanning.

    Returns:
        List of discovered RTSPStream objects.
    """
    if paths is None:
        paths = RTSP_PATHS

    # Step 1: Find open ports
    print("\n" + "=" * 50)
    print("RTSP STREAM AUTO-DISCOVERY")
    print("=" * 50)

    open_endpoints = scan_subnet(subnet, ports, timeout, max_workers)

    if not open_endpoints:
        print("\nNo open RTSP ports found on the network.")
        return []

    print(f"\nFound {len(open_endpoints)} potential RTSP endpoint(s)")

    # Step 2: Build stream URLs
    streams = []
    for ip, port in open_endpoints:
        for path in paths:
            url = f"rtsp://{ip}:{port}{path}"
            streams.append(RTSPStream(
                ip=ip,
                port=port,
                path=path,
                url=url,
                responsive=True
            ))

    # Step 3: Validate streams (optional)
    if validate and streams:
        print(f"\nValidating {len(streams)} potential stream URL(s)...")
        print("(This may take a moment...)\n")

        validated_streams = []
        for i, stream in enumerate(streams):
            print(f"  [{i+1}/{len(streams)}] Testing {stream.url}...", end=" ", flush=True)
            if validate_rtsp_stream(stream.url):
                stream.validated = True
                print("✓ VALID")
                validated_streams.append(stream)
            else:
                print("✗")

        # Return validated streams first, then responsive ones
        if validated_streams:
            return validated_streams

    # Return all responsive streams if no validation or none validated
    return [s for s in streams if s.responsive]


def print_discovered_streams(streams: List[RTSPStream]) -> None:
    """Print discovered streams in a formatted table."""
    if not streams:
        print("\nNo RTSP streams discovered.")
        return

    print("\n" + "=" * 50)
    print("DISCOVERED RTSP STREAMS")
    print("=" * 50)

    validated = [s for s in streams if s.validated]
    responsive = [s for s in streams if s.responsive and not s.validated]

    if validated:
        print(f"\n✓ Validated streams ({len(validated)}):")
        for i, stream in enumerate(validated, 1):
            print(f"  {i}. {stream.url}")

    if responsive and not validated:
        print(f"\n? Responsive endpoints ({len(responsive)}):")
        print("  (Could not validate - ffprobe may not be installed)")
        for i, stream in enumerate(responsive, 1):
            print(f"  {i}. {stream.url}")

    print()


def select_stream(streams: List[RTSPStream], auto: bool = False) -> Optional[str]:
    """
    Let user select a stream or auto-select first validated one.

    Args:
        streams: List of discovered streams.
        auto: If True, automatically select first validated stream.

    Returns:
        Selected RTSP URL or None.
    """
    if not streams:
        return None

    # Sort: validated first
    streams = sorted(streams, key=lambda s: (not s.validated, s.url))

    if auto:
        # Auto-select first validated, or first responsive
        validated = [s for s in streams if s.validated]
        if validated:
            print(f"\nAuto-selected: {validated[0].url}")
            return validated[0].url
        elif streams:
            print(f"\nAuto-selected (unvalidated): {streams[0].url}")
            return streams[0].url
        return None

    # Interactive selection
    print_discovered_streams(streams)

    print("Enter stream number to connect (or 'q' to quit): ", end="", flush=True)
    try:
        choice = input().strip()
        if choice.lower() == 'q':
            return None

        idx = int(choice) - 1
        if 0 <= idx < len(streams):
            return streams[idx].url
        else:
            print("Invalid selection")
            return None
    except (ValueError, EOFError, KeyboardInterrupt):
        return None
