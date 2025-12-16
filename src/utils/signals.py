"""Signal handling utilities."""

import signal
import asyncio
import sys
from typing import Callable, Optional

_cleanup_callback: Optional[Callable] = None
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    print("\nShutdown requested...")
    _shutdown_requested = True

def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    return _shutdown_requested

def setup_signal_handlers(cleanup_callback: Optional[Callable] = None) -> None:
    """Setup signal handlers for graceful shutdown."""
    global _cleanup_callback
    _cleanup_callback = cleanup_callback
    signal.signal(signal.SIGINT, signal_handler)