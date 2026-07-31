#!/usr/bin/env python3
"""NKey authentication hotfix for the headless vision service."""

from __future__ import annotations

import asyncio
import os
import stat
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import nats

from deploy import vision_service
from src.services.communication.service import OverwatchCommunication


def _secure_file(path_value: str, label: str) -> str:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} file does not exist: {path}")

    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise RuntimeError(f"{label} file must not be accessible by group/other")
    return str(path)


async def _connect_nats_nkey(self: OverwatchCommunication) -> None:
    """Connect with exactly one NKey/JWT credential mode."""
    creds_file = os.getenv("NATS_CREDS_FILE", "").strip()
    seed_file = os.getenv("NATS_NKEY_SEED_FILE", "").strip()
    seed_string = os.getenv("NATS_NKEY_SEED_STR", "").strip()

    configured = [
        bool(creds_file),
        bool(seed_file),
        bool(seed_string),
    ]
    if sum(configured) != 1:
        raise RuntimeError(
            "Configure exactly one of NATS_CREDS_FILE, "
            "NATS_NKEY_SEED_FILE, or NATS_NKEY_SEED_STR"
        )

    print(f"Attempting to connect to NATS at: {self.nats_config['url']}")
    connect_opts = {
        "servers": [self.nats_config["url"]],
        "name": "vision2constellation-c4isr",
        "allow_reconnect": True,
        "reconnect_time_wait": self.nats_config["reconnect_time_wait"],
        "max_reconnect_attempts": self.nats_config["max_reconnect_attempts"],
        "connect_timeout": self.nats_config["connect_timeout"],
        "ping_interval": self.nats_config["ping_interval"],
        "max_outstanding_pings": self.nats_config["max_outstanding_pings"],
        "disconnected_cb": self._on_disconnected,
        "reconnected_cb": self._on_reconnected,
        "error_cb": self._on_error,
        "closed_cb": self._on_closed,
    }

    if creds_file:
        connect_opts["user_credentials"] = _secure_file(
            creds_file,
            "NATS credentials",
        )
        auth_mode = "JWT credentials file"
    elif seed_file:
        connect_opts["nkeys_seed"] = _secure_file(
            seed_file,
            "NKey seed",
        )
        auth_mode = "NKey seed file"
    else:
        if not seed_string.startswith("S"):
            raise RuntimeError("NATS_NKEY_SEED_STR is not a valid seed value")
        connect_opts["nkeys_seed_str"] = seed_string
        auth_mode = "NKey seed string"

    print(f"Using {auth_mode} authentication")
    self.nc = await nats.connect(**connect_opts)
    print("Connected to NATS server with NKey authentication")


async def main() -> None:
    # Patch only the token-limited connection seam. All JetStream, KV, event,
    # detector, and RTSP behavior remains in the upstream implementation.
    OverwatchCommunication._connect_nats = _connect_nats_nkey
    await vision_service.main()


if __name__ == "__main__":
    asyncio.run(main())
