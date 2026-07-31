#!/usr/bin/env python3
"""Whitespace-safe raw NKey seed-file adapter."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deploy import vision_service, vision_service_nkey
from src.services.communication.service import OverwatchCommunication


async def _connect_with_trimmed_seed(self: OverwatchCommunication) -> None:
    seed_path = os.environ.get("NATS_NKEY_SEED_FILE", "").strip()
    if not seed_path:
        raise RuntimeError("NATS_NKEY_SEED_FILE is required")

    secure_path = vision_service_nkey._secure_file(seed_path, "NKey seed")
    seed = Path(secure_path).read_text(encoding="ascii").strip()
    if not seed.startswith("S"):
        raise RuntimeError("NKey seed file does not contain a valid seed")

    os.environ.pop("NATS_NKEY_SEED_FILE", None)
    os.environ["NATS_NKEY_SEED_STR"] = seed
    try:
        await vision_service_nkey._connect_nats_nkey(self)
    finally:
        os.environ.pop("NATS_NKEY_SEED_STR", None)


async def main() -> None:
    OverwatchCommunication._connect_nats = _connect_with_trimmed_seed
    await vision_service.main()


if __name__ == "__main__":
    asyncio.run(main())
