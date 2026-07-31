#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${REPO_ROOT}/.venv/bin/python" -u \
  "${REPO_ROOT}/deploy/vision_service_live_single_writer.py"
