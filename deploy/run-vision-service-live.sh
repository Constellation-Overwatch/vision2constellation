#!/usr/bin/env bash
set -euo pipefail

required_values=(
  VISION_INPUT_RTSP_URL
  VISION_OUTPUT_RTSP_URL
  CONSTELLATION_ORG_ID
  CONSTELLATION_ENTITY_ID
  NATS_URL
  NATS_NKEY_SEED_FILE
)

for name in "${required_values[@]}"; do
  value="${!name:-}"
  if [[ -z "${value}" || "${value}" == REPLACE_* ]]; then
    echo "Required configuration is missing: ${name}" >&2
    exit 78
  fi
done

exec .venv/bin/python -u deploy/vision_service_live.py
