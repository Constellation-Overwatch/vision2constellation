#!/usr/bin/env bash
set -euo pipefail

required_values=(
  VISION_INPUT_RTSP_URL
  VISION_OUTPUT_RTSP_URL
  CONSTELLATION_ORG_ID
  CONSTELLATION_ENTITY_ID
  NATS_URL
)

for name in "${required_values[@]}"; do
  value="${!name:-}"
  if [[ -z "${value}" || "${value}" == REPLACE_* ]]; then
    echo "Required configuration is missing: ${name}" >&2
    exit 78
  fi
done

auth_count=0
for name in NATS_CREDS_FILE NATS_NKEY_SEED_FILE NATS_NKEY_SEED_STR; do
  if [[ -n "${!name:-}" ]]; then
    auth_count=$((auth_count + 1))
  fi
done

if [[ "${auth_count}" -ne 1 ]]; then
  echo "Configure exactly one NATS NKey credential mode" >&2
  exit 78
fi

if [[ "${VISION_INPUT_RTSP_URL}" == "${VISION_OUTPUT_RTSP_URL}" ]]; then
  echo "VISION_INPUT_RTSP_URL and VISION_OUTPUT_RTSP_URL must differ" >&2
  exit 78
fi

exec .venv/bin/python deploy/vision_service_nkey.py
