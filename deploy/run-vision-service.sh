#!/usr/bin/env bash
set -euo pipefail

required_values=(
  VISION_INPUT_RTSP_URL
  VISION_OUTPUT_RTSP_URL
  CONSTELLATION_ORG_ID
  CONSTELLATION_ENTITY_ID
  NATS_URL
  NATS_AUTH_TOKEN
)

for name in "${required_values[@]}"; do
  value="${!name:-}"
  if [[ -z "${value}" || "${value}" == REPLACE_* ]]; then
    echo "Required configuration is missing: ${name}" >&2
    exit 78
  fi
done

if [[ "${VISION_INPUT_RTSP_URL}" == "${VISION_OUTPUT_RTSP_URL}" ]]; then
  echo "VISION_INPUT_RTSP_URL and VISION_OUTPUT_RTSP_URL must differ" >&2
  exit 78
fi

if [[ "${VISION_INPUT_RTSP_URL}" != rtsp://* && "${VISION_INPUT_RTSP_URL}" != rtsps://* ]]; then
  echo "VISION_INPUT_RTSP_URL must use rtsp:// or rtsps://" >&2
  exit 78
fi

if [[ "${VISION_OUTPUT_RTSP_URL}" != rtsp://* && "${VISION_OUTPUT_RTSP_URL}" != rtsps://* ]]; then
  echo "VISION_OUTPUT_RTSP_URL must use rtsp:// or rtsps://" >&2
  exit 78
fi

exec /home/galaxy-sim/.local/bin/uv run deploy/vision_service.py
