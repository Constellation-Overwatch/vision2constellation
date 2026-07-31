# Deployment Boundary Decision

DECISION: Run inference on SIM; read and publish RTSP/TCP through the remote
MediaMTX peer; publish detection state into SIM-local Constellation NATS.

TIER: 2

STATUS: proceed

CONFIDENCE: HIGH

## Why

- EXTRACTED: SIM Constellation Overwatch is active and its embedded NATS listener
  is `127.0.0.1:4224` with JetStream enabled and authentication required.
- EXTRACTED: the tailnet is healthy, MagicDNS is enabled, and the intended remote
  peer is `constellation.tail8e4fe5.ts.net`; that peer is currently offline.
- EXTRACTED: upstream `vision2constellation` already supports `yoloe_c4isr`,
  RTSP input, NATS event/KV publication, and RTSP-over-TCP input options.
- EXTRACTED: upstream does not support headless display suppression or publishing
  annotated frames to an RTSP server.
- INFERRED: a bounded FFmpeg subprocess is the smallest locally coherent adapter
  for returning annotated H.264 to MediaMTX without coupling model inference to
  MediaMTX internals.

## Local contracts

- Remote MediaMTX accepts a camera/input publisher or pull source and exposes an
  RTSP input path over Tailscale. It does not know inference internals.
- The vision worker accepts decoded frames, produces threat detections and
  annotated frames, and tolerates a temporarily unavailable output sink.
- The RTSP output adapter accepts annotated BGR frames and produces H.264/RTSP
  on a distinct remote MediaMTX path. It does not know detection semantics.
- Constellation Overwatch accepts NATS events/KV state and uses the remote
  MediaMTX API/WebRTC endpoints for video discovery and playback.

## Structural check

HYBRID: Tailscale DNS, distinct MediaMTX paths, bounded queues, and systemd
restart policy provide the durable structure. ML inference and video encoding
are the irreducible computed work.

## Rejected alternatives

- Install MediaMTX on SIM: rejected by operator direction.
- Run inference on the MediaMTX host: rejected by operator direction.
- Use only NATS frame streaming: rejected because the required viewer path is
  annotated video returned to MediaMTX.

SWITCHING COST: HOURS. The RTSP input/output URLs, model, codec, and service
settings are isolated configuration or adapter choices.

## What I cannot verify yet

- The final remote MediaMTX input path, output path authorization, API exposure,
  and WebRTC exposure.
- End-to-end media flow while the remote Tailscale peer remains offline.
- Which existing Constellation entity should own the vision detections.
