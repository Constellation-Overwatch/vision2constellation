# Fresh Overwatch target

- Base URL: `https://overwatch.tail8e4fe5.ts.net/`
- NATS URL: `nats://overwatch.tail8e4fe5.ts.net:4222`
- MediaMTX source:
  `rtsp://galaxy-gcs-2-950sbe-951sbe.tail8e4fe5.ts.net:8554/live/stream`
- MediaMTX processed path:
  `rtsp://galaxy-gcs-2-950sbe-951sbe.tail8e4fe5.ts.net:8554/vision/c4threatisr`

Verified from the SIM host:

- Overwatch MagicDNS resolves and responds over HTTPS.
- NATS 2.12.6 responds with JetStream enabled and authentication required.
- The MediaMTX source authenticates and exposes H.264 1280x720 at 15 FPS.
- MediaMTX currently rejects publishing to `vision/c4threatisr`; add `publish`
  and `read` permission for that path before starting this service.

Pending configuration:

- `CONSTELLATION_ENTITY_ID`
- NATS authentication material and its mode (token, NKey seed, or credentials
  file)
