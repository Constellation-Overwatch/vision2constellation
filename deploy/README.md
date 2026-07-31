# SIM C4 threat ISR deployment

This deployment keeps MediaMTX on the remote Tailscale peer and runs inference
on this SIM host:

1. Pull `live/stream` from MediaMTX over RTSP/TCP.
2. Run the `yoloe_c4isr` detector on the RTX GPU.
3. Publish detections and entity state to the fresh Constellation Overwatch
   deployment's NATS endpoint over Tailscale.
4. Push annotated H.264 over RTSP/TCP to `vision/c4threatisr`.
5. Let Constellation Overwatch consume the processed MediaMTX path.

The input and output paths must differ. The remote MediaMTX configuration must
authorize this SIM Tailscale identity to read `live/stream` and publish
`vision/c4threatisr`. Prefer MagicDNS over a hard-coded `100.x` address.

Install:

```bash
mkdir -p ~/.config/vision2constellation ~/.config/systemd/user
cp deploy/runtime.env.example ~/.config/vision2constellation/runtime.env
cp deploy/secret.env.example ~/.config/vision2constellation/secret.env
chmod 600 ~/.config/vision2constellation/*.env
ln -sfn "$PWD/deploy/vision2constellation.service" \
  ~/.config/systemd/user/vision2constellation.service
systemctl --user daemon-reload
systemctl --user enable vision2constellation.service
```

Set the new Overwatch MagicDNS/NATS endpoint, Pulsar entity ID, and vision
client credential before starting. The wrapper exits with status 78 while any
required value is missing or still a placeholder.

The Overwatch HTTPS URL is not an RTSP endpoint. On the new Overwatch install,
set `MEDIAMTX_API_URL` and `MEDIAMTX_WEBRTC_URL` to the MediaMTX peer's
tailnet-reachable API and WebRTC endpoints after those ports are available.
