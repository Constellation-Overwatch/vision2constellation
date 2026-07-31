# Cross-Model Review Launch

## [1. CONTEXT]

Prepare `vision2constellation` on the SIM workstation to consume an RTSP stream
from `constellation.tail8e4fe5.ts.net` over Tailscale, run the repository's
`yoloe_c4isr` C4 threat ISR model, publish detections to the local Constellation
Overwatch NATS server at `127.0.0.1:4224`, and publish annotated H.264 video back
to a distinct path on the remote MediaMTX server.

The current upstream code already forces RTSP input over TCP but assumes a GUI
display and has no RTSP output publisher. The remote Tailscale peer is currently
offline, so review must distinguish configuration readiness from live end-to-end
proof.

## [2. TEAM]

- Claude adversary: identify structural, lifecycle, security, and failure-mode
  risks. Write `output/claude_review.md`.
- AGY alternate reviewer: independently assess implementation shape, deployment
  operability, and verification coverage. Write `output/agy_review.md`.
- Codex lead: owns all implementation, host changes, test loops, reconciliation,
  and live evidence.

## [3. RULES]

- Load the `tmux-prompt-injection` skill before sending any pane injections.
- Reviewers must not edit source, configuration, service units, or host state.
- Do not print or copy credentials, NATS tokens, private keys, or credentialed
  RTSP URLs.
- Label direct evidence `EXTRACTED`, reasoning `INFERRED`, and gaps `UNVERIFIED`.
- Each reviewer must include exactly three explicit adversary challenges and
  prioritize findings by severity.
- Use the ao-ops queue/crosstalk contract: claim one task, write the artifact,
  complete with evidence, and post WHAT + EVIDENCE + NEXT.
- Heartbeat through ao-ops at meaningful phase boundaries. This is an attended,
  bounded review; no unattended paid loop or open-ended cost is authorized.

## [4. DONE]

Done when both independent review artifacts exist, each has findings ordered by
severity and a pass/block verdict, Codex has compared and dispositioned every
finding in `output/reconciliation.md`, implementation tests pass, service
configuration is installed, and remaining live-stream uncertainty is explicit.

Cleanup: reviewers stop after their artifact and completion post. Codex closes
the ao-ops operation after final reconciliation and reports any retained tmux
session.
