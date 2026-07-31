# NKey authentication hotfix

The upstream communication service currently supports token authentication
only. The deployment adapter patches only `_connect_nats` and supports:

- `NATS_CREDS_FILE` for a combined NATS user JWT/seed `.creds` file;
- `NATS_NKEY_SEED_FILE` for a raw user seed stored in a mode-0600 file;
- `NATS_NKEY_SEED_STR` for environments where file mounting is unavailable.

Exactly one mode is required. File-backed credentials are rejected when group
or other permissions are present. The token configuration is intentionally not
passed when the NKey adapter is active.
