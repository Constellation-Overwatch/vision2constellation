"""Helpers for handling stream URLs without leaking credentials."""

from urllib.parse import urlsplit, urlunsplit


def redact_stream_url(url: str) -> str:
    """Return a URL safe for logs by removing credentials, query, and fragment."""
    if not url:
        return url

    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"

    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def stream_identity(url: str) -> tuple[str, str, int | None, str]:
    """Return the non-secret endpoint identity used to compare stream paths."""
    parsed = urlsplit(url)
    return (
        parsed.scheme.lower(),
        (parsed.hostname or "").lower(),
        parsed.port,
        parsed.path.rstrip("/") or "/",
    )
