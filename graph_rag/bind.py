"""Refuse to serve an open API on a reachable address (GRA-136, THREAT-MODEL.md section 1).

``REQUIRE_AUTH`` defaults off, so a deployment that forgets it would serve every tenant's memory to
anyone who can reach the port. Serving with auth off is allowed only on a loopback address, or when
``ALLOW_OPEN_BIND`` says the operator means it (the local compose stack does).
"""

from __future__ import annotations

import ipaddress
import logging

from graph_rag.config import Settings


class OpenBindError(RuntimeError):
    """Auth is off and the service would listen beyond this machine."""


def is_loopback(host: str) -> bool:
    """True for ``localhost`` and loopback IPs; names, ``0.0.0.0`` and ``::`` are not."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # a hostname other than localhost could resolve anywhere


def check_bind(host: str, *, require_auth: bool, allow_open_bind: bool) -> str | None:
    """Raise :class:`OpenBindError` for an open service on a reachable address.

    Returns a warning to log when the opt-out let it through, else ``None``.
    """
    if require_auth or is_loopback(host):
        return None
    if allow_open_bind:
        return (
            f"serving on {host} with REQUIRE_AUTH off (ALLOW_OPEN_BIND is set): anyone who can "
            "reach this port can read, write and delete every tenant's memory"
        )
    raise OpenBindError(
        f"refusing to serve on {host} with REQUIRE_AUTH off: the API would be open to anyone "
        "who can reach it. Set REQUIRE_AUTH=true (and create keys with graph-rag-keys), bind "
        "to 127.0.0.1, or set ALLOW_OPEN_BIND=true if an open service is what you want."
    )


def enforce_bind(host: str, settings: Settings, logger: logging.Logger) -> None:
    """Exit with the reason, before anything is served, when :func:`check_bind` refuses."""
    try:
        warning = check_bind(
            host, require_auth=settings.require_auth, allow_open_bind=settings.allow_open_bind
        )
    except OpenBindError as error:
        raise SystemExit(f"error: {error}") from error
    if warning:
        logger.warning(warning)
