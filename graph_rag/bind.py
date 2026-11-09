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


def _local_dev(field: str) -> object:
    return Settings.model_fields[field].default  # the field default is the local-dev value


def default_credentials(settings: Settings) -> list[str]:
    """Names of store credentials still at their local-dev value (Elasticsearch's: unset)."""
    found = []
    if settings.minio_access_key == _local_dev("minio_access_key"):
        found.append("MINIO_ACCESS_KEY")
    if settings.minio_secret_key == _local_dev("minio_secret_key"):
        found.append("MINIO_SECRET_KEY")
    if settings.neo4j_password == _local_dev("neo4j_password"):
        found.append("NEO4J_PASSWORD")
    if not settings.elasticsearch_password:
        found.append("ELASTICSEARCH_PASSWORD")
    return found


def enforce_credentials(settings: Settings) -> None:
    """With auth on, exit when a store credential is a local-dev default (GRA-138).

    Auth on means a hosted deployment, where default passwords are the gap. With auth off this is
    the open local stack, which uses them on purpose, so nothing is checked.
    """
    if not settings.require_auth or settings.allow_default_credentials:
        return
    names = default_credentials(settings)
    if names:
        raise SystemExit(
            "error: REQUIRE_AUTH is on but these still have their local-dev values: "
            + ", ".join(names)
            + ". Set real ones (docker-compose.hosted.yml requires them), or set "
            "ALLOW_DEFAULT_CREDENTIALS=true for a staging run."
        )
