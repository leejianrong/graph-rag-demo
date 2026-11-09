"""Refusing to serve an open API on a reachable address (GRA-136)."""

from __future__ import annotations

import logging

import pytest

from graph_rag.bind import OpenBindError, check_bind, enforce_bind, is_loopback
from graph_rag.config import Settings


@pytest.mark.parametrize("host", ["127.0.0.1", "127.9.9.9", "::1", "localhost", "LOCALHOST"])
def test_loopback_hosts(host: str) -> None:
    assert is_loopback(host)


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "::", "10.0.0.5", "192.168.1.2", "api.example.com", ""]
)
def test_everything_else_is_reachable(host: str) -> None:
    assert not is_loopback(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.5", "api.example.com"])
def test_open_on_a_reachable_address_is_refused(host: str) -> None:
    with pytest.raises(OpenBindError, match="REQUIRE_AUTH"):
        check_bind(host, require_auth=False, allow_open_bind=False)


def test_the_refusal_says_how_to_fix_it() -> None:
    with pytest.raises(OpenBindError) as refusal:
        check_bind("0.0.0.0", require_auth=False, allow_open_bind=False)
    text = str(refusal.value)
    assert "REQUIRE_AUTH=true" in text and "127.0.0.1" in text and "ALLOW_OPEN_BIND" in text


def test_auth_on_or_loopback_or_the_opt_out_is_allowed() -> None:
    assert check_bind("0.0.0.0", require_auth=True, allow_open_bind=False) is None
    assert check_bind("127.0.0.1", require_auth=False, allow_open_bind=False) is None
    warning = check_bind("0.0.0.0", require_auth=False, allow_open_bind=True)
    assert warning and "anyone who can reach" in warning  # allowed, but loudly


def test_enforce_exits_with_the_reason_and_logs_the_opt_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    log = logging.getLogger("test")
    with pytest.raises(SystemExit) as exit_:
        enforce_bind("0.0.0.0", Settings(require_auth=False), log)
    assert "refusing to serve" in str(exit_.value)
    with caplog.at_level(logging.WARNING):
        enforce_bind("0.0.0.0", Settings(allow_open_bind=True), log)
    assert "ALLOW_OPEN_BIND" in caplog.text


def test_the_settings_defaults_are_the_safe_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("REQUIRE_AUTH", "ALLOW_OPEN_BIND", "BIND_HOST"):
        monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=None)
    assert not s.require_auth and not s.allow_open_bind
    # the container's default bind is all addresses, so with nothing set it must not start
    with pytest.raises(OpenBindError):
        check_bind(s.bind_host, require_auth=s.require_auth, allow_open_bind=s.allow_open_bind)


def test_the_entry_points_check_before_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    from graph_rag import main as service
    from graph_rag import offline_server

    served: list[object] = []
    monkeypatch.setattr(offline_server.uvicorn, "run", lambda *a, **k: served.append(k))
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.delenv("ALLOW_OPEN_BIND", raising=False)
    monkeypatch.delenv("REQUIRE_AUTH", raising=False)
    with pytest.raises(SystemExit):
        offline_server.main()
    assert served == []
    monkeypatch.setenv("ALLOW_OPEN_BIND", "true")
    offline_server.main()
    assert served

    # the real service refuses before it connects to any store
    monkeypatch.setenv("BIND_HOST", "0.0.0.0")
    monkeypatch.delenv("ALLOW_OPEN_BIND")
    with pytest.raises(SystemExit):
        service.main()
