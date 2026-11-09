"""Rate limits: a budget per key and a throttle on bad keys (GRA-137)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.auth import ApiKeys, InMemoryKeyStore
from graph_rag.config import Settings
from graph_rag.ratelimit import RateLimiter
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_v1 import Queue

QUESTION = {"question": "Where is Acme Corp based?"}


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# --- the limiter itself ---------------------------------------------------------------------


def test_a_burst_up_to_the_limit_is_allowed_then_refused_with_a_wait() -> None:
    clock = Clock()
    limiter = RateLimiter(3, clock)
    assert [limiter.spend("k") for _ in range(3)] == [None, None, None]
    wait = limiter.spend("k")
    assert wait == 20  # 3 a minute is one token every 20 seconds


def test_tokens_come_back_with_time() -> None:
    clock = Clock()
    limiter = RateLimiter(60, clock)
    for _ in range(60):
        limiter.spend("k")
    assert limiter.spend("k") == 1
    clock.now += 1
    assert limiter.spend("k") is None
    assert limiter.spend("k") == 1


def test_keys_have_separate_budgets() -> None:
    limiter = RateLimiter(1, Clock())
    assert limiter.spend("a") is None
    assert limiter.spend("a") is not None
    assert limiter.spend("b") is None


def test_zero_turns_it_off() -> None:
    limiter = RateLimiter(0, Clock())
    assert all(limiter.spend("k") is None for _ in range(1000))


def test_idle_buckets_are_dropped_so_memory_stays_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    import graph_rag.ratelimit as module

    monkeypatch.setattr(module, "_MAX_TRACKED", 10)
    clock = Clock()
    limiter = RateLimiter(60, clock)
    for i in range(10):
        limiter.spend(f"k{i}")
    clock.now += 120  # all refilled
    limiter.spend("new")
    assert len(limiter._buckets) <= 2  # noqa: SLF001


# --- on the endpoints -----------------------------------------------------------------------


def stack(**settings: int):
    components = build_offline_components()
    keys = ApiKeys(InMemoryKeyStore())
    app = create_app(
        components.object_store,
        Queue(),
        Settings(require_auth=True, **settings),
        retriever=components.retriever,
        keys=keys,
    )
    return app, keys


def bearer(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


def test_a_burst_gets_429_in_the_contracts_error_shape() -> None:
    app, keys = stack(rate_limit_per_minute=3)
    secret, _ = keys.create("acme")
    client = TestClient(app)
    codes = [
        client.post("/v1/recall", json=QUESTION, headers=bearer(secret)).status_code
        for _ in range(5)
    ]
    assert codes == [200, 200, 200, 429, 429]
    refused = client.post("/v1/recall", json=QUESTION, headers=bearer(secret))
    assert refused.json()["error"]["code"] == "rate_limited"
    assert refused.json()["error"]["details"]["retry_after_seconds"] >= 1
    assert int(refused.headers["retry-after"]) >= 1


def test_the_legacy_endpoints_are_limited_too_with_a_detail_body() -> None:
    app, keys = stack(rate_limit_per_minute=1)
    secret, _ = keys.create("acme")
    client = TestClient(app)
    assert client.post("/query", json=QUESTION, headers=bearer(secret)).status_code == 200
    refused = client.post("/query", json=QUESTION, headers=bearer(secret))
    assert refused.status_code == 429 and "limit" in refused.json()["detail"]
    assert "retry-after" in refused.headers


def test_one_keys_burst_does_not_slow_another() -> None:
    app, keys = stack(rate_limit_per_minute=2)
    noisy, _ = keys.create("acme")
    quiet, _ = keys.create("globex")
    client = TestClient(app)
    for _ in range(5):
        client.post("/v1/recall", json=QUESTION, headers=bearer(noisy))
    assert client.post("/v1/recall", json=QUESTION, headers=bearer(quiet)).status_code == 200


def test_repeated_bad_keys_are_throttled_and_a_good_key_is_not_affected() -> None:
    app, keys = stack(failed_key_limit_per_minute=3)
    good, _ = keys.create("acme")
    client = TestClient(app)  # every request here comes from the same address
    bad = bearer("grk_deadbeef_nope")
    codes = [client.post("/v1/recall", json=QUESTION, headers=bad).status_code for _ in range(5)]
    assert codes == [401, 401, 401, 429, 429]
    throttled = client.post("/v1/recall", json=QUESTION, headers=bad).json()
    assert throttled["error"]["code"] == "rate_limited"
    # The same address, now with a good key, is served.
    assert client.post("/v1/recall", json=QUESTION, headers=bearer(good)).status_code == 200


def test_a_missing_key_counts_as_a_failure() -> None:
    app, _ = stack(failed_key_limit_per_minute=1)
    client = TestClient(app)
    assert client.post("/v1/recall", json=QUESTION).status_code == 401
    assert client.post("/v1/recall", json=QUESTION).status_code == 429


def test_another_addresss_failures_do_not_throttle_this_one() -> None:
    app, _ = stack(failed_key_limit_per_minute=1)
    flooder = TestClient(app, client=("203.0.113.9", 1))
    for _ in range(3):
        flooder.post("/v1/recall", json=QUESTION, headers=bearer("grk_deadbeef_nope"))
    other = TestClient(app, client=("198.51.100.4", 1))
    assert (
        other.post("/v1/recall", json=QUESTION, headers=bearer("grk_deadbeef_nope")).status_code
        == 401
    )


def test_health_is_never_limited() -> None:
    app, _ = stack(rate_limit_per_minute=1, failed_key_limit_per_minute=1)
    client = TestClient(app)
    assert all(client.get("/health").status_code == 200 for _ in range(10))


def test_with_auth_off_nothing_is_limited() -> None:
    components = build_offline_components()
    app = create_app(
        components.object_store,
        Queue(),
        Settings(rate_limit_per_minute=1, failed_key_limit_per_minute=1),
        retriever=components.retriever,
    )
    client = TestClient(app)
    assert all(client.post("/query", json=QUESTION).status_code == 200 for _ in range(5))
