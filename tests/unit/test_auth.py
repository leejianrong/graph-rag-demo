"""API keys: shown once, stored hashed, verified, revoked (GRA-22)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from graph_rag.auth import ApiKeys, AuthError, InMemoryKeyStore, JsonFileKeyStore, main


@pytest.fixture(params=["memory", "file"])
def keys(request: pytest.FixtureRequest, tmp_path: Path) -> ApiKeys:
    store = (
        InMemoryKeyStore() if request.param == "memory" else JsonFileKeyStore(tmp_path / "k.json")
    )
    return ApiKeys(store)


def test_a_created_key_verifies_to_its_tenant(keys: ApiKeys) -> None:
    secret, record = keys.create("acme", "ci")
    assert secret.startswith("grk_")
    found = keys.verify(secret)
    assert (found.tenant, found.key_id, found.label) == ("acme", record.key_id, "ci")


def test_the_secret_is_not_stored_only_its_hash(tmp_path: Path) -> None:
    path = tmp_path / "k.json"
    secret, _ = ApiKeys(JsonFileKeyStore(path)).create("acme")
    on_disk = path.read_text()
    assert secret not in on_disk
    assert secret.split("_", 2)[2] not in on_disk
    assert json.loads(on_disk)  # one record, with a key_hash
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_a_revoked_key_is_rejected_and_others_still_work(keys: ApiKeys) -> None:
    secret, record = keys.create("acme")
    other, _ = keys.create("acme")
    assert keys.revoke(record.key_id) is True
    with pytest.raises(AuthError, match="revoked"):
        keys.verify(secret)
    assert keys.verify(other).tenant == "acme"


def test_revoking_twice_keeps_the_first_time_and_an_unknown_id_is_false(keys: ApiKeys) -> None:
    _, record = keys.create("acme")
    keys.revoke(record.key_id)
    first = keys.list()[0].revoked_at
    assert keys.revoke(record.key_id) is True
    assert keys.list()[0].revoked_at == first
    assert keys.revoke("nope") is False


@pytest.mark.parametrize("bad", ["", "nonsense", "grk_", "grk_abcd", "xyz_abcd_token"])
def test_a_malformed_key_is_rejected(keys: ApiKeys, bad: str) -> None:
    with pytest.raises(AuthError):
        keys.verify(bad)


def test_a_wrong_token_with_a_real_id_is_rejected(keys: ApiKeys) -> None:
    secret, record = keys.create("acme")
    with pytest.raises(AuthError):
        keys.verify(f"grk_{record.key_id}_not-the-token")
    with pytest.raises(AuthError):
        keys.verify(secret + "x")


def test_a_key_from_another_store_is_rejected(keys: ApiKeys) -> None:
    secret, _ = ApiKeys(InMemoryKeyStore()).create("acme")
    with pytest.raises(AuthError):
        keys.verify(secret)


def test_list_filters_by_tenant_and_never_carries_a_secret(keys: ApiKeys) -> None:
    secret, _ = keys.create("a")
    keys.create("b")
    assert [k.tenant for k in keys.list("a")] == ["a"]
    assert len(keys.list()) == 2
    assert all(secret not in repr(k) for k in keys.list())


@pytest.mark.parametrize("tenant", ["", "a/b", "a|b"])
def test_a_tenant_that_cannot_be_a_scope_gets_no_key(keys: ApiKeys, tenant: str) -> None:
    with pytest.raises(ValueError):
        keys.create(tenant)
    assert keys.list() == []


def test_a_revoke_by_another_process_takes_effect_on_the_next_check(tmp_path: Path) -> None:
    path = tmp_path / "k.json"
    secret, record = ApiKeys(JsonFileKeyStore(path)).create("acme")
    server = ApiKeys(JsonFileKeyStore(path))
    assert server.verify(secret).tenant == "acme"
    ApiKeys(JsonFileKeyStore(path)).revoke(record.key_id)
    with pytest.raises(AuthError):
        server.verify(secret)


def test_the_command_creates_lists_and_revokes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = str(tmp_path / "k.json")
    assert main(["--file", path, "create", "acme", "--label", "laptop"]) == 0
    out = capsys.readouterr().out
    secret = next(line.split()[1] for line in out.splitlines() if line.startswith("key:"))
    key_id = next(line.split()[1] for line in out.splitlines() if line.startswith("key_id:"))
    assert ApiKeys(JsonFileKeyStore(path)).verify(secret).tenant == "acme"

    assert main(["--file", path, "list"]) == 0
    listed = capsys.readouterr().out
    assert key_id in listed and "laptop" in listed and secret not in listed

    assert main(["--file", path, "revoke", key_id]) == 0
    assert main(["--file", path, "revoke", "nope"]) == 1
    assert main(["--file", path, "create", "a/b"]) == 2


def test_the_default_tenant_gets_no_key(
    keys: ApiKeys, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(ValueError, match="default"):
        keys.create("default")
    assert keys.list() == []
    assert main(["--file", str(tmp_path / "k.json"), "create", "default"]) == 2
    assert "predates tenancy" in capsys.readouterr().out
    assert not (tmp_path / "k.json").exists()


def test_an_old_key_for_the_default_tenant_no_longer_verifies(tmp_path: Path) -> None:
    store = JsonFileKeyStore(tmp_path / "k.json")
    secret, record = ApiKeys(store).create("acme")
    data = json.loads((tmp_path / "k.json").read_text())
    raw = json.dumps(data).replace('"acme"', '"default"')  # as written before the refusal
    (tmp_path / "k.json").write_text(raw)
    with pytest.raises(AuthError, match="not recognised"):
        ApiKeys(JsonFileKeyStore(tmp_path / "k.json")).verify(secret)
