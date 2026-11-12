"""Locking down the backing services (GRA-138): credentials, the ES login, forged triggers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from graph_rag.adapters.es_client import es_client
from graph_rag.bind import default_credentials, enforce_credentials
from graph_rag.config import Settings
from graph_rag.models import IngestTrigger
from graph_rag.scope import DEFAULT_SCOPE, Scope
from graph_rag.wiring import build_offline_components

ROOT = Path(__file__).resolve().parents[2]
REAL = {
    "minio_access_key": "svc-account",
    "minio_secret_key": "s3cret-value",
    "neo4j_password": "n4j-value",
    "elasticsearch_password": "es-value",
    "redis_password": "valkey-value",
}


def hosted(**kw: object) -> Settings:
    return Settings(require_auth=True, **(REAL | kw))  # type: ignore[arg-type]


def test_with_auth_on_each_local_dev_credential_is_refused() -> None:
    enforce_credentials(hosted())  # all real: starts
    for field, name in [
        ("minio_access_key", "MINIO_ACCESS_KEY"),
        ("minio_secret_key", "MINIO_SECRET_KEY"),
        ("neo4j_password", "NEO4J_PASSWORD"),
    ]:
        default = Settings.model_fields[field].default
        with pytest.raises(SystemExit, match=name):
            enforce_credentials(hosted(**{field: default}))
    with pytest.raises(SystemExit, match="ELASTICSEARCH_PASSWORD"):
        enforce_credentials(hosted(elasticsearch_password=""))
    with pytest.raises(SystemExit, match="REDIS_PASSWORD"):
        enforce_credentials(hosted(redis_password=""))
    enforce_credentials(hosted(redis_password="", trigger_backend="kafka"))  # Kafka has none


def test_the_message_names_every_offender_but_never_a_value() -> None:
    with pytest.raises(SystemExit) as refusal:
        enforce_credentials(Settings(require_auth=True, minio_secret_key="minioadmin"))
    text = str(refusal.value)
    assert (
        "MINIO_SECRET_KEY" in text and "NEO4J_PASSWORD" in text and "ELASTICSEARCH_PASSWORD" in text
    )
    assert "ALLOW_DEFAULT_CREDENTIALS" in text


def test_auth_off_or_the_staging_opt_out_checks_nothing() -> None:
    enforce_credentials(Settings(require_auth=False))  # the open local stack uses the defaults
    enforce_credentials(Settings(require_auth=True, allow_default_credentials=True))
    assert default_credentials(Settings(require_auth=False))  # they are defaults, just allowed


def test_the_elasticsearch_client_signs_in_only_with_a_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict] = []
    monkeypatch.setattr("graph_rag.adapters.es_client.Elasticsearch", lambda **kw: seen.append(kw))
    es_client(
        Settings(
            elasticsearch_url="http://es:9200",
            elasticsearch_password="pw",
            elasticsearch_user="svc",
        )
    )
    es_client(Settings(elasticsearch_url="http://es:9200", elasticsearch_password=""))
    assert seen[0] == {"hosts": ["http://es:9200"], "basic_auth": ("svc", "pw")}
    assert seen[1] == {"hosts": ["http://es:9200"]}


def test_a_forged_trigger_cannot_write_into_a_tenant_without_a_document() -> None:
    """Kafka triggers name a tenant, but processing reads that tenant's own stored object."""
    c = build_offline_components()
    victim = Scope("victim")
    # Nothing stored for the victim: a trigger naming them finds no object and writes nothing.
    assert (
        c.orchestrator.process_document(IngestTrigger(bucket="b", object_key="k", tenant="victim"))
        is None
    )
    # An object that exists only in another tenant is not reachable through the victim's scope.
    c.object_store.put(Scope("other"), "b", "k", b"Ada Lovelace works for Acme Corp.")
    assert (
        c.orchestrator.process_document(IngestTrigger(bucket="b", object_key="k", tenant="victim"))
        is None
    )
    for scope in (victim, DEFAULT_SCOPE):
        assert c.entity_store.all(scope) == []
        assert c.document_store.all(scope) == []
    assert c.entity_store.scopes("victim") == []


# --- the hosted compose file ------------------------------------------------------------


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.hosted.yml").read_text())


def test_no_store_publishes_a_port_and_none_joins_a_public_network(compose: dict) -> None:
    for name in ("elasticsearch", "minio", "valkey", "kafka", "neo4j"):
        service = compose["services"][name]
        assert "ports" not in service and "expose" not in service, name
        assert service["networks"] == ["backend"], name
    assert compose["networks"]["backend"]["internal"] is True


def test_every_credential_is_required_and_has_no_default(compose: dict) -> None:
    text = (ROOT / "docker-compose.hosted.yml").read_text()
    values = json.dumps(compose)  # the parsed file: comments, which may name the defaults, are gone
    for name in (
        "ELASTICSEARCH_PASSWORD",
        "MINIO_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "NEO4J_PASSWORD",
        "VALKEY_PASSWORD",
    ):
        assert "${" + name + ":?" in text, name
        assert "${" + name + ":-" not in text, name
    for banned in ("minioadmin", "devpassword"):
        assert banned not in values


def test_the_app_runs_with_auth_on_and_without_the_open_opt_out(compose: dict) -> None:
    env = compose["services"]["app"]["environment"]
    assert env["REQUIRE_AUTH"] == "true" and env["ALLOW_OPEN_BIND"] == "false"
    assert compose["services"]["elasticsearch"]["environment"]["xpack.security.enabled"] == "true"
    assert "127.0.0.1" in compose["services"]["app"]["ports"][0]  # loopback unless told otherwise


def test_valkey_is_the_default_trigger_store_and_kafka_is_a_profile(compose: dict) -> None:
    services = compose["services"]
    assert "profiles" not in services["valkey"] and services["kafka"]["profiles"] == ["kafka"]
    for name in ("app", "worker"):
        needs = services[name]["depends_on"]
        # kafka is a profile: not started by default, so the dependency must be optional
        assert needs["kafka"]["required"] is False, name
        assert needs["valkey"]["condition"] == "service_healthy", name
    # The password reaches the server and the clients by environment, never a literal.
    valkey = services["valkey"]
    assert valkey["environment"]["VALKEY_PASSWORD"].startswith("${VALKEY_PASSWORD:?")
    assert "REDISCLI_AUTH" in valkey["environment"]  # valkey-cli's healthcheck reads it
    command = json.dumps(valkey["command"])
    assert "--requirepass" in command and "${" not in command
    assert "noeviction" in command and "valkey-data:/data" in valkey["volumes"]
    assert services["app"]["environment"]["REDIS_PASSWORD"].startswith("${VALKEY_PASSWORD:?")
