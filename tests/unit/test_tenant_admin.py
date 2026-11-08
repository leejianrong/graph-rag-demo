"""Export and delete a tenant's data (GRA-59), on the in-memory stores."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_rag.fakes import (
    InMemoryDocumentStore,
    InMemoryEntityStore,
    InMemoryGraphStore,
    InMemoryObjectStore,
)
from graph_rag.tenant_admin import Stores, main
from tests.isolation import check_tenant_admin


def fakes() -> Stores:
    return Stores(
        InMemoryObjectStore(), InMemoryDocumentStore(), InMemoryEntityStore(), InMemoryGraphStore()
    )


def test_export_then_delete_removes_one_tenant_and_leaves_the_other(tmp_path: Path) -> None:
    check_tenant_admin(fakes(), dims=4, tmp_path=tmp_path)


def test_the_default_tenant_is_refused(tmp_path: Path, capsys) -> None:
    for argv in (["export", "default", "--out", str(tmp_path)], ["delete", "default", "--yes"]):
        assert main(argv, stores=fakes()) == 1
        assert "default tenant" in capsys.readouterr().err


def test_a_tenant_that_cannot_be_a_scope_is_refused(capsys) -> None:
    assert main(["show", "a/b"], stores=fakes()) == 1
    assert "error:" in capsys.readouterr().err


def test_delete_without_yes_only_shows(tmp_path: Path, capsys) -> None:
    stores = fakes()
    from graph_rag.scope import Scope

    stores.objects.put(Scope("acme"), "b", "k.txt", b"x")
    assert main(["delete", "acme"], stores=stores) == 0
    out = capsys.readouterr().out
    assert "1 objects" in out and "nothing deleted" in out
    assert stores.objects.keys(Scope("acme")) == [("b", "k.txt")]
    assert main(["delete", "acme", "--yes"], stores=stores) == 0
    assert stores.objects.keys(Scope("acme")) == []


def test_export_writes_a_manifest_and_object_files_not_named_by_their_keys(tmp_path: Path) -> None:
    from graph_rag.scope import Scope

    stores = fakes()
    stores.objects.put(Scope("acme"), "b", "../../etc/passwd", b"x")
    assert main(["export", "acme", "--out", str(tmp_path)], stores=stores) == 0
    manifest = json.loads((tmp_path / "acme" / "manifest.json").read_text())
    assert manifest["scopes"]["acme"]["objects"] == 1
    files = sorted(p.name for p in (tmp_path / "acme" / "_root" / "objects").iterdir())
    assert files == ["0.bin"]
    assert not (tmp_path / "etc").exists()


def test_delete_fails_loudly_if_a_store_still_lists_the_tenant() -> None:
    from graph_rag.scope import Scope
    from graph_rag.tenant_admin import delete_tenant

    class Stuck(InMemoryGraphStore):
        def delete_nodes(self, scope, canonical_ids) -> None:  # silently does nothing
            return None

    stores = fakes()
    stores.graph = Stuck()
    from graph_rag.models import CanonicalEntity

    entity = CanonicalEntity(canonical_id="e1", name="E", type="ORG")
    stores.entities.upsert(Scope("acme"), entity)
    stores.graph.upsert_entities(Scope("acme"), [entity])
    with pytest.raises(RuntimeError, match="still listed in"):
        delete_tenant(stores, "acme")
