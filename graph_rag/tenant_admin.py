"""Operator commands: export a tenant's data to files, and remove it from every store (GRA-59).

Not part of the HTTP API: an operator runs ``graph-rag-admin`` against the real stack's stores.
A tenant's data is every scope under it (its default namespace and each named one). The default
tenant is refused: records written before tenancy existed carry no tenant, so it cannot be told
apart safely from the tenant of the same name.

Export writes, per scope, ``documents.jsonl``, ``entities.jsonl``, ``graph.json`` and the raw
objects, plus a ``manifest.json``. Vectors are left out: they are derived and large, and a
re-ingest rebuilds them. Delete removes the records, the edges, the objects and the entities,
then checks that no store still lists the tenant.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graph_rag.ports import DocumentStore, EntityStore, GraphStore, ObjectStore
from graph_rag.scope import DEFAULT_SCOPE, Scope

__all__ = ["Stores", "delete_tenant", "export_tenant", "main", "tenant_scopes"]


@dataclass
class Stores:
    """The four stores a tenant's data lives in."""

    objects: ObjectStore
    documents: DocumentStore
    entities: EntityStore
    graph: GraphStore


def _check(tenant: str) -> None:
    Scope(tenant)  # a tenant that cannot be a scope is refused with a ValueError
    if tenant == DEFAULT_SCOPE.tenant:
        raise ValueError(
            "the default tenant holds data written before tenancy existed and cannot be "
            "enumerated safely; export and delete work on named tenants only"
        )


def tenant_scopes(stores: Stores, tenant: str) -> list[Scope]:
    """Every scope of ``tenant`` that holds data in any store."""
    _check(tenant)
    found: set[Scope] = set()
    for store in (stores.objects, stores.documents, stores.entities, stores.graph):
        found |= set(store.scopes(tenant))
    return sorted(found, key=lambda s: s.key)


def _dump(model: Any, drop: set[str]) -> dict[str, Any]:
    return model.model_dump(mode="json", exclude=drop)  # type: ignore[no-any-return]


def export_tenant(stores: Stores, tenant: str, out: Path) -> dict[str, Any]:
    """Write ``tenant``'s data under ``out/<tenant>/`` and return the manifest."""
    scopes = tenant_scopes(stores, tenant)
    root = out / tenant
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"tenant": tenant, "scopes": {}}
    for scope in scopes:
        folder = root / (scope.namespace or "_root")
        (folder / "objects").mkdir(parents=True, exist_ok=True)

        documents = stores.documents.all(scope)
        with (folder / "documents.jsonl").open("w") as handle:
            for doc in documents:
                handle.write(json.dumps(_dump(doc, {"sentence_vectors"})) + "\n")

        entities = stores.entities.all(scope)
        with (folder / "entities.jsonl").open("w") as handle:
            for entity in entities:
                handle.write(json.dumps(_dump(entity, {"vector"})) + "\n")

        graph = stores.graph.khop(scope, [e.canonical_id for e in entities], 1)
        (folder / "graph.json").write_text(
            json.dumps(
                {
                    "nodes": [_dump(n, {"vector"}) for n in graph.nodes],
                    "edges": [_dump(e, set()) for e in graph.edges],
                }
            )
        )

        objects = []
        for n, (bucket, key) in enumerate(sorted(stores.objects.keys(scope))):
            (folder / "objects" / f"{n}.bin").write_bytes(stores.objects.get(scope, bucket, key))
            objects.append({"file": f"objects/{n}.bin", "bucket": bucket, "key": key})
        (folder / "objects.json").write_text(json.dumps(objects))

        manifest["scopes"][scope.key] = {
            "folder": folder.name,
            "documents": len(documents),
            "entities": len(entities),
            "edges": len(graph.edges),
            "objects": len(objects),
        }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def delete_tenant(stores: Stores, tenant: str) -> dict[str, int]:
    """Remove ``tenant`` from every store; raise if any store still lists it afterwards."""
    scopes = tenant_scopes(stores, tenant)
    counts = {"documents": 0, "objects": 0, "entities": 0}
    for scope in scopes:
        for doc in stores.documents.all(scope):
            stores.graph.delete_document_edges(scope, doc.document_id)
            stores.documents.delete(scope, doc.document_id)
            counts["documents"] += 1
        for bucket, key in stores.objects.keys(scope):
            stores.objects.delete(scope, bucket, key)
            counts["objects"] += 1
        while entities := stores.entities.all(scope):  # a store may cap how many one call returns
            ids = [e.canonical_id for e in entities]
            stores.graph.delete_nodes(scope, ids)
            for entity_id in ids:
                stores.entities.delete(scope, entity_id)
            counts["entities"] += len(ids)
    left = {
        name: [s.key for s in store.scopes(tenant)]
        for name, store in (
            ("objects", stores.objects),
            ("documents", stores.documents),
            ("entities", stores.entities),
            ("graph", stores.graph),
        )
        if store.scopes(tenant)
    }
    if left:
        raise RuntimeError(f"tenant {tenant!r} is not fully removed; still listed in: {left}")
    return counts


def _real_stores() -> Stores:
    from graph_rag.adapters.es_document_store import EsDocumentStore
    from graph_rag.adapters.es_entity_store import EsEntityStore
    from graph_rag.adapters.minio_object_store import MinioObjectStore
    from graph_rag.adapters.neo4j_graph_store import Neo4jGraphStore
    from graph_rag.config import get_settings

    settings = get_settings()
    return Stores(
        objects=MinioObjectStore.from_settings(settings),
        documents=EsDocumentStore.from_settings(settings),
        entities=EsEntityStore.from_settings(settings),
        graph=Neo4jGraphStore.from_settings(settings),
    )


def main(argv: list[str] | None = None, stores: Stores | None = None) -> int:
    """The ``graph-rag-admin`` command. ``stores`` is for tests; otherwise the real stack's."""
    parser = argparse.ArgumentParser(
        prog="graph-rag-admin",
        description="Operator commands for a tenant's data, against the stores named by the "
        "service's environment (MINIO_*, ELASTICSEARCH_URL, NEO4J_*).",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    sub.add_parser("show", help="count what a tenant holds").add_argument("tenant")
    export = sub.add_parser("export", help="write a tenant's data to files")
    export.add_argument("tenant")
    export.add_argument("--out", required=True, type=Path, help="directory to write into")
    delete = sub.add_parser("delete", help="remove a tenant from every store")
    delete.add_argument("tenant")
    delete.add_argument("--yes", action="store_true", help="really delete; without it, only show")
    args = parser.parse_args(argv)

    try:
        active = stores or _real_stores()
        if args.command == "export":
            manifest = export_tenant(active, args.tenant, args.out)
            print(json.dumps(manifest["scopes"], indent=2))
            print(f"exported {len(manifest['scopes'])} scope(s) to {args.out / args.tenant}")
        elif args.command == "show" or not args.yes:
            for scope in tenant_scopes(active, args.tenant):
                print(
                    f"{scope.key}: {len(active.documents.all(scope))} documents, "
                    f"{len(active.entities.all(scope))} entities, "
                    f"{len(active.objects.keys(scope))} objects"
                )
            if args.command == "delete":
                print("nothing deleted; add --yes to delete")
        else:
            counts = delete_tenant(active, args.tenant)
            print(f"deleted tenant {args.tenant}: {counts}")
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
