"""Entity merge (GRA-144) checks, run on the fakes in the fast suite and on real stores in contract.

``check_merge_store`` asserts the ``MergeRecordStore`` port; ``check_entity_merge`` drives
``EntityCurator`` over any entity, graph, document and merge store, including a merge that stops
after the record is written and is finished by a retry. Ids and scopes are new each call, because
a contract store can be shared across tests.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from graph_rag.agent_api import AgentError
from graph_rag.curation_api import EntityCurator
from graph_rag.merge_records import DroppedEntity, MergeRecord, MergeRecordStore
from graph_rag.models import CanonicalEntity, DocumentRecord, EdgeProvenance, EntityLink, Triple
from graph_rag.ports import DocumentStore, EntityStore, GraphStore
from graph_rag.scope import Scope


def check_merge_store(store: MergeRecordStore) -> None:
    tag = uuid.uuid4().hex[:8]
    a, b, a_notes = Scope(f"a{tag}"), Scope(f"b{tag}"), Scope(f"a{tag}", "notes")

    def record(merge_id: str, keep: str, dropped: str, at: str) -> MergeRecord:
        return MergeRecord(
            merge_id=merge_id,
            keep_id=keep,
            status="pending",
            created_at=at,
            keep_aliases_before=["x"],
            dropped=[DroppedEntity(canonical_id=dropped, name=dropped, type="ORG", vector=[0.5])],
        )

    store.put(a, record("m2", "k", "d2", "2026-01-02T00:00:00Z"))
    store.put(a, record("m1", "k", "d1", "2026-01-01T00:00:00Z"))
    store.put(a, record("m3", "other", "d3", "2026-01-03T00:00:00Z"))
    assert [r.merge_id for r in store.for_entity(a, "k")] == ["m1", "m2"]  # oldest first
    assert [r.merge_id for r in store.containing(a, "d2")] == ["m2"]
    assert [r.merge_id for r in store.all(a)] == ["m1", "m2", "m3"]
    got = store.get(a, "m1")
    assert got is not None and got.dropped[0].vector == [0.5] and got.keep_aliases_before == ["x"]

    done = got.model_copy(update={"status": "done", "completed_at": "2026-01-01T00:00:01Z"})
    store.put(a, done)  # same id overwrites
    again = store.get(a, "m1")
    assert again is not None and again.status == "done" and len(store.all(a)) == 3

    for other in (b, a_notes):
        assert store.get(other, "m1") is None and store.all(other) == []
        assert store.for_entity(other, "k") == [] and store.containing(other, "d1") == []
    assert a in store.scopes(a.tenant) and a_notes not in store.scopes(a.tenant)
    store.put(a_notes, record("m9", "k", "d9", "2026-01-04T00:00:00Z"))
    assert {s.key for s in store.scopes(a.tenant)} == {a.key, a_notes.key}
    assert store.scopes(b.tenant) == []

    store.delete(a, "m1")
    store.delete(a, "m1")  # a no-op the second time
    assert store.get(a, "m1") is None and [r.merge_id for r in store.all(a)] == ["m2", "m3"]


class _FailsOnce:
    """A graph store whose ``merge_nodes`` fails on its first call, as a crash mid-merge would."""

    def __init__(self, inner: GraphStore) -> None:
        self._inner = inner
        self.failed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def merge_nodes(self, scope: Scope, keep_id: str, drop_ids: list[str]) -> None:
        if not self.failed:
            self.failed = True
            raise RuntimeError("injected failure")
        self._inner.merge_nodes(scope, keep_id, drop_ids)


def check_entity_merge(
    entities: EntityStore,
    graph: GraphStore,
    documents: DocumentStore,
    merges: MergeRecordStore,
    dims: int,
) -> None:
    tag = uuid.uuid4().hex[:8]
    scope, other = Scope(f"a{tag}"), Scope(f"b{tag}")
    keep, dup, dup2, bob, cat = (f"{n}-{tag}" for n in ("keep", "dup", "dup2", "bob", "cat"))
    vec = [1.0] + [0.0] * (dims - 1)

    def edge(s: str, o: str, doc: str, sentence: int = 0) -> Triple:
        provenance = EdgeProvenance(
            source_doc_id=doc, sentence_index=sentence, source_sentence="Text."
        )
        return Triple(subject_id=s, predicate="WORKS_FOR", object_id=o, provenance=provenance)

    seeded = [
        CanonicalEntity(canonical_id=keep, name="Acme Corp", type="ORG", vector=vec),
        CanonicalEntity(
            canonical_id=dup, name="Acme Holdings", type="ORG", aliases=["AH"], vector=vec
        ),
        CanonicalEntity(canonical_id=dup2, name="Acme Intl", type="ORG", vector=vec),
        CanonicalEntity(canonical_id=bob, name="Bob Stone", type="PERSON", vector=vec),
        CanonicalEntity(canonical_id=cat, name="Cat Lee", type="PERSON", vector=vec),
    ]
    for entity in seeded:
        entities.upsert(scope, entity)
    graph.upsert_entities(scope, [e.model_copy(update={"vector": None}) for e in seeded])
    graph.write_triples(
        scope,
        [
            edge(bob, keep, f"d1-{tag}"),
            edge(cat, dup, f"d2-{tag}"),
            edge(bob, dup, f"d2-{tag}", 1),
            edge(keep, dup2, f"d3-{tag}"),  # becomes a loop, so it is dropped
        ],
    )
    for n, canonical in ((1, keep), (2, dup), (3, dup2)):
        link = EntityLink(
            mention_text="Acme", canonical_id=canonical, entity_type="ORG", score=1.0, is_new=True
        )
        documents.upsert(
            scope,
            DocumentRecord(
                document_id=f"d{n}-{tag}",
                bucket="b",
                object_key=f"d{n}.txt",
                text="t",
                el_result=[link],
            ),
        )
    first_seen = entities.first_seen(scope, dup)
    assert first_seen

    flaky = _FailsOnce(graph)  # type: ignore[arg-type]
    curator = EntityCurator(entities, flaky, documents, merges)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        curator.merge(scope, keep, [dup, dup2])
    [pending] = merges.all(scope)
    assert pending.status == "pending" and entities.get(scope, dup) is not None

    done = curator.merge(scope, keep, [dup, dup2])  # the retry finishes the pending merge
    assert done.already_merged == [dup, dup2] and done.merged == []
    assert done.merge_id == pending.merge_id

    survivor = entities.get(scope, keep)
    assert survivor is not None and survivor.aliases == ["Acme Holdings", "AH", "Acme Intl"]
    assert survivor.vector == vec
    assert entities.get(scope, dup) is None and entities.get(scope, dup2) is None
    assert graph.get_node(scope, dup) is None and graph.get_node(scope, dup2) is None
    kept_edges = {
        (e.subject_id, e.object_id, e.provenance.source_doc_id, e.provenance.sentence_index)
        for e in graph.get_node_edges(scope, keep)
    }
    assert kept_edges == {
        (bob, keep, f"d1-{tag}", 0),
        (cat, keep, f"d2-{tag}", 0),
        (bob, keep, f"d2-{tag}", 1),
    }
    assert [(s.document_id, s.mentions) for s in documents.entity_sources(scope, keep)] == [
        (f"d1-{tag}", 1),
        (f"d2-{tag}", 1),
        (f"d3-{tag}", 1),
    ]
    assert documents.entity_sources(scope, dup) == []

    [record] = merges.for_entity(scope, keep)
    assert record.status == "done" and record.completed_at and record.keep_aliases_before == []
    by_id = {d.canonical_id: d for d in record.dropped}
    assert (
        set(by_id) == {dup, dup2} and merges.containing(scope, dup)[0].merge_id == record.merge_id
    )
    assert (by_id[dup].name, by_id[dup].aliases, by_id[dup].vector) == (
        "Acme Holdings",
        ["AH"],
        vec,
    )
    assert by_id[dup].first_seen == first_seen
    assert {(e.subject_id, e.object_id) for e in by_id[dup].edges} == {(cat, dup), (bob, dup)}
    assert [(link.document_id, link.link.canonical_id) for link in by_id[dup].document_links] == [
        (f"d2-{tag}", dup)
    ]
    assert {(e.subject_id, e.object_id) for e in by_id[dup2].edges} == {(keep, dup2)}

    assert merges.all(other) == [] and merges.containing(other, dup) == []
    repeat = curator.merge(scope, keep, [dup])
    assert repeat.merged == [] and repeat.merge_id == record.merge_id
    assert len(merges.all(scope)) == 1


def _edge(s: str, o: str, doc: str, sentence: int = 0) -> Triple:
    provenance = EdgeProvenance(source_doc_id=doc, sentence_index=sentence, source_sentence="Text.")
    return Triple(subject_id=s, predicate="WORKS_FOR", object_id=o, provenance=provenance)


def _snapshot(
    scope: Scope,
    ids: list[str],
    docs: list[str],
    entities: EntityStore,
    graph: GraphStore,
    documents: DocumentStore,
) -> dict[str, Any]:
    """Everything a split must give back, as plain data: entities, nodes, edges and links."""
    edges: set[str] = set()
    for i in ids:
        edges |= {e.model_dump_json() for e in graph.get_node_edges(scope, i)}
    records = {d: documents.get(scope, d) for d in docs}
    found = {i: entities.get(scope, i) for i in ids}
    return {
        "entities": {
            i: (e.name, e.aliases, e.vector, entities.first_seen(scope, i)) if e else None
            for i, e in found.items()
        },
        "nodes": {i: graph.get_node(scope, i) is not None for i in ids},
        "edges": sorted(edges),
        "links": {
            d: [x.model_dump() for x in r.el_result] if r else None for d, r in records.items()
        },
    }


def _refused(
    curator: EntityCurator, scope: Scope, *args: Any, code: str, reason: str | None
) -> None:
    with pytest.raises(AgentError) as caught:
        curator.split(scope, *args)
    assert caught.value.code == code, caught.value.message
    if reason:
        assert caught.value.details and caught.value.details["reason"] == reason


def check_entity_split(
    entities: EntityStore,
    graph: GraphStore,
    documents: DocumentStore,
    merges: MergeRecordStore,
    dims: int,
) -> None:
    """Merge then split gives back what was there: entities, edges, aliases, links, first_seen."""
    tag = uuid.uuid4().hex[:8]
    scope, other = Scope(f"a{tag}"), Scope(f"b{tag}")
    names = ("keep", "dup", "dup2", "bob", "cat", "org2")
    keep, dup, dup2, bob, cat, org2 = (f"{n}-{tag}" for n in names)
    ids = [keep, dup, dup2, bob, cat, org2]
    docs = [f"d{n}-{tag}" for n in (1, 2, 3, 9)]
    vec = [1.0] + [0.0] * (dims - 1)
    vec2 = [0.0, 1.0] + [0.0] * (dims - 2)

    seeded = [
        CanonicalEntity(
            canonical_id=keep, name="Acme Corp", type="ORG", aliases=["ACME"], vector=vec
        ),
        CanonicalEntity(
            canonical_id=dup, name="Acme Holdings", type="ORG", aliases=["AH"], vector=vec2
        ),
        CanonicalEntity(canonical_id=dup2, name="Acme Intl", type="ORG", vector=vec),
        CanonicalEntity(canonical_id=bob, name="Bob Stone", type="PERSON", vector=vec),
        CanonicalEntity(canonical_id=cat, name="Cat Lee", type="PERSON", vector=vec),
        CanonicalEntity(canonical_id=org2, name="Globex", type="ORG", vector=vec2),
    ]
    for entity in seeded:
        entities.upsert(scope, entity)
    graph.upsert_entities(scope, [e.model_copy(update={"vector": None}) for e in seeded])
    graph.write_triples(
        scope,
        [
            _edge(bob, keep, docs[0]),
            _edge(cat, dup, docs[1]),
            _edge(bob, dup, docs[1], 1),
            _edge(bob, keep, docs[1], 1),  # the survivor's own copy of what dup brings: a repeat
            _edge(keep, dup2, docs[2]),  # becomes a loop when merged
            _edge(dup, dup2, docs[2], 1),  # between two dropped entities
        ],
    )
    for doc, canonical in zip(docs[:3], (keep, dup, dup2), strict=True):
        link = EntityLink(
            mention_text="Acme", canonical_id=canonical, entity_type="ORG", score=1.0, is_new=True
        )
        record = DocumentRecord(
            document_id=doc, bucket="b", object_key=f"{doc}.txt", text="t", el_result=[link]
        )
        documents.upsert(scope, record)
    before = _snapshot(scope, ids, docs, entities, graph, documents)
    assert before["entities"][dup][3]  # first_seen was stamped

    curator = EntityCurator(entities, graph, documents, merges)
    merged = curator.merge(scope, keep, [dup, dup2])
    assert merged.merged == [dup, dup2]
    graph.write_triples(scope, [_edge(cat, keep, docs[3])])  # after the merge: stays
    later = EntityLink(
        mention_text="Acme Holdings", canonical_id=keep, entity_type="ORG", score=0.9, is_new=False
    )
    documents.upsert(
        scope,
        DocumentRecord(
            document_id=docs[3], bucket="b", object_key="d9.txt", text="t", el_result=[later]
        ),
    )
    after_merge = _snapshot(scope, ids, docs, entities, graph, documents)

    # Refusals leave everything as it was.
    _refused(curator, scope, bob, None, None, code="conflict", reason="no_merge_record")
    _refused(curator, scope, keep, "no-such-merge", None, code="not_found", reason=None)
    _refused(curator, scope, "no-such-entity", merged.merge_id, None, code="not_found", reason=None)
    _refused(
        curator, scope, keep, merged.merge_id, [bob], code="conflict", reason="no_merge_record"
    )
    _refused(curator, other, keep, merged.merge_id, None, code="not_found", reason=None)
    assert _snapshot(scope, ids, docs, entities, graph, documents) == after_merge

    # Split one entity at a time, then the other.
    first = curator.split(scope, keep, merged.merge_id, [dup2])
    assert [e.canonical_id for e in first.split] == [dup2] and first.already_split == []
    assert entities.get(scope, dup2) is not None and entities.get(scope, dup) is None
    [record] = merges.for_entity(scope, keep)
    assert record.status == "done" and record.split_ids == [dup2]
    again = curator.split(scope, keep, merged.merge_id, [dup2])  # idempotent
    assert again.split == [] and again.already_split == [dup2]
    second = curator.split(scope, keep, None, None)  # the only merge, the entities left
    assert [e.canonical_id for e in second.split] == [dup] and second.entity.canonical_id == keep

    # Back to what it was before the merge, plus what happened after it.
    ref = _snapshot(scope, ids, docs, entities, graph, documents)
    expected = dict(before)
    expected["edges"] = sorted({*before["edges"], _edge(cat, keep, docs[3]).model_dump_json()})
    expected["links"] = {**before["links"], docs[3]: after_merge["links"][docs[3]]}
    assert ref == expected
    [record] = merges.for_entity(scope, keep)
    assert record.status == "split" and record.split_ids == [dup2, dup] and record.split_at
    assert merges.all(other) == []
    nothing_left = curator.split(scope, keep, merged.merge_id, None)
    assert nothing_left.split == [] and nothing_left.already_split == []
    _refused(curator, scope, keep, None, None, code="conflict", reason="no_merge_record")

    # A dropped id that is taken blocks the split.
    again_merged = curator.merge(scope, keep, [dup])
    entities.upsert(scope, CanonicalEntity(canonical_id=dup, name="Other", type="ORG"))
    _refused(
        curator, scope, keep, again_merged.merge_id, None, code="conflict", reason="entity_exists"
    )
    entities.delete(scope, dup)

    # A chain is split in dependency order: keep out of org2 first, then dup out of keep.
    curator.merge(scope, org2, [keep])
    _refused(
        curator, scope, keep, again_merged.merge_id, None, code="conflict", reason="survivor_merged"
    )
    [into_org2] = merges.for_entity(scope, org2)
    curator.split(scope, org2, into_org2.merge_id, None)
    curator.split(scope, keep, again_merged.merge_id, None)
    assert _snapshot(scope, ids, docs, entities, graph, documents) == ref

    # An edge's other end merged away blocks the split until that end is split out first.
    curator.merge(scope, keep, [dup])
    curator.merge(scope, cat, [bob])
    [into_keep] = [r for r in merges.for_entity(scope, keep) if r.holds(dup)]
    _refused(
        curator, scope, keep, into_keep.merge_id, None, code="conflict", reason="neighbour_merged"
    )
    [into_cat] = merges.for_entity(scope, cat)
    curator.split(scope, cat, into_cat.merge_id, None)
    curator.split(scope, keep, into_keep.merge_id, None)
    assert _snapshot(scope, ids, docs, entities, graph, documents) == ref
