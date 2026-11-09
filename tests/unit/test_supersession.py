"""A newer fact ends an older one, and both stay queryable (GRA-43)."""

from __future__ import annotations

import pytest

from graph_rag.ids import document_id
from graph_rag.models import (
    EdgeProvenance,
    IngestTrigger,
    QueryRequest,
    Triple,
)
from graph_rag.scope import DEFAULT_SCOPE
from graph_rag.supersession import SINGLE_VALUED, supersede
from graph_rag.wiring import build_offline_components


def edge(
    obj: str,
    doc: str,
    frm: str | None,
    to: str | None = None,
    predicate: str = "WORKS_FOR",
    subject: str = "ada",
    sentence: int = 0,
) -> Triple:
    return Triple(
        subject_id=subject,
        predicate=predicate,
        object_id=obj,
        valid_from=frm,
        valid_to=to,
        provenance=EdgeProvenance(source_doc_id=doc, sentence_index=sentence, source_sentence="s"),
    )


# --- the pure rule ---------------------------------------------------------------------


def test_a_newer_fact_closes_the_older_one_on_its_start_date() -> None:
    old = edge("acme", "d1", "2018-01-01")
    written, closures = supersede([edge("initech", "d2", "2020-06-01")], [old], "d2")
    assert [c.edge for c in closures] == [old]
    assert (closures[0].valid_to, closures[0].by_doc) == ("2020-06-01", "d2")
    assert written[0].valid_to is None  # the newer fact is open


def test_an_older_fact_with_no_start_date_is_closed_too() -> None:
    _, closures = supersede([edge("initech", "d2", "2020-06-01")], [edge("acme", "d1", None)], "d2")
    assert [c.valid_to for c in closures] == ["2020-06-01"]


def test_a_new_edge_that_is_the_older_fact_ends_where_the_existing_one_starts() -> None:
    newer = edge("initech", "d2", "2020-06-01")
    written, closures = supersede([edge("acme", "d1", "2018-01-01")], [newer], "d1")
    assert closures == []
    assert (written[0].valid_to, written[0].superseded_by) == ("2020-06-01", "d2")


def test_it_ends_at_the_nearest_later_fact_not_the_farthest() -> None:
    later = [edge("b", "d2", "2022-01-01"), edge("c", "d3", "2020-01-01")]
    written, _ = supersede([edge("a", "d1", "2018-01-01")], later, "d1")
    assert (written[0].valid_to, written[0].superseded_by) == ("2020-01-01", "d3")


@pytest.mark.parametrize(
    ("new", "existing", "why"),
    [
        (edge("acme", "d2", "2020-01-01"), edge("acme", "d1", "2018-01-01"), "same object"),
        (edge("x", "d2", "2020-01-01"), edge("y", "d1", "2020-01-01"), "same start: order unknown"),
        (edge("x", "d2", None), edge("y", "d1", "2018-01-01"), "new fact undated"),
        (edge("x", "d2", "2020-01-01"), edge("y", "d2", "2018-01-01"), "same document"),
        (edge("x", "d2", "2020-01-01"), edge("y", "d1", "2018-01-01", subject="bob"), "subject"),
        (
            edge("x", "d2", "2020-01-01", predicate="FOUNDED"),
            edge("y", "d1", "2018-01-01", predicate="FOUNDED"),
            "FOUNDED can hold many objects",
        ),
        (
            edge("x", "d2", "2020-01-01"),
            edge("y", "d1", "2018-01-01", predicate="LOCATED_IN"),
            "different predicate",
        ),
        (
            edge("x", "d2", "2020-01-01"),
            edge("y", "d1", "2015-01-01", "2019-01-01"),
            "already ended",
        ),
    ],
)
def test_nothing_is_closed_when_order_or_identity_is_unclear(new, existing, why) -> None:
    written, closures = supersede([new], [existing], "d2")
    assert closures == [] and written == [new], why


def test_only_predicates_with_one_current_answer_supersede() -> None:
    assert SINGLE_VALUED == {"LOCATED_IN", "WORKS_FOR", "HAS_ROLE", "GOVERNED_BY"}


# --- through the pipeline and recall ----------------------------------------------------


class Stage:
    """A KG stage that says: the first person-like entity WORKS_FOR the other one."""

    def __init__(self, predicate: str = "WORKS_FOR") -> None:
        self.predicate = predicate

    def build(self, document_id, text, sentences, el_result, canonical_entities):  # noqa: ANN001
        by_name = {e.name: e for e in canonical_entities}
        ada = by_name["Ada Lovelace"]
        org = next(e for e in canonical_entities if e is not ada)
        return [
            Triple(
                subject_id=ada.canonical_id,
                predicate=self.predicate,
                object_id=org.canonical_id,
                provenance=EdgeProvenance(
                    source_doc_id=document_id,
                    sentence_index=0,
                    source_sentence=sentences[0].text,
                    char_start=sentences[0].char_start,
                    char_end=sentences[0].char_end,
                ),
            )
        ]


@pytest.fixture
def world():
    components = build_offline_components()
    components.orchestrator._kg_build_stage = Stage()  # type: ignore[assignment]
    return components


def ingest(components, key: str, org: str, date: str | None, verb: str = "works for") -> str:
    text = f"Ada Lovelace {verb} {org}."
    components.object_store.put(DEFAULT_SCOPE, "b", key, text.encode())
    trigger = IngestTrigger(bucket="b", object_key=key, document_date=date)
    assert components.orchestrator.process_document(trigger)
    return document_id("b", key, DEFAULT_SCOPE)


def edges(components) -> dict[str, Triple]:
    out = {}
    for entity in components.entity_store.all(DEFAULT_SCOPE):
        for e in components.graph_store.get_node_edges(DEFAULT_SCOPE, entity.canonical_id):
            out[e.provenance.source_doc_id] = e
    return out


def names_in_recall(components, **kw) -> set[str]:
    response = components.retriever.retrieve(
        QueryRequest(question="Where does Ada Lovelace work?", **kw)
    )
    return {n.name for n in response.subgraph.nodes} - {"Ada Lovelace"}


def test_a_later_document_ends_the_earlier_fact_and_recall_prefers_the_open_one(world) -> None:
    first = ingest(world, "a.txt", "Acme Corp", "2018-01-01")
    second = ingest(world, "b.txt", "Initech Corp", "2020-06-01")
    stored = edges(world)
    assert (stored[first].valid_to, stored[first].superseded_by) == ("2020-06-01", second)
    assert stored[second].valid_to is None
    assert names_in_recall(world) == {"Initech Corp"}


def test_both_remain_queryable_as_history(world) -> None:
    ingest(world, "a.txt", "Acme Corp", "2018-01-01")
    ingest(world, "b.txt", "Initech Corp", "2020-06-01")
    assert names_in_recall(world, as_of="2019-01-01") == {"Acme Corp"}
    assert names_in_recall(world, as_of="2020-05-31") == {"Acme Corp"}
    assert names_in_recall(world, as_of="2020-06-01") == {"Initech Corp"}  # the switch day
    assert names_in_recall(world, as_of="2017-01-01") == set()


def test_the_order_documents_arrive_in_does_not_matter(world) -> None:
    ingest(world, "b.txt", "Initech Corp", "2020-06-01")
    first = ingest(world, "a.txt", "Acme Corp", "2018-01-01")  # the older one arrives later
    assert edges(world)[first].valid_to == "2020-06-01"
    assert names_in_recall(world) == {"Initech Corp"}
    assert names_in_recall(world, as_of="2019-01-01") == {"Acme Corp"}


def test_deleting_the_newer_document_reopens_the_older_fact(world) -> None:
    first = ingest(world, "a.txt", "Acme Corp", "2018-01-01")
    second = ingest(world, "b.txt", "Initech Corp", "2020-06-01")
    assert world.orchestrator.delete_document(second, DEFAULT_SCOPE) is not None
    assert (edges(world)[first].valid_to, edges(world)[first].superseded_by) == (None, None)
    assert names_in_recall(world) == {"Acme Corp"}


def test_re_ingesting_the_newer_document_with_a_new_date_moves_the_end(world) -> None:
    first = ingest(world, "a.txt", "Acme Corp", "2018-01-01")
    ingest(world, "b.txt", "Initech Corp", "2020-06-01")
    ingest(world, "b.txt", "Initech Corp", "2021-03-01")  # the same source, dated differently
    assert edges(world)[first].valid_to == "2021-03-01"


def test_an_undated_newer_document_ends_nothing(world) -> None:
    first = ingest(world, "a.txt", "Acme Corp", "2018-01-01")
    ingest(world, "b.txt", "Initech Corp", None)
    assert edges(world)[first].valid_to is None
    assert names_in_recall(world) == {"Acme Corp", "Initech Corp"}  # nothing says which is newer


def test_a_predicate_that_holds_many_objects_is_not_superseded(world) -> None:
    world.orchestrator._kg_build_stage = Stage("FOUNDED")  # type: ignore[assignment]
    first = ingest(world, "a.txt", "Acme Corp", "2018-01-01", verb="founded")
    ingest(world, "b.txt", "Initech Corp", "2020-06-01", verb="founded")
    assert edges(world)[first].valid_to is None
    assert names_in_recall(world) == {"Acme Corp", "Initech Corp"}


def test_a_closed_neighbour_is_listed_last_with_its_interval(world) -> None:
    from graph_rag.lookup import EntityLookup

    ingest(world, "a.txt", "Acme Corp", "2018-01-01")
    ingest(world, "b.txt", "Initech Corp", "2020-06-01")
    found = EntityLookup(world.entity_store, world.graph_store).find(
        DEFAULT_SCOPE, name="Ada Lovelace", entity_id=None, limit=10
    )
    shown = [(n.entity.name, n.valid_from, n.valid_to) for n in found[0].neighbours]
    assert shown == [
        ("Initech Corp", "2020-06-01", None),
        ("Acme Corp", "2018-01-01", "2020-06-01"),
    ]
