"""Recall as of a date (GRA-44): the edges that held then, and only what depends on them."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from graph_rag.api import create_app
from graph_rag.config import Settings
from graph_rag.models import (
    CanonicalEntity,
    EdgeProvenance,
    QueryRequest,
    Subgraph,
    SupportingSentence,
    Triple,
)
from graph_rag.query.temporal import drop_sentences, holds, subgraph_as_of
from graph_rag.scope import DEFAULT_SCOPE
from graph_rag.wiring import build_offline_components
from tests.unit.test_agent_api_v1 import Queue, drain


def node(cid: str) -> CanonicalEntity:
    return CanonicalEntity(canonical_id=cid, name=cid, type="PERSON", aliases=[])


def edge(a: str, b: str, frm: str | None = None, to: str | None = None, doc: str = "d") -> Triple:
    return Triple(
        subject_id=a,
        predicate="WORKS_FOR",
        object_id=b,
        valid_from=frm,
        valid_to=to,
        provenance=EdgeProvenance(source_doc_id=doc, sentence_index=0, source_sentence=f"{a} {b}"),
    )


@pytest.mark.parametrize(
    ("frm", "to", "day", "expected"),
    [
        ("2019-05-01", None, "2019-04-30", False),  # before it began
        ("2019-05-01", None, "2019-05-01", True),  # valid_from is inclusive
        ("2019-05-01", None, "2030-01-01", True),  # no end known: still holds
        (None, "2020-01-01", "2019-12-31", True),
        (None, "2020-01-01", "2020-01-01", False),  # valid_to is exclusive
        ("2015-01-01", "2018-06-30", "2016-06-01", True),
        (None, None, "1999-01-01", True),  # unknown on both sides: not guessed away
    ],
)
def test_an_edge_holds_between_from_inclusive_and_to_exclusive(frm, to, day, expected) -> None:
    assert holds(edge("a", "b", frm, to), day) is expected


def test_excluded_edges_take_their_dependents_with_them() -> None:
    #   a -old- b -new- c      seeds: a.  As of a day the 'old' edge has ended,
    #   a -new2- d             so b and c (only reachable through it) go, d stays.
    graph = Subgraph(
        nodes=[node(x) for x in "abcd"],
        edges=[
            edge("a", "b", None, "2020-01-01", doc="old"),
            edge("b", "c", None, None, doc="far"),
            edge("a", "d", "2020-01-01", None, doc="new"),
        ],
    )
    now, gone = subgraph_as_of(graph, ["a"], hops=2, day="2021-01-01")
    assert {n.canonical_id for n in now.nodes} == {"a", "d"}
    assert [(e.subject_id, e.object_id) for e in now.edges] == [("a", "d")]
    assert gone == {("old", 0)}

    then, gone = subgraph_as_of(graph, ["a"], hops=2, day="2019-01-01")
    assert {n.canonical_id for n in then.nodes} == {"a", "b", "c"}
    assert gone == {("new", 0)}


def test_a_node_with_no_edges_is_kept_because_nothing_dates_it() -> None:
    graph = Subgraph(nodes=[node("a"), node("lonely")], edges=[])
    now, _ = subgraph_as_of(graph, ["a", "lonely"], hops=2, day="2000-01-01")
    assert {n.canonical_id for n in now.nodes} == {"a", "lonely"}


def test_a_seed_whose_every_edge_began_later_is_dropped() -> None:
    graph = Subgraph(nodes=[node("a"), node("b")], edges=[edge("a", "b", "2022-01-01")])
    before, _ = subgraph_as_of(graph, ["a"], hops=2, day="2021-01-01")
    assert before.nodes == [] and before.edges == []


def test_the_hop_limit_still_applies_after_filtering() -> None:
    graph = Subgraph(nodes=[node(x) for x in "abc"], edges=[edge("a", "b"), edge("b", "c")])
    shallow, _ = subgraph_as_of(graph, ["a"], hops=1, day="2021-01-01")
    assert {n.canonical_id for n in shallow.nodes} == {"a", "b"}


def test_a_sentence_is_dropped_only_when_no_kept_edge_cites_it() -> None:
    graph = Subgraph(
        nodes=[node(x) for x in "abc"],
        edges=[edge("a", "b", None, "2020-01-01", doc="shared"), edge("a", "c", doc="shared")],
    )
    _, gone = subgraph_as_of(graph, ["a"], hops=1, day="2021-01-01")
    assert gone == set()  # the kept a-c edge cites the same sentence

    sentences = [
        SupportingSentence(
            document_id=d, text="t", char_start=0, char_end=1, sentence_index=i, score=1.0
        )
        for d, i in [("old", 0), ("other", 0)]
    ]
    assert [s.document_id for s in drop_sentences(sentences, {("old", 0)})] == ["other"]


# --- through the retriever and the API ------------------------------------------------

TEXT = "Ada Lovelace works for Acme Corp. Acme Corp is based in London."


@pytest.fixture
def stack():
    components = build_offline_components()
    queue = Queue()
    app = create_app(components.object_store, queue, Settings(), retriever=components.retriever)
    return TestClient(app), components, queue


def _end_the_facts(components, on: str) -> None:
    """Stand in for supersession (GRA-43), which would set valid_to: end every edge on ``on``."""
    graph = components.graph_store
    by_doc: dict[str, dict[tuple[str, str, str, int], Triple]] = {}
    for entity in components.entity_store.all(DEFAULT_SCOPE):
        for e in graph.get_node_edges(DEFAULT_SCOPE, entity.canonical_id):
            key = (e.subject_id, e.predicate, e.object_id, e.provenance.sentence_index)
            by_doc.setdefault(e.provenance.source_doc_id, {})[key] = e
    for doc, edges in by_doc.items():
        graph.delete_document_edges(DEFAULT_SCOPE, doc)
        graph.write_triples(
            DEFAULT_SCOPE, [e.model_copy(update={"valid_to": on}) for e in edges.values()]
        )


def _recall(client: TestClient, **extra: object) -> dict:
    response = client.post("/v1/recall", json={"question": "Where is Acme Corp based?"} | extra)
    assert response.status_code == 200, response.text
    return response.json()


def test_recall_as_of_returns_the_facts_of_that_day(stack) -> None:
    client, components, queue = stack
    client.post(
        "/v1/remember", json={"text": TEXT, "source_id": "n", "document_date": "2018-01-01"}
    )
    drain(components, queue)
    _end_the_facts(components, on="2020-01-01")

    # Without as_of the view is the current one, and these facts ended in 2020 (GRA-43).
    assert _recall(client)["subgraph"]["edges"] == []

    then = _recall(client, as_of="2019-06-01")
    assert then["subgraph"]["edges"] and then["supporting_sentences"]

    after = _recall(client, as_of="2021-01-01")
    assert after["subgraph"]["edges"] == []
    assert after["supporting_sentences"] == []  # their only edges had ended

    before = _recall(client, as_of="2017-01-01")
    assert before["subgraph"]["edges"] == []  # the document did not exist yet


def test_omitting_as_of_changes_nothing(stack) -> None:
    client, components, queue = stack
    client.post(
        "/v1/remember", json={"text": TEXT, "source_id": "n", "document_date": "2018-01-01"}
    )
    drain(components, queue)
    plain = client.post("/query", json={"question": "Where is Acme Corp based?"}).json()
    unfiltered = components.retriever.retrieve(
        QueryRequest(question="Where is Acme Corp based?", as_of=None)
    )
    assert plain == unfiltered.model_dump(mode="json")


def test_undated_facts_are_still_returned_as_of_any_day(stack) -> None:
    client, components, queue = stack
    client.post("/v1/remember", json={"text": TEXT, "source_id": "n"})  # no document_date
    drain(components, queue)
    assert _recall(client, as_of="1990-01-01")["subgraph"]["edges"]


def test_a_malformed_as_of_is_refused(stack) -> None:
    client, *_ = stack
    for bad in ("2019", "2019-13-01", "last year"):
        response = client.post("/v1/recall", json={"question": "q", "as_of": bad})
        assert response.status_code in (400, 422)
        assert response.json()["error"]["code"] == "invalid_request"
