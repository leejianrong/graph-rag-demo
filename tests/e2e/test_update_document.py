"""Fast E2E for changing a document that is already stored (GRA-27).

Ingesting a source again with different bytes replaces what the old text contributed: its
sentences, mentions and edges are swapped for the new text's, and an entity only the old text
named goes with it. An entity another document still mentions stays. No Docker and no model.
"""

from __future__ import annotations

import pytest

from graph_rag.scope import DEFAULT_SCOPE

from .test_delete_document import ADA, GRACE, Stack

pytestmark = pytest.mark.e2e

MOVED = b"Ada Lovelace works for Initech. Initech is based in Paris."


@pytest.fixture
def stack() -> Stack:
    return Stack()


def edge_targets(stack: Stack) -> set[frozenset[str]]:
    """Every edge as a pair of entity names, without direction (the offline stage picks its own)."""
    name = {e.canonical_id: e.name for e in stack.entities.all(DEFAULT_SCOPE)}
    return {
        frozenset((name[t.subject_id], name[t.object_id]))
        for n in name
        for t in stack.graph.get_node_edges(DEFAULT_SCOPE, n)
        if t.subject_id == n
    }


def test_a_changed_document_yields_new_edges_and_drops_the_old_ones(stack: Stack) -> None:
    doc = stack.ingest("ada.txt", ADA)
    assert frozenset(("Ada Lovelace", "Acme Corp")) in edge_targets(stack)

    assert stack.ingest("ada.txt", MOVED) == doc  # the same source keeps its id

    edges = edge_targets(stack)
    assert frozenset(("Ada Lovelace", "Initech")) in edges
    assert not any("Acme Corp" in pair for pair in edges)
    assert stack.documents.get(DEFAULT_SCOPE, doc).text == MOVED.decode()
    assert all(s.text in MOVED.decode() for s in stack.sentences_of(doc))


def test_an_entity_only_the_old_text_named_is_removed_from_both_stores(stack: Stack) -> None:
    stack.ingest("ada.txt", ADA)
    old_ids = {e.canonical_id for e in stack.entities.all(DEFAULT_SCOPE)}

    stack.ingest("ada.txt", MOVED)

    assert stack.names() >= {"Ada Lovelace", "Initech", "Paris"}
    assert not stack.names() & {"Acme Corp", "London"}
    gone = [i for i in old_ids if stack.entities.get(DEFAULT_SCOPE, i) is None]
    assert gone and all(stack.graph.get_node(DEFAULT_SCOPE, i) is None for i in gone)


def test_an_entity_another_document_mentions_stays(stack: Stack) -> None:
    stack.ingest("ada.txt", ADA)
    grace = stack.ingest("grace.txt", GRACE)

    stack.ingest("ada.txt", MOVED)

    assert "Acme Corp" in stack.names()  # Grace's document still names it
    assert frozenset(("Grace Hopper", "Acme Corp")) in edge_targets(stack)
    assert stack.documents.get(DEFAULT_SCOPE, grace) is not None
    assert frozenset(("Ada Lovelace", "Acme Corp")) not in edge_targets(stack)


def test_the_same_text_again_changes_nothing(stack: Stack) -> None:
    stack.ingest("ada.txt", ADA)
    before = (stack.names(), edge_targets(stack), stack.graph.node_count(DEFAULT_SCOPE))

    stack.ingest("ada.txt", ADA)

    assert (stack.names(), edge_targets(stack), stack.graph.node_count(DEFAULT_SCOPE)) == before
