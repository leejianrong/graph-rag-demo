"""Fast E2E for merging duplicate entities when a later document links them (GRA-30).

Two documents name one person differently, so the linker keeps two entities. A third says, in one
coreference cluster, that the two names are the same person. The two entities fold into one:
the entity store keeps both names, the graph moves both entities' edges onto the survivor with
their provenance, and the documents' links follow. Whatever order the three arrive in, the same
entities and edges come out. No Docker, no model.
"""

from __future__ import annotations

from itertools import permutations

import pytest

from graph_rag.fakes import (
    FakeEmbedder,
    InMemoryDocumentStore,
    InMemoryEntityStore,
    InMemoryGraphStore,
    InMemoryObjectStore,
)
from graph_rag.ids import document_id
from graph_rag.models import (
    CorefCluster,
    EdgeProvenance,
    IngestTrigger,
    Mention,
    Sentence,
    Triple,
)
from graph_rag.orchestrator import Orchestrator
from graph_rag.scope import DEFAULT_SCOPE
from graph_rag.stages.entity_linking import EntityLinkingStage
from graph_rag.stages.ner import NerResult

pytestmark = pytest.mark.e2e

NAMES = {
    "Ada Lovelace": "PERSON",
    "Countess of Lovelace": "PERSON",
    "Acme": "ORG",
    "Initech": "ORG",
    "London": "LOCATION",
}
DOCS = {
    "a.txt": "Ada Lovelace works for Acme.",
    "b.txt": "Countess of Lovelace works for Initech.",
    "c.txt": "Ada Lovelace, also known as Countess of Lovelace, lives in London.",
}


class Ner:
    """Finds the known names in a one-sentence document."""

    def analyze(self, text: str) -> NerResult:
        found = sorted((text.index(n), n) for n in NAMES if n in text)
        mentions = [
            Mention(text=n, type=NAMES[n], char_start=i, char_end=i + len(n)) for i, n in found
        ]
        return NerResult(
            mentions=mentions,
            sentences=[Sentence(text=text, char_start=0, char_end=len(text), index=0)],
        )


class Coref:
    """The two names in 'also known as' text are one cluster; everything else stands alone."""

    def resolve(self, text: str, mentions: list[Mention]) -> list[CorefCluster]:
        if "also known as" in text:
            return [CorefCluster(canonical="Ada Lovelace", members=["Countess of Lovelace"])]
        return []


class Kg:
    """An edge from each document's first entity to each other one."""

    def build(self, document_id, text, sentences, el_result, canonical_entities):  # noqa: ANN001
        first, *rest = canonical_entities
        return [
            Triple(
                subject_id=first.canonical_id,
                predicate="WORKS_FOR",
                object_id=other.canonical_id,
                provenance=EdgeProvenance(
                    source_doc_id=document_id,
                    sentence_index=0,
                    source_sentence=sentences[0].text,
                ),
            )
            for other in rest
        ]


class Stack:
    def __init__(self) -> None:
        self.objects = InMemoryObjectStore()
        self.documents = InMemoryDocumentStore()
        self.entities = InMemoryEntityStore()
        self.graph = InMemoryGraphStore()
        self.orchestrator = Orchestrator(
            object_store=self.objects,
            document_store=self.documents,
            ner_stage=Ner(),  # type: ignore[arg-type]
            coref_stage=Coref(),  # type: ignore[arg-type]
            entity_linking_stage=EntityLinkingStage(
                self.entities, FakeEmbedder(), threshold=2.0, knn_top_k=0
            ),
            graph_store=self.graph,
            kg_build_stage=Kg(),  # type: ignore[arg-type]
            entity_store=self.entities,
        )

    def ingest(self, key: str) -> None:
        self.objects.put(DEFAULT_SCOPE, "b", key, DOCS[key].encode())
        assert self.orchestrator.process_document(IngestTrigger(bucket="b", object_key=key))

    def groups(self) -> set[frozenset[str]]:
        """Each entity as the set of names it answers to."""
        return {frozenset({e.name, *e.aliases}) for e in self.entities.all(DEFAULT_SCOPE)}

    def edges(self) -> set[tuple[frozenset[str], frozenset[str], str]]:
        """Each edge as (names of its subject, names of its object, source document)."""
        by_id = {
            e.canonical_id: frozenset({e.name, *e.aliases})
            for e in self.entities.all(DEFAULT_SCOPE)
        }
        return {
            (by_id[t.subject_id], by_id[t.object_id], t.provenance.source_doc_id)
            for n in by_id
            for t in self.graph.get_node_edges(DEFAULT_SCOPE, n)
            if t.subject_id == n
        }


def test_a_late_document_merges_two_earlier_entities() -> None:
    stack = Stack()
    stack.ingest("a.txt")
    stack.ingest("b.txt")
    assert (
        len([g for g in stack.groups() if "Ada Lovelace" in g or "Countess of Lovelace" in g]) == 2
    )

    stack.ingest("c.txt")

    person = frozenset({"Ada Lovelace", "Countess of Lovelace"})
    assert person in stack.groups() and len(stack.groups()) == 4  # person, Acme, Initech, London
    assert stack.graph.node_count(DEFAULT_SCOPE) == 4


def test_edges_and_their_provenance_survive_the_merge() -> None:
    stack = Stack()
    for key in ("a.txt", "b.txt", "c.txt"):
        stack.ingest(key)

    person = frozenset({"Ada Lovelace", "Countess of Lovelace"})
    docs = {k: document_id("b", k, DEFAULT_SCOPE) for k in DOCS}
    assert stack.edges() == {
        (person, frozenset({"Acme"}), docs["a.txt"]),
        (person, frozenset({"Initech"}), docs["b.txt"]),
        (person, frozenset({"London"}), docs["c.txt"]),
    }
    sentences = {
        t.provenance.source_sentence
        for n in stack.entities.all(DEFAULT_SCOPE)
        for t in stack.graph.get_node_edges(DEFAULT_SCOPE, n.canonical_id)
    }
    assert sentences == set(DOCS.values())


def test_every_document_link_names_an_entity_that_exists() -> None:
    stack = Stack()
    for key in ("a.txt", "b.txt", "c.txt"):
        stack.ingest(key)
    known = {e.canonical_id for e in stack.entities.all(DEFAULT_SCOPE)}
    links = {link.canonical_id for r in stack.documents.all(DEFAULT_SCOPE) for link in r.el_result}
    assert links <= known
    assert all(stack.graph.get_node(DEFAULT_SCOPE, i) is not None for i in known)


@pytest.mark.parametrize("order", list(permutations(DOCS)))
def test_any_arrival_order_gives_the_same_entities_and_edges(order: tuple[str, ...]) -> None:
    reference = Stack()
    for key in DOCS:
        reference.ingest(key)

    stack = Stack()
    for key in order:
        stack.ingest(key)

    assert stack.groups() == reference.groups()
    assert stack.edges() == reference.edges()
    assert stack.graph.node_count(DEFAULT_SCOPE) == reference.graph.node_count(DEFAULT_SCOPE)


def test_forgetting_the_linking_document_does_not_split_them_again() -> None:
    stack = Stack()
    for key in ("a.txt", "b.txt", "c.txt"):
        stack.ingest(key)

    stack.orchestrator.delete_document(document_id("b", "c.txt", DEFAULT_SCOPE))

    person = frozenset({"Ada Lovelace", "Countess of Lovelace"})
    assert person in stack.groups()  # a split is for a person to ask for, not a side effect
    assert {e[1] for e in stack.edges()} == {frozenset({"Acme"}), frozenset({"Initech"})}
