"""The LLM response cache is kept per tenant (GRA-139)."""

from __future__ import annotations

from pathlib import Path

from graph_rag.adapters.llm_client import LiteLLMClient, tenant_cache_dir
from graph_rag.config import Settings
from graph_rag.fakes import (
    FakeEmbedder,
    InMemoryDocumentStore,
    InMemoryEntityStore,
    InMemoryGraphStore,
    InMemoryObjectStore,
)
from graph_rag.models import IngestTrigger, QueryRequest
from graph_rag.orchestrator import Orchestrator
from graph_rag.query.retriever import QueryRetriever
from graph_rag.query.synthesis import AnswerSynthesizer
from graph_rag.scope import Scope, acting_for, current_tenant
from graph_rag.stages.ner import NerResult
from graph_rag.tenant_admin import Stores, delete_tenant

PROMPT = "Who is Ada Lovelace?"


class Backend:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, model: str, prompt: str, **params: object) -> str:
        self.calls += 1
        return f"answer {self.calls}"


def client(tmp_path: Path) -> tuple[LiteLLMClient, Backend]:
    backend = Backend()
    return LiteLLMClient("m", cache_dir=tmp_path, completion_fn=backend), backend


def test_identical_text_in_two_tenants_does_not_share_an_entry(tmp_path: Path) -> None:
    llm, backend = client(tmp_path)
    with acting_for("acme"):
        first = llm.complete(PROMPT)
    with acting_for("globex"):
        second = llm.complete(PROMPT)
    assert backend.calls == 2 and first != second  # globex did not get acme's response
    assert llm.cache_hits == 0


def test_a_tenant_still_hits_its_own_entry(tmp_path: Path) -> None:
    llm, backend = client(tmp_path)
    for _ in range(2):
        with acting_for("acme"):
            llm.complete(PROMPT)
    assert backend.calls == 1 and llm.cache_hits == 1


def test_a_tenant_does_not_read_the_default_tenants_entry(tmp_path: Path) -> None:
    llm, backend = client(tmp_path)
    llm.complete(PROMPT)  # default tenant
    with acting_for("acme"):
        llm.complete(PROMPT)
    assert backend.calls == 2


def test_the_default_tenant_keeps_the_cache_root(tmp_path: Path) -> None:
    llm, _ = client(tmp_path)
    llm.complete(PROMPT)
    assert len(list(tmp_path.glob("*.json"))) == 1  # entries from before tenancy still hit


def test_a_tenant_name_cannot_name_a_path(tmp_path: Path) -> None:
    path = tenant_cache_dir(tmp_path, "..")
    assert path.parent.parent == tmp_path and ".." not in path.parts


def test_the_acting_tenant_is_restored_even_after_an_error() -> None:
    try:
        with acting_for("acme"):
            raise RuntimeError
    except RuntimeError:
        pass
    assert current_tenant() == "default"


def test_ingest_runs_its_llm_calls_as_the_documents_tenant() -> None:
    seen: list[str] = []

    class Probe:
        def analyze(self, text: str) -> NerResult:
            return NerResult(mentions=[], sentences=[])

    class Coref:
        def resolve(self, *args: object, **kwargs: object) -> object:
            seen.append(current_tenant())
            raise RuntimeError("stop here")

    objects = InMemoryObjectStore()
    objects.put(Scope("acme"), "b", "k.txt", b"hello")
    Orchestrator(
        object_store=objects,
        document_store=InMemoryDocumentStore(),
        ner_stage=Probe(),
        coref_stage=Coref(),
    ).process_document(IngestTrigger(bucket="b", object_key="k.txt", tenant="acme"))
    assert seen == ["acme"]
    assert current_tenant() == "default"


def test_deleting_a_tenant_removes_its_cache_and_leaves_the_others(tmp_path: Path) -> None:
    llm, _ = client(tmp_path)
    for tenant in ("acme", "globex"):
        with acting_for(tenant):
            llm.complete(PROMPT)
    stores = Stores(
        objects=InMemoryObjectStore(),
        documents=InMemoryDocumentStore(),
        entities=InMemoryEntityStore(),
        graph=InMemoryGraphStore(),
    )
    delete_tenant(stores, "acme", tmp_path)
    assert not tenant_cache_dir(tmp_path, "acme").exists()
    assert tenant_cache_dir(tmp_path, "globex").exists()


def test_synthesis_runs_its_llm_call_as_the_askers_tenant() -> None:
    seen: list[str] = []

    class Llm:
        def complete(self, prompt: str, **params: object) -> str:
            seen.append(current_tenant())
            return "prose"

    retriever = QueryRetriever.from_settings(
        Settings(),
        embedder=FakeEmbedder(),
        entity_store=InMemoryEntityStore(),
        document_store=InMemoryDocumentStore(),
        graph_store=InMemoryGraphStore(),
        synthesizer=AnswerSynthesizer(Llm()),  # type: ignore[arg-type]
    )
    retriever.retrieve(QueryRequest(question="Who?", synthesize=True), Scope("acme"))
    assert seen == ["acme"] and current_tenant() == "default"
