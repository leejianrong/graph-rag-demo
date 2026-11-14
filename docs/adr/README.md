# Architecture decision records

One file per decision, with the alternatives weighed. The number is the order they were written. A later ADR can
change part of an earlier one; the earlier file keeps its text and says so in its status line.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-single-inprocess-modular-pipeline.md) | One in-process, modular pipeline consumer, not microservices | Accepted; the trigger transport is now chosen by ADR-0016 |
| [0002](0002-local-ner-with-spacy.md) | Local NER with spaCy (`en_core_web_trf`, falling back to `lg` then `sm`) | Accepted |
| [0003](0003-within-document-coreference.md) | Within-document coreference as a cluster map; the text is not rewritten | Accepted |
| [0004](0004-corpus-local-entity-linking.md) | Corpus-local entity linking, no external knowledge base | Accepted; GRA-157 added a name guard to the cosine merge |
| [0005](0005-elasticsearch-index-split.md) | Elasticsearch split into a documents index and an entities index | Accepted |
| [0006](0006-knowledge-graph-builder-and-model.md) | Schema-guided triples, a closed predicate set, a Neo4j graph with edge-level provenance | Accepted |
| [0007](0007-graph-rag-query-retrieval.md) | Non-LLM vector-anchored retrieval with graph expansion; LLM synthesis optional | Accepted; ingest is no longer Kafka-only (ADR-0016) |
| [0008](0008-llm-abstraction-caching-structured-output.md) | LiteLLM client, prompt-hash response cache, structured output | Accepted |
| [0009](0009-benchmarking-strategy.md) | Benchmarking strategy: 2WikiMultihopQA subset, non-LLM metrics | Accepted |
| [0010](0010-external-dependency-port-boundary.md) | Ports for every external dependency; fakes-first tests | Accepted |
| [0011](0011-in-repo-evaluation-bench.md) | An in-repo evaluation bench with gold the pipeline never touches | Accepted |
| [0012](0012-standalone-data-generator.md) | The synthetic data generator is a standalone package | Accepted |
| [0013](0013-question-gold-schema.md) | Questions carry a typed gold record | Accepted |
| [0014](0014-agent-api-contract.md) | The agent API: four verbs, one error shape (addenda: entity browsing, key management, merge, split; answer selection is in the decision) | Accepted |
| [0015](0015-tenancy-model.md) | Tenants are a field on every record, scoped by a `Scope` argument | Accepted |
| [0016](0016-trigger-backend.md) | Ingest triggers go through a Valkey stream by default; Kafka stays selectable | Accepted |
| [0017](0017-consolidate-stores-on-postgres.md) | Consolidate the stateful stores on Postgres with pgvector | Accepted as direction; prototyped in `spikes/pg_stores/`, nothing migrated (GRA-169 is the build) |
