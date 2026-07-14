# Supply-chain demo corpus

A tiny, three-document corpus that reproduces the multi-hop example from the
project README: no single document connects a supplier to the law that governs
it, but the graph does, in two hops.

| File | What it says |
|------|--------------|
| `01-aurelia-components.md` | *Aurelia Components* is based in *Berlin*. |
| `02-german-supply-chain-act.md` | The *German Supply Chain Act* applies to companies in *Berlin*. |
| `03-manufacturing-in-berlin.md` | Context on *Berlin* manufacturing (reinforces the hub). |

The connection lives in the structure between documents. *Berlin* is the bridge:

```
Aurelia Components ──▶ Berlin ──▶ German Supply Chain Act
```

The demo question is:

> **How is Aurelia Components connected to the German Supply Chain Act?**

Neither document mentions both endpoints, so ordinary keyword or vector search
over a single passage misses the link. Graph retrieval seeds on the two named
entities, walks two hops through *Berlin*, and returns the connected subgraph.

## Running it

```bash
make demo-offline   # no Docker, no API key — deterministic heuristic pipeline
make demo           # the real stack over HTTP (needs `make up` + OPENAI_API_KEY)
```

Both read *these same files*. The offline run extracts entities with a
Title-Case heuristic and connects entities that share a sentence, so every edge
is a generic `RELATED_TO`. The real run uses spaCy NER and an LLM to build the
graph, so edges are typed (`LOCATED_IN`, `SUBJECT_TO`, and so on) and you can add
`synthesize: true` for a prose answer. The corpus is written to work under both.
