# hard-small: one world, two texts, several pipelines

`PATH_RETRIEVAL=true uv run --extra embed kgbench hard --dir datagen/examples/hard-small --questions 300 --embedder bge --k 5 --llm-cache bench/kgbench/data/llm-cache-hard-small-trf --spacy en_core_web_trf --texts hard --pipelines llm+spacy`; graph-rag with bge-small-en-v1.5 embeddings, one world, so the differences are one sample, not a distribution. `LLM` is the real coref and graph-build stages (Claude Haiku through semula, answers cached); `spaCy` is the real NER stage; the rest are the offline heuristics.

| | hard + LLM + spaCy |
|---|---|
| words per document | 43 |
| mention F1 | 0.900 |
| entity F1 | 0.737 |
| entity-pair F1 | 0.881 |
| triple F1 (strict)* | 0.876 |
| linking F1 | 0.870 |

\* `n/a`: that pipeline's graph builder emits only the `RELATED_TO` fallback, so strict triple F1 is 0 by construction (drawn as 0 in the charts).

## Retrieval: hard + LLM + spaCy

```
path@5 / recall@5, mean over seeds [0]
  hops  n     BM25                Dense               graph-rag passages  graph-rag graph     graph-rag path    
  all   300   0.24 / 0.50         0.24 / 0.47         0.24 / 0.47         0.26 / 0.51         0.31 / 0.51       
  1     75    0.89 / 0.89         0.89 / 0.89         0.89 / 0.89         0.92 / 0.92         0.85 / 0.85       
  2     75    0.03 / 0.49         0.07 / 0.49         0.07 / 0.49         0.12 / 0.54         0.28 / 0.55       
  3     75    0.03 / 0.36         0.00 / 0.30         0.00 / 0.30         0.01 / 0.35         0.09 / 0.40       
  4     75    0.00 / 0.25         0.00 / 0.22         0.00 / 0.22         0.00 / 0.23         0.00 / 0.25       
```
