# hard-small: one world, two texts, several pipelines

`PATH_RETRIEVAL=true uv run --extra embed kgbench hard --dir datagen/examples/hard-small --questions 300 --embedder bge --k 5 --llm-cache bench/kgbench/data/llm-cache-hard-small --spacy en_core_web_sm --texts hard --pipelines llm+spacy`; graph-rag with bge-small-en-v1.5 embeddings, one world, so the differences are one sample, not a distribution. `LLM` is the real coref and graph-build stages (Claude Haiku through semula, answers cached); `spaCy` is the real NER stage; the rest are the offline heuristics.

| | hard + LLM + spaCy |
|---|---|
| words per document | 43 |
| mention F1 | 0.856 |
| entity F1 | 0.728 |
| entity-pair F1 | 0.796 |
| triple F1 (strict)* | 0.789 |
| linking F1 | 0.859 |

\* `n/a`: that pipeline's graph builder emits only the `RELATED_TO` fallback, so strict triple F1 is 0 by construction (drawn as 0 in the charts).

## Retrieval: hard + LLM + spaCy

```
path@5 / recall@5, mean over seeds [0]
  hops  n     BM25                Dense               graph-rag passages  graph-rag graph     graph-rag path    
  all   300   0.24 / 0.50         0.24 / 0.47         0.24 / 0.47         0.21 / 0.41         0.27 / 0.48       
  1     75    0.89 / 0.89         0.89 / 0.89         0.89 / 0.89         0.73 / 0.73         0.81 / 0.81       
  2     75    0.03 / 0.49         0.07 / 0.49         0.07 / 0.49         0.11 / 0.44         0.19 / 0.51       
  3     75    0.03 / 0.36         0.00 / 0.30         0.00 / 0.30         0.01 / 0.29         0.07 / 0.36       
  4     75    0.00 / 0.25         0.00 / 0.22         0.00 / 0.22         0.00 / 0.18         0.00 / 0.23       
```
