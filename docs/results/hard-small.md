# The same world as template text and as harder LLM-written text

`uv run --extra embed kgbench hard --questions 300 --embedder bge --k 5`; offline graph-rag, bge-small-en-v1.5 embeddings, one world (the committed `hard-small` spec), so the differences are one sample, not a distribution.

| | template | hard |
|---|---|---|
| words per document | 13 | 43 |
| mention F1 | 0.907 | 0.756 |
| entity F1 | 0.740 | 0.573 |
| entity pair F1 (relation ignored) | 0.681 | 0.690 |
| linking B-cubed F1 | 0.843 | 0.833 |

Strict triple F1 is left out: the offline graph builder links co-occurring entities with `RELATED_TO` only, so it is 0 by construction. Judge relations with an LLM graph builder (`kgbench extract --real`).

## Retrieval on template text

```
path@5 / recall@5, mean over seeds [0]
  hops  n     BM25                Dense               graph-rag passages  graph-rag graph   
  all   300   0.25 / 0.48         0.25 / 0.49         0.22 / 0.45         0.21 / 0.44       
  1     75    0.95 / 0.95         0.96 / 0.96         0.84 / 0.84         0.77 / 0.77       
  2     75    0.04 / 0.46         0.03 / 0.46         0.04 / 0.45         0.08 / 0.47       
  3     75    0.00 / 0.32         0.01 / 0.33         0.00 / 0.30         0.00 / 0.31       
  4     75    0.00 / 0.22         0.00 / 0.22         0.00 / 0.20         0.00 / 0.21       
```

## Retrieval on hard text

```
path@5 / recall@5, mean over seeds [0]
  hops  n     BM25                Dense               graph-rag passages  graph-rag graph   
  all   300   0.24 / 0.50         0.24 / 0.47         0.21 / 0.42         0.21 / 0.42       
  1     75    0.89 / 0.89         0.89 / 0.89         0.80 / 0.80         0.75 / 0.75       
  2     75    0.03 / 0.49         0.07 / 0.49         0.05 / 0.43         0.05 / 0.43       
  3     75    0.03 / 0.36         0.00 / 0.30         0.00 / 0.26         0.03 / 0.30       
  4     75    0.00 / 0.25         0.00 / 0.22         0.00 / 0.19         0.00 / 0.20       
```
