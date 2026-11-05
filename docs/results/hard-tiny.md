# template vs hard vs template + LLM vs hard + LLM: the same world as different text and pipelines

`uv run --extra embed kgbench hard --dir datagen/examples/hard-tiny --questions 60 --embedder bge --k 5 --llm-cache bench/kgbench/data/llm-cache-hard-tiny`; graph-rag with bge-small-en-v1.5 embeddings, one world, so the differences are one sample, not a distribution. A `+ LLM` column runs the real coref and graph-build stages (Claude Haiku through semula, answers cached); the others use the offline heuristics.

| | template | hard | template + LLM | hard + LLM |
|---|---|---|---|---|
| words per document | 10 | 36 | 10 | 36 |
| mention F1 | 0.985 | 0.753 | 0.985 | 0.753 |
| entity F1 | 0.827 | 0.610 | 0.861 | 0.640 |
| entity pair F1 (relation ignored) | 0.774 | 0.752 | 0.911 | 0.957 |
| triple F1 (strict) | n/a | n/a | 0.917 | 0.959 |
| linking B-cubed F1 | 0.847 | 0.905 | 0.870 | 0.908 |

`n/a`: that pipeline's graph builder emits only the `RELATED_TO` fallback, so strict triple F1 is 0 by construction.

## Retrieval: template

```
path@5 / recall@5, mean over seeds [0]
  hops  n     BM25                Dense               graph-rag passages  graph-rag graph   
  all   60    0.30 / 0.61         0.28 / 0.55         0.28 / 0.55         0.30 / 0.56       
  1     15    0.93 / 0.93         1.00 / 1.00         1.00 / 1.00         1.00 / 1.00       
  2     15    0.27 / 0.60         0.13 / 0.57         0.13 / 0.57         0.20 / 0.60       
  3     15    0.00 / 0.49         0.00 / 0.33         0.00 / 0.33         0.00 / 0.38       
  4     15    0.00 / 0.42         0.00 / 0.30         0.00 / 0.28         0.00 / 0.27       
```

## Retrieval: hard

```
path@5 / recall@5, mean over seeds [0]
  hops  n     BM25                Dense               graph-rag passages  graph-rag graph   
  all   60    0.30 / 0.62         0.25 / 0.51         0.22 / 0.44         0.20 / 0.45       
  1     15    1.00 / 1.00         0.93 / 0.93         0.80 / 0.80         0.80 / 0.80       
  2     15    0.20 / 0.60         0.07 / 0.50         0.07 / 0.47         0.00 / 0.43       
  3     15    0.00 / 0.44         0.00 / 0.33         0.00 / 0.27         0.00 / 0.33       
  4     15    0.00 / 0.43         0.00 / 0.28         0.00 / 0.23         0.00 / 0.25       
```

## Retrieval: template + LLM

```
path@5 / recall@5, mean over seeds [0]
  hops  n     BM25                Dense               graph-rag passages  graph-rag graph   
  all   60    0.30 / 0.61         0.28 / 0.55         0.28 / 0.55         0.27 / 0.55       
  1     15    0.93 / 0.93         1.00 / 1.00         1.00 / 1.00         1.00 / 1.00       
  2     15    0.27 / 0.60         0.13 / 0.57         0.13 / 0.57         0.07 / 0.53       
  3     15    0.00 / 0.49         0.00 / 0.33         0.00 / 0.33         0.00 / 0.42       
  4     15    0.00 / 0.42         0.00 / 0.30         0.00 / 0.28         0.00 / 0.25       
```

## Retrieval: hard + LLM

```
path@5 / recall@5, mean over seeds [0]
  hops  n     BM25                Dense               graph-rag passages  graph-rag graph   
  all   60    0.30 / 0.62         0.25 / 0.51         0.22 / 0.44         0.25 / 0.53       
  1     15    1.00 / 1.00         0.93 / 0.93         0.80 / 0.80         0.93 / 0.93       
  2     15    0.20 / 0.60         0.07 / 0.50         0.07 / 0.47         0.07 / 0.47       
  3     15    0.00 / 0.44         0.00 / 0.33         0.00 / 0.27         0.00 / 0.40       
  4     15    0.00 / 0.43         0.00 / 0.28         0.00 / 0.23         0.00 / 0.33       
```
