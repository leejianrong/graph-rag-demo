# 2WikiMultihopQA slice: 200 questions, 1585 paragraphs, 5295 sentences

The first 200 validation rows of `framolfese/2WikiMultihopQA` sorted by id; bge-small-en-v1.5 embeddings. A question is solved at k when every gold supporting sentence is in the top k; `hops` counts the paragraphs holding its gold sentences.

## Whole-path@5, with 95% intervals

| group | n | BM25 | Dense | graph-rag passages | graph-rag graph |
|---|---|---|---|---|---|
| all | 200 | 16% (31/200, 11%–21%) | 26% (51/200, 20%–32%) | 24% (48/200, 19%–30%) | 23% (46/200, 18%–29%) |
| 2 paragraphs | 160 | 19% (31/160, 14%–26%) | 32% (51/160, 25%–39%) | 30% (48/160, 23%–38%) | 28% (45/160, 22%–36%) |
| 4 paragraphs | 40 | 0% (0/40, 0%–9%) | 0% (0/40, 0%–9%) | 0% (0/40, 0%–9%) | 2% (1/40, 0%–13%) |

Ranking by whole-path@5 over all questions: Dense > graph-rag passages > graph-rag graph > BM25.

## BM25

```text
Retrieval vs gold: recall@k and whole-path@k
--------------------------------------------
  hops  n    R@1  path@1   R@3  path@3   R@5  path@5   R@10 path@10
  all   200  0.20  0.00    0.37  0.11    0.42  0.15    0.50  0.22  
  2     160  0.21  0.00    0.37  0.14    0.42  0.19    0.51  0.28  
  4     40   0.15  0.00    0.36  0.00    0.39  0.00    0.45  0.00  
```

## Dense

```text
Retrieval vs gold: recall@k and whole-path@k
--------------------------------------------
  hops  n    R@1  path@1   R@3  path@3   R@5  path@5   R@10 path@10
  all   200  0.26  0.00    0.48  0.19    0.54  0.26    0.60  0.32  
  2     160  0.28  0.00    0.50  0.24    0.56  0.32    0.62  0.39  
  4     40   0.22  0.00    0.42  0.00    0.46  0.00    0.48  0.00  
```

## graph-rag passages

```text
Retrieval vs gold: recall@k and whole-path@k
--------------------------------------------
  hops  n    R@1  path@1   R@3  path@3   R@5  path@5   R@10 path@10
  all   200  0.26  0.00    0.47  0.18    0.51  0.24    0.51  0.24  
  2     160  0.28  0.00    0.48  0.23    0.53  0.30    0.53  0.30  
  4     40   0.21  0.00    0.41  0.00    0.45  0.00    0.45  0.00  
```

## graph-rag graph

```text
Retrieval vs gold: recall@k and whole-path@k
--------------------------------------------
  hops  n    R@1  path@1   R@3  path@3   R@5  path@5   R@10 path@10
  all   200  0.25  0.00    0.42  0.18    0.48  0.23    0.53  0.28  
  2     160  0.26  0.00    0.44  0.23    0.49  0.28    0.55  0.33  
  4     40   0.21  0.00    0.36  0.00    0.42  0.03    0.47  0.05  
```
