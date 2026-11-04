# Complexity sweeps: small worlds

Where does each retriever degrade as the graph gets harder? Each section changes one knob of the world, holds the rest at the base, and scores BM25, dense retrieval and graph-rag on generated questions. path@5 is the share of questions whose whole path is in the top 5 sentences.

## Setup

- Preset `small` with n_entities=500, world seeds [0, 1, 2], every other knob at its base value.
- Embedder for the dense baseline and graph-rag: bge-small-en-v1.5 (local sentence-transformers).
- Reproduce: `uv run --extra embed kgbench sweep --knob all --preset small --set n_entities=500 --seeds 0 1 2 --questions 160 --embedder bge --k 5 --out docs/results/sweeps`
- **The verdict reads 2-hop path@5**, not the score over every question. A 1-hop path is solved by every system and nobody solves a 3-hop or 4-hop path in the top 5, so the all-hops score mostly counts how many 1-hop questions there were. (The `hops` sweep reads each hop count on its own.)
- A point is a win for graph-rag when its graph evidence beats the better baseline by at least 0.05. A smaller gap is inside the spread between seeds.

## Verdicts

- **n_entities**: graph-rag beat the baselines on 2-hop path@5 at every value of n_entities tried; the margin was smallest at 1000 (+0.09).
- **avg_degree**: graph-rag beat the baselines on 2-hop path@5 up to 6 and stopped beating them from 10 on.
- **degree_skew**: graph-rag beat the baselines on 2-hop path@5 at every value of degree_skew tried; the margin was smallest at 1.5 (+0.09).
- **homonym_rate**: graph-rag beat the baselines on 2-hop path@5 up to 0.3 and stopped beating them from 0.5 on.
- **churn_rate**: graph-rag beat the baselines on 2-hop path@5 up to 0.1 and stopped beating them from 0.3 on.
- **hops**: graph-rag beat the baselines on path@5 only at 2 of hops.

## n_entities

![2-hop path@5 against n_entities](./sweep-n_entities.svg)

2-hop path@5, mean over seeds; n is the 2-hop questions behind each row:

| n_entities | n | BM25 | Dense | graph-rag passages | graph-rag graph | margin |
|---|---|---|---|---|---|---|
| 100 | 120 | 0.03 | 0.02 | 0.02 | 0.18 | +0.15 |
| 300 | 120 | 0.01 | 0.02 | 0.02 | 0.14 | +0.12 |
| 1000 | 120 | 0.01 | 0.02 | 0.02 | 0.11 | +0.09 |
| 2000 | 120 | 0.00 | 0.00 | 0.00 | 0.20 | +0.20 |

Over every hop count (path@5 / recall@5), for reference:

| n_entities | n | BM25 | Dense | graph-rag passages | graph-rag graph |
|---|---|---|---|---|---|
| 100 | 480 | 0.25 / 0.50 | 0.25 / 0.52 | 0.25 / 0.52 | 0.28 / 0.51 |
| 300 | 480 | 0.24 / 0.50 | 0.25 / 0.50 | 0.25 / 0.50 | 0.27 / 0.51 |
| 1000 | 480 | 0.25 / 0.51 | 0.25 / 0.50 | 0.25 / 0.50 | 0.26 / 0.51 |
| 2000 | 480 | 0.25 / 0.50 | 0.25 / 0.49 | 0.25 / 0.49 | 0.28 / 0.51 |

Worlds behind each point (realised, per seed):

| n_entities | seed | entities | facts | mean degree | max degree | shared names | replaced | retracted |
|---|---|---|---|---|---|---|---|---|
| 100 | 0 | 100 | 200 | 4.00 | 15 | 0 | 0 | 0 |
| 100 | 1 | 100 | 200 | 4.00 | 25 | 0 | 0 | 0 |
| 100 | 2 | 100 | 200 | 4.00 | 15 | 0 | 0 | 0 |
| 300 | 0 | 300 | 600 | 4.00 | 33 | 0 | 0 | 0 |
| 300 | 1 | 300 | 600 | 4.00 | 31 | 0 | 0 | 0 |
| 300 | 2 | 300 | 600 | 4.00 | 38 | 0 | 0 | 0 |
| 1000 | 0 | 1000 | 2000 | 4.00 | 44 | 0 | 0 | 0 |
| 1000 | 1 | 1000 | 2000 | 4.00 | 90 | 0 | 0 | 0 |
| 1000 | 2 | 1000 | 2000 | 4.00 | 47 | 0 | 0 | 0 |
| 2000 | 0 | 2000 | 4000 | 4.00 | 132 | 0 | 0 | 0 |
| 2000 | 1 | 2000 | 4000 | 4.00 | 59 | 0 | 0 | 0 |
| 2000 | 2 | 2000 | 4000 | 4.00 | 69 | 0 | 0 | 0 |

## avg_degree

![2-hop path@5 against avg_degree](./sweep-avg_degree.svg)

2-hop path@5, mean over seeds; n is the 2-hop questions behind each row:

| avg_degree | n | BM25 | Dense | graph-rag passages | graph-rag graph | margin |
|---|---|---|---|---|---|---|
| 2 | 120 | 0.01 | 0.02 | 0.02 | 0.30 | +0.28 |
| 4 | 120 | 0.01 | 0.01 | 0.01 | 0.18 | +0.18 |
| 6 | 120 | 0.00 | 0.00 | 0.00 | 0.12 | +0.12 |
| 10 | 120 | 0.00 | 0.00 | 0.00 | 0.03 | +0.03 |

Over every hop count (path@5 / recall@5), for reference:

| avg_degree | n | BM25 | Dense | graph-rag passages | graph-rag graph |
|---|---|---|---|---|---|
| 2 | 480 | 0.25 / 0.51 | 0.25 / 0.52 | 0.25 / 0.52 | 0.33 / 0.57 |
| 4 | 480 | 0.25 / 0.50 | 0.25 / 0.51 | 0.25 / 0.51 | 0.27 / 0.50 |
| 6 | 480 | 0.24 / 0.49 | 0.24 / 0.48 | 0.24 / 0.48 | 0.25 / 0.46 |
| 10 | 480 | 0.22 / 0.46 | 0.23 / 0.46 | 0.23 / 0.46 | 0.21 / 0.41 |

Worlds behind each point (realised, per seed):

| avg_degree | seed | entities | facts | mean degree | max degree | shared names | replaced | retracted |
|---|---|---|---|---|---|---|---|---|
| 2 | 0 | 500 | 500 | 2.00 | 23 | 0 | 0 | 0 |
| 2 | 1 | 500 | 500 | 2.00 | 23 | 0 | 0 | 0 |
| 2 | 2 | 500 | 500 | 2.00 | 17 | 0 | 0 | 0 |
| 4 | 0 | 500 | 1000 | 4.00 | 34 | 0 | 0 | 0 |
| 4 | 1 | 500 | 1000 | 4.00 | 35 | 0 | 0 | 0 |
| 4 | 2 | 500 | 1000 | 4.00 | 33 | 0 | 0 | 0 |
| 6 | 0 | 500 | 1500 | 6.00 | 55 | 0 | 0 | 0 |
| 6 | 1 | 500 | 1500 | 6.00 | 46 | 0 | 0 | 0 |
| 6 | 2 | 500 | 1500 | 6.00 | 41 | 0 | 0 | 0 |
| 10 | 0 | 500 | 2500 | 10.00 | 82 | 0 | 0 | 0 |
| 10 | 1 | 500 | 2500 | 10.00 | 64 | 0 | 0 | 0 |
| 10 | 2 | 500 | 2500 | 10.00 | 63 | 0 | 0 | 0 |

## degree_skew

![2-hop path@5 against degree_skew](./sweep-degree_skew.svg)

2-hop path@5, mean over seeds; n is the 2-hop questions behind each row:

| degree_skew | n | BM25 | Dense | graph-rag passages | graph-rag graph | margin |
|---|---|---|---|---|---|---|
| 0 | 120 | 0.02 | 0.00 | 0.00 | 0.12 | +0.10 |
| 0.5 | 120 | 0.01 | 0.01 | 0.01 | 0.18 | +0.18 |
| 1 | 120 | 0.01 | 0.01 | 0.01 | 0.13 | +0.12 |
| 1.5 | 120 | 0.03 | 0.00 | 0.00 | 0.12 | +0.09 |

Over every hop count (path@5 / recall@5), for reference:

| degree_skew | n | BM25 | Dense | graph-rag passages | graph-rag graph |
|---|---|---|---|---|---|
| 0 | 480 | 0.24 / 0.50 | 0.24 / 0.49 | 0.24 / 0.49 | 0.26 / 0.50 |
| 0.5 | 480 | 0.25 / 0.50 | 0.25 / 0.51 | 0.25 / 0.51 | 0.27 / 0.50 |
| 1 | 480 | 0.25 / 0.51 | 0.25 / 0.51 | 0.25 / 0.51 | 0.27 / 0.52 |
| 1.5 | 480 | 0.25 / 0.51 | 0.25 / 0.51 | 0.25 / 0.51 | 0.27 / 0.52 |

Worlds behind each point (realised, per seed):

| degree_skew | seed | entities | facts | mean degree | max degree | shared names | replaced | retracted |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 500 | 1000 | 4.00 | 14 | 0 | 0 | 0 |
| 0 | 1 | 500 | 1000 | 4.00 | 18 | 0 | 0 | 0 |
| 0 | 2 | 500 | 1000 | 4.00 | 13 | 0 | 0 | 0 |
| 0.5 | 0 | 500 | 1000 | 4.00 | 34 | 0 | 0 | 0 |
| 0.5 | 1 | 500 | 1000 | 4.00 | 35 | 0 | 0 | 0 |
| 0.5 | 2 | 500 | 1000 | 4.00 | 33 | 0 | 0 | 0 |
| 1 | 0 | 500 | 1000 | 4.00 | 92 | 0 | 0 | 0 |
| 1 | 1 | 500 | 1000 | 4.00 | 99 | 0 | 0 | 0 |
| 1 | 2 | 500 | 1000 | 4.00 | 86 | 0 | 0 | 0 |
| 1.5 | 0 | 500 | 1000 | 4.00 | 138 | 0 | 0 | 0 |
| 1.5 | 1 | 500 | 1000 | 4.00 | 118 | 0 | 0 | 0 |
| 1.5 | 2 | 500 | 1000 | 4.00 | 130 | 0 | 0 | 0 |

## homonym_rate

![2-hop path@5 against homonym_rate](./sweep-homonym_rate.svg)

2-hop path@5, mean over seeds; n is the 2-hop questions behind each row:

| homonym_rate | n | BM25 | Dense | graph-rag passages | graph-rag graph | margin |
|---|---|---|---|---|---|---|
| 0 | 120 | 0.01 | 0.01 | 0.01 | 0.18 | +0.18 |
| 0.1 | 120 | 0.01 | 0.00 | 0.00 | 0.10 | +0.09 |
| 0.3 | 120 | 0.03 | 0.00 | 0.00 | 0.12 | +0.09 |
| 0.5 | 120 | 0.03 | 0.00 | 0.00 | 0.05 | +0.02 |

Over every hop count (path@5 / recall@5), for reference:

| homonym_rate | n | BM25 | Dense | graph-rag passages | graph-rag graph |
|---|---|---|---|---|---|
| 0 | 480 | 0.25 / 0.50 | 0.25 / 0.51 | 0.25 / 0.51 | 0.27 / 0.50 |
| 0.1 | 480 | 0.24 / 0.49 | 0.25 / 0.50 | 0.25 / 0.50 | 0.26 / 0.49 |
| 0.3 | 480 | 0.25 / 0.50 | 0.25 / 0.50 | 0.25 / 0.50 | 0.26 / 0.49 |
| 0.5 | 480 | 0.25 / 0.49 | 0.25 / 0.50 | 0.25 / 0.50 | 0.24 / 0.47 |

Worlds behind each point (realised, per seed):

| homonym_rate | seed | entities | facts | mean degree | max degree | shared names | replaced | retracted |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 500 | 1000 | 4.00 | 34 | 0 | 0 | 0 |
| 0 | 1 | 500 | 1000 | 4.00 | 35 | 0 | 0 | 0 |
| 0 | 2 | 500 | 1000 | 4.00 | 33 | 0 | 0 | 0 |
| 0.1 | 0 | 500 | 1000 | 4.00 | 34 | 50 | 0 | 0 |
| 0.1 | 1 | 500 | 1000 | 4.00 | 35 | 50 | 0 | 0 |
| 0.1 | 2 | 500 | 1000 | 4.00 | 33 | 50 | 0 | 0 |
| 0.3 | 0 | 500 | 1000 | 4.00 | 34 | 150 | 0 | 0 |
| 0.3 | 1 | 500 | 1000 | 4.00 | 35 | 150 | 0 | 0 |
| 0.3 | 2 | 500 | 1000 | 4.00 | 33 | 150 | 0 | 0 |
| 0.5 | 0 | 500 | 1000 | 4.00 | 34 | 250 | 0 | 0 |
| 0.5 | 1 | 500 | 1000 | 4.00 | 35 | 250 | 0 | 0 |
| 0.5 | 2 | 500 | 1000 | 4.00 | 33 | 250 | 0 | 0 |

## churn_rate

![2-hop path@5 against churn_rate](./sweep-churn_rate.svg)

2-hop path@5, mean over seeds; n is the 2-hop questions behind each row:

| churn_rate | n | BM25 | Dense | graph-rag passages | graph-rag graph | margin |
|---|---|---|---|---|---|---|
| 0 | 120 | 0.01 | 0.01 | 0.01 | 0.18 | +0.18 |
| 0.1 | 120 | 0.02 | 0.02 | 0.02 | 0.12 | +0.11 |
| 0.3 | 120 | 0.02 | 0.00 | 0.00 | 0.06 | +0.03 |

Over every hop count (path@5 / recall@5), for reference:

| churn_rate | n | BM25 | Dense | graph-rag passages | graph-rag graph |
|---|---|---|---|---|---|
| 0 | 480 | 0.25 / 0.50 | 0.25 / 0.51 | 0.25 / 0.51 | 0.27 / 0.50 |
| 0.1 | 480 | 0.24 / 0.49 | 0.25 / 0.50 | 0.25 / 0.50 | 0.26 / 0.49 |
| 0.3 | 480 | 0.24 / 0.49 | 0.24 / 0.48 | 0.24 / 0.48 | 0.24 / 0.46 |

Worlds behind each point (realised, per seed):

| churn_rate | seed | entities | facts | mean degree | max degree | shared names | replaced | retracted |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 500 | 1000 | 4.00 | 34 | 0 | 0 | 0 |
| 0 | 1 | 500 | 1000 | 4.00 | 35 | 0 | 0 | 0 |
| 0 | 2 | 500 | 1000 | 4.00 | 33 | 0 | 0 | 0 |
| 0.1 | 0 | 500 | 1100 | 4.00 | 34 | 0 | 100 | 0 |
| 0.1 | 1 | 500 | 1100 | 4.00 | 35 | 0 | 100 | 0 |
| 0.1 | 2 | 500 | 1100 | 4.00 | 33 | 0 | 100 | 0 |
| 0.3 | 0 | 500 | 1300 | 4.00 | 34 | 0 | 300 | 0 |
| 0.3 | 1 | 500 | 1300 | 4.00 | 35 | 0 | 300 | 0 |
| 0.3 | 2 | 500 | 1300 | 4.00 | 33 | 0 | 300 | 0 |

## hops

![path@5 against hops](./sweep-hops.svg)

path@5, mean over seeds; n is the questions at that hop count behind each row:

| hops | n | BM25 | Dense | graph-rag passages | graph-rag graph | margin |
|---|---|---|---|---|---|---|
| 1 | 120 | 0.97 | 0.99 | 0.99 | 0.91 | -0.08 |
| 2 | 120 | 0.01 | 0.01 | 0.01 | 0.18 | +0.18 |
| 3 | 120 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 |
| 4 | 120 | 0.00 | 0.00 | 0.00 | 0.00 | +0.00 |

## Caveats

- Templated text with made-up names; graph-rag is the offline pipeline (see the compare report's caveats). A knob that moves the result here may not move it on real prose.
- Questions are plain entity questions, equal weight on each hop count, so each point has a quarter of its questions at 2 hops. Gaps under 0.05 are noise; the margin column shows how far each point is from that.
- Only 2-hop paths separate the systems, so the verdicts say where graph-rag keeps its advantage on that one hop count and say nothing about 3 and 4 hops, where none of them solves a path in the top 5.
