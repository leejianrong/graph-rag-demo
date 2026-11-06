# Shared names: hard-small

`uv run --extra embed kgbench split --dir datagen/examples/hard-small --embedder bge --k 5 --llm-cache bench/kgbench/data/llm-cache-hard-small --spacy en_core_web_sm --pipelines offline spacy llm+spacy`; 300 questions, 177 with a shared name; bge-small-en-v1.5 embeddings. 44 of 180 entities share a name.

## Retrieval

| system | hops | no shared name: path@5 | shared name: path@5 | gap (points) |
|---|---|---|---|---|
| BM25 | all | 41% (51/123, 33%–50%) | 11% (20/177, 7%–17%) | -30 |
| BM25 | 1 | 93% (50/54, 82%–97%) | 81% (17/21, 60%–92%) | -12 |
| BM25 | 2 | 3% (1/34, 1%–15%) | 2% (1/41, 0%–13%) | -1 |
| BM25 | 3 | 0% (0/19, 0%–17%) | 4% (2/56, 1%–12%) | +4 |
| BM25 | 4 | 0% (0/16, 0%–19%) | 0% (0/59, 0%–6%) | +0 |
| Dense | all | 41% (50/123, 32%–49%) | 12% (22/177, 8%–18%) | -28 |
| Dense | 1 | 91% (49/54, 80%–96%) | 86% (18/21, 65%–95%) | -5 |
| Dense | 2 | 3% (1/34, 1%–15%) | 10% (4/41, 4%–23%) | +7 |
| Dense | 3 | 0% (0/19, 0%–17%) | 0% (0/56, 0%–6%) | +0 |
| Dense | 4 | 0% (0/16, 0%–19%) | 0% (0/59, 0%–6%) | +0 |
| graph-rag passages (offline) | all | 37% (45/123, 29%–45%) | 11% (19/177, 7%–16%) | -26 |
| graph-rag passages (offline) | 1 | 81% (44/54, 69%–90%) | 76% (16/21, 55%–89%) | -5 |
| graph-rag passages (offline) | 2 | 3% (1/34, 1%–15%) | 7% (3/41, 3%–19%) | +4 |
| graph-rag passages (offline) | 3 | 0% (0/19, 0%–17%) | 0% (0/56, 0%–6%) | +0 |
| graph-rag passages (offline) | 4 | 0% (0/16, 0%–19%) | 0% (0/59, 0%–6%) | +0 |
| graph-rag graph (offline) | all | 34% (42/123, 26%–43%) | 11% (20/177, 7%–17%) | -23 |
| graph-rag graph (offline) | 1 | 74% (40/54, 61%–84%) | 76% (16/21, 55%–89%) | +2 |
| graph-rag graph (offline) | 2 | 6% (2/34, 2%–19%) | 5% (2/41, 1%–16%) | -1 |
| graph-rag graph (offline) | 3 | 0% (0/19, 0%–17%) | 4% (2/56, 1%–12%) | +4 |
| graph-rag graph (offline) | 4 | 0% (0/16, 0%–19%) | 0% (0/59, 0%–6%) | +0 |
| graph-rag passages (spaCy) | all | 41% (50/123, 32%–49%) | 12% (22/177, 8%–18%) | -28 |
| graph-rag passages (spaCy) | 1 | 91% (49/54, 80%–96%) | 86% (18/21, 65%–95%) | -5 |
| graph-rag passages (spaCy) | 2 | 3% (1/34, 1%–15%) | 10% (4/41, 4%–23%) | +7 |
| graph-rag passages (spaCy) | 3 | 0% (0/19, 0%–17%) | 0% (0/56, 0%–6%) | +0 |
| graph-rag passages (spaCy) | 4 | 0% (0/16, 0%–19%) | 0% (0/59, 0%–6%) | +0 |
| graph-rag graph (spaCy) | all | 32% (39/123, 24%–40%) | 14% (24/177, 9%–19%) | -18 |
| graph-rag graph (spaCy) | 1 | 65% (35/54, 51%–76%) | 86% (18/21, 65%–95%) | +21 |
| graph-rag graph (spaCy) | 2 | 12% (4/34, 5%–27%) | 10% (4/41, 4%–23%) | -2 |
| graph-rag graph (spaCy) | 3 | 0% (0/19, 0%–17%) | 4% (2/56, 1%–12%) | +4 |
| graph-rag graph (spaCy) | 4 | 0% (0/16, 0%–19%) | 0% (0/59, 0%–6%) | +0 |
| graph-rag passages (LLM + spaCy) | all | 41% (50/123, 32%–49%) | 12% (22/177, 8%–18%) | -28 |
| graph-rag passages (LLM + spaCy) | 1 | 91% (49/54, 80%–96%) | 86% (18/21, 65%–95%) | -5 |
| graph-rag passages (LLM + spaCy) | 2 | 3% (1/34, 1%–15%) | 10% (4/41, 4%–23%) | +7 |
| graph-rag passages (LLM + spaCy) | 3 | 0% (0/19, 0%–17%) | 0% (0/56, 0%–6%) | +0 |
| graph-rag passages (LLM + spaCy) | 4 | 0% (0/16, 0%–19%) | 0% (0/59, 0%–6%) | +0 |
| graph-rag graph (LLM + spaCy) | all | 36% (44/123, 28%–45%) | 11% (20/177, 7%–17%) | -24 |
| graph-rag graph (LLM + spaCy) | 1 | 72% (39/54, 59%–82%) | 81% (17/21, 60%–92%) | +9 |
| graph-rag graph (LLM + spaCy) | 2 | 15% (5/34, 6%–30%) | 5% (2/41, 1%–16%) | -10 |
| graph-rag graph (LLM + spaCy) | 3 | 0% (0/19, 0%–17%) | 2% (1/56, 0%–9%) | +2 |
| graph-rag graph (LLM + spaCy) | 4 | 0% (0/16, 0%–19%) | 0% (0/59, 0%–6%) | +0 |

## Extraction

| pipeline | mention recall, not shared | mention recall, shared | triple recall, no shared end | triple recall, a shared end |
|---|---|---|---|---|
| offline | 95% (451/477, 92%–96%) | 96% (149/156, 91%–98%) | 0% (0/184, 0%–2%) | 0% (0/144, 0%–3%) |
| spaCy | 82% (389/477, 78%–85%) | 83% (129/156, 76%–88%) | 0% (0/184, 0%–2%) | 0% (0/144, 0%–3%) |
| LLM + spaCy | 82% (389/477, 78%–85%) | 83% (129/156, 76%–88%) | 78% (143/184, 71%–83%) | 65% (93/144, 56%–72%) |
