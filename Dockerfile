# Pipeline/API service image (ARCHITECTURE §8): Python 3.12 + uv.
FROM python:3.12-slim

# uv for fast, reproducible installs from the locked pyproject.toml/uv.lock.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install runtime dependencies first (cached until the lock changes). Install
# only the deps, not the project, so this layer is reused across source edits.
# The real stack uses transformer NER + the local embedder, so include the heavy
# `trf` (spacy-transformers + torch) and `embed` (sentence-transformers) extras
# here — neither is in the fast CI/pre-push install.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra trf --extra embed --no-install-project

# Download the spaCy NER models the real stack loads: the transformer model
# (Settings.ner_model default) plus the small model as the fallback in the chain.
RUN uv run --extra trf python -m spacy download en_core_web_trf \
    && uv run python -m spacy download en_core_web_sm

# Pre-download the sentence-transformer embedding model (V4, B1) so the real
# stack is ready to embed without a first-request download.
RUN uv run --extra embed python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Now the source, then install the project itself against the frozen lock.
COPY graph_rag ./graph_rag
COPY README.md ./
RUN uv sync --frozen --extra trf --extra embed

EXPOSE 8000

# Composition root: wires the real stack and serves FastAPI (python -m graph_rag.main).
CMD ["uv", "run", "python", "-m", "graph_rag.main"]
