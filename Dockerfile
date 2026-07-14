# Pipeline/API service image (ARCHITECTURE §8): Python 3.12 + uv.
FROM python:3.12-slim

# uv for fast, reproducible installs from the locked pyproject.toml/uv.lock.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install runtime dependencies first (cached until the lock changes). Install
# only the deps, not the project, so this layer is reused across source edits.
# The real stack uses transformer NER, so include the heavy `trf` extra
# (spacy-transformers + torch) here — not in the fast CI/pre-push install.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra trf --no-install-project

# Download the spaCy NER models the real stack loads: the transformer model
# (Settings.ner_model default) plus the small model as the fallback in the chain.
RUN uv run --extra trf python -m spacy download en_core_web_trf \
    && uv run python -m spacy download en_core_web_sm

# Now the source, then install the project itself against the frozen lock.
COPY graph_rag ./graph_rag
COPY README.md ./
RUN uv sync --frozen --extra trf

EXPOSE 8000

# Composition root: wires the real stack and serves FastAPI (python -m graph_rag.main).
CMD ["uv", "run", "python", "-m", "graph_rag.main"]
