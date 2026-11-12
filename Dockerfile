# Web UI build stage (GRA-53): Vite + Svelte. `npm run build` regenerates the typed API client from
# docs/openapi.json first, so the spec is copied beside ui/ the way the repo lays it out. Only the
# files the build reads are copied (there is no .dockerignore, and a local node_modules must not
# come along). Nothing from this stage but the built files reaches the final image.
FROM node:24.11.0-slim AS ui
WORKDIR /build/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY docs/openapi.json ../docs/openapi.json
COPY ui/index.html ui/svelte.config.js ui/tsconfig.json ui/vite.config.ts ./
COPY ui/src ./src
RUN npm run build

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
RUN uv sync --frozen --extra trf --extra embed --extra otel --no-install-project

# Download the spaCy NER models the real stack loads: the transformer model
# (Settings.ner_model default) plus the small model as the fallback in the chain.
# --no-sync: run in the deps-only env from the step above WITHOUT re-syncing.
# A sync here would try to build+install the project (which needs README.md, not
# yet copied) and would churn the trf/embed extras in and out.
RUN uv run --no-sync python -m spacy download en_core_web_trf \
    && uv run --no-sync python -m spacy download en_core_web_sm

# Pre-download the sentence-transformer embedding model (V4, B1) so the real
# stack is ready to embed without a first-request download. (Cached to a data
# dir, not a package, so the project install below leaves it untouched.)
RUN uv run --no-sync python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Now the source, then install the project itself. Use `uv pip install --no-deps`
# rather than another `uv sync --frozen`: a frozen sync makes the env match the
# lock EXACTLY and would prune the spaCy model packages downloaded above (they
# aren't in uv.lock). `--no-deps` installs only graph_rag against the deps that
# are already present, leaving the models in place.
COPY graph_rag ./graph_rag
COPY README.md ./
# bench/ and datagen/ are uv workspace members (the root pyproject lists them); uv needs
# their manifests present to parse the project, but neither is shipped in the image.
COPY bench/pyproject.toml ./bench/pyproject.toml
COPY datagen/pyproject.toml ./datagen/pyproject.toml
RUN uv pip install --no-deps .

# The built web UI, served at / by the API process (Settings.ui_dir, default /app/ui-dist).
COPY --from=ui /build/ui/dist ./ui-dist

EXPOSE 8000

# Composition root: wires the real stack and serves FastAPI (python -m graph_rag.main).
# --no-sync so container start does NOT re-sync (which would prune the spaCy models).
CMD ["uv", "run", "--no-sync", "python", "-m", "graph_rag.main"]
