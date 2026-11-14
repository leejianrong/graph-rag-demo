# pg_stores: GRA-54 spike (ADR-0017)

Scratch prototype of the four store ports over Postgres + pgvector. Not wired into the product, not in the
mypy file list, not collected by the fast suite (`testpaths` excludes it, and every test skips without `PG_DSN`).

```bash
docker volume create grsmoke6-pg-data
docker run -d --name grsmoke6-pg -e POSTGRES_PASSWORD=spike -e POSTGRES_DB=spike \
  -p 127.0.0.1:55432:5432 -v grsmoke6-pg-data:/var/lib/postgresql/data \
  pgvector/pgvector:0.8.1-pg17
export PG_DSN=postgresql://postgres:spike@127.0.0.1:55432/spike
uv sync --frozen --extra dev && uv pip install "psycopg[binary]"
.venv/bin/python -m pytest spikes/pg_stores -p no:cacheprovider     # contract + behaviour tests
.venv/bin/python -m spikes.pg_stores.run_bench --questions 120      # one bench run, memory vs postgres
.venv/bin/python -m spikes.pg_stores.scale                          # synthetic 50k entities, 150k edges
docker rm -f grsmoke6-pg && docker volume rm grsmoke6-pg-data
```
