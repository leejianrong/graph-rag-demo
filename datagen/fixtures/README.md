# Fixtures

Seeded datasets, committed so runs start from fixed inputs. Regenerate with `uv run kggen fixtures write`; a test fails if they drift from the generator.

| Fixture | Entities | Seed | Facts | Documents | Questions | Size |
|---|---|---|---|---|---|---|
| `small` | 150 | 0 | 300 | 117 | 50 | 228 KB |
| `medium` | 1000 | 0 | 2500 | 817 | 50 | 1600 KB |

Each folder is a `kggen export` of that preset (see its own `README.md`). Documents are template text. Update/delete/reorder scenarios regenerate from the world with `kggen scenario generate`. LLM-written documents need a cache, see `examples/`.
