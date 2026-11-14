# ui

The product web UI: Vite, Svelte 5 and TypeScript over the HTTP API. Views (one file each in `src/views/`): sign-in,
ingest, query, graph explorer (with entity merge and split in `CurationPanel`) and API keys. The key is held in memory only.
This is not the generator UI (`make ui`, `kggen ui`).

Needs Node 22 or later (CI and the Dockerfile use 24).

```bash
make ui-install    # npm ci
make ui-dev        # dev server on :5173, proxies the API at :8000 (API_URL=... to change)
make ui-build      # type-check, then build into ui/dist
make ui-e2e        # build, then Playwright against graph-rag-offline (once: cd ui && npx playwright install chromium)
cd ui && npm test  # vitest
```

The API client types (`src/api/schema.d.ts`, gitignored) are generated from `../docs/openapi.json` by `npm run gen:api`, which `dev`,
`build`, `check` and `test` run first. After an API change run `make openapi` and commit the new `docs/openapi.json`; a
fast-suite test fails when it is stale. The API process serves `ui/dist` at `/` when `UI_DIR` points at it (the image
sets `/app/ui-dist`), under a CSP that allows only its own files, so fonts are bundled (`@fontsource/ibm-plex-*`).
`node_modules/`, `dist/`, `playwright-report/` and `test-results/` are gitignored.
