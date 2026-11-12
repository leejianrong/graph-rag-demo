<script lang="ts">
  import { onMount, tick } from "svelte";
  import {
    centreOn,
    claimExplorer,
    explorer,
    goBack,
    loadMore,
    pick,
    retryDetail,
    retrySearch,
    setQuery,
  } from "../lib/explorer.svelte";
  import { firstSeen, moreCount, spanLabel, splitNeighbours, validity, type Neighbour } from "../lib/explorer";

  let text = $state("");
  let evidence = $state<Neighbour | null>(null);
  let centreEl = $state<HTMLElement | null>(null);

  onMount(() => {
    claimExplorer();
    text = explorer.search.q;
    if (!explorer.search.loaded) setQuery(text, true);
  });

  const detail = $derived(explorer.detail);
  const focusId = $derived(explorer.trail[explorer.trail.length - 1]?.id ?? null);
  const sides = $derived(detail ? splitNeighbours(detail) : { incoming: [], outgoing: [] });
  const search = $derived(explorer.search);

  // A new focus clears the evidence of the last one and takes keyboard focus, since the button
  // that was pressed is gone.
  let lastId: string | null = null;
  $effect(() => {
    const id = detail?.canonical_id ?? null;
    if (id !== lastId) {
      lastId = id;
      evidence = null;
      if (id) void tick().then(() => centreEl?.focus());
    }
  });

  const key = (n: Neighbour, i: number) => `${n.direction}|${n.predicate}|${n.entity.canonical_id}|${i}`;
  const same = (a: Neighbour | null, b: Neighbour) => a !== null && a === b;
  const mentions = (n: number) => `${n} ${n === 1 ? "mention" : "mentions"}`;
</script>

<header class="head">
  <h1>Graph explorer</h1>
</header>

<form class="find" onsubmit={(e) => { e.preventDefault(); setQuery(text, true); }}>
  <label for="entity-search">Find an entity by name or alias</label>
  <input
    id="entity-search"
    type="search"
    maxlength="200"
    autocomplete="off"
    placeholder="Leave empty to list the newest"
    bind:value={text}
    oninput={() => setQuery(text)}
  />
</form>

<section class="results" aria-label="Search results" aria-busy={explorer.searching}>
  {#if explorer.searchError}
    <div class="problem" role="alert">
      <span>{explorer.searchError.message}</span>
      <button type="button" class="btn" onclick={retrySearch}>Try again</button>
    </div>
  {:else if search.loaded && search.results.length === 0}
    <div class="empty">
      {#if search.q.trim()}
        <strong>Nothing matches "{search.q.trim()}".</strong>
        <span>
          Search reads names and aliases. Try a shorter word, or check that this key's namespace holds the
          document you expect.
        </span>
      {:else}
        <strong>This namespace has no entities yet.</strong>
        <span>Entities appear once a document has been read and linked.</span>
      {/if}
      <a href="#/ingest">Go to Ingest</a>
    </div>
  {:else if search.results.length > 0}
    <ul class="chips">
      {#each search.results as e (e.canonical_id)}
        <li>
          <button
            type="button"
            class="chip"
            aria-current={e.canonical_id === focusId ? "true" : undefined}
            onclick={() => pick(e)}
          >
            {e.name} <span class="kind">{e.type}</span>
          </button>
        </li>
      {/each}
    </ul>
    {#if search.cursor}
      <button type="button" class="btn more" disabled={explorer.searching} onclick={loadMore}>
        {explorer.searching ? "Loading…" : "Load more"}
      </button>
    {/if}
  {:else}
    <p class="hint" role="status">Searching…</p>
  {/if}
</section>

{#if explorer.trail.length > 0}
  <div class="cols">
    <section class="hood" aria-labelledby="hood-title" aria-busy={explorer.loadingDetail}>
      <h2 id="hood-title">Neighbourhood of {detail?.name ?? explorer.trail[explorer.trail.length - 1].name}</h2>

      {#if explorer.trail.length > 1}
        <nav class="trail" aria-label="Entities you have visited">
          <ol>
            {#each explorer.trail as step, i (step.id)}
              <li>
                {#if i === explorer.trail.length - 1}
                  <span aria-current="location">{step.name}</span>
                {:else}
                  <button type="button" class="link" onclick={() => goBack(i)}>{step.name}</button>
                {/if}
              </li>
            {/each}
          </ol>
        </nav>
      {/if}

      {#if explorer.detailError}
        <div class="problem" role="alert">
          <span>{explorer.detailError.message}</span>
          {#if explorer.detailError.status !== 404}
            <button type="button" class="btn" onclick={retryDetail}>Try again</button>
          {/if}
        </div>
      {:else if !detail}
        <p class="hint" role="status">Loading…</p>
      {:else}
        <div class="board" class:stale={explorer.loadingDetail}>
          <div class="side">
            <h3 class="label">Points in</h3>
            {#each sides.incoming as n, i (key(n, i))}
              {@render node(n)}
            {:else}
              <span class="none">Nothing points here.</span>
            {/each}
            {#if moreCount(detail.incoming_total, sides.incoming.length) > 0}
              <span class="more-n">+{moreCount(detail.incoming_total, sides.incoming.length)} more</span>
            {/if}
          </div>

          <div class="centre" tabindex="-1" bind:this={centreEl} aria-current="true">
            <span class="c-name">{detail.name}</span>
            <span class="mono c-meta">{detail.type} · {detail.canonical_id}</span>
            <span class="c-count">
              {sides.incoming.length + sides.outgoing.length} of {detail.incoming_total + detail.outgoing_total}
              neighbours shown
            </span>
          </div>

          <div class="side">
            <h3 class="label">Points out</h3>
            {#each sides.outgoing as n, i (key(n, i))}
              {@render node(n)}
            {:else}
              <span class="none">Nothing points out.</span>
            {/each}
            {#if moreCount(detail.outgoing_total, sides.outgoing.length) > 0}
              <span class="more-n">+{moreCount(detail.outgoing_total, sides.outgoing.length)} more</span>
            {/if}
          </div>
        </div>
        <p class="hint">Select a neighbour to centre on it. "Evidence" shows the sentence an edge came from.</p>
      {/if}
    </section>

    {#if detail}
      <section class="panel" aria-label="Entity detail">
        <div class="p-head">
          <span class="label">Entity detail</span>
          <span class="p-name">{detail.name}</span>
          <span class="mono p-meta">{detail.type} · {detail.canonical_id}</span>
          <span class="p-meta">First seen {firstSeen(detail.first_seen)}</span>
        </div>

        <div class="block">
          <h3>Aliases</h3>
          {#if detail.aliases.length}
            <ul class="tags">
              {#each detail.aliases as a (a)}<li class="tag">{a}</li>{/each}
            </ul>
          {:else}
            <span class="none">No other names seen.</span>
          {/if}
        </div>

        <!-- A "Merged from" section and the merge and split actions go here (GRA-51). -->

        {#if evidence}
          <div class="block" aria-live="polite">
            <h3>Evidence for {evidence.predicate}</h3>
            <p class="quote">
              <mark>{evidence.provenance.source_sentence}</mark>
            </p>
            <span class="mono p-meta">
              {evidence.provenance.source_doc_id}, sentence {evidence.provenance.sentence_index}{#if spanLabel(evidence.provenance.char_start, evidence.provenance.char_end)}, {spanLabel(evidence.provenance.char_start, evidence.provenance.char_end)}{/if}
            </span>
            {#if evidence.provenance.confidence != null}
              <span class="p-meta">Confidence {evidence.provenance.confidence.toFixed(2)}</span>
            {/if}
            <button type="button" class="link" onclick={() => (evidence = null)}>Hide evidence</button>
          </div>
        {/if}

        <div class="block">
          <h3>
            {detail.sources_total === 0
              ? "Not in any document"
              : `Mentioned in ${detail.sources_total} ${detail.sources_total === 1 ? "document" : "documents"}`}
          </h3>
          {#if detail.sources.length}
            <ul class="docs">
              {#each detail.sources as s (s.document_id)}
                <li class="mono">{s.name} <span class="muted">· {mentions(s.mentions)}</span></li>
              {/each}
            </ul>
            {#if detail.sources_total > detail.sources.length}
              <span class="p-meta">+{detail.sources_total - detail.sources.length} more documents</span>
            {/if}
          {/if}
        </div>
      </section>
    {/if}
  </div>
{/if}

{#snippet node(n: Neighbour)}
  {@const when = validity(n.valid_from, n.valid_to)}
  <div class="nrow">
    <button
      type="button"
      class="node"
      onclick={() => centreOn({ id: n.entity.canonical_id, name: n.entity.name })}
    >
      <span class="n-name">{n.entity.name}</span>
      <span class="edge">{n.predicate}</span>
      {#if when}<span class="when">{when}</span>{/if}
    </button>
    <button
      type="button"
      class="link ev"
      aria-pressed={same(evidence, n)}
      aria-label={`Evidence for ${n.predicate} ${n.entity.name}`}
      onclick={() => (evidence = same(evidence, n) ? null : n)}
    >
      Evidence
    </button>
  </div>
{/snippet}

<style>
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 24px;
  }
  h1 {
    margin: 0;
    font-size: 26px;
    font-weight: 600;
  }
  h2 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
  }
  h3 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
  }
  .find {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-width: 560px;
    margin-bottom: 16px;
  }
  .find label {
    font-size: 13px;
    font-weight: 500;
  }
  input {
    padding: 11px 14px;
    font-size: 15px;
    border: 1px solid var(--line);
    border-radius: var(--radius-m);
    background: var(--panel);
    color: var(--ink);
    min-width: 0;
  }
  .results {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 24px;
  }
  .chips,
  .tags,
  .docs {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .chip {
    padding: 6px 12px;
    min-height: 36px;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: var(--panel);
    color: var(--ink);
    font-size: 13px;
  }
  .chip:hover {
    background: var(--hover);
  }
  .chip[aria-current="true"] {
    border-color: var(--accent);
    background: var(--soft);
    color: var(--accent);
  }
  .kind {
    margin-left: 4px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--muted);
  }
  .btn {
    padding: 9px 16px;
    min-height: 44px;
    border: 1px solid var(--accent);
    border-radius: var(--radius-m);
    background: var(--panel);
    color: var(--accent);
    font-size: 14px;
    font-weight: 600;
  }
  .btn:disabled {
    opacity: 0.6;
    cursor: default;
  }
  .more {
    align-self: flex-start;
  }
  .problem,
  .empty {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    padding: 14px 16px;
    border-radius: var(--radius-m);
    background: var(--panel);
    max-width: 720px;
    font-size: 14px;
  }
  .problem {
    border: 1px solid var(--bad);
    color: var(--bad);
  }
  .empty {
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
    border: 1px dashed var(--line);
  }
  .empty span {
    color: var(--muted);
  }
  .hint,
  .none,
  .muted {
    margin: 0;
    font-size: 13px;
    color: var(--muted);
  }
  .cols {
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    align-items: flex-start;
  }
  .hood {
    flex: 2 1 520px;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .trail ol {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin: 0;
    padding: 0;
    list-style: none;
    font-size: 13px;
  }
  .trail li + li::before {
    content: "›";
    margin-right: 4px;
    color: var(--muted);
  }
  .trail [aria-current] {
    font-weight: 600;
  }
  .link {
    padding: 4px 0;
    border: 0;
    background: none;
    color: var(--accent);
    font-size: 13px;
    text-decoration: underline;
  }
  .board {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    align-items: flex-start;
    justify-content: space-between;
    padding: 20px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-l);
  }
  .board.stale {
    opacity: 0.55;
  }
  .side {
    flex: 1 1 150px;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .label {
    margin: 0;
    font-size: 11px;
    font-weight: 400;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .nrow {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0;
  }
  .node {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 8px 12px;
    text-align: left;
    color: var(--ink);
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-m);
  }
  .node:hover {
    background: var(--hover);
  }
  .n-name {
    font-weight: 600;
    overflow-wrap: anywhere;
  }
  .edge {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--accent);
  }
  .when {
    font-size: 11px;
    color: var(--muted);
  }
  .ev {
    font-size: 12px;
    padding: 4px 2px;
  }
  .ev[aria-pressed="true"] {
    font-weight: 600;
  }
  .more-n {
    font-size: 13px;
    color: var(--muted);
  }
  .centre {
    flex: 0 1 210px;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 16px;
    border: 2px solid var(--accent);
    background: var(--soft);
    border-radius: var(--radius-l);
  }
  .c-name {
    font-size: 18px;
    font-weight: 600;
    overflow-wrap: anywhere;
  }
  .c-meta {
    font-size: 11px;
    color: var(--muted);
    overflow-wrap: anywhere;
  }
  .c-count {
    font-size: 12px;
    color: var(--muted);
  }
  .panel {
    flex: 1 1 320px;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 20px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-l);
  }
  .p-head {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .p-name {
    font-size: 20px;
    font-weight: 600;
    overflow-wrap: anywhere;
  }
  .p-meta {
    font-size: 12px;
    color: var(--muted);
    overflow-wrap: anywhere;
  }
  .block {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
    padding-top: 14px;
    border-top: 1px solid var(--line);
  }
  .p-head + .block {
    border-top: 0;
    padding-top: 0;
  }
  .tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .tag {
    padding: 2px 8px;
    border: 1px solid var(--line);
    border-radius: 999px;
    font-size: 12px;
  }
  .docs {
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 13px;
    overflow-wrap: anywhere;
  }
  .quote {
    margin: 0;
    font-size: 14px;
  }
  mark {
    background: #ffe7a3;
    color: #14211f;
    padding: 0 2px;
  }
  @media (max-width: 719px) {
    .board {
      flex-direction: column;
      align-items: stretch;
    }
    .side,
    .centre {
      flex: 0 0 auto;
    }
  }
</style>
