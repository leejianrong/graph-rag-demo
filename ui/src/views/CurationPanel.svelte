<script lang="ts">
  import { tick } from "svelte";
  import {
    backToPicker,
    closeCuration,
    confirmMerge,
    confirmSplit,
    curation,
    openEntity,
    reviewMerge,
    selectionProblem,
    setPickerQuery,
    toggleSelected,
  } from "../lib/curation.svelte";
  import { MAX_MERGE, mergeLines, mergeSummary, splitLines } from "../lib/curation";
  import type { Neighbourhood } from "../lib/explorer";

  let { keep }: { keep: Neighbourhood } = $props();

  let headingEl = $state<HTMLElement | null>(null);

  // Each step of a flow takes focus, so a keyboard user lands on what just opened.
  $effect(() => {
    void curation.mode;
    void tick().then(() => headingEl?.focus());
  });

  const problem = $derived(curation.selected.length > 0 ? selectionProblem() : null);
  const lines = $derived(
    curation.mode === "confirm-merge" && curation.previews
      ? mergeLines(keep.name, curation.selected.map((e) => e.name), mergeSummary(keep, curation.previews))
      : curation.mode === "confirm-split" && curation.target
        ? splitLines(keep.name, curation.target)
        : [],
  );
  const isChosen = (id: string) => curation.selected.some((s) => s.canonical_id === id);
</script>

<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
<section
  class="cur"
  class:danger={curation.mode !== "pick"}
  aria-labelledby="cur-title"
  onkeydown={(e) => e.key === "Escape" && !curation.busy && closeCuration()}
>
  {#if curation.mode === "pick"}
    <h3 id="cur-title" tabindex="-1" bind:this={headingEl}>Merge another entity into {keep.name}</h3>
    <p class="sub">
      {keep.name} stays, and keeps its name. Pick up to {MAX_MERGE} entities of the same type ({keep.type}) to fold into it.
    </p>

    <div class="field">
      <label for="pick-search">Find entities to merge in</label>
      <input
        id="pick-search"
        type="search"
        maxlength="200"
        autocomplete="off"
        placeholder="Leave empty to list the newest"
        value={curation.q}
        oninput={(e) => setPickerQuery(e.currentTarget.value)}
      />
    </div>

    {#if curation.searchProblem}
      <div class="problem" role="alert">
        <span>{curation.searchProblem}</span>
        <button type="button" class="btn" onclick={() => setPickerQuery(curation.q, true)}>Try again</button>
      </div>
    {:else if curation.loaded && curation.results.length === 0}
      <p class="none">Nothing else matches{curation.q.trim() ? ` "${curation.q.trim()}"` : ""}.</p>
    {:else if !curation.loaded}
      <p class="none" role="status">Searching…</p>
    {:else}
      <ul class="rows" aria-label="Entities that can be merged">
        {#each curation.results as e (e.canonical_id)}
          {@const mismatch = e.type !== keep.type}
          <li class:off={mismatch}>
            <label>
              <input
                type="checkbox"
                checked={isChosen(e.canonical_id)}
                disabled={mismatch || (!isChosen(e.canonical_id) && curation.selected.length >= MAX_MERGE)}
                onchange={() => toggleSelected(e)}
              />
              <span class="who">
                <span class="n">{e.name}</span>
                <span class="mono m">{e.type} · {e.canonical_id}</span>
                {#if mismatch}
                  <span class="m warn">Different type from {keep.name} ({keep.type}), so it cannot be merged.</span>
                {/if}
              </span>
            </label>
          </li>
        {/each}
      </ul>
    {/if}

    <p class="count" role="status">
      {curation.selected.length} of {MAX_MERGE} selected{#if curation.selected.length}: {curation.selected
          .map((e) => e.name)
          .join(", ")}{/if}
    </p>
    {#if problem}<p class="warn" role="alert">{problem}</p>{/if}
    {#if curation.failure}
      {@render failure()}
    {/if}

    <div class="acts">
      <button type="button" class="go" disabled={curation.busy || curation.selected.length === 0 || problem !== null} onclick={reviewMerge}>
        {curation.busy ? "Loading…" : "Review merge"}
      </button>
      <button type="button" class="btn" onclick={() => closeCuration()}>Cancel</button>
    </div>
  {:else if curation.mode === "confirm-merge"}
    <h3 id="cur-title" tabindex="-1" bind:this={headingEl}>
      Merge {curation.selected.map((e) => e.name).join(", ")} into {keep.name}?
    </h3>
    <ul class="lines">
      {#each lines as l (l)}<li>{l}</li>{/each}
    </ul>
    {#if curation.failure}
      {@render failure()}
    {/if}
    <div class="acts">
      <button type="button" class="go bad" disabled={curation.busy} onclick={confirmMerge}>
        {curation.busy ? "Merging…" : "Merge entities"}
      </button>
      <button type="button" class="btn" disabled={curation.busy} onclick={backToPicker}>Back</button>
      <button type="button" class="btn" disabled={curation.busy} onclick={() => closeCuration()}>Cancel</button>
    </div>
  {:else if curation.mode === "confirm-split" && curation.target}
    <h3 id="cur-title" tabindex="-1" bind:this={headingEl}>Split {curation.target.name} out of {keep.name}?</h3>
    <ul class="lines">
      {#each lines as l (l)}<li>{l}</li>{/each}
    </ul>
    {#if curation.failure}
      {@render failure()}
    {/if}
    <div class="acts">
      <button type="button" class="go bad" disabled={curation.busy} onclick={confirmSplit}>
        {curation.busy ? "Splitting…" : "Split entity"}
      </button>
      <button type="button" class="btn" disabled={curation.busy} onclick={() => closeCuration()}>Cancel</button>
    </div>
  {/if}
</section>

{#snippet failure()}
  {#if curation.failure}
    <div class="problem" role="alert">
      <span>{curation.failure.message}</span>
      {#if curation.failure.action}
        {@const a = curation.failure.action}
        <button type="button" class="btn" onclick={() => openEntity(a.id)}>{a.label}</button>
      {/if}
    </div>
  {/if}
{/snippet}

<style>
  .cur {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 18px 20px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-l);
  }
  .cur.danger {
    border: 2px solid var(--bad);
  }
  h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    overflow-wrap: anywhere;
  }
  h3:focus-visible {
    outline-offset: 4px;
  }
  .sub,
  .none,
  .count {
    margin: 0;
    font-size: 13px;
    color: var(--muted);
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-width: 480px;
  }
  .field label {
    font-size: 13px;
    font-weight: 500;
  }
  input[type="search"] {
    padding: 11px 14px;
    font-size: 15px;
    border: 1px solid var(--line);
    border-radius: var(--radius-m);
    background: var(--panel);
    color: var(--ink);
    min-width: 0;
  }
  .rows {
    list-style: none;
    margin: 0;
    padding: 0;
    max-height: 260px;
    overflow-y: auto;
    border: 1px solid var(--line);
    border-radius: var(--radius-m);
  }
  .rows li + li {
    border-top: 1px solid var(--line);
  }
  .rows label {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 10px 12px;
    min-height: 44px;
    cursor: pointer;
  }
  .rows label:hover {
    background: var(--hover);
  }
  .off label {
    cursor: not-allowed;
    opacity: 0.75;
  }
  .rows input {
    width: 18px;
    height: 18px;
    margin: 2px 0 0;
    accent-color: var(--accent);
    flex: none;
  }
  .who {
    display: flex;
    flex-direction: column;
    gap: 1px;
    min-width: 0;
  }
  .n {
    font-weight: 600;
    overflow-wrap: anywhere;
  }
  .m {
    font-size: 11px;
    color: var(--muted);
    overflow-wrap: anywhere;
  }
  .warn {
    margin: 0;
    font-size: 13px;
    color: var(--bad);
  }
  .m.warn {
    font-size: 12px;
  }
  .lines {
    margin: 0;
    padding-left: 18px;
    font-size: 14px;
    line-height: 1.6;
    overflow-wrap: anywhere;
  }
  .acts {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
  }
  .go,
  .btn {
    padding: 9px 20px;
    min-height: 44px;
    border-radius: var(--radius-m);
    font-size: 14px;
    font-weight: 600;
  }
  .go {
    border: 0;
    background: var(--accent);
    color: var(--on-accent);
  }
  .go.bad {
    background: var(--bad);
    color: var(--on-accent);
  }
  .btn {
    border: 1px solid var(--accent);
    background: var(--panel);
    color: var(--accent);
  }
  .go:disabled,
  .btn:disabled {
    opacity: 0.55;
    cursor: default;
  }
  .problem {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    border: 1px solid var(--bad);
    border-radius: var(--radius-m);
    color: var(--bad);
    font-size: 14px;
  }
  @media (max-width: 719px) {
    .acts > button {
      flex: 1 1 100%;
    }
  }
</style>
