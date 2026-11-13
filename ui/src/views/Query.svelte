<script lang="ts">
  import { api, session } from "../lib/auth.svelte";
  import {
    highlight,
    hopOf,
    runQuery,
    type NodeRef,
    type Path,
    type QueryView,
    type SentenceView,
  } from "../lib/query";

  let question = $state("");
  let synthesize = $state(false);
  let busy = $state(false);
  let error = $state<string | null>(null);
  let asked = $state("");
  let view = $state<QueryView | null>(null);
  let ms = $state(0);
  let usedLlm = $state(false);
  let pathIndex = $state(0);
  let picked = $state<string | null>(null);

  const path = $derived(view?.paths[pathIndex]);
  const sentence = $derived<SentenceView | undefined>(view?.sentences.find((s) => s.key === picked));
  const names = $derived.by(() => {
    if (!sentence) return [];
    const own = sentence.supports.flatMap((s) => [s.subject.name, s.object.name]);
    if (own.length) return own;
    return path ? [path.start.name, ...path.hops.map((h) => h.to.name)] : [];
  });

  async function ask(event: SubmitEvent) {
    event.preventDefault();
    busy = true;
    error = null;
    const wanted = synthesize;
    const result = await runQuery(api, question, wanted);
    busy = false;
    if (!result.ok) {
      error = result.message;
      view = null;
      return;
    }
    asked = question.trim();
    view = result.view;
    ms = result.ms;
    usedLlm = wanted;
    pathIndex = 0;
    picked = null;
  }

  const pathLabel = (p: Path, i: number) =>
    `Path ${i + 1}: ${[p.start.name, ...p.hops.map((h) => `${h.predicate} ${h.to.name}`)].join(", ")}`;
  const label = (n: NodeRef) => n.name;
  const tag = (s: SentenceView) => {
    const hop = hopOf(s, path);
    if (hop) return `on path · hop ${hop}`;
    return s.supports.length ? "supports an edge off this path" : "context";
  };
</script>

{#snippet route(p: Path)}
            <span class="node"><b>{label(p.start)}</b><span class="mono sub">{p.start.type ?? "entity"}</span></span>
            {#each p.hops as h, n (n)}
              <span class="hop" aria-hidden="true">
                <span class="mono pred">{h.predicate}</span>
                <span>{h.forward ? "→" : "←"}</span>
                <span class="sub">hop {n + 1}</span>
              </span>
              <span class="node" class:target={n === p.hops.length - 1}>
                <b>{label(h.to)}</b><span class="mono sub">{h.to.type ?? "entity"}</span>
              </span>
            {/each}
{/snippet}

<div class="page">
  <header>
    <h1>Query playground</h1>
    <span class="mono meta">POST /query</span>
  </header>

  <form onsubmit={ask}>
    <div class="fields">
      <div class="field grow">
        <label for="q">Question</label>
        <input id="q" bind:value={question} autocomplete="off" aria-describedby="ask-error" />
      </div>
      <button type="submit" class="primary" disabled={busy}>{busy ? "Asking..." : "Ask"}</button>
    </div>
    <label class="check">
      <input type="checkbox" bind:checked={synthesize} />
      <span>
        Also write a prose answer with an LLM
        <span class="hint">Off by default. It calls a language model and costs money; the default path is deterministic and costs $0.</span>
      </span>
    </label>
    <p id="ask-error" class="error" role="alert">
      {#if error}<span aria-hidden="true">&#9888;</span> {error}{/if}
    </p>
  </form>

  {#if view === null}
    <div class="chips">
      <span class="chip">Default path: no LLM, deterministic, $0</span>
      {#if session.namespace}<span class="chip">Namespace {session.namespace}</span>{/if}
    </div>
    <p class="empty">Ask a question about what you have ingested. The answer, the path through the graph and the sentences behind it show here.</p>
  {:else}
    <div role="status" class="sr-only">
      {view.answer ? `Answer: ${[view.answer, ...view.moreAnswers.map((m) => m.name)].join(", ")}` : "No answer found"}
    </div>

    <section class="card answer" aria-labelledby="answer-title">
      <div class="answer-main">
        <h2 id="answer-title" class="label">Answer</h2>
        {#if view.answer}
          <span class="big">{view.answer}</span>
          {#if view.moreAnswers.length}
            <span class="also">
              And: {#each view.moreAnswers as more, i (more.id)}{i ? ", " : ""}<strong>{more.name}</strong>{/each}
            </span>
          {/if}
          <span class="muted">
            {#if path}
              Found by following {path.hops.length} {path.hops.length === 1 ? "edge" : "edges"} from {path.start.name},
              supported by {view.sentences.length} {view.sentences.length === 1 ? "sentence" : "sentences"}.
            {:else}
              Ranked first, with no path from another matched entity in the subgraph.
            {/if}
          </span>
        {:else}
          <span class="big none">No answer</span>
        {/if}
      </div>
      <dl class="stats">
        <div><dt>Retriever</dt><dd class="mono">graph-rag</dd></div>
        <div><dt>Latency</dt><dd class="mono">{ms} ms</dd></div>
        <div><dt>LLM</dt><dd class="mono">{usedLlm ? "called (billed)" : "not called, $0"}</dd></div>
      </dl>
    </section>

    {#if view.prose}
      <section class="card" aria-labelledby="prose-title">
        <h2 id="prose-title" class="label">Prose answer from an LLM</h2>
        <p class="prose">{view.prose}</p>
      </section>
    {:else if usedLlm}
      <p class="hint">You asked for a prose answer, but the service returned none. It may have no LLM configured, or nothing to ground one on.</p>
    {/if}

    {#if view.empty}
      <section class="notice" aria-labelledby="empty-title">
        <h2 id="empty-title">
          {view.empty === "no-match" ? `Nothing in the graph matched "${asked}"` : "No path found to the answer"}
        </h2>
        {#if view.empty === "no-match"}
          <p>
            No entity from your question was found, so there is no subgraph to read an answer from. Check the
            spelling and use the names as your documents write them. If you have not ingested anything yet,
            or your documents do not mention it, the graph for this key may simply be empty.
          </p>
        {:else}
          <p>
            An answer was ranked first, but no other matched entity connects to it in the {view.nodeCount} nodes
            and {view.edgeCount} edges returned, so there is no path to draw. The sentences below are the
            closest evidence.
          </p>
        {/if}
        <a href="#/ingest">Add a document</a>
      </section>
    {/if}

    {#if view.empty !== "no-match"}
    <div class="cols">
      <section class="left" aria-labelledby="path-title">
        <h2 id="path-title">Path through the graph</h2>
        {#if view.paths.length > 1}
          <p class="hint">Several routes reach the answer. Pick one to see which sentences back it.</p>
        {/if}
        {#each view.paths as p, i (i)}
          {#if view.paths.length > 1}
            <button type="button" class="path" aria-pressed={i === pathIndex} aria-label={pathLabel(p, i)} onclick={() => (pathIndex = i)}>
              {@render route(p)}
            </button>
          {:else}
            <div class="path" role="group" aria-label={pathLabel(p, i)}>
              {@render route(p)}
            </div>
          {/if}
        {/each}
        {#if view.paths.length === 0 && view.nodeCount > 0}
          <p class="muted">No path to draw.</p>
        {/if}
        {#if view.offPath.length}
          <p class="off">
            Also in the subgraph, off the path:
            {#each view.offPath as n (n.id)}<span class="chip">{n.name}</span>{/each}
          </p>
        {/if}
      </section>

      <section class="right" aria-labelledby="sent-title">
        <h2 id="sent-title">Supporting sentences</h2>
        {#if view.sentences.length === 0}
          <p class="muted">No sentences came back with this result.</p>
        {/if}
        <ul class="sentences">
          {#each view.sentences as s (s.key)}
            <li>
              <button
                type="button"
                class="sentence"
                aria-pressed={picked === s.key}
                onclick={() => (picked = picked === s.key ? null : s.key)}
              >
                <span class="text">{s.text}</span>
                <span class="meta-row">
                  <span class="chip mono">{tag(s)}</span>
                  <span class="mono sub">{s.documentId}</span>
                </span>
              </button>
            </li>
          {/each}
        </ul>

        {#if sentence}
          <div class="card source" aria-live="polite">
            <h3 class="label">Source of the selected sentence</h3>
            <span class="mono doc">{sentence.documentId}</span>
            <dl class="kv">
              <dt>Sentence</dt>
              <dd class="mono">index {sentence.sentenceIndex}</dd>
              <dt>Characters</dt>
              <dd class="mono">{sentence.charStart} to {sentence.charEnd} of the document</dd>
              <dt>Match score</dt>
              <dd class="mono">{sentence.score.toFixed(3)}</dd>
              <dt>Supports</dt>
              <dd class="mono">
                {#if sentence.supports.length}
                  {#each sentence.supports as sup, i (i)}<div>{sup.subject.name} {sup.predicate} {sup.object.name}</div>{/each}
                {:else}no edge cites it (context only){/if}
              </dd>
            </dl>
            <p class="quote">
              {#each highlight(sentence.text, names) as seg, i (i)}{#if seg.hit}<mark>{seg.text}</mark>{:else}{seg.text}{/if}{/each}
            </p>
            <p class="hint">Marked: the entities of the edges this sentence backs. The document's name and full text are not part of the result.</p>
          </div>
        {:else if view.sentences.length}
          <p class="hint">Select a sentence to see where it came from.</p>
        {/if}
      </section>
    </div>
    {/if}
  {/if}
</div>

<style>
  .page {
    display: flex;
    flex-direction: column;
    gap: 24px;
    max-width: 1100px;
  }
  header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
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
  .meta,
  .sub,
  .hint,
  .muted {
    font-size: 12px;
    color: var(--muted);
  }
  .hint {
    display: block;
    margin: 0;
    font-size: 13px;
  }
  .muted {
    font-size: 14px;
  }
  form {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .fields {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: flex-end;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
  }
  .grow {
    flex: 1 1 360px;
  }
  label {
    font-size: 13px;
    font-weight: 500;
  }
  #q {
    font-size: 16px;
    padding: 12px 14px;
    border: 1px solid var(--line);
    border-radius: var(--radius-m);
    background: var(--panel);
    color: var(--ink);
    min-width: 0;
  }
  .check {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    font-weight: 400;
    font-size: 14px;
  }
  .check input {
    width: 18px;
    height: 18px;
    margin: 2px 0 0;
    accent-color: var(--accent);
    flex: none;
  }
  .primary {
    min-height: 44px;
    padding: 12px 24px;
    border: 0;
    border-radius: var(--radius-m);
    background: var(--accent);
    color: var(--on-accent);
    font-weight: 600;
    font-size: 15px;
  }
  .primary:disabled {
    opacity: 0.6;
    cursor: progress;
  }
  .error {
    margin: 0;
    color: var(--bad);
    font-size: 14px;
  }
  .chips,
  .off {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin: 0;
  }
  .off {
    font-size: 13px;
    color: var(--muted);
  }
  .chip {
    display: inline-block;
    padding: 2px 8px;
    border: 1px solid var(--line);
    border-radius: 999px;
    font-size: 12px;
    background: var(--panel);
    color: var(--ink);
    overflow-wrap: anywhere;
  }
  .empty {
    margin: 0;
    padding: 24px;
    border: 1px dashed var(--line);
    border-radius: var(--radius-l);
    color: var(--muted);
    font-size: 14px;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-l);
    padding: 20px 24px;
  }
  .answer {
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    align-items: center;
    justify-content: space-between;
  }
  .answer-main {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
  }
  .label {
    margin: 0;
    font-size: 12px;
    font-weight: 400;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .big {
    font-size: 44px;
    font-weight: 600;
    line-height: 1.1;
    color: var(--accent);
    overflow-wrap: anywhere;
  }
  .also {
    font-size: 20px;
    font-weight: 600;
    overflow-wrap: anywhere;
  }
  .also strong {
    color: var(--accent);
  }
  .big.none {
    color: var(--muted);
  }
  .stats {
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 8px 28px;
  }
  .stats dt {
    font-size: 12px;
    color: var(--muted);
  }
  .stats dd {
    margin: 0;
    font-size: 15px;
  }
  .prose {
    margin: 8px 0 0;
    line-height: 1.6;
  }
  .notice {
    border: 1px dashed var(--muted);
    border-radius: var(--radius-l);
    padding: 16px 20px;
    background: var(--panel);
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .notice h2 {
    font-size: 16px;
  }
  .notice p {
    margin: 0;
    font-size: 14px;
    color: var(--muted);
  }
  .notice a {
    font-size: 14px;
    align-self: flex-start;
  }
  .cols {
    display: flex;
    flex-wrap: wrap;
    gap: 24px;
    align-items: flex-start;
  }
  .left,
  .right {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .left {
    flex: 1 1 480px;
  }
  .right {
    flex: 1 1 380px;
  }
  .path {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
    padding: 20px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-l);
    color: var(--ink);
    text-align: left;
    width: 100%;
  }
  button.path[aria-pressed="true"] {
    border-color: var(--accent);
    box-shadow: inset 0 0 0 1px var(--accent);
  }
  .node {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 10px 12px;
    border: 1px solid var(--line);
    border-radius: var(--radius-m);
    min-width: 0;
  }
  .node b {
    overflow-wrap: anywhere;
  }
  .node.target {
    border: 2px solid var(--accent);
    background: var(--soft);
  }
  .hop {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    color: var(--accent);
  }
  .pred {
    font-size: 12px;
  }
  .sentences {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .sentence {
    width: 100%;
    text-align: left;
    border: 1px solid var(--line);
    background: var(--panel);
    border-radius: 10px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    color: var(--ink);
  }
  .sentence[aria-pressed="true"] {
    background: var(--soft);
    border-color: var(--accent);
  }
  .text {
    font-size: 15px;
    line-height: 1.45;
  }
  .meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
  }
  .meta-row .sub {
    overflow-wrap: anywhere;
  }
  .source {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 16px 18px;
  }
  .source h3 {
    margin: 0;
  }
  .doc {
    font-weight: 500;
    overflow-wrap: anywhere;
  }
  .kv {
    margin: 0;
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 16px;
    font-size: 13px;
  }
  .kv dt {
    color: var(--muted);
  }
  .kv dd {
    margin: 0;
    overflow-wrap: anywhere;
  }
  .quote {
    margin: 0;
    font-size: 14px;
    line-height: 1.6;
    border-left: 3px solid var(--line);
    padding-left: 12px;
  }
  mark {
    background: var(--mark);
    color: var(--ink);
    padding: 0 2px;
  }
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip-path: inset(50%);
    white-space: nowrap;
  }
  @media (max-width: 719px) {
    .card {
      padding: 16px;
    }
    .big {
      font-size: 34px;
    }
    .primary {
      width: 100%;
    }
    .path {
      padding: 14px;
    }
  }
</style>
