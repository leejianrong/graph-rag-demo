<script lang="ts">
  import { onMount } from "svelte";
  import {
    FILE_ACCEPT,
    SIZE_HINT_BYTES,
    defaultTextName,
    textSource,
  } from "../lib/ingest";
  import { ingest, jobs, pauseStreams, recheck, resumeStreams, retry } from "../lib/jobs.svelte";
  import { segments, statusText, type Source, type TrackedJob } from "../lib/jobs";

  type Mode = "text" | "url" | "file";
  const modes: { id: Mode; label: string }[] = [
    { id: "text", label: "Text" },
    { id: "url", label: "URL" },
    { id: "file", label: "File (txt, html, pdf)" },
  ];

  let mode = $state<Mode>("url");
  let text = $state("");
  let textName = $state("");
  let url = $state("");
  let file = $state<File | null>(null);
  let busy = $state(false);
  let error = $state<string | null>(null);
  let announcement = $state("");

  onMount(() => {
    resumeStreams();
    return pauseStreams;
  });

  // Say a job's outcome once, when it finishes. Stage ticks are not announced.
  const told = new Map<number, string>();
  $effect(() => {
    for (const job of jobs.list) {
      const final = job.state === "done" || job.state === "failed" || job.state === "unchanged";
      if (final && told.get(job.id) !== job.state) {
        told.set(job.id, job.state);
        announcement = `${job.name}: ${statusText(job).toLowerCase()}`;
      }
    }
  });

  const tooBig = $derived(file !== null && file.size > SIZE_HINT_BYTES);

  function onTabKey(event: KeyboardEvent) {
    const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (!step) return;
    event.preventDefault();
    const next = modes[(modes.findIndex((m) => m.id === mode) + step + modes.length) % modes.length];
    mode = next.id;
    error = null;
    document.getElementById(`tab-${next.id}`)?.focus();
  }

  function pick(event: Event) {
    file = (event.currentTarget as HTMLInputElement).files?.[0] ?? null;
  }

  function sourceFromForm(): { name: string; source: Source } | string {
    if (mode === "url") {
      const value = url.trim();
      if (!value) return "Enter the address of a page to fetch.";
      return { name: value.replace(/^https?:\/\//, ""), source: { kind: "url", url: value } };
    }
    if (mode === "file") {
      if (!file) return "Choose a file to upload.";
      return { name: file.name, source: { kind: "file", file, name: file.name } };
    }
    if (!text.trim()) return "Paste some text to ingest.";
    const source = textSource(text, textName || defaultTextName());
    return { name: source.kind === "file" ? source.name : "", source };
  }

  async function submit(event: SubmitEvent) {
    event.preventDefault();
    const built = sourceFromForm();
    if (typeof built === "string") {
      error = built;
      return;
    }
    busy = true;
    error = null;
    const result = await ingest(built.name, built.source);
    busy = false;
    if (result.ok) {
      if (mode === "text") text = "";
      if (mode === "url") url = "";
    } else {
      error = result.message;
    }
  }

  async function again(job: TrackedJob) {
    const result = await retry(job);
    if (!result.ok) error = result.message;
  }
</script>

<div class="page">
  <header>
    <h1>Ingest console</h1>
    <span class="mono meta">POST /ingest · GET /jobs/&#123;id&#125;/events</span>
  </header>

  <section class="card">
    <div role="tablist" aria-label="What to ingest">
      {#each modes as m (m.id)}
        <button
          type="button"
          role="tab"
          id={`tab-${m.id}`}
          aria-selected={mode === m.id}
          aria-controls="source-panel"
          tabindex={mode === m.id ? 0 : -1}
          onclick={() => {
            mode = m.id;
            error = null;
          }}
          onkeydown={onTabKey}>{m.label}</button
        >
      {/each}
    </div>

    <div id="source-panel" role="tabpanel" aria-labelledby={`tab-${mode}`}>
    <form onsubmit={submit}>
      <div class="fields">
        {#if mode === "url"}
          <div class="field grow">
            <label for="url">Page to fetch</label>
            <input
              id="url"
              class="mono"
              type="url"
              inputmode="url"
              placeholder="https://example.com/page"
              autocomplete="off"
              spellcheck="false"
              bind:value={url}
              aria-describedby="hint form-error"
            />
          </div>
        {:else if mode === "file"}
          <div class="field grow">
            <label for="file">File to upload</label>
            <input
              id="file"
              type="file"
              accept={FILE_ACCEPT}
              onchange={pick}
              aria-describedby="hint form-error"
            />
            {#if tooBig}
              <span class="warn">
                That file is over 10 MB. The service may refuse it; its limit is set by the operator.
              </span>
            {/if}
          </div>
        {:else}
          <div class="field grow">
            <label for="text">Text to ingest</label>
            <textarea id="text" rows="6" bind:value={text} aria-describedby="hint form-error"></textarea>
          </div>
          <div class="field name">
            <label for="name">Document name</label>
            <input
              id="name"
              class="mono"
              autocomplete="off"
              spellcheck="false"
              placeholder="pasted-yyyymmdd-hhmmss.txt"
              bind:value={textName}
            />
          </div>
        {/if}
        <button type="submit" class="primary" disabled={busy}>{busy ? "Sending..." : "Ingest"}</button>
      </div>

      <p id="form-error" class="error" role="alert">
        {#if error}<span aria-hidden="true">&#9888;</span> {error}{/if}
      </p>

      <p id="hint" class="hint">
        {#if mode === "url"}
          The address becomes the document's identity. Ingesting it again after the page changes
          updates the graph instead of duplicating it. Private and local addresses are refused.
        {:else if mode === "file"}
          The file's name becomes the document's identity. A PDF or an HTML page is stored as its
          text; one with no text in it is refused and nothing is queued.
        {:else}
          The service takes files, not bare text, so your text is sent as a plain-text file with the
          name above. The name is the document's identity: reuse it to update that document.
        {/if}
      </p>
    </form>
    </div>
  </section>

  <section class="jobs" aria-labelledby="jobs-title">
    <div class="jobs-head">
      <h2 id="jobs-title">Jobs</h2>
      <span class="live">Live: updates arrive as each stage finishes</span>
    </div>
    <p class="hint">Only documents you ingest in this tab are listed. Reloading or signing out clears it.</p>
    <p class="sr-only" role="status" aria-live="polite">{announcement}</p>

    {#if jobs.list.length === 0}
      <p class="empty">No jobs yet. Ingest something above and its progress shows here.</p>
    {/if}

    {#each jobs.list as job (job.id)}
      <article class="job" class:failed={job.state === "failed"} class:running={job.state === "running" || job.state === "queued" || job.state === "submitting"}>
        <div class="row">
          <span class="mono name-text">{job.name}</span>
          <span class="chip {job.state}">{statusText(job)}</span>
        </div>
        {#if job.state !== "unchanged"}
          <ol class="segs" aria-label="Pipeline stages">
            {#each segments(job) as seg (seg.stage)}
              <li class={seg.state}>
                <i></i>
                <span>{seg.stage}<span class="sr-only"> ({seg.state})</span></span>
              </li>
            {/each}
          </ol>
        {/if}
        {#if job.state === "failed"}
          <p class="reason">
            <strong>Reason:</strong>
            {job.error}
            <button type="button" class="link" onclick={() => again(job)}>Ingest it again</button>
          </p>
        {:else if job.state === "unchanged"}
          <p class="reason">This source already holds exactly this content, so nothing was queued.</p>
        {:else if job.link === "polling"}
          <p class="note">The live link dropped. Checking the job every few seconds instead.</p>
        {:else if job.link === "idle" && job.state !== "submitting" && job.state !== "done"}
          <p class="note">
            Lost contact with the service.
            <button type="button" class="link" onclick={() => recheck(job)}>Check again</button>
          </p>
        {/if}
      </article>
    {/each}
  </section>
</div>

<style>
  .page {
    display: flex;
    flex-direction: column;
    gap: 28px;
    max-width: 960px;
  }
  header,
  .jobs-head,
  .row {
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
  .live,
  .hint {
    font-size: 12px;
    color: var(--muted);
  }
  .hint {
    margin: 0;
    font-size: 13px;
  }
  .live::before {
    content: "";
    display: inline-block;
    width: 8px;
    height: 8px;
    margin-right: 6px;
    border-radius: 50%;
    background: var(--ok);
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-l);
    padding: 20px 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  [role="tablist"] {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    border-bottom: 1px solid var(--line);
  }
  [role="tab"] {
    padding: 10px 14px;
    border: 0;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    background: none;
    color: var(--muted);
    font-size: 14px;
  }
  [role="tab"][aria-selected="true"] {
    color: var(--accent);
    font-weight: 600;
    border-bottom-color: var(--accent);
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
    flex: 1 1 380px;
  }
  .name {
    flex: 1 1 240px;
  }
  label {
    font-size: 13px;
    font-weight: 500;
  }
  input:not([type="file"]),
  textarea {
    font: inherit;
    font-size: 15px;
    padding: 12px 14px;
    border: 1px solid var(--line);
    border-radius: var(--radius-m);
    background: var(--panel);
    color: var(--ink);
    min-width: 0;
  }
  input.mono {
    font-family: var(--font-mono);
    font-size: 14px;
  }
  input[type="file"] {
    font-size: 14px;
    padding: 8px 0;
    max-width: 100%;
  }
  textarea {
    resize: vertical;
    font-family: var(--font-sans);
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
    min-height: 0;
    color: var(--bad);
    font-size: 14px;
  }
  .warn {
    font-size: 13px;
    color: var(--bad);
  }
  .empty {
    margin: 0;
    padding: 24px;
    border: 1px dashed var(--line);
    border-radius: var(--radius-l);
    color: var(--muted);
    font-size: 14px;
  }
  .jobs {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .job {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-l);
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .job.running {
    border-color: var(--accent);
  }
  .job.failed {
    border-color: var(--bad);
  }
  .name-text {
    font-size: 14px;
    font-weight: 500;
    overflow-wrap: anywhere;
    min-width: 0;
  }
  .chip {
    display: inline-block;
    padding: 2px 8px;
    border: 1px solid var(--line);
    border-radius: 999px;
    font-size: 12px;
    background: var(--panel);
  }
  .chip.running,
  .chip.queued,
  .chip.submitting {
    color: var(--accent);
    border-color: var(--accent);
  }
  .chip.done {
    color: var(--ok);
    border-color: var(--ok);
  }
  .chip.failed {
    color: var(--bad);
    border-color: var(--bad);
  }
  .segs {
    display: flex;
    gap: 6px;
    margin: 0;
    padding: 0;
    list-style: none;
  }
  .segs li {
    flex: 1 1 0;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 11px;
    color: var(--muted);
  }
  .segs i {
    display: block;
    height: 6px;
    border-radius: 3px;
    background: var(--line);
  }
  .segs .done i {
    background: var(--accent);
  }
  .segs .running {
    color: var(--accent);
    font-weight: 600;
  }
  .segs .running i {
    background: var(--soft);
    box-shadow: inset 0 0 0 1px var(--accent);
    animation: pulse 1.4s ease-in-out infinite;
  }
  .segs .failed {
    color: var(--bad);
    font-weight: 600;
  }
  .segs .failed i {
    background: var(--bad);
  }
  @keyframes pulse {
    50% {
      opacity: 0.45;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .segs .running i {
      animation: none;
    }
  }
  .reason,
  .note {
    margin: 0;
    font-size: 14px;
    line-height: 1.5;
  }
  .note {
    color: var(--muted);
  }
  .link {
    padding: 0;
    border: 0;
    background: none;
    color: var(--accent);
    text-decoration: underline;
    font-size: inherit;
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
    .segs {
      gap: 4px;
    }
    .segs li span:not(.sr-only) {
      font-size: 10px;
      overflow-wrap: anywhere;
    }
    .primary {
      width: 100%;
    }
  }
</style>
