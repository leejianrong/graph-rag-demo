<script lang="ts">
  import { onDestroy, onMount, tick } from "svelte";
  import { api, session } from "../lib/auth.svelte";
  import {
    MAX_LABEL,
    copySecret,
    createBlockedReason,
    createKey,
    displayLabel,
    formatCreated,
    listKeys,
    maskedSuffix,
    nextReveal,
    revokeBlockedReason,
    revokeKey,
    type KeyInfo,
    type Reveal,
  } from "../lib/keys";

  let keys = $state<KeyInfo[]>([]);
  let loaded = $state(false);
  let loadError = $state<string | null>(null);
  let label = $state("");
  let creating = $state(false);
  let createError = $state<string | null>(null);
  // The only place the secret lives: this variable, while the panel is shown.
  let reveal = $state<Reveal>(null);
  let secretEl = $state<HTMLElement>();
  let panelEl = $state<HTMLElement>();
  let confirming = $state<string | null>(null);
  let revoking = $state(false);
  let rowError = $state<string | null>(null);
  let announcement = $state("");

  const blocked = $derived(createBlockedReason(keys, creating));
  const target = $derived(keys.find((k) => k.key_id === confirming) ?? null);

  async function refresh() {
    const result = await listKeys(api);
    if (result.ok) {
      keys = result.value;
      loadError = null;
    } else if (!loaded) {
      loadError = result.message;
    } else {
      rowError = result.message;
    }
    loaded = true;
  }

  onMount(refresh);
  onDestroy(() => {
    reveal = nextReveal(reveal, { type: "left" });
  });
  $effect(() => {
    if (session.key === null) reveal = nextReveal(reveal, { type: "signed-out" });
  });

  async function create(event: SubmitEvent) {
    event.preventDefault();
    if (blocked) return;
    creating = true;
    createError = null;
    rowError = null;
    announcement = "";
    reveal = nextReveal(reveal, { type: "dismissed" });
    const result = await createKey(api, label);
    creating = false;
    if (!result.ok) {
      createError = result.message;
      announcement = result.message;
      return;
    }
    reveal = nextReveal(reveal, { type: "created", key: result.value });
    label = "";
    announcement = `Key ${displayLabel(result.value)} created. Copy it now: it is shown only once.`;
    await tick();
    panelEl?.focus();
    await refresh();
  }

  function dismiss() {
    reveal = nextReveal(reveal, { type: "dismissed" });
    announcement = "Key hidden.";
  }

  async function copy() {
    if (!reveal) return;
    const outcome = await copySecret(reveal.secret);
    if (outcome === "copied") {
      announcement = "Key copied to the clipboard.";
      return;
    }
    if (secretEl) {
      const range = document.createRange();
      range.selectNodeContents(secretEl);
      const selection = getSelection();
      selection?.removeAllRanges();
      selection?.addRange(range);
    }
    announcement = "Could not copy automatically. The key is selected: press Ctrl+C or Cmd+C to copy it.";
  }

  async function ask(key: KeyInfo) {
    confirming = key.key_id;
    rowError = null;
    await tick();
    document.getElementById("revoke-cancel")?.focus();
  }

  async function cancel() {
    const id = confirming;
    confirming = null;
    await tick();
    document.getElementById(`revoke-${id}`)?.focus();
  }

  async function confirmRevoke() {
    if (!target) return;
    const name = `${displayLabel(target)} ${maskedSuffix(target)}`;
    revoking = true;
    const result = await revokeKey(api, target.key_id);
    revoking = false;
    confirming = null;
    if (result.ok) {
      announcement = `Key ${name} revoked. Requests with it now get 401.`;
    } else {
      rowError = result.message;
      announcement = result.message;
    }
    await refresh();
  }
</script>

<div class="page">
  <header>
    <h1>API keys</h1>
    <span class="mono meta">tenant {session.tenant}</span>
  </header>

  <div class="sr-only" role="status" aria-live="polite">{announcement}</div>

  {#if reveal}
    <section class="reveal" aria-labelledby="reveal-title" tabindex="-1" bind:this={panelEl}>
      <h2 id="reveal-title">
        {reveal.label ? `Key "${reveal.label}" created.` : "Key created."} Copy it now.
      </h2>
      <p class="note">
        This is the only time the full key is shown. After you leave this screen only its last four
        characters remain.
      </p>
      <div class="secret-row">
        <code class="mono secret" bind:this={secretEl} data-testid="secret">{reveal.secret}</code>
        <button type="button" class="primary" onclick={copy}>Copy key</button>
        <button type="button" class="ghost" onclick={dismiss}>I have copied it</button>
      </div>
    </section>
  {/if}

  <form onsubmit={create}>
    <div class="field">
      <label for="label">Label for a new key</label>
      <input
        id="label"
        bind:value={label}
        maxlength={MAX_LABEL}
        autocomplete="off"
        placeholder="staging-worker"
        aria-describedby="create-hint"
      />
    </div>
    <button type="submit" class="outline" disabled={blocked !== null}>Create key</button>
    <p id="create-hint" class="hint" class:warn={blocked !== null && !creating}>
      {blocked ?? "Optional. Up to 64 characters."}
    </p>
  </form>
  {#if createError}<p class="error" role="alert">{createError}</p>{/if}

  {#if rowError}<p class="error" role="alert">{rowError}</p>{/if}

  {#if target}
    <div class="confirm" role="alertdialog" aria-labelledby="confirm-title" aria-describedby="confirm-body">
      <p id="confirm-title" class="strong">
        Revoke {displayLabel(target)} <span class="mono">{maskedSuffix(target)}</span>?
      </p>
      <p id="confirm-body" class="note">
        Anything using this key gets 401 from its next request. This cannot be undone. The key stays
        listed as revoked.
      </p>
      <div class="actions">
        <button type="button" class="danger solid" onclick={confirmRevoke} disabled={revoking}
          >Revoke key</button
        >
        <button type="button" class="ghost" id="revoke-cancel" onclick={cancel}>Cancel</button>
      </div>
    </div>
  {/if}

  {#if loadError}
    <p class="empty">
      {loadError}
      <button type="button" class="link" onclick={refresh}>Try again</button>
    </p>
  {:else if !loaded}
    <p class="empty">Loading keys…</p>
  {:else}
    <section class="table-wrap" aria-label="Your keys">
      <table>
        <thead>
          <tr>
            <th scope="col">Label</th>
            <th scope="col">Key</th>
            <th scope="col">Created (UTC)</th>
            <th scope="col">Status</th>
            <th scope="col"><span class="sr-only">Actions</span></th>
          </tr>
        </thead>
        <tbody>
          {#each keys as key (key.key_id)}
            {@const reason = revokeBlockedReason(key)}
            <tr class:revoked={key.status === "revoked"}>
              <th scope="row" class="label-cell">
                <span class="lbl">{displayLabel(key)}</span>
                {#if key.current}<span class="chip">This session</span>{/if}
              </th>
              <td class="mono">{maskedSuffix(key)}</td>
              <td class="mono">{formatCreated(key.created_at)}</td>
              <td>
                <span class="chip {key.status}">{key.status === "active" ? "Active" : "Revoked"}</span>
              </td>
              <td>
                {#if key.status === "revoked"}
                  <span class="small">Requests with it now get 401</span>
                {:else if reason}
                  <span class="small">{reason}</span>
                {:else}
                  <button
                    type="button"
                    class="danger"
                    id={`revoke-${key.key_id}`}
                    aria-label={`Revoke ${displayLabel(key)} ${maskedSuffix(key)}`}
                    onclick={() => ask(key)}
                    disabled={revoking}>Revoke</button
                  >
                {/if}
              </td>
            </tr>
          {:else}
            <tr><td colspan="5" class="small">No keys yet. Create one above.</td></tr>
          {/each}
        </tbody>
      </table>
    </section>
  {/if}
</div>

<style>
  .page {
    display: flex;
    flex-direction: column;
    gap: 24px;
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
  .small {
    font-size: 12px;
    color: var(--muted);
  }
  .note,
  .hint {
    margin: 0;
    font-size: 14px;
    color: var(--muted);
  }
  .hint {
    flex-basis: 100%;
    font-size: 13px;
  }
  .hint.warn,
  .error {
    color: var(--bad);
  }
  .error {
    margin: 0;
    font-size: 14px;
  }
  .strong {
    margin: 0;
    font-weight: 600;
  }
  .reveal {
    background: var(--panel);
    border: 2px solid var(--accent);
    border-radius: var(--radius-l);
    padding: 20px 24px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .secret-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    align-items: center;
  }
  .secret {
    flex: 1 1 320px;
    min-width: 0;
    padding: 12px 14px;
    border-radius: var(--radius-m);
    background: var(--soft);
    font-size: 14px;
    overflow-wrap: anywhere;
    user-select: all;
  }
  form {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 12px;
    align-items: flex-end;
  }
  .field {
    flex: 1 1 280px;
    max-width: 420px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
  }
  label {
    font-size: 13px;
    font-weight: 500;
  }
  input {
    font-size: 15px;
    padding: 11px 14px;
    border: 1px solid var(--line);
    border-radius: var(--radius-m);
    background: var(--panel);
    color: var(--ink);
    min-width: 0;
  }
  button {
    min-height: 44px;
    padding: 9px 16px;
    border-radius: var(--radius-m);
    border: 1px solid var(--accent);
    background: var(--panel);
    color: var(--accent);
    font-weight: 600;
    font-size: 14px;
  }
  button:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }
  .primary {
    border: 0;
    background: var(--accent);
    color: var(--on-accent);
    padding: 9px 20px;
  }
  .ghost {
    border-color: var(--line);
    color: var(--ink);
  }
  .danger {
    color: var(--bad);
    border-color: var(--bad);
  }
  .danger.solid {
    background: var(--bad);
    color: var(--ground);
  }
  .link {
    min-height: 0;
    padding: 0;
    border: 0;
    background: none;
    text-decoration: underline;
    font-size: inherit;
  }
  .empty {
    margin: 0;
    padding: 24px;
    border: 1px dashed var(--line);
    border-radius: var(--radius-l);
    color: var(--muted);
    font-size: 14px;
  }
  .table-wrap {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius-l);
    padding: 8px 8px 4px;
    overflow-x: auto;
    position: relative; /* keeps the sr-only header inside the scroll box */
  }
  table {
    width: 100%;
    min-width: 620px;
    border-collapse: collapse;
  }
  thead th {
    text-align: left;
    font-size: 12px;
    font-weight: 500;
    color: var(--muted);
    padding: 8px 12px;
    border-bottom: 1px solid var(--line);
  }
  tbody th,
  td {
    text-align: left;
    padding: 12px;
    border-bottom: 1px solid var(--line);
    font-size: 14px;
    vertical-align: middle;
  }
  .label-cell {
    font-weight: 600;
  }
  .label-cell .lbl {
    margin-right: 8px;
    overflow-wrap: anywhere;
  }
  tr.revoked {
    color: var(--muted);
  }
  tr.revoked .lbl {
    text-decoration: line-through;
  }
  .confirm {
    background: var(--soft);
    border: 1px solid var(--bad);
    border-radius: var(--radius-l);
    padding: 16px 20px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
  }
  .chip {
    display: inline-block;
    padding: 2px 8px;
    border: 1px solid var(--line);
    border-radius: 999px;
    font-size: 12px;
    font-weight: 400;
    background: var(--panel);
    color: var(--ink);
  }
  .chip.active {
    color: var(--ok);
    border-color: var(--ok);
  }
  .chip.revoked {
    color: var(--bad);
    border-color: var(--bad);
  }
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip-path: inset(50%);
    white-space: nowrap;
  }
  .reveal:focus-visible {
    outline-offset: 3px;
  }
  @media (max-width: 719px) {
    .reveal {
      padding: 16px;
    }
    .actions button,
    .secret-row button,
    form > button {
      width: 100%;
    }
  }
</style>
