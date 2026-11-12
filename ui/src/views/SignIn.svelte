<script lang="ts">
  import { session, signIn } from "../lib/auth.svelte";

  let key = $state("");
  let error = $state<string | null>(null);
  let busy = $state(false);
  const message = $derived(error ?? session.notice);

  async function submit(event: SubmitEvent) {
    event.preventDefault();
    busy = true;
    const result = await signIn(key);
    busy = false;
    if (result.ok) {
      key = "";
    } else {
      error = result.message;
    }
  }
</script>

<div class="page">
  <section class="intro">
    <div class="brand">
      <span class="name">Graph RAG</span>
      <span class="mono small">memory store for agents and developers</span>
    </div>
    <div class="pitch">
      <h1>Your documents, as a graph you can question.</h1>
      <p>
        Sign in with the API key your operator gave you. The key names your tenant, and everything
        you see is scoped to it.
      </p>
    </div>
    <span class="mono small">The key stays in this tab's memory. Closing the tab signs you out.</span>
  </section>

  <section class="form">
    <div class="box">
      <h2>Sign in</h2>
      <form onsubmit={submit}>
        <label for="key">API key</label>
        <input
          id="key"
          type="password"
          autocomplete="off"
          spellcheck="false"
          bind:value={key}
          aria-invalid={error ? "true" : undefined}
          aria-describedby="key-message"
        />
        <p id="key-message" role="alert" class="message" class:shown={message}>
          {#if message}
            <span aria-hidden="true">&#9888;</span>
            <span>{message}</span>
          {/if}
        </p>
        <button type="submit" disabled={busy}>{busy ? "Checking..." : "Sign in"}</button>
      </form>
    </div>
  </section>
</div>

<style>
  .page {
    display: flex;
    flex-wrap: wrap;
    min-height: 100vh;
  }
  section {
    flex: 1 1 420px;
    min-width: 0;
    padding: 56px;
  }
  .intro {
    background: var(--accent);
    color: var(--on-accent);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 40px;
  }
  .brand {
    display: flex;
    flex-direction: column;
  }
  .name {
    font-size: 20px;
    font-weight: 600;
  }
  .small {
    font-size: 12px;
    opacity: 0.85;
  }
  .pitch {
    max-width: 460px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  h1 {
    margin: 0;
    font-size: 36px;
    line-height: 1.15;
    font-weight: 600;
  }
  .pitch p {
    margin: 0;
    font-size: 16px;
    line-height: 1.6;
    opacity: 0.92;
  }
  .form {
    flex-basis: 460px;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .box {
    max-width: 440px;
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  h2 {
    margin: 0;
    font-size: 24px;
    font-weight: 600;
  }
  form {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  label {
    font-size: 13px;
    font-weight: 500;
  }
  input {
    padding: 12px 14px;
    border: 1px solid var(--line);
    border-radius: var(--radius-m);
    background: var(--panel);
    color: var(--ink);
    font-size: 16px;
    min-width: 0;
  }
  input[aria-invalid="true"] {
    border: 2px solid var(--bad);
    padding: 11px 13px;
  }
  .message {
    margin: 0;
    font-size: 14px;
    color: var(--bad);
    display: none;
    gap: 8px;
    align-items: baseline;
  }
  .message.shown {
    display: flex;
  }
  button {
    margin-top: 4px;
    min-height: 44px;
    padding: 12px 24px;
    border: 0;
    border-radius: var(--radius-m);
    background: var(--accent);
    color: var(--on-accent);
    font-weight: 600;
    font-size: 15px;
  }
  button:disabled {
    opacity: 0.6;
    cursor: progress;
  }
  @media (max-width: 719px) {
    section {
      padding: 32px 16px;
    }
    .intro {
      gap: 24px;
    }
    h1 {
      font-size: 28px;
    }
  }
</style>
