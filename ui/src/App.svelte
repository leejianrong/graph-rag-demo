<script lang="ts">
  import "./app.css";
  import { session, signOut } from "./lib/auth.svelte";
  import { route, routes } from "./lib/router.svelte";
  import Explorer from "./views/Explorer.svelte";
  import Ingest from "./views/Ingest.svelte";
  import Keys from "./views/Keys.svelte";
  import Query from "./views/Query.svelte";
  import SignIn from "./views/SignIn.svelte";

  const views = { ingest: Ingest, query: Query, explorer: Explorer, keys: Keys };
  const View = $derived(views[route.current]);
</script>

{#if session.key === null}
  <SignIn />
{:else}
  <div class="shell">
    <aside>
      <div class="brand">
        <span class="name">Graph RAG</span>
        <span class="mono sub">memory store</span>
      </div>
      <nav aria-label="Main">
        {#each routes as r (r.path)}
          <a href={`#/${r.path}`} aria-current={route.current === r.path ? "page" : undefined}>
            {r.label}
          </a>
        {/each}
      </nav>
      <div class="tenant">
        <span class="small">Signed in as tenant</span>
        <span class="mono who" data-testid="tenant">{session.tenant}</span>
        <span class="mono small">
          {#if session.namespace}namespace {session.namespace} · {/if}key ••••{session.keySuffix ?? ""}
        </span>
        <button type="button" class="link" onclick={() => signOut()}>Sign out</button>
      </div>
    </aside>
    <main><View /></main>
  </div>
{/if}

<style>
  .shell {
    display: flex;
    flex-wrap: wrap;
    min-height: 100vh;
  }
  aside {
    flex: 1 1 232px;
    max-width: 100%;
    padding: 24px 16px;
    background: var(--panel);
    border-right: 1px solid var(--line);
    display: flex;
    flex-direction: column;
    gap: 24px;
  }
  main {
    flex: 999 1 560px;
    min-width: 0;
    padding: 32px 40px;
  }
  .brand {
    display: flex;
    flex-direction: column;
    padding: 0 12px;
  }
  .name {
    font-size: 18px;
    font-weight: 600;
  }
  .sub,
  .small {
    font-size: 12px;
    color: var(--muted);
  }
  nav {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  nav a {
    display: block;
    padding: 10px 12px;
    border-radius: var(--radius-s);
    color: var(--ink);
    text-decoration: none;
    font-size: 14px;
  }
  nav a:hover {
    background: var(--hover);
  }
  nav a[aria-current="page"] {
    background: var(--soft);
    color: var(--accent);
    font-weight: 600;
  }
  .tenant {
    margin-top: auto;
    border: 1px solid var(--line);
    border-radius: var(--radius-m);
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
  }
  .who {
    font-size: 14px;
    font-weight: 500;
    overflow-wrap: anywhere;
  }
  .link {
    align-self: flex-start;
    padding: 4px 0;
    border: 0;
    background: none;
    color: var(--accent);
    font-size: 13px;
    text-decoration: underline;
  }
  @media (max-width: 719px) {
    .shell {
      align-content: flex-start;
    }
    aside {
      flex-basis: 100%;
      border-right: 0;
      border-bottom: 1px solid var(--line);
      padding: 16px;
      gap: 12px;
    }
    nav {
      flex-direction: row;
      flex-wrap: wrap;
    }
    .tenant {
      margin-top: 0;
    }
    main {
      padding: 24px 16px;
    }
  }
</style>
