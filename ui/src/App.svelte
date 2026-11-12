<script lang="ts">
  import { route, routes } from "./lib/router.svelte";
  import Explorer from "./views/Explorer.svelte";
  import Health from "./views/Health.svelte";
  import Ingest from "./views/Ingest.svelte";
  import Keys from "./views/Keys.svelte";
  import Query from "./views/Query.svelte";
  import SignIn from "./views/SignIn.svelte";

  const views = {
    health: Health,
    "sign-in": SignIn,
    ingest: Ingest,
    query: Query,
    explorer: Explorer,
    keys: Keys,
  };
  const View = $derived(views[route.current]);
</script>

<header>
  <strong>Graph RAG</strong>
  <nav>
    {#each routes as r (r.path)}
      <a href={`#/${r.path}`} aria-current={route.current === r.path ? "page" : undefined}>
        {r.label}
      </a>
    {/each}
  </nav>
</header>
<main><View /></main>

<style>
  :global(body) {
    margin: 0;
    font-family: system-ui, sans-serif;
    color: #1c1c1c;
  }
  header {
    display: flex;
    gap: 1.5rem;
    align-items: baseline;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #ddd;
  }
  nav {
    display: flex;
    gap: 1rem;
  }
  a {
    color: #1a56a0;
    text-decoration: none;
  }
  a[aria-current="page"] {
    font-weight: 600;
    text-decoration: underline;
  }
  main {
    padding: 1rem;
  }
</style>
