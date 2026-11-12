<script lang="ts">
  import { checkHealth } from "../api/client";

  let state = $state<"checking" | "ok" | "down">("checking");

  async function refresh() {
    state = "checking";
    state = (await checkHealth()) ? "ok" : "down";
  }
  refresh();
</script>

<h1>Health</h1>
<p data-testid="health">
  {#if state === "checking"}
    Checking the API...
  {:else if state === "ok"}
    The API is up.
  {:else}
    The API did not answer.
  {/if}
</p>
<button onclick={refresh}>Check again</button>
