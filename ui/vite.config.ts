import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// The dev server forwards API paths to the service, so the API needs no CORS change.
const api = process.env.API_URL ?? "http://localhost:8000";
const paths = ["/health", "/ingest", "/jobs", "/documents", "/query", "/v1"];

export default defineConfig({
  plugins: [svelte()],
  server: { proxy: Object.fromEntries(paths.map((p) => [p, api])) },
  test: { environment: "node", exclude: ["e2e/**", "node_modules/**", "dist/**"] },
});
