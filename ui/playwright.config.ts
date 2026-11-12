import { defineConfig } from "@playwright/test";

// One browser test over the real service: `graph-rag-offline` serving the built UI (e2e/global-setup.ts).
// Run `npm run build` first. Chromium only, a fixed viewport, and nothing leaves 127.0.0.1.
export default defineConfig({
  testDir: "./e2e",
  globalSetup: "./e2e/global-setup.ts",
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  use: {
    trace: "on-first-retry",
    viewport: { width: 1280, height: 800 },
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
