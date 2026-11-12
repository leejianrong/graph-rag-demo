import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import type { FullConfig } from "@playwright/test";

// Boots `graph-rag-offline` (in-memory stores, no Docker, model or LLM) with auth on, serving the built
// UI, and makes a key for it in a throwaway file. The key goes into this process's environment for the
// tests and is never printed or written anywhere else; the temp directory is removed on teardown.

const root = resolve(import.meta.dirname, "../..");

function freePort(): Promise<number> {
  return new Promise((done, fail) => {
    const server = createServer();
    server.once("error", fail);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as { port: number };
      server.close(() => done(port));
    });
  });
}

async function waitFor(url: string, until: number): Promise<void> {
  while (Date.now() < until) {
    try {
      if ((await fetch(url)).ok) return;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  throw new Error(`the offline server did not answer ${url} in time`);
}

export default async function globalSetup(_config: FullConfig) {
  const dir = mkdtempSync(join(tmpdir(), "graph-rag-e2e-"));
  const keyFile = join(dir, "keys.json");
  const made = spawnSync(
    "uv",
    ["run", "--no-sync", "graph-rag-keys", "--file", keyFile, "create", "e2e-tenant"],
    { cwd: root, encoding: "utf8" },
  );
  const secret = /^key:\s+(\S+)$/m.exec(made.stdout ?? "")?.[1];
  if (made.status !== 0 || !secret) {
    rmSync(dir, { recursive: true, force: true });
    throw new Error("could not create a key with graph-rag-keys");
  }

  const port = await freePort();
  const server = spawn("uv", ["run", "--no-sync", "graph-rag-offline"], {
    cwd: root,
    detached: true, // its own process group, so teardown stops uv and the python it starts
    stdio: "ignore",
    env: {
      ...process.env,
      HOST: "127.0.0.1",
      PORT: String(port),
      REQUIRE_AUTH: "true",
      API_KEYS_PATH: keyFile,
      UI_DIR: join(root, "ui", "dist"),
    },
  });
  const pid = server.pid;
  const stop = () => {
    if (pid) {
      try {
        process.kill(-pid, "SIGTERM");
      } catch {
        /* already gone */
      }
    }
    rmSync(dir, { recursive: true, force: true });
  };

  try {
    await waitFor(`http://127.0.0.1:${port}/health`, Date.now() + 90_000);
  } catch (error) {
    stop();
    throw error;
  }
  process.env.E2E_BASE_URL = `http://127.0.0.1:${port}`;
  process.env.E2E_KEY = secret;
  return stop;
}
