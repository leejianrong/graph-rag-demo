import { afterEach, beforeEach, expect, test, vi } from "vitest";

// The shared client has a relative base URL, which Node cannot resolve, so the tests swap in one with an absolute URL.
const hooked = vi.hoisted(() => ({ impl: (async () => new Response("{}", { status: 500 })) as typeof fetch }));
vi.mock("./auth.svelte", async (original) => {
  const real = await original<typeof import("./auth.svelte")>();
  const { makeClient } = await import("../api/client");
  return { ...real, api: makeClient("http://x", (...a) => hooked.impl(...a)) };
});
import { curation, closeCuration, confirmMerge, confirmSplit, openPicker, reviewMerge, startSplit, toggleSelected } from "./curation.svelte";
import { centreOn, explorer, resetExplorer } from "./explorer.svelte";
import { session } from "./auth.svelte";

type Reply = [number, unknown];
let routes: Record<string, Reply>;
let calls: { path: string; body: unknown }[];

const hood = (id: string, over: Record<string, unknown> = {}) => ({
  canonical_id: id,
  name: id.toUpperCase(),
  type: "ORG",
  aliases: [],
  neighbours: [],
  sources: [],
  incoming_total: 1,
  outgoing_total: 1,
  sources_total: 1,
  merged_from: [],
  ...over,
});
const sum = (id: string) => ({ canonical_id: id, name: id.toUpperCase(), type: "ORG", aliases: [] });

beforeEach(() => {
  resetExplorer();
  closeCuration(false);
  session.namespace = "support";
  calls = [];
  routes = {};
  hooked.impl = async (input: RequestInfo | URL, init?: RequestInit) => {
    const req = new Request(input as Request, init);
    const path = decodeURIComponent(new URL(req.url).pathname);
    calls.push({ path, body: req.method === "POST" ? await req.json() : null });
    const [status, body] = routes[`${req.method} ${path}`] ?? [200, { entities: [] }];
    return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
  };
});
afterEach(() => {
  session.namespace = null;
});

const settle = () => new Promise((r) => setTimeout(r, 0));

test("a merge-away 404 with merged_into moves the focus to the survivor with a notice", async () => {
  routes["GET /v1/entities/old"] = [404, { error: { code: "not_found", message: "x", details: { merged_into: "keep" } } }];
  routes["GET /v1/entities/keep"] = [200, hood("keep", { name: "Keeper" })];
  centreOn({ id: "old", name: "Old" });
  await settle();
  expect(explorer.detail?.canonical_id).toBe("keep");
  expect(explorer.trail).toEqual([{ id: "keep", name: "Keeper" }]);
  expect(explorer.notice).toMatch(/Old was merged into another entity/);
  expect(explorer.detailError).toBeNull();
});

test("a confirmed merge sends the ids and namespace, announces, and refreshes the detail", async () => {
  routes["GET /v1/entities/k"] = [200, hood("k")];
  routes["GET /v1/entities/d"] = [200, hood("d")];
  routes["POST /v1/entities/k/merge"] = [200, { merge_id: "m1", merged: ["d"], already_merged: [], entity: sum("k") }];
  centreOn({ id: "k", name: "K" });
  await settle();
  openPicker(null);
  toggleSelected(sum("d") as never);
  await reviewMerge();
  expect(curation.mode).toBe("confirm-merge");
  const pending = confirmMerge();
  expect(curation.busy).toBe(true);
  await pending;
  expect(calls.find((c) => c.path.endsWith("/merge"))?.body).toEqual({ merge_ids: ["d"], namespace: "support" });
  expect(explorer.notice).toBe("Merged 1 entity into K.");
  expect(curation.mode).toBe("idle");
});

test("a failed split stays open with the reason and an action", async () => {
  routes["GET /v1/entities/k"] = [200, hood("k")];
  routes["POST /v1/entities/k/split"] = [
    409,
    { error: { code: "conflict", message: "m", details: { reason: "survivor_merged", merged_into: "p", merge_id: "m1" } } },
  ];
  centreOn({ id: "k", name: "K" });
  await settle();
  startSplit({ canonical_id: "d", name: "D", type: "ORG", aliases: [], merge_id: "m1", merged_at: "t" }, null);
  await confirmSplit();
  expect(calls.find((c) => c.path.endsWith("/split"))?.body).toEqual({ merge_id: "m1", entity_ids: ["d"], namespace: "support" });
  expect(curation.mode).toBe("confirm-split");
  expect(curation.busy).toBe(false);
  expect(curation.failure?.message).toMatch(/merged into p/);
  expect(curation.failure?.action?.id).toBe("p");
});
