import { expect, test } from "vitest";
import { entityDetail, makeClient, searchEntities } from "./client";

/** A fetch that records the request and answers with `body`. */
function spy(body: unknown, status = 200) {
  const seen: Request[] = [];
  const impl: typeof fetch = async (input, init) => {
    seen.push(new Request(input, init));
    return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
  };
  return { impl, seen };
}

const hooks = (signedOut: string[] = []) => ({ key: () => "test-token", onUnauthorized: () => void signedOut.push("out") });

test("search sends q, limit and cursor with the bearer key and returns the page", async () => {
  const page = { entities: [{ canonical_id: "a", name: "A", type: "ORG", aliases: [] }], next_cursor: "n1" };
  const { impl, seen } = spy(page);
  const out = await searchEntities(makeClient("http://x", impl, hooks()), { q: "hal", limit: 20, cursor: "c0" });
  expect(out).toEqual({ ok: true, data: page });
  const url = new URL(seen[0].url);
  expect(url.pathname).toBe("/v1/entities");
  expect(url.searchParams.get("q")).toBe("hal");
  expect(url.searchParams.get("limit")).toBe("20");
  expect(url.searchParams.get("cursor")).toBe("c0");
  expect(seen[0].headers.get("authorization")).toBe("Bearer test-token");
});

test("search leaves out parameters that are not set", async () => {
  const { impl, seen } = spy({ entities: [] });
  await searchEntities(makeClient("http://x", impl), { limit: 20 });
  expect(new URL(seen[0].url).search).toBe("?limit=20");
});

test("detail puts the id in the path and carries the namespace", async () => {
  const body = { canonical_id: "org:1/x", name: "X", type: "ORG", aliases: [], neighbours: [], sources: [] };
  const { impl, seen } = spy(body);
  const out = await entityDetail(makeClient("http://x", impl), "org:1/x", { namespace: "support" });
  expect(out.ok).toBe(true);
  const url = new URL(seen[0].url);
  expect(decodeURIComponent(url.pathname)).toBe("/v1/entities/org:1/x");
  expect(url.searchParams.get("namespace")).toBe("support");
});

test("an API error keeps its status and message", async () => {
  const { impl } = spy({ error: { code: "not_found", message: "no such entity" } }, 404);
  expect(await entityDetail(makeClient("http://x", impl), "zz")).toEqual({
    ok: false,
    status: 404,
    message: "no such entity",
  });
});

test("a 401 signs out and a network failure is status 0", async () => {
  const out: string[] = [];
  const { impl } = spy({ error: { code: "unauthorized", message: "bad key" } }, 401);
  const r = await searchEntities(makeClient("http://x", impl, hooks(out)), {});
  expect(r.ok).toBe(false);
  expect(out).toEqual(["out"]);
  const broken: typeof fetch = async () => {
    throw new TypeError("offline");
  };
  expect(await searchEntities(makeClient("http://x", broken), {})).toEqual({ ok: false, status: 0 });
});
