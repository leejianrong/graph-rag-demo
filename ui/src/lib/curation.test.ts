import { afterEach, expect, test, vi } from "vitest";
import { makeClient, mergeEntities, splitEntity } from "../api/client";
import {
  describeCurationError,
  guardMerge,
  mergeAnnouncement,
  mergeLines,
  mergeSummary,
  splitAnnouncement,
  splitLines,
  trailAfterMerge,
  type Failure,
} from "./curation";
import type { EntitySummary, Neighbourhood } from "./explorer";

const ent = (id: string, type = "ORG", name = id.toUpperCase()): EntitySummary => ({
  canonical_id: id,
  name,
  type: type as EntitySummary["type"],
  aliases: [],
});
const hood = (over: Partial<Neighbourhood>): Neighbourhood =>
  ({
    canonical_id: "x",
    name: "X",
    type: "ORG",
    aliases: [],
    neighbours: [],
    sources: [],
    incoming_total: 0,
    outgoing_total: 0,
    sources_total: 0,
    merged_from: [],
    ...over,
  }) as Neighbourhood;

const conflict = (reason: string, extra: Record<string, unknown> = {}): Failure => ({
  status: 409,
  details: { reason, ...extra },
});
const nameOf = (id: string) => `<${id}>`;

test("every 409 reason has its own plain message", () => {
  const reasons: [Failure, RegExp][] = [
    [conflict("type_mismatch", { entity_ids: ["b"] }), /different types.*<b>/],
    [conflict("no_merge_record"), /no merge record/],
    [conflict("merge_pending", { merge_id: "m1" }), /never finished/],
    [conflict("survivor_merged", { merged_into: "p", merge_id: "m1" }), /merged into <p>.*first/],
    [conflict("entity_exists", { entity_ids: ["b"] }), /<b>.*cannot be restored/],
    [conflict("neighbour_merged", { entity_ids: ["c"] }), /<c>.*merged away/],
  ];
  const seen = new Set<string>();
  for (const [f, re] of reasons) {
    const out = describeCurationError("split", f, nameOf);
    expect(out.message).toMatch(re);
    seen.add(out.message);
  }
  expect(seen.size).toBe(reasons.length);
});

test("survivor_merged and neighbour_merged offer a button that opens the entity to deal with first", () => {
  expect(describeCurationError("split", conflict("survivor_merged", { merged_into: "p" }), nameOf).action).toEqual({
    label: "Open <p>",
    id: "p",
  });
  expect(describeCurationError("split", conflict("neighbour_merged", { entity_ids: ["c", "d"] }), nameOf).action?.id).toBe("c");
  expect(describeCurationError("split", conflict("merge_pending"), nameOf).action).toBeUndefined();
});

test("an unknown 409 reason still explains that nothing changed", () => {
  expect(describeCurationError("merge", { status: 409, message: "busy" }).message).toBe(
    "The merge conflicts with the current graph: busy. Nothing changed.",
  );
});

test("other statuses are explained by what to do next", () => {
  const m = (f: Failure) => describeCurationError("merge", f, nameOf);
  expect(m({ status: 0 }).message).toMatch(/did not answer/);
  expect(m({ status: 400, message: "merge_ids repeat" }).message).toMatch(/refused.*merge_ids repeat/);
  expect(m({ status: 400, details: { field: "merge_id", merge_ids: ["a", "b"] } }).message).toMatch(/several merges.*a, b/);
  expect(m({ status: 401 }).message).toMatch(/signed out/);
  expect(m({ status: 404 }).message).toMatch(/no longer in this namespace/);
  expect(m({ status: 404, details: { merged_into: "s" } })).toEqual({
    message: "That entity was merged into <s> in the meantime. Nothing changed.",
    action: { label: "Open <s>", id: "s" },
  });
  expect(m({ status: 429 }).message).toMatch(/Too many/);
  expect(m({ status: 503 }).message).toMatch(/not switched on/);
  expect(m({ status: 500 }).message).toMatch(/answered 500/);
});

test("the guard blocks an empty pick, self-merge, repeats, too many and a type mismatch", () => {
  const keep = ent("k", "ORG", "Kept");
  expect(guardMerge(keep, [ent("a")])).toEqual({ ok: true });
  expect(guardMerge(keep, [])).toMatchObject({ ok: false });
  expect(guardMerge(keep, [ent("k")])).toEqual({ ok: false, message: "An entity cannot be merged into itself." });
  expect(guardMerge(keep, [ent("a"), ent("a")])).toMatchObject({ ok: false, message: expect.stringMatching(/twice/) });
  const eleven = Array.from({ length: 11 }, (_, i) => ent(`e${i}`));
  expect(guardMerge(keep, eleven)).toMatchObject({ ok: false, message: expect.stringMatching(/at most 10/) });
  expect(guardMerge(keep, Array.from({ length: 10 }, (_, i) => ent(`e${i}`)))).toEqual({ ok: true });
  const g = guardMerge(keep, [ent("p", "PERSON", "Pat")]);
  expect(g).toMatchObject({ ok: false });
  expect(g.ok === false && g.message).toBe(
    "Kept is ORG. Pat (PERSON) has a different type, and entities of different types cannot be merged.",
  );
});

test("the merge summary adds up edges and documents and lists only names the keeper lacks", () => {
  const keep = hood({ name: "Halden Robotics", aliases: ["Halden"] });
  const a = hood({ name: "Halden Robotics OU", aliases: ["halden", "HR OU"], incoming_total: 1, outgoing_total: 2, sources_total: 2 });
  const b = hood({ name: "HR OU", incoming_total: 0, outgoing_total: 1, sources_total: 1 });
  const s = mergeSummary(keep, [a, b]);
  expect(s).toEqual({ edges: 4, documents: 3, aliases: ["Halden Robotics OU", "HR OU"] });
  const lines = mergeLines("Halden Robotics", ["Halden Robotics OU", "HR OU"], s);
  expect(lines[0]).toBe("4 edges and 3 document links move to Halden Robotics.");
  expect(lines[1]).toBe("2 names become aliases of Halden Robotics: Halden Robotics OU, HR OU.");
  expect(lines[3]).toMatch(/Nothing is deleted.*split it again/);
  expect(mergeLines("K", ["D"], { edges: 1, documents: 1, aliases: [] })[1]).toMatch(/no aliases are added/);
});

test("the split lines say what returns and what stays", () => {
  const lines = splitLines("Halden Robotics", { name: "Halden Robotics OU", aliases: ["HR OU"] });
  expect(lines[0]).toMatch(/Halden Robotics OU comes back as its own entity/);
  expect(lines[1]).toBe("Names that only it brought leave Halden Robotics: Halden Robotics OU, HR OU.");
  expect(lines[2]).toBe("Edges and document links created after the merge stay with Halden Robotics.");
});

test("merged entities leave the trail and the keeper stays", () => {
  const trail = [{ id: "a", name: "A" }, { id: "b", name: "B" }, { id: "k", name: "K" }];
  expect(trailAfterMerge(trail, ["a", "b"])).toEqual([{ id: "k", name: "K" }]);
});

test("a repeat is explained as a no-op, not an error", () => {
  expect(mergeAnnouncement("K", 2, 0)).toBe("Merged 2 entities into K.");
  expect(mergeAnnouncement("K", 1, 1)).toBe("Merged 1 entity into K. 1 other had already been merged.");
  expect(mergeAnnouncement("K", 0, 1)).toBe("Nothing to do: 1 entity was already merged into K.");
  expect(splitAnnouncement("K", ["D"], 0)).toBe("Split D out of K.");
  expect(splitAnnouncement("K", [], 1)).toBe("Nothing to do: that entity was already split out of K.");
});

/** A fetch that records requests and answers with `body`. */
function spy(body: unknown, status = 200) {
  const seen: Request[] = [];
  const impl: typeof fetch = async (input, init) => {
    seen.push(new Request(input, init));
    return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
  };
  return { impl, seen };
}

test("merge posts merge_ids and the namespace to the keeper's path with the key", async () => {
  const reply = { merge_id: "m1", merged: ["b"], already_merged: [], entity: ent("a") };
  const { impl, seen } = spy(reply);
  const client = makeClient("http://x", impl, { key: () => "test-token", onUnauthorized: () => {} });
  const out = await mergeEntities(client, "org:1/a", ["b", "c"], "support");
  expect(out).toEqual({ ok: true, data: reply });
  expect(seen[0].method).toBe("POST");
  expect(decodeURIComponent(new URL(seen[0].url).pathname)).toBe("/v1/entities/org:1/a/merge");
  expect(await seen[0].json()).toEqual({ merge_ids: ["b", "c"], namespace: "support" });
  expect(seen[0].headers.get("authorization")).toBe("Bearer test-token");
});

test("split posts merge_id and entity_ids", async () => {
  const reply = { merge_id: "m1", split: [ent("b")], already_split: [], entity: ent("a") };
  const { impl, seen } = spy(reply);
  const out = await splitEntity(makeClient("http://x", impl), "a", "m1", ["b"]);
  expect(out.ok).toBe(true);
  expect(new URL(seen[0].url).pathname).toBe("/v1/entities/a/split");
  expect(await seen[0].json()).toEqual({ merge_id: "m1", entity_ids: ["b"] });
});

test("a 409 keeps its code and details so the reason can be explained", async () => {
  const error = { code: "conflict", message: "types differ", details: { reason: "type_mismatch", entity_ids: ["b"] } };
  const { impl } = spy({ error }, 409);
  const out = await mergeEntities(makeClient("http://x", impl), "a", ["b"]);
  expect(out).toEqual({ ok: false, status: 409, message: "types differ", code: "conflict", details: error.details });
  if (!out.ok) expect(describeCurationError("merge", out, nameOf).message).toMatch(/different types/);
});

test("404 with merged_into, 429, 503 and a dead network each come back with their status", async () => {
  const cases: [number, unknown][] = [
    [404, { error: { code: "not_found", message: "gone", details: { merged_into: "s" } } }],
    [429, { error: { code: "rate_limited", message: "slow" } }],
    [503, { error: { code: "unavailable", message: "off" } }],
  ];
  for (const [status, body] of cases) {
    const out = await splitEntity(makeClient("http://x", spy(body, status).impl), "a", "m", ["b"]);
    expect(out).toMatchObject({ ok: false, status });
  }
  const dead: typeof fetch = async () => {
    throw new TypeError("offline");
  };
  expect(await mergeEntities(makeClient("http://x", dead), "a", ["b"])).toEqual({ ok: false, status: 0 });
});

afterEach(() => vi.unstubAllGlobals());
