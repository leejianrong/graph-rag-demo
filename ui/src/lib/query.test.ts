import { expect, test } from "vitest";
import { makeClient } from "../api/client";
import { findPaths, highlight, hopOf, runQuery, toView, type NodeRef, type Path, type QueryResponse } from "./query";

const ref = (id: string, name = id): NodeRef => ({ id, name, type: "Person" });
const edge = (s: string, p: string, o: string, doc = "d1", idx = 0) => ({
  subject_id: s,
  predicate: p,
  object_id: o,
  provenance: { source_doc_id: doc, sentence_index: idx, source_sentence: "x" },
});
const refs = (...ids: string[]) => new Map(ids.map((i) => [i, ref(i)]));
const route = (p: Path) => p.hops.map((h) => `${h.from.id}-${h.predicate}->${h.to.id}`);

test("a single hop reports predicate and direction", () => {
  const [p] = findPaths([edge("a", "WORKS_AT", "b")], [ref("a")], ref("b"), refs("a", "b"));
  expect(route(p)).toEqual(["a-WORKS_AT->b"]);
  expect(p.hops[0].forward).toBe(true);
});

test("a hop against the stored edge is marked backward", () => {
  const [p] = findPaths([edge("b", "EMPLOYS", "a")], [ref("a")], ref("b"), refs("a", "b"));
  expect(p.hops[0].forward).toBe(false);
});

test("multi hop takes the shortest route", () => {
  const edges = [edge("a", "P1", "b"), edge("b", "P2", "c"), edge("c", "P3", "d"), edge("a", "SHORT", "d")];
  const all = refs("a", "b", "c", "d");
  expect(route(findPaths(edges, [ref("a")], ref("d"), all)[0])).toEqual(["a-SHORT->d"]);
  expect(route(findPaths(edges.slice(0, 3), [ref("a")], ref("d"), all)[0])).toEqual([
    "a-P1->b",
    "b-P2->c",
    "c-P3->d",
  ]);
});

test("cycles terminate", () => {
  const edges = [edge("a", "P", "b"), edge("b", "P", "a"), edge("b", "Q", "c"), edge("c", "R", "a")];
  expect(findPaths(edges, [ref("a")], ref("c"), refs("a", "b", "c"))[0].hops).toHaveLength(1);
});

test("no path when the nodes are not connected, or the start is the target", () => {
  const edges = [edge("a", "P", "b"), edge("x", "P", "y")];
  expect(findPaths(edges, [ref("a")], ref("y"), refs("a", "b", "x", "y"))).toEqual([]);
  expect(findPaths(edges, [ref("b")], ref("b"), refs("a", "b"))).toEqual([]);
  expect(findPaths(edges, [ref("zzz")], ref("b"), refs("a", "b"))).toEqual([]);
});

test("ties go to the earlier edge, and the same route is listed once", () => {
  const edges = [edge("a", "FIRST", "t"), edge("a", "SECOND", "t")];
  const paths = findPaths(edges, [ref("a"), ref("a")], ref("t"), refs("a", "t"));
  expect(paths).toHaveLength(1);
  expect(route(paths[0])).toEqual(["a-FIRST->t"]);
});

test("shorter paths come first, then the order of the starts", () => {
  const edges = [edge("a", "P", "m"), edge("m", "P", "t"), edge("b", "P", "t")];
  const paths = findPaths(edges, [ref("a"), ref("b")], ref("t"), refs("a", "b", "m", "t"));
  expect(paths.map((p) => p.start.id)).toEqual(["b", "a"]);
});

test("at most `limit` paths, in the order of the starts", () => {
  const edges = [edge("a", "P", "t"), edge("b", "P", "t"), edge("c", "P", "t")];
  const paths = findPaths(edges, [ref("c"), ref("a"), ref("b")], ref("t"), refs("a", "b", "c", "t"), 2);
  expect(paths.map((p) => p.start.id)).toEqual(["c", "a"]);
});

const node = (id: string, name: string, type = "PERSON") => ({ canonical_id: id, name, type, aliases: [] });
const rank = (id: string, name: string, score: number, type = "PERSON") => ({ canonical_id: id, name, type, score });
const response = (): QueryResponse =>
  ({
    answer: "Tallinn",
    answer_entity: rank("c", "Tallinn", 0.9, "GPE"),
    prose: null,
    ranked_nodes: [rank("c", "Tallinn", 0.9, "GPE"), rank("a", "Priya", 0.5)],
    matched_entity_ids: ["a"],
    subgraph: {
      nodes: [node("a", "Priya"), node("b", "Halden", "ORG"), node("c", "Tallinn", "GPE"), node("z", "Tartu", "GPE")],
      edges: [edge("a", "WORKS_AT", "b", "d1", 2), edge("b", "HEADQUARTERED_IN", "c", "d2", 0)],
    },
    supporting_sentences: [
      { document_id: "d1", text: "Priya joined Halden.", char_start: 10, char_end: 30, sentence_index: 2, score: 0.8 },
      { document_id: "d3", text: "Unrelated.", char_start: 0, char_end: 10, sentence_index: 0, score: 0.1 },
    ],
  }) as QueryResponse;

test("the mapper builds the path, off-path nodes and sentence support", () => {
  const view = toView(response());
  expect(view.answer).toBe("Tallinn");
  expect(route(view.paths[0])).toEqual(["a-WORKS_AT->b", "b-HEADQUARTERED_IN->c"]);
  expect(view.offPath.map((n) => n.name)).toEqual(["Tartu"]);
  expect(view.empty).toBeNull();
  expect(view.sentences[0].supports[0].predicate).toBe("WORKS_AT");
  expect(hopOf(view.sentences[0], view.paths[0])).toBe(1);
  expect(hopOf(view.sentences[1], view.paths[0])).toBeNull();
  expect(view.prose).toBeNull();
});

test("paths start from the matched entities, not from other ranked nodes", () => {
  const r = response();
  expect(toView({ ...r, matched_entity_ids: [] }).paths).toEqual([]);
  expect(toView({ ...r, matched_entity_ids: ["z", "a"] }).paths.map((p) => p.start.id)).toEqual(["a"]);
  expect(toView({ ...r, matched_entity_ids: undefined }).paths).toEqual([]);
});

test("an empty response is explained as no match; an answer with no path as no path", () => {
  const none = toView({ answer: null, answer_entity: null, subgraph: { nodes: [], edges: [] } } as QueryResponse);
  expect(none.empty).toBe("no-match");
  const alone = toView({
    answer: "X",
    answer_entity: rank("x", "X", 1),
    ranked_nodes: [rank("x", "X", 1)],
    subgraph: { nodes: [node("x", "X")], edges: [] },
  } as QueryResponse);
  expect(alone.empty).toBe("no-path");
  expect(alone.answer).toBe("X");
});

test("highlight marks every mention, case-insensitively, and merges overlaps", () => {
  const hits = highlight("Priya met priya at Halden.", ["Priya", "Halden"]).filter((s) => s.hit);
  expect(hits.map((s) => s.text)).toEqual(["Priya", "priya", "Halden"]);
  expect(highlight("Halden Robotics", ["Halden", "Halden Robotics"])).toEqual([{ text: "Halden Robotics", hit: true }]);
  expect(highlight("plain", ["nope"])).toEqual([{ text: "plain", hit: false }]);
});

const json =
  (body: unknown, status = 200): typeof fetch =>
  async () =>
    new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

test("runQuery posts the question and measures latency", async () => {
  let sent: { url: string; body: unknown; auth: string | null } | null = null;
  const spy: typeof fetch = async (input, init) => {
    const req = input instanceof Request ? input : new Request(input, init);
    sent = { url: req.url, body: await req.json(), auth: req.headers.get("authorization") };
    return json(response())(input, init);
  };
  const clock = [100, 142];
  const client = makeClient("http://x", spy, { key: () => "k", onUnauthorized: () => {} });
  const result = await runQuery(client, "  Where? ", true, () => clock.shift()!);
  expect(result.ok && result.ms).toBe(42);
  expect(sent).toEqual({ url: "http://x/query", body: { question: "Where?", synthesize: true }, auth: "Bearer k" });
});

test("runQuery explains errors and signs out on 401", async () => {
  let out = 0;
  const hooks = { key: () => "k", onUnauthorized: () => out++ };
  const ask = (status: number) => runQuery(makeClient("http://x", json({ detail: "n" }, status), hooks), "q", false);
  expect(await ask(401)).toMatchObject({ ok: false, status: 401 });
  expect(out).toBe(1);
  expect(await ask(429)).toMatchObject({ ok: false, message: expect.stringContaining("Wait a minute") });
  expect(await ask(500)).toMatchObject({ ok: false, message: expect.stringContaining("500") });
  expect(await ask(503)).toMatchObject({ ok: false, status: 503 });
  const down: typeof fetch = async () => {
    throw new TypeError("offline");
  };
  expect(await runQuery(makeClient("http://x", down), "q", false)).toMatchObject({ ok: false, status: null });
  expect(await runQuery(makeClient("http://x", down), "  ", false)).toMatchObject({
    ok: false,
    message: "Type a question first.",
  });
});

test("a two-part question lists the later answers after the first", () => {
  const r = response();
  expect(toView(r).moreAnswers).toEqual([]);
  const two = toView({ ...r, answer_entities: [r.answer_entity!, rank("a", "Priya", 0.5)] as QueryResponse["answer_entities"] });
  expect(two.moreAnswers.map((n) => n.name)).toEqual(["Priya"]);
});
