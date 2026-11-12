import { expect, test } from "vitest";
import {
  applyPage,
  backTo,
  describeError,
  emptySearch,
  firstSeen,
  moreCount,
  spanLabel,
  splitNeighbours,
  validity,
  visit,
  type EntitySummary,
  type Neighbour,
} from "./explorer";

const ent = (id: string): EntitySummary => ({ canonical_id: id, name: id.toUpperCase(), type: "ORG", aliases: [] });
const edge = (direction: string, id: string): Neighbour => ({
  direction,
  predicate: "works_at",
  entity: { canonical_id: id, name: id, type: "ORG", aliases: [] },
  provenance: { source_doc_id: "d", sentence_index: 0, source_sentence: "s" },
});

test("a first page replaces the list and a later page extends it without repeats", () => {
  const first = applyPage(emptySearch("hal"), { entities: [ent("a"), ent("b")], next_cursor: "c1" }, false);
  expect(first.results.map((e) => e.canonical_id)).toEqual(["a", "b"]);
  expect(first.cursor).toBe("c1");
  expect(first.loaded).toBe(true);
  const next = applyPage(first, { entities: [ent("b"), ent("c")], next_cursor: null }, true);
  expect(next.results.map((e) => e.canonical_id)).toEqual(["a", "b", "c"]);
  expect(next.cursor).toBeNull();
  const again = applyPage(next, { entities: [ent("z")] }, false);
  expect(again.results.map((e) => e.canonical_id)).toEqual(["z"]);
});

test("an empty first page is loaded with no results", () => {
  const s = applyPage(emptySearch("x"), { entities: [] }, false);
  expect(s.loaded).toBe(true);
  expect(s.results).toEqual([]);
});

test("the trail grows on neighbour clicks, restarts on a search pick and cuts back on a revisit", () => {
  const a = { id: "a", name: "A" };
  const b = { id: "b", name: "B" };
  const c = { id: "c", name: "C" };
  let trail = visit([], a, true);
  trail = visit(trail, b);
  trail = visit(trail, c);
  expect(trail.map((s) => s.id)).toEqual(["a", "b", "c"]);
  expect(visit(trail, a).map((s) => s.id)).toEqual(["a"]);
  expect(visit(trail, b).map((s) => s.id)).toEqual(["a", "b"]);
  expect(visit(trail, c, true).map((s) => s.id)).toEqual(["c"]);
});

test("backTo drops later crumbs and ignores a bad index", () => {
  const trail = [{ id: "a", name: "A" }, { id: "b", name: "B" }, { id: "c", name: "C" }];
  expect(backTo(trail, 0).map((s) => s.id)).toEqual(["a"]);
  expect(backTo(trail, 1).map((s) => s.id)).toEqual(["a", "b"]);
  expect(backTo(trail, 5)).toBe(trail);
  expect(backTo(trail, -1)).toBe(trail);
});

test("neighbours split by direction and keep their order", () => {
  const { incoming, outgoing } = splitNeighbours({
    neighbours: [edge("outgoing", "o1"), edge("incoming", "i1"), edge("outgoing", "o2"), edge("incoming", "i2")],
  });
  expect(outgoing.map((n) => n.entity.canonical_id)).toEqual(["o1", "o2"]);
  expect(incoming.map((n) => n.entity.canonical_id)).toEqual(["i1", "i2"]);
});

test("the more count is the unshown part of a total and never negative", () => {
  expect(moreCount(30, 25)).toBe(5);
  expect(moreCount(3, 3)).toBe(0);
  expect(moreCount(2, 5)).toBe(0);
});

test("validity names whichever ends are set", () => {
  expect(validity("2026-01-05", "2026-03-01")).toBe("2026-01-05 to 2026-03-01");
  expect(validity("2026-01-05", null)).toBe("from 2026-01-05");
  expect(validity(undefined, "2026-03-01")).toBe("until 2026-03-01");
  expect(validity(null, null)).toBeNull();
});

test("the span label needs both offsets and a non-empty span", () => {
  expect(spanLabel(10, 40)).toBe("characters 10–40");
  expect(spanLabel(null, 40)).toBeNull();
  expect(spanLabel(5, 5)).toBeNull();
});

test("first seen shows the date only", () => {
  expect(firstSeen("2026-11-12T06:43:57.936Z")).toBe("2026-11-12");
  expect(firstSeen(null)).toBe("not recorded");
});

test("errors read differently by status", () => {
  expect(describeError(400, "bad cursor").message).toContain("bad cursor");
  expect(describeError(404).message).toMatch(/not in this namespace/);
  expect(describeError(429).message).toMatch(/Too many/);
  expect(describeError(503).message).toMatch(/cannot browse/);
  expect(describeError(0).message).toMatch(/did not answer/);
  expect(describeError(500).message).toContain("500");
});
