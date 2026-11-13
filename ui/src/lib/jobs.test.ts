import { expect, test } from "vitest";
import { countsText, newJob, reduceJob, segments, statusText, type JobEvent } from "./jobs";

const start = () => newJob(1, "a.txt", { kind: "url", url: "https://x.test/a" });
const run = (...events: JobEvent[]) => events.reduce(reduceJob, start());
const states = (job: ReturnType<typeof start>) => segments(job).map((s) => s.state);
const accepted: JobEvent = { type: "accepted", documentId: "d1", status: "accepted" };

test("an accepted source becomes a queued job; an unchanged one is final", () => {
  expect(run(accepted).state).toBe("queued");
  const same = run({ type: "accepted", documentId: "d1", status: "unchanged" });
  expect(same.state).toBe("unchanged");
  expect(statusText(same)).toMatch(/Unchanged/);
});

test("stages move the running segment and finished ones turn done", () => {
  const job = run(accepted, { type: "stage", stage: "read" }, { type: "stage", stage: "entity recognition" });
  expect(job.state).toBe("running");
  expect(states(job)).toEqual(["done", "running", "pending", "pending", "pending"]);
  expect(statusText(job)).toBe("Running · entity recognition");
});

test("a replayed stage is not added twice", () => {
  const job = run(accepted, { type: "stage", stage: "read" }, { type: "stage", stage: "read" });
  expect(job.stages).toEqual(["read"]);
});

test("done fills every segment", () => {
  const job = run(accepted, { type: "stage", stage: "read" }, {
    type: "snapshot",
    job: { document_id: "d1", object_key: "k", state: "done", stages: ["read"], queued_at: "t0", updated_at: "t1" },
  });
  expect(states(job)).toEqual(Array(5).fill("done"));
  expect(statusText(job)).toBe("Done");
});

test("failed marks the last stage and keeps the reason", () => {
  const job = run(accepted, {
    type: "snapshot",
    job: {
      document_id: "d1",
      object_key: "k",
      state: "failed",
      error: "coreference failed: TimeoutError",
      stages: ["read", "entity recognition", "coreference"],
      queued_at: "t0",
      updated_at: "t1",
    },
  });
  expect(states(job)).toEqual(["done", "done", "failed", "pending", "pending"]);
  expect(job.error).toBe("coreference failed: TimeoutError");
  expect(statusText(job)).toBe("Failed · coreference");
});

test("a finished job ignores late events", () => {
  const done = run(accepted, {
    type: "snapshot",
    job: { document_id: "d1", object_key: "k", state: "done", stages: [], queued_at: "t0", updated_at: "t1" },
  });
  expect(reduceJob(done, { type: "stage", stage: "read" })).toBe(done);
  expect(reduceJob(done, { type: "link", link: "polling" })).toBe(done);
});

test("a snapshot with fewer stages than already seen keeps the seen ones", () => {
  const job = run(accepted, { type: "stage", stage: "read" }, { type: "stage", stage: "coreference" }, {
    type: "snapshot",
    job: { document_id: "d1", object_key: "k", state: "running", stages: ["read"], queued_at: "t0", updated_at: "t1" },
  });
  expect(job.stages).toEqual(["read", "coreference"]);
});

test("failing to start shows the reason", () => {
  const job = run({ type: "failed-to-start", error: "gone" });
  expect(job.state).toBe("failed");
  expect(job.error).toBe("gone");
});

test("a done job with counts shows them, and one without shows none", () => {
  const snap = (extra: object): JobEvent => ({
    type: "snapshot",
    job: { document_id: "d1", object_key: "k", state: "done", stages: ["read"], queued_at: "t0", updated_at: "t1", ...extra },
  });
  expect(countsText(run(accepted, snap({ entities: 5, edges: 1 })))).toBe("5 entities · 1 edge");
  expect(countsText(run(accepted, snap({ entities: 1, edges: 0 })))).toBe("1 entity · 0 edges");
  expect(countsText(run(accepted, snap({})))).toBeNull();
});
