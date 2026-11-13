import { expect, test, vi } from "vitest";
import { authedFetch } from "../api/client";
import { defaultTextName, submitSource, textSource, watchJob } from "./ingest";
import type { JobEvent } from "./jobs";

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });

const sse = (text: string, split = 7) => {
  const enc = new TextEncoder().encode(text);
  return new Response(
    new ReadableStream<Uint8Array>({
      start(c) {
        for (let i = 0; i < enc.length; i += split) c.enqueue(enc.slice(i, i + split));
        c.close();
      },
    }),
    { headers: { "content-type": "text/event-stream" } },
  );
};

const sent = (spy: ReturnType<typeof vi.fn>) => spy.mock.calls[0] as [string, RequestInit];

test("a url goes up as the url field and nothing else", async () => {
  const spy = vi.fn(async () => json({ document_id: "d1", status: "accepted" }));
  const result = await submitSource({ kind: "url", url: "https://x.test/p" }, spy as unknown as typeof fetch);
  expect(result).toEqual({ ok: true, documentId: "d1", status: "accepted" });
  const [path, init] = sent(spy);
  expect(path).toBe("/ingest");
  const body = init.body as FormData;
  expect([...body.keys()]).toEqual(["url"]);
  expect(body.get("url")).toBe("https://x.test/p");
});

test("a file goes up as the file field under its own name", async () => {
  const spy = vi.fn(async () => json({ document_id: "d2", status: "unchanged" }));
  const file = new File(["hello"], "notes.txt", { type: "text/plain" });
  const result = await submitSource({ kind: "file", file, name: "notes.txt" }, spy as unknown as typeof fetch);
  expect(result).toMatchObject({ ok: true, status: "unchanged" });
  const body = sent(spy)[1].body as FormData;
  expect([...body.keys()]).toEqual(["file"]);
  expect((body.get("file") as File).name).toBe("notes.txt");
});

test("pasted text goes up as a text/plain file with the given or a generated name", async () => {
  const spy = vi.fn(async () => json({ document_id: "d3", status: "accepted" }));
  await submitSource(textSource("some words", "memo.txt"), spy as unknown as typeof fetch, true);
  const [path, init] = sent(spy);
  expect(path).toBe("/ingest?force=true");
  const part = (init.body as FormData).get("file") as File;
  expect(part.name).toBe("memo.txt");
  expect(part.type).toBe("text/plain");
  expect(await part.text()).toBe("some words");
  const blank = textSource("x", "  ");
  expect(blank.kind === "file" && blank.name).toMatch(/^pasted-\d{8}-\d{6}\.txt$/);
  expect(defaultTextName(new Date("2026-01-02T03:04:05Z"))).toBe("pasted-20260102-030405.txt");
});

test("refusals become form messages", async () => {
  const src = { kind: "url", url: "https://x.test" } as const;
  const refuse = (status: number, body: unknown, headers = {}) =>
    submitSource(src, (async () => json(body, status, headers)) as unknown as typeof fetch);
  expect(await refuse(400, { detail: "Send either a file or a url." })).toEqual({
    ok: false,
    status: 400,
    message: "Send either a file or a url.",
  });
  expect(await refuse(422, { detail: "That page has no text." })).toMatchObject({
    message: "Nothing was queued. That page has no text.",
  });
  expect(await refuse(413, { detail: "The upload is over the 10-byte limit." })).toMatchObject({
    message: "The upload is over the 10-byte limit.",
  });
  expect(await refuse(429, { error: { message: "slow" } }, { "retry-after": "12" })).toMatchObject({
    message: "Too many requests. Wait 12 seconds and try again.",
  });
  expect(await refuse(401, { error: { message: "no" } })).toMatchObject({ message: expect.stringMatching(/signed out/) });
  expect(await refuse(500, "oops")).toMatchObject({ message: "The API answered 500. Try again in a moment." });
});

test("a network failure is reported, not thrown", async () => {
  const down = (async () => {
    throw new TypeError("offline");
  }) as unknown as typeof fetch;
  expect(await submitSource({ kind: "url", url: "https://x.test" }, down)).toMatchObject({ ok: false, status: null });
});

test("authedFetch attaches the key and reports a 401", async () => {
  const onUnauthorized = vi.fn();
  const inner = vi.fn(async () => new Response("", { status: 401 }));
  const f = authedFetch({ key: () => "k-1", onUnauthorized }, inner as unknown as typeof fetch);
  await f("/ingest", { method: "POST" });
  expect(new Headers((inner.mock.calls[0] as unknown as [string, RequestInit])[1].headers).get("authorization")).toBe("Bearer k-1");
  expect(onUnauthorized).toHaveBeenCalledOnce();
});

const watch = async (fetchImpl: typeof fetch, signal = new AbortController().signal) => {
  const events: JobEvent[] = [];
  await watchJob("d1", { fetchImpl, signal, onEvent: (e) => events.push(e), sleep: async () => {}, maxPolls: 4 });
  return events;
};

const final = { document_id: "d1", object_key: "k", state: "done", stages: ["read"], queued_at: "t0", updated_at: "t1" };

test("the stream's stages and final event come through, in order", async () => {
  const events = await watch((async () =>
    sse(`: keep-alive\n\nevent: stage\ndata: {"stage":"read"}\n\nevent: done\ndata: ${JSON.stringify(final)}\n\n`)) as unknown as typeof fetch);
  expect(events).toEqual([
    { type: "link", link: "stream" },
    { type: "stage", stage: "read" },
    { type: "snapshot", job: final },
  ]);
});

test("a stream that drops before done falls back to polling until the job ends", async () => {
  const calls: string[] = [];
  const fetchImpl = (async (path: string) => {
    calls.push(path);
    if (path.endsWith("/events")) return sse('event: stage\ndata: {"stage":"read"}\n\n');
    return calls.length < 4 ? json({ ...final, state: "running" }) : json(final);
  }) as unknown as typeof fetch;
  const events = await watch(fetchImpl);
  expect(events).toContainEqual({ type: "link", link: "polling" });
  expect(events.at(-1)).toEqual({ type: "snapshot", job: final });
  expect(calls.filter((c) => c === "/jobs/d1")).toHaveLength(3);
});

test("polling reports a forgotten job as failed", async () => {
  const events = await watch((async (path: string) =>
    path.endsWith("/events") ? new Response("", { status: 500 }) : json({ detail: "No job" }, 404)) as unknown as typeof fetch);
  expect(events.at(-1)).toMatchObject({ type: "failed-to-start" });
});

test("polling that never finishes gives up with an idle link, not a spinner", async () => {
  const events = await watch((async (path: string) =>
    path.endsWith("/events") ? new Response("", { status: 500 }) : json({ ...final, state: "running" })) as unknown as typeof fetch);
  expect(events.at(-1)).toEqual({ type: "link", link: "idle" });
});

test("an abort stops the stream without polling or events", async () => {
  const controller = new AbortController();
  const fetchImpl = (async (_p: string, init?: RequestInit) =>
    new Promise<Response>((_res, rej) => init?.signal?.addEventListener("abort", () => rej(new DOMException("a", "AbortError"))))) as unknown as typeof fetch;
  const pending = watch(fetchImpl, controller.signal);
  controller.abort();
  expect(await pending).toEqual([{ type: "link", link: "stream" }]);
});

test("the stream is read to its end after the final event, not cancelled while open", async () => {
  const cancelled = vi.fn();
  const enc = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(enc.encode(`event: done\ndata: ${JSON.stringify(final)}\n\n`));
      setTimeout(() => c.close(), 20);
    },
    cancel: cancelled,
  });
  const events = await watch((async () => new Response(body)) as unknown as typeof fetch);
  expect(events.map((e) => e.type)).toEqual(["link", "snapshot"]);
  expect(cancelled).not.toHaveBeenCalled();
});
