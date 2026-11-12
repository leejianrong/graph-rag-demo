import { expect, test } from "vitest";
import { SseParser, readEvents } from "./sse";

test("one event in one chunk", () => {
  expect(new SseParser().push('event: stage\ndata: {"stage":"read"}\n\n')).toEqual([
    { event: "stage", data: '{"stage":"read"}' },
  ]);
});

test("several events in one chunk", () => {
  const out = new SseParser().push("event: stage\ndata: a\n\nevent: stage\ndata: b\n\n");
  expect(out.map((e) => e.data)).toEqual(["a", "b"]);
});

test("an event split at any byte arrives whole, once", () => {
  const text = "event: done\ndata: {\"x\":1}\n\n";
  for (let cut = 1; cut < text.length; cut++) {
    const p = new SseParser();
    expect([...p.push(text.slice(0, cut)), ...p.push(text.slice(cut))]).toEqual([
      { event: "done", data: '{"x":1}' },
    ]);
  }
});

test("comments and unknown fields are skipped, and a comment alone makes no event", () => {
  const p = new SseParser();
  expect(p.push(": keep-alive\n\n")).toEqual([]);
  expect(p.push("id: 7\n: note\nevent: stage\ndata: a\n\n")).toEqual([{ event: "stage", data: "a" }]);
});

test("CRLF line ends, multi-line data and a default event name", () => {
  expect(new SseParser().push("data: a\r\ndata: b\r\n\r\n")).toEqual([{ event: "message", data: "a\nb" }]);
});

test("readEvents ends when the stream closes and drops a half frame", async () => {
  const enc = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(enc.encode("event: stage\nda"));
      c.enqueue(enc.encode("ta: a\n\nevent: stage\ndata: b"));
      c.close();
    },
  });
  const seen: string[] = [];
  for await (const e of readEvents(body)) seen.push(e.data);
  expect(seen).toEqual(["a"]);
});
