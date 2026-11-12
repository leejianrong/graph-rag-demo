/** One server-sent event: its `event:` name (default "message") and its joined `data:` lines. */
export interface SseEvent {
  event: string;
  data: string;
}

/**
 * An incremental parser for a text/event-stream body. Feed it decoded text as chunks arrive, in
 * any split; it returns the events completed so far. Comment lines (`: keep-alive`) and unknown
 * fields are skipped, `\r\n` and `\n` both end a line, and a half-received frame waits for more.
 */
export class SseParser {
  private buffer = "";
  private event = "";
  private data: string[] = [];

  push(chunk: string): SseEvent[] {
    this.buffer += chunk;
    const out: SseEvent[] = [];
    let start = 0;
    for (;;) {
      const match = /\r\n|\n|\r/.exec(this.buffer.slice(start));
      if (!match) break;
      const end = start + match.index;
      // A lone "\r" at the very end may be the first half of "\r\n": wait for the next chunk.
      if (match[0] === "\r" && end + 1 === this.buffer.length) break;
      this.line(this.buffer.slice(start, end), out);
      start = end + match[0].length;
    }
    this.buffer = this.buffer.slice(start);
    return out;
  }

  private line(line: string, out: SseEvent[]) {
    if (line === "") {
      if (this.data.length > 0 || this.event) {
        out.push({ event: this.event || "message", data: this.data.join("\n") });
      }
      this.event = "";
      this.data = [];
      return;
    }
    if (line.startsWith(":")) return;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") this.event = value;
    else if (field === "data") this.data.push(value);
  }
}

/** Read a response body as events until it closes (or the reader is cancelled by an abort). */
export async function* readEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<SseEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      yield* parser.push(decoder.decode(value, { stream: true }));
    }
  } finally {
    reader.cancel().catch(() => {});
  }
}
