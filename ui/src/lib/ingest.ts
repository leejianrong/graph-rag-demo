import type { components } from "../api/schema";
import type { JobEvent, JobJson, Source } from "./jobs";
import { readEvents } from "./sse";

/** Types `/ingest` takes as a file; the server decides from the bytes and the name. */
export const FILE_ACCEPT = ".txt,.html,.htm,.pdf,text/plain,text/html,application/pdf";
/** A hint only: the server enforces MAX_UPLOAD_BYTES (10 MB by default) and says so with a 413. */
export const SIZE_HINT_BYTES = 10_000_000;

/** A pasted text goes up as a `text/plain` file; this is the name it gets when the user gives none. */
export function defaultTextName(now = new Date()): string {
  const stamp = now.toISOString().slice(0, 19).replace(/[-:]/g, "").replace("T", "-");
  return `pasted-${stamp}.txt`;
}

/** The form body for `/ingest`: exactly one of the `url` and `file` fields, as the API requires. */
export function ingestBody(source: Source): FormData {
  const body = new FormData();
  if (source.kind === "url") body.set("url", source.url);
  else body.set("file", source.file, source.name);
  return body;
}

export function textSource(text: string, name: string): Source {
  const clean = name.trim() || defaultTextName();
  return { kind: "file", file: new Blob([text], { type: "text/plain" }), name: clean };
}

export type SubmitResult =
  | { ok: true; documentId: string; status: "accepted" | "unchanged" }
  | { ok: false; status: number | null; message: string };

/** The reason the API gave: FastAPI's `detail` string, or our `{error: {message}}` envelope. */
async function reason(response: Response): Promise<string | null> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail) && typeof body.detail[0]?.msg === "string") return body.detail[0].msg;
    if (typeof body?.error?.message === "string") return body.error.message;
  } catch {
    /* not JSON */
  }
  return null;
}

/** Turn a refused `/ingest` into a sentence for the form. */
export async function describeRefusal(response: Response): Promise<string> {
  const detail = await reason(response);
  switch (response.status) {
    case 400:
      return detail ?? "The API refused the request.";
    case 401:
      return "Your key was rejected, so you were signed out. Sign in again.";
    case 413:
      return detail ?? "That upload is over the size limit.";
    case 422:
      return `Nothing was queued. ${detail ?? "No text could be extracted from that source."}`;
    case 429: {
      const wait = Number(response.headers.get("retry-after"));
      return wait > 0
        ? `Too many requests. Wait ${Math.ceil(wait)} seconds and try again.`
        : "Too many requests. Wait a moment and try again.";
    }
    default:
      return detail ?? `The API answered ${response.status}. Try again in a moment.`;
  }
}

/** Send one source to `/ingest`. `fetchImpl` is the authenticated fetch. */
export async function submitSource(
  source: Source,
  fetchImpl: typeof fetch,
  force = false,
): Promise<SubmitResult> {
  let response: Response;
  try {
    response = await fetchImpl(force ? "/ingest?force=true" : "/ingest", {
      method: "POST",
      body: ingestBody(source),
    });
  } catch {
    return { ok: false, status: null, message: "The API did not answer. Check that the service is running." };
  }
  if (!response.ok) {
    return { ok: false, status: response.status, message: await describeRefusal(response) };
  }
  const data = (await response.json()) as components["schemas"]["IngestAccepted"];
  return {
    ok: true,
    documentId: data.document_id,
    status: data.status === "unchanged" ? "unchanged" : "accepted",
  };
}

export interface WatchOptions {
  fetchImpl: typeof fetch;
  signal: AbortSignal;
  onEvent: (event: JobEvent) => void;
  /** Waits `ms`; tests pass an instant one. Rejects never: an abort just ends the wait early. */
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>;
  /** Polling gives up after this many reads (about five minutes at the backoff cap). */
  maxPolls?: number;
}

const wait = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => (clearTimeout(timer), resolve()), { once: true });
  });

const path = (documentId: string, tail = "") => `/jobs/${encodeURIComponent(documentId)}${tail}`;

/**
 * Follow one job to its end: read the SSE stream, and if it drops before `done` or `failed`, poll
 * `GET /jobs/{id}` with backoff. Returns when the job finished, the user aborted or polling gave up.
 */
export async function watchJob(documentId: string, options: WatchOptions): Promise<void> {
  const { signal, onEvent } = options;
  onEvent({ type: "link", link: "stream" });
  if (await stream(documentId, options)) return;
  if (signal.aborted) return;
  onEvent({ type: "link", link: "polling" });
  await poll(documentId, options);
}

/**
 * True when the stream reached a terminal event, or the job is gone. After the terminal event the
 * body is read to its end instead of cancelled: the server closes it right away, and cancelling a
 * body still open makes the browser report the request as failed (GRA-156).
 */
async function stream(documentId: string, { fetchImpl, signal, onEvent }: WatchOptions): Promise<boolean> {
  let terminal = false;
  try {
    const response = await fetchImpl(path(documentId, "/events"), {
      headers: { Accept: "text/event-stream" },
      signal,
    });
    if (!response.ok || !response.body) return false;
    for await (const { event, data } of readEvents(response.body)) {
      if (terminal) continue;
      if (event === "stage") {
        onEvent({ type: "stage", stage: (JSON.parse(data) as components["schemas"]["StageEvent"]).stage });
      } else if (event === "done" || event === "failed") {
        onEvent({ type: "snapshot", job: JSON.parse(data) as JobJson });
        terminal = true;
      }
    }
  } catch {
    /* dropped or aborted: the caller checks the signal */
  }
  return terminal;
}

async function poll(documentId: string, options: WatchOptions): Promise<void> {
  const { fetchImpl, signal, onEvent, sleep = wait, maxPolls = 40 } = options;
  let delay = 1000;
  for (let n = 0; n < maxPolls && !signal.aborted; n++) {
    await sleep(delay, signal);
    if (signal.aborted) return;
    delay = Math.min(delay * 2, 10_000);
    try {
      const response = await fetchImpl(path(documentId), { signal });
      if (response.status === 404) {
        onEvent({
          type: "failed-to-start",
          error: "The service no longer knows this job. It may have restarted; ingest it again.",
        });
        return;
      }
      if (response.status === 401) return; // the auth hook already signed out
      if (!response.ok) continue;
      const job = (await response.json()) as JobJson;
      onEvent({ type: "snapshot", job });
      if (job.state === "done" || job.state === "failed") return;
    } catch {
      /* try again after the next wait */
    }
  }
  if (!signal.aborted) onEvent({ type: "link", link: "idle" });
}
