import { authedFetch } from "../api/client";
import { session, signOut } from "./auth.svelte";
import { submitSource, watchJob, type SubmitResult } from "./ingest";
import { newJob, reduceJob, type JobEvent, type Source, type TrackedJob } from "./jobs";

/**
 * The jobs submitted in this tab, newest first. The API has no list-jobs route, so this is
 * session-local: a reload or a sign-out empties it, and jobs from other tabs or clients are absent.
 */
export const jobs = $state<{ list: TrackedJob[] }>({ list: [] });

const fetchAs = authedFetch({
  key: () => session.key,
  onUnauthorized: () => signOut("Your key was rejected, so you were signed out. Sign in again."),
});

const streams = new Map<number, AbortController>();
let nextId = 1;

function apply(id: number, event: JobEvent) {
  const i = jobs.list.findIndex((j) => j.id === id);
  if (i !== -1) jobs.list[i] = reduceJob(jobs.list[i], event);
}

function watch(id: number, documentId: string) {
  streams.get(id)?.abort();
  const controller = new AbortController();
  streams.set(id, controller);
  void watchJob(documentId, {
    fetchImpl: fetchAs,
    signal: controller.signal,
    onEvent: (event) => apply(id, event),
  }).finally(() => {
    if (streams.get(id) === controller) streams.delete(id);
  });
}

/** Abort every open stream, keeping the list. The view calls it when it unmounts. */
export function pauseStreams() {
  for (const controller of streams.values()) controller.abort();
  streams.clear();
}

/** Follow again the jobs that were still running when the view was left. */
export function resumeStreams() {
  for (const job of jobs.list) {
    if (job.documentId && (job.state === "queued" || job.state === "running")) {
      watch(job.id, job.documentId);
    }
  }
}

/** Stop every stream and forget every job. Runs on sign-out. */
export function clearJobs() {
  pauseStreams();
  jobs.list = [];
}

/** Send a source and, if the API accepts it, add a job and follow it live. */
export async function ingest(name: string, source: Source, force = false): Promise<SubmitResult> {
  const result = await submitSource(source, fetchAs, force);
  if (!result.ok) return result;
  const id = nextId++;
  jobs.list.unshift(
    reduceJob(newJob(id, name, source), {
      type: "accepted",
      documentId: result.documentId,
      status: result.status,
    }),
  );
  if (result.status === "accepted") watch(id, result.documentId);
  return result;
}

/** Ingest a failed job's source again, as a new job; `force` because the first attempt was kept. */
export function retry(job: TrackedJob) {
  return ingest(job.name, job.source, true);
}

/** Read a job whose live link gave out. */
export function recheck(job: TrackedJob) {
  if (job.documentId) watch(job.id, job.documentId);
}

// Sign-out, from the button or from a rejected key, ends every stream and empties the list.
$effect.root(() => {
  $effect(() => {
    if (session.key === null) clearJobs();
  });
});
