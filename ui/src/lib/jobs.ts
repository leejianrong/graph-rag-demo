import type { components } from "../api/schema";

/** The pipeline's stages in order, as the job's `stages` list names them (graph_rag/orchestrator.py). */
export const STAGES = [
  "read",
  "entity recognition",
  "coreference",
  "entity linking",
  "knowledge graph",
] as const;

/** The job from `GET /jobs/{id}` and the final `done`/`failed` event, as the API schema has it. */
export type JobJson = components["schemas"]["Job"];

export type JobState = "submitting" | "queued" | "running" | "done" | "failed" | "unchanged";

/** What the user submitted, kept so a failed job can be ingested again. */
export type Source =
  | { kind: "url"; url: string }
  | { kind: "file"; file: File | Blob; name: string };

/** One job in the list. */
export interface TrackedJob {
  id: number;
  name: string;
  source: Source;
  documentId: string | null;
  state: JobState;
  stages: string[];
  error: string | null;
  /** "polling" once the live stream dropped and the job is read by polling instead. */
  link: "stream" | "polling" | "idle";
}

export type JobEvent =
  | { type: "accepted"; documentId: string; status: string }
  | { type: "stage"; stage: string }
  | { type: "snapshot"; job: JobJson }
  | { type: "link"; link: TrackedJob["link"] }
  | { type: "failed-to-start"; error: string };

export function newJob(id: number, name: string, source: Source): TrackedJob {
  return {
    id,
    name,
    source,
    documentId: null,
    state: "submitting",
    stages: [],
    error: null,
    link: "idle",
  };
}

/** A finished job never moves again, whatever arrives late. */
const finished = (job: TrackedJob) =>
  job.state === "done" || job.state === "failed" || job.state === "unchanged";

/** The pure step from a job and one event to the next job. Replayed stages are not added twice. */
export function reduceJob(job: TrackedJob, event: JobEvent): TrackedJob {
  switch (event.type) {
    case "accepted":
      return {
        ...job,
        documentId: event.documentId,
        state: event.status === "unchanged" ? "unchanged" : "queued",
      };
    case "failed-to-start":
      return { ...job, state: "failed", error: event.error };
    case "link":
      return finished(job) ? job : { ...job, link: event.link };
    case "stage":
      if (finished(job) || job.stages.includes(event.stage)) return job;
      return { ...job, state: "running", stages: [...job.stages, event.stage] };
    case "snapshot": {
      if (finished(job)) return job;
      const seen = event.job.stages ?? [];
      return {
        ...job,
        state: event.job.state,
        stages: seen.length >= job.stages.length ? seen : job.stages,
        error: event.job.state === "failed" ? (event.job.error ?? "The job failed.") : null,
      };
    }
  }
}

export type SegmentState = "done" | "running" | "failed" | "pending";

/** The state of each of the five stage segments for a job. */
export function segments(job: TrackedJob): { stage: string; state: SegmentState }[] {
  const lastStage = job.stages[job.stages.length - 1];
  const last = lastStage === undefined ? -1 : STAGES.findIndex((s) => s === lastStage);
  return STAGES.map((stage, i) => {
    let state: SegmentState = "pending";
    if (job.state === "done") state = "done";
    else if (job.state === "failed" || job.state === "running") {
      const here = job.state === "failed" ? "failed" : "running";
      state = i < last ? "done" : i === last ? here : "pending";
    }
    return { stage, state };
  });
}

/** The chip text for a job's current or final state. */
export function statusText(job: TrackedJob): string {
  switch (job.state) {
    case "submitting":
      return "Sending";
    case "queued":
      return "Queued";
    case "running":
      return `Running · ${job.stages[job.stages.length - 1] ?? "starting"}`;
    case "done":
      return "Done";
    case "failed": {
      const last = job.stages[job.stages.length - 1];
      return last ? `Failed · ${last}` : "Failed";
    }
    case "unchanged":
      return "Unchanged · nothing to do";
  }
}
