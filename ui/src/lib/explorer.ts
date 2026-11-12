import type { components } from "../api/schema";

/** Generated API shapes the explorer works with. */
export type EntitySummary = components["schemas"]["EntitySummary"];
export type Neighbourhood = components["schemas"]["EntityNeighbourhood"];
export type Neighbour = components["schemas"]["Neighbour"];

/** Why a call failed, in words the view can show next to the control that caused it. */
export interface ExplorerError {
  status: number;
  message: string;
}

/** One entity visited in the neighbourhood view. */
export interface Step {
  id: string;
  name: string;
}

/** A search: the text, the results so far and where the next page starts. */
export interface SearchState {
  q: string;
  results: EntitySummary[];
  cursor: string | null;
  /** True once a page for `q` has arrived, so an empty list can mean "no match". */
  loaded: boolean;
}

export const emptySearch = (q = ""): SearchState => ({ q, results: [], cursor: null, loaded: false });

/** A first page replaces the list; a later page extends it without repeating an id. */
export function applyPage(
  state: SearchState,
  page: { entities: EntitySummary[]; next_cursor?: string | null },
  append: boolean,
): SearchState {
  const seen = new Set(append ? state.results.map((e) => e.canonical_id) : []);
  const fresh = page.entities.filter((e) => !seen.has(e.canonical_id));
  return {
    ...state,
    results: append ? [...state.results, ...fresh] : fresh,
    cursor: page.next_cursor ?? null,
    loaded: true,
  };
}

/** The entities the user has centred on, oldest first; the last one is the focus. */
export type Trail = Step[];

/** Centre on `step`. A search pick (`fresh`) starts a new trail; a neighbour click extends it; re-visiting an id cuts back to it. */
export function visit(trail: Trail, step: Step, fresh = false): Trail {
  if (fresh) return [step];
  const at = trail.findIndex((s) => s.id === step.id);
  return at === -1 ? [...trail, step] : trail.slice(0, at + 1);
}

/** Go back to the crumb at `index`, dropping everything after it. */
export function backTo(trail: Trail, index: number): Trail {
  return index < 0 || index >= trail.length ? trail : trail.slice(0, index + 1);
}

/** The edges split by `direction`, in the order the API sent them. */
export function splitNeighbours(detail: Pick<Neighbourhood, "neighbours">): {
  incoming: Neighbour[];
  outgoing: Neighbour[];
} {
  return {
    incoming: detail.neighbours.filter((n) => n.direction === "incoming"),
    outgoing: detail.neighbours.filter((n) => n.direction === "outgoing"),
  };
}

/** How many edges of a side the response left out; never negative. */
export const moreCount = (total: number, shown: number): number => Math.max(0, total - shown);

/** "characters 120-158", or null when the API gave no offsets. They locate the sentence in its document. */
export function spanLabel(start: number | null | undefined, end: number | null | undefined): string | null {
  return start == null || end == null || end <= start ? null : `characters ${start}–${end}`;
}

/** "2026-01-05 to 2026-03-01", "from 2026-01-05", "until 2026-03-01" or null when neither end is set. */
export function validity(from?: string | null, to?: string | null): string | null {
  if (from && to) return `${from} to ${to}`;
  if (from) return `from ${from}`;
  if (to) return `until ${to}`;
  return null;
}

/** The date part of the API's timestamp, or "not recorded". */
export const firstSeen = (value?: string | null): string => (value ? value.slice(0, 10) : "not recorded");

/** A message for a failed explorer call, by status (0 is a network failure). */
export function describeError(status: number, apiMessage?: string): ExplorerError {
  const message =
    status === 400
      ? `The API rejected the request: ${apiMessage ?? "invalid input"}`
      : status === 404
        ? "That entity is not in this namespace. It may have been removed with its last document."
        : status === 429
          ? "Too many requests. Wait a moment and try again."
          : status === 503
            ? "This service cannot browse entities yet. Ask your operator to enable the explorer."
            : status === 0
              ? "The API did not answer. Check that the service is running."
              : `The API answered ${status}. Try again in a moment.`;
  return { status, message };
}
