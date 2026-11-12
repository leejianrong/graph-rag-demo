import { entityDetail, searchEntities } from "../api/client";
import { api, session } from "./auth.svelte";
import {
  applyPage,
  backTo,
  describeError,
  emptySearch,
  visit,
  type EntitySummary,
  type ExplorerError,
  type Neighbourhood,
  type SearchState,
  type Step,
  type Trail,
} from "./explorer";

/** Explorer state for this tab. Search and the focused entity load independently, each with its own error. */
export const explorer = $state<{
  search: SearchState;
  searching: boolean;
  searchError: ExplorerError | null;
  trail: Trail;
  detail: Neighbourhood | null;
  loadingDetail: boolean;
  detailError: ExplorerError | null;
  /** A polite one-line outcome (a merge, a split, a redirect) for the view's live region. */
  notice: string | null;
}>({
  search: emptySearch(),
  searching: false,
  searchError: null,
  trail: [],
  detail: null,
  loadingDetail: false,
  detailError: null,
  notice: null,
});

export const PAGE_SIZE = 20;
const DEBOUNCE_MS = 250;

// A later request makes an earlier one's answer stale; these numbers tell them apart.
let searchSeq = 0;
let detailSeq = 0;
let timer: ReturnType<typeof setTimeout> | undefined;

const namespace = () => session.namespace ?? undefined;

async function runSearch(q: string, cursor: string | null) {
  const mine = ++searchSeq;
  explorer.searching = true;
  explorer.searchError = null;
  const result = await searchEntities(api, { q: q || undefined, limit: PAGE_SIZE, cursor: cursor ?? undefined, namespace: namespace() });
  if (mine !== searchSeq) return;
  explorer.searching = false;
  if (result.ok) explorer.search = applyPage({ ...explorer.search, q }, result.data, cursor !== null);
  else explorer.searchError = describeError(result.status, result.message);
}

/** Search for `q` after a short pause, so typing sends one request, not one per key. `now` skips the pause. */
export function setQuery(q: string, now = false) {
  clearTimeout(timer);
  explorer.search = emptySearch(q);
  const trimmed = q.trim();
  if (now) void runSearch(trimmed, null);
  else timer = setTimeout(() => void runSearch(trimmed, null), DEBOUNCE_MS);
}

/** Fetch the next page of the current search. */
export function loadMore() {
  const { q, cursor } = explorer.search;
  if (cursor && !explorer.searching) void runSearch(q.trim(), cursor);
}

/** Retry whatever failed last in the search box. */
export function retrySearch() {
  setQuery(explorer.search.q, true);
}

async function load(id: string) {
  const mine = ++detailSeq;
  explorer.loadingDetail = true;
  explorer.detailError = null;
  const result = await entityDetail(api, id, { namespace: namespace() });
  if (mine !== detailSeq) return;
  explorer.loadingDetail = false;
  if (result.ok) {
    explorer.detail = result.data;
    // A step opened by id alone (a redirect to a survivor) learns its name here.
    const last = explorer.trail[explorer.trail.length - 1];
    if (last && last.id === id && last.name !== result.data.name) {
      explorer.trail = [...explorer.trail.slice(0, -1), { id, name: result.data.name }];
    }
    return;
  }
  const into = result.details?.merged_into;
  if (result.status === 404 && typeof into === "string") {
    // The entity was merged away: show the survivor, not a blank page.
    const gone = explorer.trail[explorer.trail.length - 1]?.name ?? id;
    explorer.trail = visit(explorer.trail.slice(0, -1), { id: into, name: into });
    explorer.notice = `${gone} was merged into another entity, so you are looking at that one.`;
    void load(into);
    return;
  }
  explorer.detailError = describeError(result.status, result.message);
}

/** Open an entity picked from the search results; the trail starts over. */
export function pick(entity: EntitySummary) {
  explorer.notice = null;
  explorer.trail = visit([], { id: entity.canonical_id, name: entity.name }, true);
  void load(entity.canonical_id);
}

/** Centre on a neighbour, keeping the trail so the user can go back. */
export function centreOn(step: Step) {
  explorer.notice = null;
  explorer.trail = visit(explorer.trail, step);
  void load(step.id);
}

/** Return to an earlier entity in the trail. */
export function goBack(index: number) {
  explorer.notice = null;
  explorer.trail = backTo(explorer.trail, index);
  const focus = explorer.trail[explorer.trail.length - 1];
  if (focus) void load(focus.id);
}

/** Say `text` in the view's live region; null clears it. */
export function announce(text: string | null) {
  explorer.notice = text;
}

/** Reload the focused entity, after a failure or a change to it. */
export function retryDetail() {
  const focus = explorer.trail[explorer.trail.length - 1];
  if (focus) void load(focus.id);
}

let owner: string | null = null;

/** Drop what a previous sign-in left behind; the view calls it on mount so one tenant never sees another's entities. */
export function claimExplorer() {
  if (owner !== session.key) resetExplorer();
  owner = session.key;
}

/** Forget everything, for sign-out and tests. */
export function resetExplorer() {
  clearTimeout(timer);
  searchSeq++;
  detailSeq++;
  explorer.search = emptySearch();
  explorer.searching = false;
  explorer.searchError = null;
  explorer.trail = [];
  explorer.detail = null;
  explorer.loadingDetail = false;
  explorer.detailError = null;
  explorer.notice = null;
}
