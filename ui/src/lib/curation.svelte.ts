import { tick } from "svelte";
import { entityDetail, mergeEntities, searchEntities, splitEntity } from "../api/client";
import { api, session } from "./auth.svelte";
import {
  describeCurationError,
  guardMerge,
  mergeAnnouncement,
  splitAnnouncement,
  trailAfterMerge,
  MAX_MERGE,
  type Explained,
  type Failure,
  type MergedFrom,
  type Operation,
} from "./curation";
import type { EntitySummary, Neighbourhood } from "./explorer";
import { announce, centreOn, explorer, retryDetail, retrySearch } from "./explorer.svelte";

export type Mode = "idle" | "pick" | "confirm-merge" | "confirm-split";

/** The merge and split flows for the focused entity: a picker, then a confirmation, then one call. */
export const curation = $state<{
  mode: Mode;
  q: string;
  results: EntitySummary[];
  loaded: boolean;
  searching: boolean;
  searchProblem: string | null;
  selected: EntitySummary[];
  /** The detail of each selected entity, for the "what moves" lines; null while it loads. */
  previews: Neighbourhood[] | null;
  target: MergedFrom | null;
  busy: boolean;
  failure: Explained | null;
}>({
  mode: "idle",
  q: "",
  results: [],
  loaded: false,
  searching: false,
  searchProblem: null,
  selected: [],
  previews: null,
  target: null,
  busy: false,
  failure: null,
});

const PICK_SIZE = 20;
const DEBOUNCE_MS = 250;
let seq = 0;
let timer: ReturnType<typeof setTimeout> | undefined;
let opener: HTMLElement | null = null;
let fallback: () => HTMLElement | null = () => null;

const namespace = () => session.namespace ?? undefined;
const keep = () => explorer.detail;

/** Tell the flows where focus goes when the control that opened one has gone (the centre box). */
export function setFocusFallback(f: () => HTMLElement | null) {
  fallback = f;
}

async function restoreFocus() {
  await tick();
  const el = opener?.isConnected ? opener : fallback();
  el?.focus();
  opener = null;
}

/** The name to show for an id, from whatever the UI has loaded. */
export function nameOf(id: string): string {
  const known =
    curation.selected.find((e) => e.canonical_id === id) ??
    curation.results.find((e) => e.canonical_id === id) ??
    explorer.detail?.merged_from.find((m) => m.canonical_id === id);
  return known?.name ?? explorer.trail.find((s) => s.id === id)?.name ?? id;
}

function reset() {
  clearTimeout(timer);
  seq++;
  curation.mode = "idle";
  curation.q = "";
  curation.results = [];
  curation.loaded = false;
  curation.searching = false;
  curation.searchProblem = null;
  curation.selected = [];
  curation.previews = null;
  curation.target = null;
  curation.busy = false;
  curation.failure = null;
}

/** Close whatever is open. `refocus` returns keyboard focus to the control that opened it. */
export function closeCuration(refocus = true) {
  const had = curation.mode !== "idle";
  reset();
  if (had && refocus) void restoreFocus();
  else opener = null;
}

async function runPickerSearch(q: string) {
  const mine = ++seq;
  curation.searching = true;
  curation.searchProblem = null;
  const r = await searchEntities(api, { q: q || undefined, limit: PICK_SIZE, namespace: namespace() });
  if (mine !== seq) return;
  curation.searching = false;
  if (r.ok) {
    curation.results = r.data.entities.filter((e) => e.canonical_id !== keep()?.canonical_id);
    curation.loaded = true;
  } else {
    curation.searchProblem = describeCurationError("merge", r).message;
  }
}

/** Open the picker for entities to merge into the focused one. `from` is the button to return focus to. */
export function openPicker(from: HTMLElement | null) {
  opener = from;
  reset();
  curation.mode = "pick";
  void runPickerSearch("");
}

/** Search the picker after a short pause; `now` skips it. */
export function setPickerQuery(q: string, now = false) {
  clearTimeout(timer);
  curation.q = q;
  curation.loaded = false;
  if (now) void runPickerSearch(q.trim());
  else timer = setTimeout(() => void runPickerSearch(q.trim()), DEBOUNCE_MS);
}

/** Add or remove `e` from the selection, up to the merge limit. */
export function toggleSelected(e: EntitySummary) {
  const has = curation.selected.some((s) => s.canonical_id === e.canonical_id);
  if (has) curation.selected = curation.selected.filter((s) => s.canonical_id !== e.canonical_id);
  else if (curation.selected.length < MAX_MERGE) curation.selected = [...curation.selected, e];
}

/** The reason the current selection cannot be merged, or null when it can. */
export function selectionProblem(): string | null {
  const k = keep();
  if (!k) return "Open an entity first.";
  const g = guardMerge(k, curation.selected);
  return g.ok ? null : g.message;
}

/** Fetch what each selected entity holds, then show the confirmation. */
export async function reviewMerge() {
  const k = keep();
  if (!k || selectionProblem()) return;
  curation.busy = true;
  curation.failure = null;
  const answers = await Promise.all(
    curation.selected.map((e) => entityDetail(api, e.canonical_id, { namespace: namespace() })),
  );
  curation.busy = false;
  const details: Neighbourhood[] = [];
  for (const a of answers) {
    if (!a.ok) {
      curation.failure = describeCurationError("merge", a, nameOf);
      return;
    }
    details.push(a.data);
  }
  curation.previews = details;
  curation.mode = "confirm-merge";
}

/** Go back from the confirmation to the picker. */
export function backToPicker() {
  curation.mode = "pick";
  curation.previews = null;
  curation.failure = null;
}

function fail(op: Operation, f: Failure) {
  curation.failure = describeCurationError(op, f, nameOf);
  // The graph moved under us: reload what is shown so the next try starts from the truth.
  if (f.status === 404 || f.status === 409) {
    retryDetail();
    retrySearch();
  }
}

/** Send the merge. Success refreshes the detail and says what happened; failure stays on the panel. */
export async function confirmMerge() {
  const k = keep();
  if (!k || curation.busy || curation.mode !== "confirm-merge") return;
  curation.busy = true;
  curation.failure = null;
  const r = await mergeEntities(api, k.canonical_id, curation.selected.map((e) => e.canonical_id), namespace());
  curation.busy = false;
  if (!r.ok) return fail("merge", r);
  const gone = [...r.data.merged, ...r.data.already_merged];
  explorer.trail = trailAfterMerge(explorer.trail, gone);
  announce(mergeAnnouncement(k.name, r.data.merged.length, r.data.already_merged.length));
  reset();
  retryDetail();
  retrySearch();
  void restoreFocus();
}

/** Show the confirmation for splitting `item` out of the focused entity. */
export function startSplit(item: MergedFrom, from: HTMLElement | null) {
  opener = from;
  reset();
  curation.target = item;
  curation.mode = "confirm-split";
}

/** Send the split with the merge id and entity id from `merged_from`. */
export async function confirmSplit() {
  const k = keep();
  const t = curation.target;
  if (!k || !t || curation.busy || curation.mode !== "confirm-split") return;
  curation.busy = true;
  curation.failure = null;
  const r = await splitEntity(api, k.canonical_id, t.merge_id, [t.canonical_id], namespace());
  curation.busy = false;
  if (!r.ok) return fail("split", r);
  announce(splitAnnouncement(k.name, r.data.split.map((e) => e.name), r.data.already_split.length));
  reset();
  retryDetail();
  retrySearch();
  void restoreFocus();
}

/** Open an entity named by an error's action button (the survivor, say). */
export function openEntity(id: string) {
  closeCuration(false);
  centreOn({ id, name: nameOf(id) });
}
