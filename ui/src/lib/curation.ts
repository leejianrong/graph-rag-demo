import type { components } from "../api/schema";
import type { EntitySummary, Neighbourhood, Trail } from "./explorer";

/** What `merged_from` lists: an entity folded into the focused one, with the merge that did it. */
export type MergedFrom = components["schemas"]["MergedFrom"];

export const MAX_MERGE = 10;

/** A failed curation call as the API reported it (status 0 is a network failure). */
export interface Failure {
  status: number;
  message?: string;
  code?: string;
  details?: Record<string, unknown>;
}

/** What the user sees for a failure: a plain sentence and, when one click helps, a button that opens an entity. */
export interface Explained {
  message: string;
  action?: { label: string; id: string };
}

export type Operation = "merge" | "split";

const ids = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []);
const str = (v: unknown): string | undefined => (typeof v === "string" ? v : undefined);
const list = (xs: string[]) => xs.join(", ");

/** The plain-language message for a failed merge or split, from the status and the API's `details.reason`. `nameOf` turns an id into a name where the UI knows one. */
export function describeCurationError(
  op: Operation,
  f: Failure,
  nameOf: (id: string) => string = (id) => id,
): Explained {
  const d = f.details ?? {};
  const mergedInto = str(d.merged_into);
  switch (f.status) {
    case 0:
      return { message: "The API did not answer, so nothing changed. Check that the service is running." };
    case 400:
      if (d.field === "merge_id") {
        return {
          message: `This entity has several merges to undo (${list(ids(d.merge_ids))}). Split from the Merged from list, which names the merge.`,
        };
      }
      return { message: `The API refused the request: ${f.message ?? "invalid input"}. Nothing changed.` };
    case 401:
      return { message: "Your key was rejected, so you were signed out. Sign in again." };
    case 404:
      if (mergedInto) {
        return {
          message: `That entity was merged into ${nameOf(mergedInto)} in the meantime. Nothing changed.`,
          action: { label: `Open ${nameOf(mergedInto)}`, id: mergedInto },
        };
      }
      return {
        message:
          "One of these entities is no longer in this namespace, so nothing changed. The list has been refreshed.",
      };
    case 409:
      return describeConflict(op, f, nameOf);
    case 429:
      return { message: "Too many requests. Wait a moment and try again. Nothing changed." };
    case 503:
      return {
        message: "Merging and splitting are not switched on for this service. Ask your operator to enable curation.",
      };
    default:
      return {
        message: `The API answered ${f.status}. Refresh the entity to see whether anything changed, then try again.`,
      };
  }
}

function describeConflict(op: Operation, f: Failure, nameOf: (id: string) => string): Explained {
  const d = f.details ?? {};
  const who = ids(d.entity_ids).map(nameOf);
  switch (d.reason) {
    case "type_mismatch":
      return {
        message: `Entities of different types cannot be merged${who.length ? ` (${list(who)})` : ""}. Pick entities of the same type as the one you keep.`,
      };
    case "no_merge_record":
      return {
        message: `There is no merge record to split${who.length ? ` for ${list(who)}` : ""}. It may have been split already; refresh the entity.`,
      };
    case "merge_pending":
      return {
        message:
          "That merge never finished, so it cannot be split yet. Merge the same entities again to finish it, then split.",
      };
    case "survivor_merged": {
      const into = str(d.merged_into);
      return {
        message: `This entity was itself merged into ${into ? nameOf(into) : "another entity"}. Split it out of that one first, then split from here.`,
        action: into ? { label: `Open ${nameOf(into)}`, id: into } : undefined,
      };
    }
    case "entity_exists":
      return {
        message: `Another entity now uses ${who.length ? list(who) : "an id this split needs"}, so it cannot be restored under that id.`,
      };
    case "neighbour_merged": {
      const first = ids(d.entity_ids)[0];
      return {
        message: `An edge to restore ends at ${who.length ? list(who) : "an entity"} that has been merged away since. Split that one out first, then split here.`,
        action: first ? { label: `Open ${nameOf(first)}`, id: first } : undefined,
      };
    }
    default:
      return {
        message: `The ${op} conflicts with the current graph${f.message ? `: ${f.message}` : ""}. Nothing changed.`,
      };
  }
}

export type Guard = { ok: true } | { ok: false; message: string };

/** Whether `picks` may be merged into `keep`: at least one, at most 10, none the keeper itself, no repeats, all of its type. */
export function guardMerge(keep: Pick<EntitySummary, "canonical_id" | "name" | "type">, picks: EntitySummary[]): Guard {
  if (picks.length === 0) return { ok: false, message: "Select at least one entity to merge in." };
  if (picks.length > MAX_MERGE) return { ok: false, message: `Merge at most ${MAX_MERGE} entities at a time.` };
  if (picks.some((p) => p.canonical_id === keep.canonical_id)) {
    return { ok: false, message: "An entity cannot be merged into itself." };
  }
  if (new Set(picks.map((p) => p.canonical_id)).size !== picks.length) {
    return { ok: false, message: "The same entity is selected twice." };
  }
  const odd = picks.filter((p) => p.type !== keep.type);
  if (odd.length) {
    return {
      ok: false,
      message: `${keep.name} is ${keep.type}. ${list(odd.map((p) => `${p.name} (${p.type})`))} ${odd.length === 1 ? "has" : "have"} a different type, and entities of different types cannot be merged.`,
    };
  }
  return { ok: true };
}

const norm = (s: string) => s.trim().toLowerCase();
const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** What a merge moves, worked out from the detail already fetched for the keeper and each entity to drop. */
export interface MergeSummary {
  edges: number;
  documents: number;
  /** Names of the dropped entities (and their aliases) the keeper does not already carry. */
  aliases: string[];
}

export function mergeSummary(keep: Neighbourhood, dropped: Neighbourhood[]): MergeSummary {
  const known = new Set([keep.name, ...keep.aliases].map(norm));
  const aliases: string[] = [];
  for (const d of dropped) {
    for (const n of [d.name, ...d.aliases]) {
      if (!known.has(norm(n))) {
        known.add(norm(n));
        aliases.push(n);
      }
    }
  }
  return {
    edges: dropped.reduce((a, d) => a + d.incoming_total + d.outgoing_total, 0),
    documents: dropped.reduce((a, d) => a + d.sources_total, 0),
    aliases,
  };
}

/** The confirmation lines for a merge: what moves, and what does not. */
export function mergeLines(keepName: string, droppedNames: string[], s: MergeSummary): string[] {
  return [
    `${plural(s.edges, "edge")} and ${plural(s.documents, "document link")} move to ${keepName}.`,
    s.aliases.length
      ? `${plural(s.aliases.length, "name")} become${s.aliases.length === 1 ? "s" : ""} aliases of ${keepName}: ${list(s.aliases)}.`
      : `${keepName} already carries every name, so no aliases are added.`,
    `${list(droppedNames)} will no longer appear as ${droppedNames.length === 1 ? "a separate entity" : "separate entities"} in search or the graph.`,
    "Nothing is deleted. The merge is recorded, so you can split it again later.",
  ];
}

/** The confirmation lines for splitting `item` back out of `keepName`. */
export function splitLines(keepName: string, item: Pick<MergedFrom, "name" | "aliases">): string[] {
  const names = [item.name, ...item.aliases];
  return [
    `${item.name} comes back as its own entity, with the edges, document links and names it had before the merge.`,
    `Names that only it brought leave ${keepName}: ${list(names)}.`,
    `Edges and document links created after the merge stay with ${keepName}.`,
  ];
}

/** The trail without the entities just merged away; the keeper stays. */
export function trailAfterMerge(trail: Trail, mergedIds: string[]): Trail {
  const gone = new Set(mergedIds);
  return trail.filter((s) => !gone.has(s.id));
}

/** The outcome sentence for a merge response; ids an earlier merge had already folded in are explained, not errors. */
export function mergeAnnouncement(keepName: string, merged: number, already: number): string {
  if (merged === 0 && already > 0) {
    return `Nothing to do: ${plural(already, "entity", "entities")} ${already === 1 ? "was" : "were"} already merged into ${keepName}.`;
  }
  const base = `Merged ${plural(merged, "entity", "entities")} into ${keepName}.`;
  return already > 0 ? `${base} ${plural(already, "other")} had already been merged.` : base;
}

/** The outcome sentence for a split response. */
export function splitAnnouncement(keepName: string, restored: string[], already: number): string {
  if (restored.length === 0 && already > 0) return `Nothing to do: that entity was already split out of ${keepName}.`;
  return `Split ${list(restored)} out of ${keepName}.${already > 0 ? " One had already been split." : ""}`;
}
