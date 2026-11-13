import type { components } from "../api/schema";
import type { makeClient } from "../api/client";

export type QueryResponse = components["schemas"]["QueryResponse"];
type Triple = components["schemas"]["Triple"];
type RankedNode = components["schemas"]["RankedNode"];
type SupportingSentence = components["schemas"]["SupportingSentence"];

export interface NodeRef {
  id: string;
  name: string;
  type: string | null;
}

/** One edge walked from `from` to `to`. `forward` is true when the stored edge points the same way. */
export interface Hop {
  from: NodeRef;
  to: NodeRef;
  predicate: string;
  forward: boolean;
  edge: Triple;
}

export interface Path {
  start: NodeRef;
  hops: Hop[];
}

interface Step {
  edge: Triple;
  other: string;
  forward: boolean;
}

/**
 * Shortest paths through the subgraph from each start node to `target`, one per start, fewest hops
 * first (starts in the order given break ties), at most `limit` of them. Edges are walked in either direction (the hop says
 * which way the stored edge points); ties between equally short routes go to the edge that comes
 * first in `edges`, so the result is the same every time. A start that equals the target, is not in
 * the subgraph or cannot reach the target gives no path, and identical paths are listed once.
 */
export function findPaths(
  edges: Triple[],
  starts: NodeRef[],
  target: NodeRef,
  refs: Map<string, NodeRef>,
  limit = 3,
): Path[] {
  const around = new Map<string, Step[]>();
  const add = (id: string, step: Step) => {
    const list = around.get(id);
    if (list) list.push(step);
    else around.set(id, [step]);
  };
  for (const edge of edges) {
    add(edge.subject_id, { edge, other: edge.object_id, forward: true });
    add(edge.object_id, { edge, other: edge.subject_id, forward: false });
  }
  const ref = (id: string): NodeRef => refs.get(id) ?? { id, name: id, type: null };

  const paths: Path[] = [];
  const seen = new Set<string>();
  for (const start of starts) {
    if (start.id === target.id || !around.has(start.id)) continue;
    const via = new Map<string, { prev: string; edge: Triple; forward: boolean }>();
    const visited = new Set([start.id]);
    const queue = [start.id];
    for (let head = 0; head < queue.length && !visited.has(target.id); head++) {
      for (const step of around.get(queue[head]) ?? []) {
        if (visited.has(step.other)) continue;
        visited.add(step.other);
        via.set(step.other, { prev: queue[head], edge: step.edge, forward: step.forward });
        queue.push(step.other);
      }
    }
    if (!via.has(target.id)) continue;
    const hops: Hop[] = [];
    for (let at = target.id; at !== start.id; ) {
      const step = via.get(at)!;
      hops.unshift({
        from: ref(step.prev),
        to: ref(at),
        predicate: step.edge.predicate,
        forward: step.forward,
        edge: step.edge,
      });
      at = step.prev;
    }
    const key = hops.map((h) => `${h.from.id}>${h.predicate}>${h.to.id}`).join("|");
    if (seen.has(key)) continue;
    seen.add(key);
    paths.push({ start: ref(start.id), hops });
  }
  return paths.sort((a, b) => a.hops.length - b.hops.length).slice(0, limit);
}

/** An edge a sentence backs: the edge cites this document and sentence index. */
export interface Support {
  predicate: string;
  subject: NodeRef;
  object: NodeRef;
}

export interface SentenceView {
  key: string;
  text: string;
  documentId: string;
  sentenceIndex: number;
  charStart: number;
  charEnd: number;
  score: number;
  supports: Support[];
}

/** Why the result has nothing (or little) to show, in the terms a user can act on. */
export type Emptiness = "no-match" | "no-path" | null;

export interface QueryView {
  answer: string | null;
  answerNode: NodeRef | null;
  /** The answers to the later asks of a two-part question, in the order asked. */
  moreAnswers: NodeRef[];
  prose: string | null;
  paths: Path[];
  /** Subgraph nodes that no shown path passes through, other than the answer. */
  offPath: NodeRef[];
  sentences: SentenceView[];
  edgeCount: number;
  nodeCount: number;
  empty: Emptiness;
}

/**
 * Turn a `/query` response into what the view shows. Paths start from the entities the question
 * matched (`matched_entity_ids`, best match first) and end at the answer.
 */
export function toView(response: QueryResponse, maxPaths = 3): QueryView {
  const ranked: RankedNode[] = response.ranked_nodes ?? [];
  const nodes = response.subgraph.nodes ?? [];
  const refs = new Map<string, NodeRef>();
  for (const n of ranked) refs.set(n.canonical_id, { id: n.canonical_id, name: n.name, type: n.type });
  for (const n of nodes) refs.set(n.canonical_id, { id: n.canonical_id, name: n.name, type: n.type });

  const answerSource = response.answer_entity ?? ranked[0] ?? null;
  const answerNode = answerSource ? (refs.get(answerSource.canonical_id) ?? null) : null;
  const moreAnswers = (response.answer_entities ?? [])
    .filter((n) => n.canonical_id !== answerNode?.id)
    .map((n) => refs.get(n.canonical_id) ?? { id: n.canonical_id, name: n.name, type: n.type });
  const edges = response.subgraph.edges ?? [];

  const starts = (response.matched_entity_ids ?? [])
    .filter((id) => id !== answerNode?.id && refs.has(id))
    .map((id) => refs.get(id)!);
  const paths = answerNode ? findPaths(edges, starts, answerNode, refs, maxPaths) : [];

  const onPath = new Set<string>(answerNode ? [answerNode.id] : []);
  for (const p of paths) {
    onPath.add(p.start.id);
    for (const h of p.hops) onPath.add(h.to.id);
  }
  const offPath = nodes.filter((n) => !onPath.has(n.canonical_id)).map((n) => refs.get(n.canonical_id)!);

  const nodeRef = (id: string): NodeRef => refs.get(id) ?? { id, name: id, type: null };
  const sentences = (response.supporting_sentences ?? []).map(
    (s: SupportingSentence): SentenceView => ({
      key: `${s.document_id}#${s.sentence_index}`,
      text: s.text,
      documentId: s.document_id,
      sentenceIndex: s.sentence_index,
      charStart: s.char_start,
      charEnd: s.char_end,
      score: s.score,
      supports: edges
        .filter((e) => e.provenance.source_doc_id === s.document_id && e.provenance.sentence_index === s.sentence_index)
        .map((e) => ({ predicate: e.predicate, subject: nodeRef(e.subject_id), object: nodeRef(e.object_id) })),
    }),
  );

  const empty: Emptiness = !answerNode && nodes.length === 0 ? "no-match" : paths.length === 0 ? "no-path" : null;
  return {
    answer: response.answer ?? answerNode?.name ?? null,
    answerNode,
    moreAnswers,
    prose: response.prose ?? null,
    paths,
    offPath,
    sentences,
    edgeCount: edges.length,
    nodeCount: nodes.length,
    empty,
  };
}

/** The 1-based hop of `path` whose edge this sentence backs, or null when it backs none of them. */
export function hopOf(sentence: SentenceView, path: Path | undefined): number | null {
  if (!path) return null;
  const i = path.hops.findIndex((h) =>
    sentence.supports.some(
      (s) => s.predicate === h.predicate && s.subject.id === h.edge.subject_id && s.object.id === h.edge.object_id,
    ),
  );
  return i < 0 ? null : i + 1;
}

export interface Segment {
  text: string;
  hit: boolean;
}

/** Split `text` so every case-insensitive occurrence of a term is its own `hit` segment. */
export function highlight(text: string, terms: string[]): Segment[] {
  const spans: [number, number][] = [];
  const lower = text.toLowerCase();
  for (const term of new Set(terms.map((t) => t.trim().toLowerCase()).filter(Boolean))) {
    for (let at = lower.indexOf(term); at >= 0; at = lower.indexOf(term, at + term.length)) {
      spans.push([at, at + term.length]);
    }
  }
  spans.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
  const merged: [number, number][] = [];
  for (const span of spans) {
    const last = merged[merged.length - 1];
    if (last && span[0] <= last[1]) last[1] = Math.max(last[1], span[1]);
    else merged.push([span[0], span[1]]);
  }
  const out: Segment[] = [];
  let at = 0;
  for (const [a, b] of merged) {
    if (a > at) out.push({ text: text.slice(at, a), hit: false });
    out.push({ text: text.slice(a, b), hit: true });
    at = b;
  }
  if (at < text.length) out.push({ text: text.slice(at), hit: false });
  return out;
}

export type QueryResult =
  | { ok: true; view: QueryView; ms: number }
  | { ok: false; status: number | null; message: string };

/** Ask `/query`. `now` is injectable so tests can fix the clock; latency is measured here, in the browser. */
export async function runQuery(
  client: ReturnType<typeof makeClient>,
  question: string,
  synthesize: boolean,
  now: () => number = () => performance.now(),
): Promise<QueryResult> {
  const q = question.trim();
  if (!q) return { ok: false, status: null, message: "Type a question first." };
  const began = now();
  try {
    const { data, response } = await client.POST("/query", { body: { question: q, synthesize } });
    const ms = Math.round(now() - began);
    if (data) return { ok: true, view: toView(data), ms };
    const s = response.status;
    const message =
      s === 401
        ? "Your key was rejected, so you were signed out."
        : s === 400 || s === 422
          ? "The service could not read that question. Rephrase it and try again."
          : s === 429
            ? "Too many questions in a short time. Wait a minute and try again."
            : s === 503
              ? "The service has no query engine configured, so it cannot answer yet."
              : `The service answered ${s}. Try again in a moment.`;
    return { ok: false, status: s, message };
  } catch {
    return { ok: false, status: null, message: "The API did not answer. Check that the service is running." };
  }
}
