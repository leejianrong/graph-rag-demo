"""Do the generated documents say what the world says, and nothing more? (GRA-122)

The renderer already checks, in code, that each fact's sentence names its object, that its subject
was named, and that no other entity of the world appears. It cannot check the relation itself
(direction, tense, years) or claims that name nobody. A model reads each document once against the
facts it should state and replies, per fact, ``STATED``, ``WRONG`` (said differently or reversed)
or ``MISSING``, and lists specific claims about the entities that are not among the facts.

The judge is one model's opinion, so it is checked on documents altered on purpose: a fact's
sentence removed (the judge should say ``MISSING``) and a fact reversed in the list (``WRONG``).
Nothing runs unless a caller supplies an endpoint or a committed cache.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

from kggen.llm import CachedChat
from kggen.render import Document
from kggen.render_llm import statement
from kggen.world import World

from kgbench.charts import grouped_bars, wilson

VERDICTS = ("STATED", "WRONG", "MISSING")
_SYSTEM = "You check a short article against a list of facts. Follow the reply format exactly."
_ASK = (
    "Article:\n{text}\n\nFacts the article should state (an entity may appear under an alias: "
    "{aliases}):\n{facts}\n\n"
    "For each numbered fact, find the words of the article that state it, then give a verdict. "
    "STATED: the article states it with the same relation, the same direction (who does it to "
    "whom), the same tense and the same years. WRONG: the article says something different or "
    'opposite about those entities ("A owns B" is not "B owns A"). MISSING: it is not in the '
    "article. Then read the article again, clause by clause, and list every clause that adds "
    'detail about an entity beyond the facts, including descriptions such as "a product that '
    'buyers compare" or "which has drawn attention". Only remarks that mention no entity (and '
    'no "it", "the company" or "the product") may be ignored. Reply in exactly this form:\n'
    '1: STATED | "words from the article"\n2: MISSING | none\nExtra: NONE\n'
    "(for the last line write NONE, or the clauses separated by ' ; ')."
)
_LINE = re.compile(
    r"^\s*(\d+)\s*[:.)-]\s*\**\s*(STATED|WRONG|MISSING)\b", re.IGNORECASE | re.MULTILINE
)
_EXTRA = re.compile(r"^\s*\**Extra\**\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


def facts_of(doc: Document, world: World) -> list[str]:
    """The statements of the facts a document carries, in order of first appearance."""
    names = {e.id: e.name for e in world.entities}
    edges = {e.id: e for e in world.edges}
    seen: list[str] = []
    for sentence in doc.sentences:
        seen += [f for f in sentence.fact_ids if f not in seen]
    return [
        statement(edges[f], names[edges[f].subject], names[edges[f].object], world.spec.start_year)
        for f in seen
    ]


def _aliases(doc: Document, world: World) -> str:
    entities = {e.id: e for e in world.entities}
    ids = {m.entity_id for s in doc.sentences for m in s.mentions}
    return (
        "; ".join(
            f"{entities[i].name} = {', '.join(entities[i].aliases)}"
            for i in sorted(ids)
            if entities[i].aliases
        )
        or "none"
    )


def messages(text: str, facts: Sequence[str], aliases: str) -> list[dict[str, str]]:
    listed = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1))
    prompt = _ASK.format(text=text, facts=listed, aliases=aliases)
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]


def parse(reply: str, n: int) -> tuple[list[str], list[str]] | None:
    """Per-fact verdicts and the unlisted claims, or ``None`` unless every fact has a verdict and
    the ``Extra`` line is there."""
    found = {int(i): v.upper() for i, v in _LINE.findall(reply)}
    extra = _EXTRA.search(reply)
    if extra is None or set(found) != set(range(1, n + 1)):
        return None
    claims = [c.strip() for c in extra.group(1).split(";") if c.strip()]
    return [found[i] for i in range(1, n + 1)], [] if claims == ["NONE"] else claims


def check(chat: CachedChat, text: str, facts: Sequence[str], aliases: str):
    """One document against its facts; a reply that does not parse is never cached."""
    prompt = messages(text, facts, aliases)
    reply = chat.ask(prompt)
    parsed = parse(reply, len(facts))
    if parsed is not None and chat.cached(prompt) is None:
        chat.store(prompt, reply)
    return parsed


def _without_fact(doc: Document, world: World) -> tuple[str, list[str], int] | None:
    """The text with the one-fact sentence of the first fact that has one removed, and the index
    (in the fact list) the judge should now call missing."""
    facts = facts_of(doc, world)
    order: list[str] = []
    for s in doc.sentences:
        order += [f for f in s.fact_ids if f not in order]
    for index, fact_id in enumerate(order):
        alone = [s for s in doc.sentences if s.fact_ids == (fact_id,)]
        if alone and len(facts) > 1:
            return " ".join(s.text for s in doc.sentences if s is not alone[0]), facts, index
    return None


def _reversed_fact(doc: Document, world: World) -> tuple[list[str], int]:
    """The facts with the first one's subject and object swapped, and its index."""
    names = {e.id: e.name for e in world.entities}
    edges = {e.id: e for e in world.edges}
    first = next(f for s in doc.sentences for f in s.fact_ids)
    edge = edges[first]
    facts = facts_of(doc, world)
    facts[0] = statement(edge, names[edge.object], names[edge.subject], world.spec.start_year)
    return facts, 0


def evaluate(
    chat: CachedChat,
    documents: Sequence[Document],
    world: World,
    controls: int = 15,
    workers: int = 4,
) -> dict:
    """Check every document, plus ``controls`` altered ones of each kind, in a fixed order."""
    real = [d for d in documents if any(s.fact_ids for s in d.sentences)]

    def doc_row(doc: Document) -> dict:
        facts = facts_of(doc, world)
        parsed = check(chat, doc.text, facts, _aliases(doc, world))
        return {"id": doc.id, "facts": facts, "parsed": parsed, "distractor": doc.distractor}

    lacking = [(d, c) for d in real if (c := _without_fact(d, world))][:controls]
    flipped = [d for d in real if _edge_predicate(d, world) != "AFFILIATED_WITH"][:controls]

    def missing_row(item: tuple) -> dict:
        doc, (text, facts, index) = item
        parsed = check(chat, text, facts, _aliases(doc, world))
        return {
            "id": doc.id,
            "kind": "missing",
            "expected": "MISSING",
            "index": index,
            "parsed": parsed,
        }

    def wrong_row(doc: Document) -> dict:
        facts, index = _reversed_fact(doc, world)
        parsed = check(chat, doc.text, facts, _aliases(doc, world))
        return {
            "id": doc.id,
            "kind": "wrong",
            "expected": "WRONG",
            "index": index,
            "parsed": parsed,
        }

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(doc_row, real))
        control_rows = list(pool.map(missing_row, lacking)) + list(pool.map(wrong_row, flipped))
    return {"documents": rows, "controls": control_rows}


def _edge_predicate(doc: Document, world: World) -> str:
    edges = {e.id: e for e in world.edges}
    return edges[next(f for s in doc.sentences for f in s.fact_ids)].predicate


def word_order(documents: Sequence[Document], world: World) -> dict[str, int]:
    """A check by code, no model: in sentences that state exactly one fact, is the subject named
    before the object? A reversed or passive sentence would fail it. Sentences that name the
    subject only by pronoun or description are counted separately."""
    edges = {e.id: e for e in world.edges}
    out = {"sentences": 0, "object_first": 0, "subject_not_named": 0}
    for doc in documents:
        for s in doc.sentences:
            if len(s.fact_ids) != 1:
                continue
            edge = edges[s.fact_ids[0]]
            subject = [m.start for m in s.mentions if m.entity_id == edge.subject]
            obj = [m.start for m in s.mentions if m.entity_id == edge.object]
            out["sentences"] += 1
            out["object_first"] += bool(subject and obj and min(obj) < min(subject))
            out["subject_not_named"] += not subject
    return out


def _share(hit: int, total: int) -> str:
    low, high = wilson(hit, total)
    return f"{hit / total:.1%} ({hit}/{total}, {low:.0%}–{high:.0%})" if total else "n/a"


def tally(result: Mapping) -> dict:
    """The counts the report and chart use."""
    rows = [r for r in result["documents"] if r["parsed"]]
    verdicts = [v for r in rows for v in r["parsed"][0]]
    with_extra = [r for r in rows if r["parsed"][1]]
    controls = [c for c in result["controls"] if c["parsed"]]
    caught = {kind: [c for c in controls if c["kind"] == kind] for kind in ("missing", "wrong")}
    return {
        "documents": len(result["documents"]),
        "unparsed": sum(1 for r in result["documents"] if not r["parsed"]),
        "facts": len(verdicts),
        "verdicts": {v: verdicts.count(v) for v in VERDICTS},
        "docs_with_extra": len(with_extra),
        "docs_checked": len(rows),
        "control": {
            kind: (sum(c["parsed"][0][c["index"]] == c["expected"] for c in items), len(items))
            for kind, items in caught.items()
        },
        "claims": [(r["id"], claim) for r in with_extra for claim in r["parsed"][1]],
    }


def format_table(result: Mapping) -> str:
    t = tally(result)
    v = t["verdicts"]
    lines = [
        "| measure | share |",
        "|---|---|",
        f"| listed facts stated | {_share(v['STATED'], t['facts'])} |",
        f"| listed facts judged wrong | {_share(v['WRONG'], t['facts'])} |",
        f"| listed facts judged missing | {_share(v['MISSING'], t['facts'])} |",
        f"| documents with an unlisted claim | {_share(t['docs_with_extra'], t['docs_checked'])} |",
        f"| control: removed fact called MISSING | {_share(*t['control']['missing'])} |",
        f"| control: reversed fact called WRONG | {_share(*t['control']['wrong'])} |",
    ]
    if t["unparsed"]:
        lines.append(f"\n{t['unparsed']} documents got no valid reply and are left out.")
    return "\n".join(lines)


def chart_svg(result: Mapping, title: str) -> str:
    t = tally(result)
    bars = {
        "facts stated": (t["verdicts"]["STATED"], t["facts"]),
        "documents with no unlisted claim": (
            t["docs_checked"] - t["docs_with_extra"],
            t["docs_checked"],
        ),
        "control: missing found": t["control"]["missing"],
        "control: reversal found": t["control"]["wrong"],
    }
    names = list(bars)
    values = [[a / b if b else 0.0 for a, b in (bars[n],)] for n in names]
    return grouped_bars(
        title,
        names,
        {"share": [v[0] for v in values]},
        ylabel="share of cases",
        intervals={"share": [wilson(*bars[n]) for n in names]},
        counts=[f"n={bars[n][1]}" for n in names],
    )
