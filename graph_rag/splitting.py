"""What a split puts back, worked out from a merge record alone (GRA-145).

Pure functions, so the in-memory and the real stores are restored the same way. A merge record
lists each dropped entity with the edges and document links it had. ``restore`` is the set of
dropped ids being split out now; ``record.split_ids`` are the ones an earlier split already
restored. An id is *inside* the survivor while it is dropped and not yet split.
"""

from __future__ import annotations

from graph_rag.merge_records import MergeRecord
from graph_rag.merging import edge_key
from graph_rag.models import DocumentRecord, Triple
from graph_rag.normalize import normalize_name

__all__ = ["plan_aliases", "plan_edges", "plan_links", "shadowed_edges"]


def _inside(record: MergeRecord) -> set[str]:
    return {d.canonical_id for d in record.dropped} - set(record.split_ids)


def _at(edge: Triple, inside: set[str], keep: str) -> Triple:
    """``edge`` with every endpoint that lives inside the survivor renamed to it."""
    return edge.model_copy(
        update={
            "subject_id": keep if edge.subject_id in inside else edge.subject_id,
            "object_id": keep if edge.object_id in inside else edge.object_id,
        }
    )


def shadowed_edges(
    dropped_edges: list[Triple], keep: str, drop: set[str], present: list[Triple]
) -> list[Triple]:
    """The survivor's own edges that equal one a merge moves onto it (so the merge skips it)."""
    moved = {edge_key(_at(e, drop, keep)) for e in dropped_edges}
    return [e for e in present if edge_key(e) in moved]


def plan_edges(record: MergeRecord, restore: set[str]) -> tuple[list[Triple], list[Triple]]:
    """``(remove, add)`` for the graph when ``restore`` leaves the survivor.

    Each recorded edge of a restored entity goes back to its original endpoints, except that an
    endpoint still inside the survivor (another dropped entity not being restored) stays the
    survivor. The copy the merge put on the survivor is removed, unless the survivor already owned
    an identical edge (``shadowed_edges``), another still-merged entity's edge maps to the same
    copy, or it was a loop (a merge never writes one). Edges created after the merge are never in
    the record, so they stay.
    """
    keep, inside = record.keep_id, _inside(record)
    after = inside - restore
    originals: dict[tuple[str, str, str, str, int], Triple] = {}
    for dropped in record.dropped:
        if dropped.canonical_id in restore:
            for edge in dropped.edges:
                originals.setdefault(edge_key(edge), edge)
    needed = {edge_key(e) for e in record.shadowed_edges}
    for dropped in record.dropped:
        if dropped.canonical_id in after:
            needed |= {edge_key(_at(e, inside, keep)) for e in dropped.edges}
    remove: list[Triple] = []
    add: list[Triple] = []
    for edge in originals.values():
        old = _at(edge, inside, keep)
        if old.subject_id != old.object_id and edge_key(old) not in needed:
            remove.append(old)
        new = _at(edge, after, keep)
        if new.subject_id != new.object_id:
            add.append(new)
    return remove, add


def plan_links(
    record: MergeRecord, restore: set[str], documents: dict[str, DocumentRecord]
) -> list[DocumentRecord]:
    """The documents whose entity links change, with the links of ``restore`` pointing back.

    A recorded link is matched on its document and on every field but the id, among the links that
    now name the survivor; a link already naming the dropped id counts as done, so repeating this
    changes nothing more. Two links identical but for the id cannot be told apart, so the first
    is taken. A document that was deleted since is skipped, and so is a link its re-ingest dropped.
    """
    recorded = [
        item
        for dropped in record.dropped
        if dropped.canonical_id in restore
        for item in dropped.document_links
    ]
    changed: list[DocumentRecord] = []
    for document_id in dict.fromkeys(item.document_id for item in recorded):
        document = documents.get(document_id)
        if document is None:
            continue
        links = list(document.el_result)
        claimed: set[int] = set()
        edited = False
        for item in (i for i in recorded if i.document_id == document_id):
            shape = item.link.model_copy(update={"canonical_id": record.keep_id})
            for wanted in (item.link, shape):
                found = next(
                    (i for i, x in enumerate(links) if i not in claimed and x == wanted), None
                )
                if found is not None:
                    claimed.add(found)
                    if wanted is shape:
                        links[found] = item.link
                        edited = True
                    break
        if edited:
            changed.append(document.model_copy(update={"el_result": links}))
    return changed


def plan_aliases(
    survivor_name: str, aliases: list[str], record: MergeRecord, restore: set[str]
) -> list[str]:
    """The survivor's aliases without the names only the restored entities brought.

    A name stays if the survivor had it before the merge (``keep_aliases_before``), is the
    survivor's own name, or another entity still inside the survivor brings it too.
    """
    after = _inside(record) - restore
    keep_names = {normalize_name(n) for n in (survivor_name, *record.keep_aliases_before)}
    still = {
        normalize_name(s)
        for d in record.dropped
        if d.canonical_id in after
        for s in (d.name, *d.aliases)
    }
    gone = (
        {
            normalize_name(s)
            for d in record.dropped
            if d.canonical_id in restore
            for s in (d.name, *d.aliases)
        }
        - keep_names
        - still
    )
    return [a for a in aliases if normalize_name(a) not in gone]
