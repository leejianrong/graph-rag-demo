"""The document faithfulness check and its controls (GRA-122)."""

from __future__ import annotations

import pytest
from kgbench import faithfulness
from kggen.llm import CachedChat, ResponseCache
from kggen.render import render_documents
from kggen.spec import load_preset
from kggen.world import generate


class Scripted:
    model = "m"

    def __init__(self, reply):
        self.reply = reply

    def complete(self, messages):
        return self.reply(messages[-1]["content"])


@pytest.fixture(scope="module")
def world():
    return generate(load_preset("small"))


def honest(prompt: str) -> str:
    """A judge that reads the prompt: a fact is STATED when every name in it is in the article."""
    article, _, rest = prompt.partition("\n\nFacts the article should state")
    lines = [ln for ln in rest.split("\n\nFor each")[0].splitlines() if ln[:1].isdigit()]
    out = []
    for ln in lines:
        number, _, fact = ln.partition(". ")
        names = [w for w in fact.split(" (")[0].split() if w[0].isupper()]
        out.append(f"{number}: {'STATED' if all(n in article for n in names) else 'MISSING'}")
    return "\n".join(out) + "\nExtra: NONE"


def test_parse_wants_every_fact_and_the_extra_line():
    assert faithfulness.parse("1: STATED\n2: WRONG\nExtra: NONE", 2) == (["STATED", "WRONG"], [])
    assert faithfulness.parse("1: STATED\nExtra: a ; b", 1) == (["STATED"], ["a", "b"])
    assert faithfulness.parse("1: STATED\nExtra: NONE", 2) is None
    assert faithfulness.parse("1: STATED", 1) is None


def test_controls_alter_one_fact_and_say_which(world):
    docs = [d for d in render_documents(world) if len(faithfulness.facts_of(d, world)) > 1]
    text, facts, index = faithfulness._without_fact(docs[0], world)
    removed = [s for s in docs[0].sentences if s.text not in text]
    assert len(removed) == 1 and removed[0].fact_ids
    assert len(text) < len(docs[0].text) and 0 <= index < len(facts)
    reversed_facts, at = faithfulness._reversed_fact(docs[0], world)
    original = faithfulness.facts_of(docs[0], world)
    assert reversed_facts[at] != original[at] and reversed_facts[1:] == original[1:]


def test_every_document_is_checked_and_controls_are_counted(world, tmp_path):
    chat = CachedChat(ResponseCache(tmp_path), Scripted(honest))
    result = faithfulness.evaluate(chat, render_documents(world), world, controls=2, workers=1)
    t = faithfulness.tally(result)
    assert t["facts"] > 0 and t["verdicts"]["WRONG"] == 0 and t["docs_with_extra"] == 0
    assert t["control"]["missing"][1] == 2 and t["control"]["wrong"][1] == 2


def test_an_unparsable_reply_is_not_cached(world, tmp_path):
    doc = render_documents(world)[0]
    chat = CachedChat(ResponseCache(tmp_path), Scripted(lambda _: "no idea"))
    assert faithfulness.check(chat, doc.text, faithfulness.facts_of(doc, world), "none") is None
    assert not list(tmp_path.rglob("*.json"))


def test_report_renders(world, tmp_path):
    chat = CachedChat(ResponseCache(tmp_path), Scripted(honest))
    result = faithfulness.evaluate(chat, render_documents(world), world, controls=2, workers=1)
    assert "listed facts stated" in faithfulness.format_table(result)
    assert faithfulness.chart_svg(result, "t").startswith("<svg")


def test_template_sentences_name_the_subject_first(world):
    order = faithfulness.word_order(render_documents(world), world)
    assert order["sentences"] > 0 and order["object_first"] == 0
