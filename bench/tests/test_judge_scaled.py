"""The cookbook's 1-5 judge beside the yes/no judge (GRA-126)."""

from __future__ import annotations

from kgbench import judge_scaled
from kggen.llm import CachedChat, ResponseCache

FACT = {"fact_id": "F1", "text": "Aa Works owns Bb Group"}


class Scripted:
    model = "m"

    def __init__(self, reply):
        self.reply = reply

    def complete(self, messages):
        return self.reply(messages[-1]["content"])


def chat(tmp_path, reply):
    return CachedChat(ResponseCache(tmp_path), Scripted(reply))


def test_the_score_is_read_from_result_and_only_valid_ones_are_cached(tmp_path):
    ok = chat(tmp_path / "a", lambda _: "Feedback: fine. [RESULT] 4")
    assert judge_scaled.score(ok, "Aa Works owns Bb Group.", FACT) == 4
    assert len(list((tmp_path / "a").rglob("*.json"))) == 1
    bad = chat(tmp_path / "b", lambda _: "Feedback: no score")
    assert judge_scaled.score(bad, "x", FACT) is None
    assert not list((tmp_path / "b").rglob("*.json"))


def test_the_prompt_carries_the_answer_and_the_reference(tmp_path):
    seen = []
    c = chat(tmp_path, lambda p: seen.append(p) or "[RESULT] 5")
    judge_scaled.score(c, "ANSWER TEXT", FACT)
    assert "ANSWER TEXT" in seen[0] and "Aa Works owns Bb Group" in seen[0]


def test_compare_and_report(tmp_path):
    cases = [
        {
            "id": "1",
            "kind": "stated",
            "label": True,
            "answer": "Aa Works owns Bb Group",
            "fact": FACT,
        },
        {"id": "2", "kind": "denied", "label": False, "answer": "It does not", "fact": FACT},
    ]
    c = chat(
        tmp_path,
        lambda p: (
            "[RESULT] 5" if "Group" in p.split("Response to evaluate")[1][:60] else "[RESULT] 1"
        ),
    )
    rows = judge_scaled.compare(cases, c, lambda answer, fact: "does not" not in answer, workers=1)
    assert [r["score"] for r in rows] == [5, 1]
    table = judge_scaled.format_table(rows)
    assert "stated" in table and "denied" in table
    assert judge_scaled.distribution(rows).count("|") > 5
    assert judge_scaled.chart_svg(rows, "t").startswith("<svg")
