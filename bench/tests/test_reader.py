"""The LLM single-hop reader (GRA-131)."""

from __future__ import annotations

from kgbench import reader
from kggen.llm import CachedChat, ResponseCache
from kggen.render import Document, Sentence


def sentence(text: str, *facts: str) -> Sentence:
    return Sentence(text, 0, len(text), tuple(facts), ())


DOCS = [
    Document(
        "D1",
        "A",
        "Aa owns Bb. Bb is in Cc.",
        (sentence("Aa owns Bb.", "F1"), sentence("Bb is in Cc.", "F2")),
        False,
    ),
    # one merged sentence expresses both facts of a second path
    Document(
        "D2",
        "B",
        "Xx owns Yy and Yy is in Zz.",
        (sentence("Xx owns Yy and Yy is in Zz.", "F3", "F4"),),
        False,
    ),
]
Q = {
    "id": "Q1",
    "hops": 2,
    "shared_name": False,
    "question": "Where is the firm Aa owns?",
    "gold_paths": [["F1", "F2"]],
    "supporting_facts": ["F1", "F2"],
    "accepted_answers": ["Cc", "C city"],
}
MERGED = {
    **Q,
    "id": "Q2",
    "gold_paths": [["F3", "F4"]],
    "supporting_facts": ["F3", "F4"],
    "accepted_answers": ["Zz"],
}


def test_last_holds_only_the_final_fact_and_chain_holds_the_path() -> None:
    out = reader.passages(Q, DOCS)
    assert out["last"] == ("Bb is in Cc.", False)
    assert out["chain"] == ("Aa owns Bb. Bb is in Cc.", True)


def test_a_merged_sentence_makes_the_last_passage_hold_the_whole_path() -> None:
    assert reader.passages(MERGED, DOCS)["last"][1] is True


def test_grade() -> None:
    assert reader.grade("The answer is Cc.", Q) == "correct"
    assert reader.grade("Unknown", Q) == "unknown"
    assert reader.grade("Dd", Q) == "wrong"
    assert reader.grade("Ccx", Q) == "wrong"  # whole word only


class Scripted:
    model = "m"

    def __init__(self, reply):
        self.reply = reply

    def complete(self, messages):
        return self.reply(messages[-1]["content"])


def test_rows_count_shortcuts_and_replay(tmp_path) -> None:
    def reply(prompt: str) -> str:  # answers right whenever the passage alone names Cc or Zz
        passage = prompt.split("Passage:\n")[1].split("\n\nQuestion")[0]
        return "Cc" if passage == "Bb is in Cc." else "unknown"

    chat = CachedChat(ResponseCache(tmp_path), Scripted(reply))
    rows = reader.evaluate(chat, [Q, MERGED], DOCS, workers=1)
    outcome = {(r["id"], r["input"]): r["outcome"] for r in rows}
    assert outcome[("Q1", "last")] == "correct"  # the shortcut: no chain needed
    assert outcome[("Q1", "chain")] == "unknown"
    assert "1 were answered correctly" in reader.shortcuts(rows)
    assert "leaving 1 true shortcuts" in reader.shortcuts(rows)
    replay = reader.evaluate(
        CachedChat(ResponseCache(tmp_path), None, "m"), [Q, MERGED], DOCS, workers=1
    )
    assert replay == rows


def test_an_empty_reply_is_not_cached_and_counts_nowhere(tmp_path) -> None:
    chat = CachedChat(ResponseCache(tmp_path), Scripted(lambda _: "  "))
    rows = reader.evaluate(chat, [Q], DOCS, workers=1)
    assert {r["outcome"] for r in rows} == {None}
    assert not list(tmp_path.rglob("*.json"))
    assert "2 replies were empty" in reader.format_table(rows)


def test_the_committed_replies_replay_offline(capsys) -> None:
    from pathlib import Path

    from kgbench.cli import main

    folder = Path(__file__).resolve().parents[2] / "datagen" / "examples" / "hard-small"
    assert main(["reader", "--dir", str(folder), "--n", "100"]) == 0
    out = capsys.readouterr().out
    assert "| all | 100 | 17% (17/100" in out and "leaving 0 true shortcuts" in out
