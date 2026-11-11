"""The nightly answer gate: within tolerance passes, a real regression turns it red (GRA-17)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from kgbench import cli, nightly
from kgbench.cli import main

BASE = {"exact_match": 0.5, "token_f1": 0.6, "supporting_fact_f1": 0.4}


def test_a_drop_within_the_tolerance_passes_and_beyond_it_fails() -> None:
    assert nightly.check({**BASE, "exact_match": 0.48}, BASE, 0.02) == []
    drops = nightly.check({**BASE, "exact_match": 0.47, "token_f1": 0.9}, BASE, 0.02)
    assert [d.metric for d in drops] == ["exact_match"]


def test_a_missing_metric_fails_and_an_improvement_passes() -> None:
    missing = nightly.check({"exact_match": 1.0}, BASE, 0.02)
    assert {d.metric for d in missing} == {"token_f1", "supporting_fact_f1"}
    assert nightly.check({m: 1.0 for m in BASE}, BASE, 0.0) == []


def test_the_summary_marks_each_metric() -> None:
    text = nightly.summary_markdown({**BASE, "token_f1": 0.1}, BASE, 0.02)
    rows = {line.split("|")[1].strip(): line for line in text.splitlines() if "_" in line[:20]}
    assert rows["token_f1"].rstrip(" |").endswith("FAIL")
    assert rows["exact_match"].rstrip(" |").endswith("ok")


def test_the_committed_baseline_passes_today(tmp_path: Path) -> None:
    summary = tmp_path / "summary.md"
    assert main(["nightly", "--summary", str(summary)]) == 0
    assert "Nightly benchmark" in summary.read_text()


def test_a_deliberately_degraded_pipeline_turns_it_red(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = cli.build_offline_components

    class Degraded:
        """A retriever that finds nothing: no answer, no supporting sentences."""

        def __init__(self, inner) -> None:
            self._inner = inner

        def retrieve(self, request):
            found = self._inner.retrieve(request)
            return found.model_copy(update={"answer": None, "supporting_sentences": []})

    def broken():
        components = real()
        return dataclasses.replace(components, retriever=Degraded(components.retriever))

    monkeypatch.setattr(cli, "build_offline_components", broken)
    assert main(["nightly"]) == 1
    out = capsys.readouterr().out
    assert "REGRESSION supporting_fact_f1" in out and "FAIL" in out


def test_update_rewrites_the_baseline(tmp_path: Path) -> None:
    path = tmp_path / "nightly.json"
    path.write_text(json.dumps({"subset": "small", "tolerance": 0.05, "baseline": {}}))
    assert main(["nightly", "--baseline", str(path), "--update"]) == 0
    saved = json.loads(path.read_text())
    assert saved["tolerance"] == 0.05 and set(saved["baseline"]) == set(nightly.GATED)
    assert main(["nightly", "--baseline", str(path)]) == 0
