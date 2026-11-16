"""A world that changes by the month or the day (GRA-197), and that years are untouched."""

from __future__ import annotations

import datetime as dt
import json

import pytest
from kggen.cli import main
from kggen.questions import QuestionSpec, generate_questions
from kggen.render import RenderSpec, render_documents
from kggen.spec import load_preset
from kggen.timeaxis import Axis
from kggen.world import generate


def world(unit: str, evolution: str = "heavy"):  # type: ignore[no-untyped-def]
    spec = load_preset("small", "clean", evolution).with_overrides(time_unit=unit)
    return generate(spec)


def test_a_year_axis_is_the_identity_and_a_default_spec_does_not_mention_it() -> None:
    axis = Axis("year", 2000, 2024)
    assert (axis.start, axis.end, axis.label(2003), axis.iso(2003)) == (
        2000,
        2024,
        "2003",
        "2003-07-01",
    )
    assert "time_unit" not in load_preset("small").model_dump()
    assert (
        load_preset("small").with_overrides(time_unit="month").model_dump()["time_unit"] == "month"
    )


def test_months_and_days_read_as_dates() -> None:
    month, day = Axis("month", 2000, 2024), Axis("day", 2000, 2024)
    assert month.label(2003 * 12 + 2) == "March 2003" and month.iso(2003 * 12 + 2) == "2003-03-01"
    t = dt.date(2003, 3, 5).toordinal()
    assert day.label(t) == "5 March 2003" and day.iso(t) == "2003-03-05"
    assert month.accepted(2003 * 12 + 2) == ("March 2003", "2003-03")
    assert day.accepted(t) == ("5 March 2003", "2003-03-05")
    assert month.start == 2000 * 12 and month.end == 2024 * 12 + 11
    assert (
        day.start == dt.date(2000, 1, 1).toordinal()
        and day.end == dt.date(2024, 12, 31).toordinal()
    )


@pytest.mark.parametrize("unit", ["month", "day"])
def test_changes_fall_inside_the_clock_and_are_finer_than_years(unit: str) -> None:
    w = world(unit)
    axis = w.spec.axis
    times = {e.valid_from for e in w.edges} | {e.valid_to for e in w.edges if e.valid_to}
    assert times and all(axis.start <= t <= axis.end + 1 for t in times)
    years = {axis.year_of(t) for t in times}
    assert len({t for t in times}) > len(years)  # more distinct moments than calendar years
    assert all(e.born is None or axis.start < e.born <= axis.end for e in w.entities)


@pytest.mark.parametrize("unit", ["month", "day"])
def test_documents_name_the_date_and_are_dated_to_it(unit: str) -> None:
    w = world(unit)
    docs = render_documents(w, RenderSpec(target_words=0, timeline=True))
    assert docs and all(d.date for d in docs)
    assert [d.date for d in docs] == sorted(d.date for d in docs)
    axis = w.spec.axis
    began = next(e for e in w.edges if e.valid_from > axis.start)
    assert any(f"since {axis.label(began.valid_from)}" in d.text for d in docs)
    assert len({d.date for d in docs}) > len({d.date[:4] for d in docs})  # not one per year


@pytest.mark.parametrize("unit", ["month", "day"])
def test_when_questions_ask_for_a_date_and_accept_it_as_written_or_iso(unit: str) -> None:
    w = world(unit)
    qs = generate_questions(
        w, QuestionSpec(n_questions=60, type_mix={"when": 1, "duration": 1, "as_of": 1})
    )
    by = {t: [q for q in qs if q.type == t] for t in ("when", "duration", "as_of")}
    assert all(by.values())
    assert all(
        f"In what {unit}" in q.question or f"Until what {unit}" in q.question for q in by["when"]
    )
    assert all(q.answer_kind == "entity" and q.number is None for q in by["when"])
    assert all(len(q.accepted_answers) == 2 for q in by["when"])
    assert all(
        f"{unit}s was it true" in q.question and q.answer_kind == "number" for q in by["duration"]
    )
    assert all(q.question.startswith("As of ") and q.as_of is not None for q in by["as_of"])
    assert not any(
        q.question.startswith(f"As of {q.as_of}") for q in by["as_of"]
    )  # a date, not a tick


def test_the_year_unit_output_is_unchanged() -> None:
    """The committed fixtures were made with years; a test elsewhere regenerates them and fails on
    drift, so this only pins that a year world carries no month or day wording."""
    w = world("year")
    docs = render_documents(w, RenderSpec(target_words=0, timeline=True))
    assert all(d.date.endswith("-07-01") for d in docs)
    assert w.spec.axis.unit == "year"


def test_the_cli_takes_a_time_unit(capsys) -> None:  # type: ignore[no-untyped-def]
    argv = ["docs", "generate", "--preset", "tiny", "--evolution", "heavy", "--timeline"]
    assert main([*argv, "--time-unit", "month"]) == 0
    months = {d["date"] for d in json.loads(capsys.readouterr().out)}
    assert main(argv) == 0
    years = {d["date"] for d in json.loads(capsys.readouterr().out)}
    assert all(d.endswith("-01") for d in months) and len(months) > len(years)
    assert all(d.endswith("-07-01") for d in years)
