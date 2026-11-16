"""The dataset wizard (GRA-195): the words on each choice, the estimate, the review, the steps."""

from __future__ import annotations

import pytest

pytest.importorskip("indah")  # the UI is an optional extra; the fast gate runs without it

from kggen_ui import model  # noqa: E402
from kggen_ui.app import build_ui  # noqa: E402


def test_every_choice_is_a_value_its_select_accepts() -> None:
    """A card sets its signal to its value, so a value the generator does not know would break
    Generate; the cards must offer exactly what the selects did."""
    assert {v for v, _, _ in model.PRESET_CHOICES} == set(model.PRESET_NAMES)
    assert {v for v, _, _ in model.DIFFICULTY_CHOICES} == set(model.DIFFICULTY_NAMES)
    assert {v for v, _, _ in model.EVOLUTION_CHOICES} == set(model.EVOLUTION_LEVELS)
    assert {v for v, _, _ in model.PERSONA_CHOICES_TEXT} == set(model.PERSONA_CHOICES)
    assert {v for v, _, _ in model.NAME_CHOICES} == {"invented", "bank"}
    assert {v for v, _, _ in model.TIMELINE_CHOICES} == {"off", "on"}
    assert {v for v, _, _ in model.WORDING_CHOICES} == {"plain", "varied"}


def test_every_choice_explains_itself_in_a_sentence() -> None:
    for choices in (
        model.PRESET_CHOICES,
        model.DIFFICULTY_CHOICES,
        model.NAME_CHOICES,
        model.EVOLUTION_CHOICES,
        model.TIMELINE_CHOICES,
        model.PERSONA_CHOICES_TEXT,
        model.WORDING_CHOICES,
    ):
        assert all(title and blurb.endswith((".", '."', "?")) for _, title, blurb in choices)


def test_the_change_cards_quote_the_generators_own_rates() -> None:
    from kggen.spec import EVOLUTION

    heavy = dict((v, b) for v, _, b in model.EVOLUTION_CHOICES)["heavy"]
    assert f"{EVOLUTION['heavy']['churn_rate']:.0%}" in heavy
    assert f"{EVOLUTION['heavy']['rename_rate']:.0%}" in heavy


def test_the_estimate_is_close_to_what_generation_makes() -> None:
    values = model.preset_values("small", "hard", "heavy")
    world = model.generate_world(values, name_style="invented")
    text = model.estimate(values)
    entities, facts = (int(t.replace(",", "")) for t in __import__("re").findall(r"[\d,]+", text))
    assert abs(entities - len(world.entities)) <= 0.15 * len(world.entities)
    assert abs(facts - len(world.edges)) <= 0.25 * len(world.edges)


def test_the_review_lists_choices_then_only_the_numbers_that_were_changed() -> None:
    ui = build_ui()
    ui.preset.set("small")
    ui.difficulty.set("hard")
    ui.evolution.set("heavy")
    lines = model.review_lines(ui.settings())
    assert "Size: Small" in lines and "Difficulty: Hard" in lines
    assert "Change over time: Heavy change" in lines
    assert not any("instead of" in line for line in lines)  # nothing fine-tuned
    ui.values["avg_degree"].set(7.0)
    changed = [line for line in model.review_lines(ui.settings()) if "instead of" in line]
    assert changed == ["Average degree: 7 instead of 4"]


def test_the_wizard_has_five_steps_and_stays_inside_them() -> None:
    ui = build_ui()
    assert model.WIZARD_STEPS == ("Start", "World", "Time", "Questions", "Review")
    assert [i["state"] for i in ui.step_items()] == ["current", *["pending"] * 4]
    ui.go(2)
    assert [i["state"] for i in ui.step_items()] == [
        "done",
        "done",
        "current",
        "pending",
        "pending",
    ]
    ui.go(99)
    assert ui.step.value == 4
    ui.go(-3)
    assert ui.step.value == 0


def test_choosing_on_a_card_is_choosing_on_the_form_so_workspaces_still_save_it() -> None:
    ui = build_ui()
    ui.evolution.set("light")
    assert ui.settings()["evolution"] == "light"


def test_generating_from_the_review_opens_the_world_tab() -> None:
    ui = build_ui()
    ui.difficulty.set("clean")
    ui.preset.set("tiny")
    ui.name_style.set("invented")
    assert ui.tab.value == 0
    ui.generate_and_show()
    assert ui.world.value is not None and ui.tab.value == 1


def test_a_failed_generate_stays_on_setup() -> None:
    ui = build_ui()
    ui.values["n_entities"].set(1)  # below what a world needs
    ui.generate_and_show()
    assert ui.world.value is None and ui.tab.value == 0
