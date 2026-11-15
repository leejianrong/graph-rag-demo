"""The name bank is deconflicted, a world drawn from it has no accidental collisions (GRA-183)."""

from __future__ import annotations

import pytest
from kggen import namebank
from kggen.names import alias_forms
from kggen.spec import PRESET_NAMES, load_preset
from kggen.world import WorldGenerationError, generate


def bank_world(preset: str = "small", **overrides):
    spec = load_preset(preset).with_overrides(name_style="bank", **overrides)
    return generate(spec)


# --- the bank itself ------------------------------------------------------------------------------


def test_the_bank_is_deconflicted() -> None:
    assert namebank.check_bank() == []


def test_check_bank_catches_a_word_in_two_pools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Revert proof: a shared word, a generator word and a prefix are each reported."""
    monkeypatch.setattr(namebank, "LOCATION_NAMES", (*namebank.LOCATION_NAMES, "Ashgrove"))
    monkeypatch.setitem(namebank.POOLS, "LOCATION", namebank.LOCATION_NAMES)
    monkeypatch.setitem(namebank.POOLS, "EVENT", (*namebank.EVENT_HEADS, "Summit"))
    monkeypatch.setitem(namebank.POOLS, "NORP", (*namebank.NORP_NAMES, "Norwoodishly"))

    problems = " ".join(namebank.check_bank())

    assert "'ashgrove' is in both" in problems or "'Ashgrove' is in both" in problems
    assert "Summit" in problems
    assert "prefix" in problems


@pytest.mark.parametrize("preset", [p for p in PRESET_NAMES if p != "stress"])
def test_the_bank_has_room_for_every_preset_but_stress(preset: str) -> None:
    spec = load_preset(preset)
    total = sum(spec.type_mix.values())
    for entity_type, capacity in namebank.CAPACITY.items():
        needed = spec.n_entities * spec.type_mix.get(entity_type, 0) / total
        assert needed <= capacity, (
            f"{preset} needs about {needed:.0f} {entity_type}, bank has {capacity}"
        )


def test_the_bank_is_a_good_deal_bigger_than_a_small_world_needs() -> None:
    small = load_preset("small")
    for entity_type, capacity in namebank.CAPACITY.items():
        assert capacity >= 3 * small.n_entities * small.type_mix.get(entity_type, 0)


# --- a world drawn from it ------------------------------------------------------------------------


def test_the_default_is_still_invented_names_and_the_spec_file_is_unchanged() -> None:
    assert load_preset("small").name_style == "invented"
    assert "name_style" not in load_preset("small").model_dump_json()
    assert '"name_style":"bank"' in load_preset("small").with_overrides(
        name_style="bank"
    ).model_dump_json().replace(" ", "")


@pytest.mark.parametrize("preset", ["small", "medium"])
def test_a_bank_world_has_no_two_entities_sharing_an_alias_word(preset: str) -> None:
    """Every alias the generator derives from a name belongs to one entity only."""
    world = bank_world(preset)
    owner: dict[str, str] = {}
    for e in world.entities:
        forms = alias_forms(e.name, e.type)
        # Initials ("CG") are built from two words and can repeat; every other form is one word's.
        keep = [f for f in forms if not (e.type in {"ORG", "EVENT", "LAW"} and f == forms[-1])]
        for form in [e.name, *keep]:
            key = form.casefold()
            assert owner.setdefault(key, e.id) == e.id, f"{form!r} names {owner[key]} and {e.id}"


def test_a_bank_world_reads_like_real_names() -> None:
    world = bank_world()
    firsts, surnames = set(namebank.FIRST_NAMES_ALL), set(namebank.POOLS["PERSON"])
    people = [e for e in world.entities if e.type == "PERSON"]
    assert people
    for p in people:
        first, last = p.name.split()
        assert first in firsts and last in surnames
    assert {e.name.split()[0] for e in world.entities if e.type == "ORG"} <= set(
        namebank.POOLS["ORG"]
    )
    assert {e.name for e in world.entities if e.type == "LOCATION"} <= set(
        namebank.POOLS["LOCATION"]
    )


def test_a_bank_world_is_repeatable_and_realises_the_same_size_as_the_spec_asks() -> None:
    """Bank and invented worlds from one seed are not the same graph: the bank draws its names from
    the world's random stream, and an invented name draws a different number of values."""
    clean = generate(load_preset("small"))
    bank = bank_world()
    assert (len(bank.entities), len(bank.edges)) == (len(clean.entities), len(clean.edges))
    assert [n.name for n in bank_world().entities] == [n.name for n in bank.entities]
    assert [n.name for n in bank.entities] != [n.name for n in clean.entities]


def test_the_ambiguity_knobs_still_work_on_bank_names() -> None:
    world = bank_world(homonym_rate=0.15, alias_rate=0.7, distractor_rate=0.2)
    assert any(e.distractor for e in world.entities)
    assert any(e.aliases for e in world.entities)
    names = [e.name for e in world.entities if not e.distractor]
    assert len(names) != len(set(names))  # the shared names the knob asked for


def test_a_world_too_big_for_the_bank_says_what_to_do() -> None:
    with pytest.raises(WorldGenerationError, match="name bank.*invented"):
        bank_world("stress")
