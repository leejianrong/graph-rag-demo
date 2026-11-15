"""Deterministic, unique, synthetic entity names (no real-world people or places)."""

from __future__ import annotations

import random

from kggen.namebank import FIRST_NAMES_ALL, POOLS

_CONSONANTS = "bdfgklmnprstvz"
_VOWELS = "aeiou"
_ORG_SUFFIX = ("Holdings", "Group", "Labs", "Industries", "Partners", "Systems", "Works")
_EVENT_SUFFIX = ("Summit", "Accord Talks", "Expo", "Conference", "Festival", "Forum")
_PLACE_SUFFIX = ("", "", "ia", "ton", "burg", "ford", "mere")
_DEMONYM_SUFFIX = ("ian", "ese", "ish")


class NameBankExhausted(RuntimeError):
    """A world asked the name bank for more names of a type than it holds."""


class NameFactory:
    """Hands out names that are unique across the whole world.

    ``style="invented"`` makes capitalised made-up words. ``style="bank"`` draws realistic names
    without replacement from ``kggen.namebank``, which is deconflicted so no two entities share an
    alias word; a type the bank has no pool for falls back to invented names.
    """

    def __init__(
        self, rng: random.Random, used: set[str] | None = None, style: str = "invented"
    ) -> None:
        self._rng = rng
        self._used: set[str] = set(used or ())
        self._style = style
        self._bank: dict[str, list[str]] = {}  # entity type -> the words still to hand out

    def _word(self, syllables: int) -> str:
        rng = self._rng
        return "".join(
            rng.choice(_CONSONANTS) + rng.choice(_VOWELS) for _ in range(syllables)
        ).capitalize()

    def _candidate(self, entity_type: str, syllables: int) -> str:
        rng = self._rng
        if entity_type == "PERSON":
            return f"{self._word(syllables)} {self._word(syllables + 1)}"
        if entity_type == "ORG":
            return f"{self._word(syllables)} {rng.choice(_ORG_SUFFIX)}"
        if entity_type == "LOCATION":
            return self._word(syllables) + rng.choice(_PLACE_SUFFIX)
        if entity_type == "EVENT":
            return f"{self._word(syllables)} {rng.choice(_EVENT_SUFFIX)}"
        if entity_type == "PRODUCT":
            return f"{self._word(syllables)} {rng.choice('XZQ')}{rng.randint(1, 99)}"
        if entity_type == "NORP":
            return self._word(syllables) + rng.choice(_DEMONYM_SUFFIX)
        if entity_type == "LAW":
            return f"{self._word(syllables)} Act"
        raise ValueError(f"no name pattern for entity type {entity_type!r}")

    def look_alike(self, name: str, entity_type: str) -> str:
        """A fresh unique name that resembles ``name`` (a retrieval distractor): it keeps the first
        word and changes the rest. A type with few endings (seven organisation suffixes) runs out
        of those when many distractors imitate the same name, so after a few tries the first word
        is varied slightly too, which always leaves more candidates."""
        first, *rest = name.split()
        for attempt in range(1, 10_000):
            if rest:
                lead = first if attempt <= 12 else first[:-2] + self._word(1 + attempt // 60)
                tail = self._candidate(entity_type, 2).split(maxsplit=1)
                candidate = f"{lead} {tail[-1]}"
            else:
                candidate = first[:-2] + self._word(1 + attempt // 8)
            if candidate not in self._used:
                self._used.add(candidate)
                return candidate
        raise RuntimeError("could not generate a look-alike name")

    def _from_bank(self, entity_type: str) -> str:
        """The next name of this type from the bank; it never repeats a word."""
        words = self._bank.get(entity_type)
        if words is None:
            words = list(POOLS[entity_type])
            self._rng.shuffle(words)
            self._bank[entity_type] = words
        rng = self._rng
        while words:
            head = words.pop()
            if entity_type == "PERSON":
                name = f"{rng.choice(FIRST_NAMES_ALL)} {head}"
            elif entity_type == "ORG":
                name = f"{head} {rng.choice(_ORG_SUFFIX)}"
            elif entity_type == "EVENT":
                name = f"{head} {rng.choice(_EVENT_SUFFIX)}"
            elif entity_type == "PRODUCT":
                name = f"{head} {rng.choice('XZQ')}{rng.randint(1, 99)}"
            else:  # LOCATION, NORP: the word is the name
                name = head
            if name not in self._used:
                self._used.add(name)
                return name
        raise NameBankExhausted(
            f"the name bank holds {len(POOLS[entity_type])} {entity_type} names and this world "
            "needs more; use name_style 'invented' or a smaller world"
        )

    def new(self, entity_type: str) -> str:
        """A fresh unique name; a collision lengthens the name until it is unique."""
        if self._style == "bank" and entity_type in POOLS:
            return self._from_bank(entity_type)
        syllables = 2
        for attempt in range(1, 10_000):
            name = self._candidate(entity_type, syllables)
            if name not in self._used:
                self._used.add(name)
                return name
            if attempt % 8 == 0:
                syllables += 1
        raise RuntimeError("could not generate a unique name")


def alias_forms(name: str, entity_type: str) -> list[str]:
    """Plausible alternative names for an entity, most natural first."""
    parts = name.split()
    first, last = parts[0], parts[-1]
    if entity_type == "PERSON" and len(parts) == 2:
        return [f"{first[0]}. {last}", last]
    if entity_type == "PRODUCT" and len(parts) == 2:
        return [f"{first}-{last}", f"{first}{last}"]
    if entity_type == "LOCATION":
        return [f"Greater {name}", name.upper()]
    if entity_type == "NORP":
        return [f"{name}s", name.upper()]
    if len(parts) >= 2:  # ORG, EVENT, LAW: drop the suffix, or use initials
        return [first, "".join(p[0] for p in parts).upper()]
    return [name.upper()]
