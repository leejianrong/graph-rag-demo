"""Deterministic, unique, synthetic entity names (no real-world people or places)."""

from __future__ import annotations

import random

_CONSONANTS = "bdfgklmnprstvz"
_VOWELS = "aeiou"
_ORG_SUFFIX = ("Holdings", "Group", "Labs", "Industries", "Partners", "Systems", "Works")
_EVENT_SUFFIX = ("Summit", "Accord Talks", "Expo", "Conference", "Festival", "Forum")
_PLACE_SUFFIX = ("", "", "ia", "ton", "burg", "ford", "mere")
_DEMONYM_SUFFIX = ("ian", "ese", "ish")


class NameFactory:
    """Hands out names that are unique across the whole world."""

    def __init__(self, rng: random.Random, used: set[str] | None = None) -> None:
        self._rng = rng
        self._used: set[str] = set(used or ())

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

    def new(self, entity_type: str) -> str:
        """A fresh unique name; a collision lengthens the name until it is unique."""
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
