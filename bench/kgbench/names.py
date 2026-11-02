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

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._used: set[str] = set()

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
