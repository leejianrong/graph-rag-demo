"""The world's clock (GRA-197): how a whole-number time becomes a date in text.

Every fact, birth, closure and rename carries an integer time (``valid_from``, ``born``, ...). With
``time_unit="year"`` that integer is the year itself, as it always was. With ``"month"`` it counts
months (``year * 12 + month - 1``) and with ``"day"`` days (a proleptic Gregorian ordinal), so one
world can change on a finer grid without any other code needing to know more than "later is a
bigger number". Only this module knows how to print or date one.
"""

from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import dataclass

UNITS: tuple[str, ...] = ("year", "month", "day")
_NOUN = {"year": "year", "month": "month", "day": "day"}
_MONTHS = tuple(calendar.month_name)[1:]


@dataclass(frozen=True)
class Axis:
    """One world's clock: its unit and the first and last time a fact can have."""

    unit: str
    start_year: int
    end_year: int

    @property
    def start(self) -> int:
        """The time every original fact starts."""
        return self._first(self.start_year)

    @property
    def end(self) -> int:
        """The last time a fact can change."""
        return self._last(self.end_year)

    def _first(self, year: int) -> int:
        return {
            "year": year,
            "month": year * 12,
            "day": dt.date(year, 1, 1).toordinal(),
        }[self.unit]

    def _last(self, year: int) -> int:
        return {
            "year": year,
            "month": year * 12 + 11,
            "day": dt.date(year, 12, 31).toordinal(),
        }[self.unit]

    def date(self, t: int) -> dt.date:
        """The calendar date a time falls on (the first of its month for a month; mid-year for a
        year, where a document is dated the way the timeline documents always were)."""
        if self.unit == "year":
            return dt.date(t, 7, 1)
        if self.unit == "month":
            return dt.date(t // 12, t % 12 + 1, 1)
        return dt.date.fromordinal(t)

    def label(self, t: int) -> str:
        """The time as a reader writes it: ``2003``, ``March 2003`` or ``5 March 2003``."""
        if self.unit == "year":
            return str(t)
        d = self.date(t)
        if self.unit == "month":
            return f"{_MONTHS[d.month - 1]} {d.year}"
        return f"{d.day} {_MONTHS[d.month - 1]} {d.year}"

    def iso(self, t: int) -> str:
        """The time as a document date, ``YYYY-MM-DD``; the date of a year or month is its own."""
        return self.date(t).isoformat()

    def year_of(self, t: int) -> int:
        return t if self.unit == "year" else self.date(t).year

    def accepted(self, t: int) -> tuple[str, ...]:
        """Every way a correct answer to "when" may be written."""
        if self.unit == "year":
            return (str(t),)
        d = self.date(t)
        short = f"{d.year}-{d.month:02d}" if self.unit == "month" else d.isoformat()
        return (self.label(t), short)

    def noun(self, plural: bool = False) -> str:
        return _NOUN[self.unit] + ("s" if plural else "")
