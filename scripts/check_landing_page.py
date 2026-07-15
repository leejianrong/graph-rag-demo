#!/usr/bin/env python3
"""Validate the GitHub Pages landing page before it ships (dev-playbook #15).

``site/index.html`` is a single hand-authored, self-contained page (no build step,
no external assets) published to GitHub Pages on merge by ``.github/workflows/pages.yml``.
There is no docs *build* to fail, so the cheap PR-time safety net is instead: the
HTML parses, and every in-page ``href="#anchor"`` resolves to a real ``id`` — so a
broken table-of-contents link is caught in CI rather than on the live site.

Run locally with ``make check-site``; CI runs it on PRs that touch ``site/``.
Exit code 0 = healthy, 1 = a problem was found (message on stderr).
"""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path

_PAGE = Path(__file__).resolve().parents[1] / "site" / "index.html"


class _AnchorCollector(HTMLParser):
    """Collect every element ``id`` and every in-page ``href="#..."`` fragment."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.fragments: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value is None:
                continue
            if name == "id":
                self.ids.add(value)
            elif name == "href" and value.startswith("#") and value != "#":
                self.fragments.append(value[1:])


def check(page: Path) -> list[str]:
    """Return a list of problems with ``page`` (empty list == healthy)."""
    if not page.is_file():
        return [f"landing page not found: {page}"]

    parser = _AnchorCollector()
    try:
        parser.feed(page.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as error:  # malformed HTML / bad encoding
        return [f"could not parse {page.name}: {error}"]

    dangling = sorted({f for f in parser.fragments if f not in parser.ids})
    return [f'in-page link "#{f}" has no matching id' for f in dangling]


def main() -> int:
    """CLI entry point. Prints problems to stderr; returns a process exit code."""
    problems = check(_PAGE)
    if problems:
        print(f"{_PAGE.name}: {len(problems)} problem(s) found", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"{_PAGE.name}: OK (parses; all in-page anchors resolve)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
