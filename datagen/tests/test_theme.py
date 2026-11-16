"""The Studio theme (GRA-196): readable colours, shipped fonts, one code for time."""

from __future__ import annotations

import pytest

pytest.importorskip("indah")  # the UI is an optional extra; the fast gate runs without it

from kggen_ui import model, theme  # noqa: E402


def luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("scheme", ["LIGHT", "DARK"])
def test_text_and_controls_meet_wcag_aa(scheme: str) -> None:
    c = getattr(theme, scheme)
    for fg, bg in (
        ("text", "bg"),
        ("text", "surface"),
        ("muted", "bg"),
        ("muted", "surface"),
        ("on-primary", "primary"),
        ("on-primary-container", "primary-container"),
        ("on-secondary", "secondary"),
        ("on-secondary-container", "secondary-container"),
    ):
        assert contrast(c[fg], c[bg]) >= 4.5, (scheme, fg, bg, contrast(c[fg], c[bg]))
    assert contrast(c["primary"], c["bg"]) >= 3  # the accent reads against the page


def test_the_time_colours_are_the_year_views_and_are_visible_on_both_papers() -> None:
    assert theme.TIME_LIGHT["ok"] == model.STATE_COLOURS[model.BORN]
    assert theme.TIME_LIGHT["err"] == model.STATE_COLOURS[model.ENDED]
    for name in ("ok", "err", "warn"):
        assert contrast(theme.TIME_LIGHT[name], theme.LIGHT["bg"]) >= 3
        assert contrast(theme.TIME_DARK[name], theme.DARK["bg"]) >= 3
    assert contrast(model.STATE_COLOURS[model.RENAMED], theme.LIGHT["bg"]) >= 3


def test_the_theme_builds_with_its_fonts_and_both_schemes() -> None:
    built = theme.studio_theme()
    css = built.css()
    assert "Bricolage Grotesque" in css and "IBM Plex Mono" in css
    assert ':root[data-theme="dark"]' in css and "--x-renamed" in css
    assert len(built.font_files) == len(theme.FONTS)
    assert all(data.startswith(b"wOF2") for data in built.font_files.values())


def test_every_font_file_is_present_and_small() -> None:
    for _, name, _ in theme.FONTS:
        path = theme.FONT_DIR / name
        assert path.is_file() and path.stat().st_size < 100_000
    assert "Font License" in (theme.FONT_DIR / "README.md").read_text().replace("\n", " ")
