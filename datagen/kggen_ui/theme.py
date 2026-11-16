"""The Studio's look (GRA-196): colour, type and radius as one indah ``Theme``.

Cool paper and ink, one accent for "selected", and a fixed four-colour code that always means time
(born, ended, renamed, replaced), the same colours the year view paints on the graph, so the page's
only saturated colour is the colour of change. Display type is Bricolage Grotesque, text IBM Plex
Sans, data and ids IBM Plex Mono, all self-hosted.
"""

from __future__ import annotations

from pathlib import Path

import indah

from kggen_ui import model

FONT_DIR = Path(__file__).parent / "fonts"

# The surface and text roles per scheme. The time colours are not here: they are the year view's
# own (model.STATE_COLOURS) so a legend and the graph cannot disagree.
LIGHT = {
    "bg": "#EEF1F0",
    "surface": "#F8FAF9",
    "surface-2": "#E6EBEA",
    "surface-3": "#DCE3E1",
    "text": "#14212B",
    "muted": "#53646C",
    "outline": "#B9C4C2",
    "outline-soft": "#D3DBD9",
    "primary": "#2457D6",
    "on-primary": "#FFFFFF",
    "primary-container": "#DCE6FB",
    "on-primary-container": "#102A66",
    "secondary": "#3B5560",
    "on-secondary": "#FFFFFF",
    "secondary-container": "#D5E3E8",
    "on-secondary-container": "#16303A",
}
DARK = {
    "bg": "#0F171C",
    "surface": "#162128",
    "surface-2": "#1D2B33",
    "surface-3": "#25363F",
    "text": "#E6ECEE",
    "muted": "#9FB0B7",
    "outline": "#3C4F58",
    "outline-soft": "#2B3C44",
    "primary": "#86A9F7",
    "on-primary": "#0B1220",
    "primary-container": "#1E3470",
    "on-primary-container": "#DCE6FB",
    "secondary": "#A9C0CA",
    "on-secondary": "#0F1D24",
    "secondary-container": "#2A4049",
    "on-secondary-container": "#D5E3E8",
}
# Time colours that hold on dark paper too (the graph keeps the light-paper values).
TIME_LIGHT = {
    "ok": model.STATE_COLOURS[model.BORN],
    "err": model.STATE_COLOURS[model.ENDED],
    "warn": "#B7791F",
}
TIME_DARK = {"ok": "#4CC38A", "err": "#E8805C", "warn": "#E0A84A"}

DISPLAY = "'Bricolage Grotesque', 'IBM Plex Sans', system-ui, sans-serif"
BODY = "'IBM Plex Sans', system-ui, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, monospace"

FONTS = (
    ("Bricolage Grotesque", "bricolage-grotesque-latin-600-normal.woff2", "600"),
    ("IBM Plex Sans", "ibm-plex-sans-latin-400-normal.woff2", "400"),
    ("IBM Plex Sans", "ibm-plex-sans-latin-600-normal.woff2", "600"),
    ("IBM Plex Mono", "ibm-plex-mono-latin-400-normal.woff2", "400"),
)


def studio_theme() -> indah.Theme:
    """The theme for the Studio app (a new one per call: it holds the font bytes)."""
    return indah.Theme(
        light={**LIGHT, **TIME_LIGHT},
        dark={**DARK, **TIME_DARK},
        extra_light={
            "x-renamed": model.STATE_COLOURS[model.RENAMED],
            "x-gone": model.STATE_COLOURS[model.GONE],
        },
        extra_dark={"x-renamed": "#B29CF0", "x-gone": "#4A5A62"},
        radius="6px",
        font_display=DISPLAY,
        font_body=BODY,
        font_mono=MONO,
        fonts=[
            indah.Font(family, FONT_DIR / name, weight=weight) for family, name, weight in FONTS
        ],
    )
