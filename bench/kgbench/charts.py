"""Small standalone SVG charts for the reports, and the interval behind their error bars."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt

PALETTE = ("#2a6f97", "#c26a3d", "#5b8e4a", "#8a5a9e", "#b0912f", "#7a7a7a")


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval for a proportion (sound at 0 and 1, unlike the normal one)."""
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def grouped_bars(
    title: str,
    categories: Sequence[str],
    series: dict[str, Sequence[float]],
    *,
    ylabel: str,
    intervals: dict[str, Sequence[tuple[float, float]]] | None = None,
    counts: Sequence[str] | None = None,
) -> str:
    """Bars for each series within each category, 0 to 1, with optional interval whiskers.

    ``counts`` is a second line under each category label (for example ``n=15``).
    """
    names = list(series)
    w = max(520, 110 * len(categories) + 190)
    h, left, right, top, bottom = 300, 46, 150, 34, 62
    plot_w, plot_h = w - left - right, h - top - bottom
    group = plot_w / len(categories)
    bar = min(34.0, group * 0.8 / len(names))

    def y(v: float) -> float:
        return top + plot_h * (1 - v)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="{title}" font-family="sans-serif" font-size="11">',
        f'<rect width="{w}" height="{h}" fill="#fffdf9"/>',
        f'<text x="{left}" y="18" fill="#2a241f" font-size="13" font-weight="bold">{title}</text>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1):
        out.append(
            f'<line x1="{left}" x2="{left + plot_w}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" '
            f'stroke="#e2d9d0"/><text x="{left - 6}" y="{y(tick) + 4:.1f}" text-anchor="end" '
            f'fill="#5b5148">{tick:.2f}</text>'
        )
    out.append(
        f'<text x="12" y="{top + plot_h / 2}" transform="rotate(-90 12 {top + plot_h / 2})" '
        f'text-anchor="middle" fill="#2a241f">{ylabel}</text>'
    )
    for c, category in enumerate(categories):
        centre = left + group * (c + 0.5)
        out.append(
            f'<text x="{centre:.1f}" y="{h - bottom + 16}" text-anchor="middle" '
            f'fill="#2a241f">{category}</text>'
        )
        if counts:
            out.append(
                f'<text x="{centre:.1f}" y="{h - bottom + 30}" text-anchor="middle" '
                f'fill="#5b5148">{counts[c]}</text>'
            )
        for s, name in enumerate(names):
            x = centre + (s - (len(names) - 1) / 2) * bar
            value = series[name][c]
            colour = PALETTE[s % len(PALETTE)]
            out.append(
                f'<rect x="{x - bar * 0.4:.1f}" y="{y(value):.1f}" width="{bar * 0.8:.1f}" '
                f'height="{max(plot_h * value, 0.6):.1f}" fill="{colour}"><title>{name}: '
                f"{value:.3f}</title></rect>"
            )
            if intervals:
                lo, hi = intervals[name][c]
                out.append(
                    f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y(lo):.1f}" y2="{y(hi):.1f}" '
                    'stroke="#2a241f" stroke-width="1.2"/>'
                    f'<line x1="{x - 3:.1f}" x2="{x + 3:.1f}" y1="{y(hi):.1f}" y2="{y(hi):.1f}" '
                    'stroke="#2a241f"/>'
                    f'<line x1="{x - 3:.1f}" x2="{x + 3:.1f}" y1="{y(lo):.1f}" y2="{y(lo):.1f}" '
                    'stroke="#2a241f"/>'
                )
    for s, name in enumerate(names):
        ly = top + 14 + 18 * s
        out.append(
            f'<rect x="{left + plot_w + 14}" y="{ly - 8}" width="14" height="10" '
            f'fill="{PALETTE[s % len(PALETTE)]}"/><text x="{left + plot_w + 34}" y="{ly + 1}" '
            f'fill="#2a241f">{name}</text>'
        )
    out.append("</svg>")
    return "\n".join(out)
