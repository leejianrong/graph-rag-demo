"""The chart helpers: a Wilson interval and a well-formed grouped bar chart."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from kgbench.charts import grouped_bars, wilson


def test_wilson_is_sound_at_the_edges_and_narrows_with_more_data():
    assert wilson(0, 0) == (0.0, 1.0)
    low, high = wilson(0, 57)
    assert low == 0.0 and 0.05 < high < 0.08  # zero errors in 57 still allows about 6%
    assert wilson(5, 10)[0] < 0.5 < wilson(5, 10)[1]
    narrow, wide = wilson(50, 100), wilson(5, 10)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]


def test_grouped_bars_is_valid_svg_with_every_series_and_category():
    svg = grouped_bars(
        "t & <u>".replace("&", "and").replace("<u>", "u"),
        ["a", "b"],
        {"x": [0.2, 0.8], "y": [1.0, 0.0]},
        ylabel="share",
        intervals={"x": [(0.1, 0.3), (0.7, 0.9)], "y": [(0.9, 1.0), (0.0, 0.1)]},
        counts=["n=3", "n=4"],
    )
    root = ET.fromstring(svg)
    text = "".join(root.itertext())
    assert all(s in text for s in ("a", "b", "x", "y", "n=3", "n=4", "share"))
    assert len(root.findall(".//{http://www.w3.org/2000/svg}rect")) >= 1 + 4 + 2  # bg, bars, key
