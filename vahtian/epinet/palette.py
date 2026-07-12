# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Heidi Andersén

"""Figure and report palettes — Wong (default) and Vahtian "Sentinel".

The report layer lets a user slide between two named palettes without touching
the science. Both are accessibility-first:

* **wong** — the Okabe–Ito colourblind-safe categorical cycle EpiNet already
  uses for figures. Best when an outcome has many classes.
* **vahtian** ("Sentinel") — the Vahtian single-hue brand palette. Neutrals
  carry structure; one purple hue carries meaning, with contrast from lightness
  steps so it survives greyscale and all colour-vision deficiencies. Caps at
  3–4 categorical series by design.

A palette is a plain dict (categorical list, highlight, sequential ramp, and a
``theme`` of named UI colours for the HTML report). :func:`apply_palette` swaps
the figure module's colours and matplotlib rcParams so the generated plots match
the report chrome. Theme colours can still be overridden per-run by the
``reporting.primary_color`` / ``accent_color`` config fields.
"""

from __future__ import annotations

# Okabe–Ito colourblind-safe cycle (the established EpiNet figure default).
WONG = {
    "name": "wong",
    "categorical": [
        "#0072B2", "#E69F00", "#009E73", "#CC79A7",
        "#56B4E9", "#D55E00", "#F0E442", "#999999",
    ],
    "highlight": "#D81B60",
    "sequential": ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
    "theme": {
        "ink": "#0B2545", "primary": "#0072B2", "accent": "#009E73",
        "canvas": "#FFFFFF", "surface": "#FFFFFF", "grid": "#E6E6E6",
        "muted": "#BBBBBB", "axis": "#3A3A3A", "text": "#1A1A1A",
    },
}

# Vahtian "Sentinel": single purple hue + neutrals, separated by lightness.
VAHTIAN = {
    "name": "vahtian",
    "categorical": ["#5E4F99", "#908C9C", "#2D2440", "#C2BFCB", "#8273C0", "#574A8C"],
    "highlight": "#2D2440",
    "sequential": ["#EDEAF4", "#CFC6E6", "#A99CD4", "#8273C0", "#574A8C", "#2D2440"],
    "theme": {
        "ink": "#2D2440", "primary": "#5E4F99", "accent": "#8273C0",
        "canvas": "#F5F4F7", "surface": "#FFFFFF", "grid": "#E2E0E6",
        "muted": "#C2BFCB", "axis": "#5C5866", "text": "#2A2730",
    },
}

PALETTES = {
    "wong": WONG, "okabe-ito": WONG, "default": WONG,
    "vahtian": VAHTIAN, "sentinel": VAHTIAN, "epinet": VAHTIAN,
}


def get_palette(name: str | None) -> dict:
    """Resolve a palette by name (case-insensitive); falls back to Wong."""
    return PALETTES.get((name or "wong").strip().lower(), WONG)


def apply_palette(name: str | None) -> dict:
    """Apply a named palette to the figure module and matplotlib rcParams.

    Returns the resolved palette dict so the HTML report can theme its chrome to
    match. Safe to call before every run; it only mutates global figure state
    that the plotting functions read at call time.
    """
    palette = get_palette(name)
    theme = palette["theme"]

    # Swap the figure module's categorical colours and highlight in place; the
    # plotting functions reference these module globals when they draw.
    from vahtian.epinet import viz

    viz.CATEGORY_COLORS = list(palette["categorical"])
    viz.HIGHLIGHT = palette["highlight"]

    try:
        import matplotlib as mpl

        mpl.rcParams.update({
            "axes.prop_cycle": mpl.cycler(color=palette["categorical"]),
            "figure.facecolor": theme["canvas"],
            "axes.facecolor": theme["canvas"],
            "savefig.facecolor": theme["canvas"],
            "axes.edgecolor": theme["axis"],
            "axes.labelcolor": theme["text"],
            "text.color": theme["text"],
            "xtick.color": theme["axis"],
            "ytick.color": theme["axis"],
            "grid.color": theme["grid"],
        })
    except Exception:  # pragma: no cover - matplotlib always present in practice
        pass
    return palette
