"""Shared figure style, so every step's figures look the same.

Usage in a step script:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from common.plotstyle import apply_style, SERIES, INK, MUTED, SURFACE

    apply_style()
"""
import matplotlib

SURFACE = "#fcfcfb"      # figure background
INK = "#0b0b0b"          # text
MUTED = "#52514e"        # axes, reference lines, de-emphasized series
GRID = "#e4e3df"

# Series colors, used in this order. The reference solver is always drawn in INK.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]

# One hue, light to dark, for an ordered quantity (temperature, strain level, ...).
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def apply_style():
    matplotlib.rcParams.update({
        "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
        "legend.fontsize": 9, "legend.frameon": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 2.0, "savefig.bbox": "tight", "savefig.dpi": 160,
    })
