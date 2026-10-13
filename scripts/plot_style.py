"""
plot_style.py - One shared look for every figure in the project (report + slides).
Colours come from a colour-blind-checked categorical palette, used in a fixed order.
"""

import matplotlib

matplotlib.use("Agg")   # draw to files only (no window needed)
import matplotlib.pyplot as plt

HUMAN_COLOR = "#2a78d6"   # blue
AI_COLOR = "#eb6834"      # orange
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "lines.linewidth": 2,
})
