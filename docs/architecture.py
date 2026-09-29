"""README architecture diagram, drawn with explicit coordinates.

Rendered to a PNG because the GitHub mobile app does not render mermaid.
Laid out top-to-bottom and narrow so it stays legible at phone width.

    .venv/bin/python docs/architecture.py     # writes docs/architecture.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "architecture.png"

SURFACE, INK, INK2, MUTED, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
BLUE, BLUE_BG, ORANGE, ORANGE_BG = "#2a78d6", "#eaf2fc", "#eb6834", "#fdeee7"
FONT = "DejaVu Sans"

Y0, Y1 = 2.15, 11.5
fig = plt.figure(figsize=(6.2, 6.2 * (Y1 - Y0) / 10), dpi=200, facecolor=SURFACE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 10)
ax.set_ylim(Y0, Y1)
ax.axis("off")


def text(x, y, s, size=9, color=INK2, **kw):
    ax.text(x, y, s, fontsize=size, color=color, family=FONT, zorder=5, **kw)


def box(cx, cy, w, h, lines, fc="white", ec=EDGE, lw=1.3):
    """lines: [(text, size, color, weight, style), ...] stacked top to bottom."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    step = 0.36
    top = cy + step * (len(lines) - 1) / 2
    for i, (s, size, color, weight, style) in enumerate(lines):
        text(cx, top - i * step, s, size=size, color=color, weight=weight, style=style,
             ha="center", va="center")


def arrow(p, q, color=INK2, lw=1.4, ms=11):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, shrinkA=0, shrinkB=0, zorder=2))


def gradient_path(points, color):
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, lw=1.7, ls=(0, (5, 3)), zorder=2, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=1.7, linestyle=(0, (5, 3)),
                                 shrinkA=0, shrinkB=0, zorder=2))


def T(s, size=9, color=INK2, weight="normal", style="normal"):
    return (s, size, color, weight, style)


# --- boxes ---------------------------------------------------------------
box(5, 10.95, 4.6, 0.8, [T("CAMELS basin attributes", 10, INK), T("+ monthly climatology", 10, INK)])
box(5, 9.5, 3.4, 0.95, [T("ParamNet", 11.5, INK, "bold"), T("LSTM + MLP (PyTorch)")],
    fc=BLUE_BG, ec=BLUE, lw=1.6)

ax.add_patch(FancyBboxPatch((1.15, 5.8), 7.7, 2.25, boxstyle="round,pad=0.02,rounding_size=0.25",
                            fc="none", ec=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1))
text(1.2, 8.3, "Two composed Tesseracts", size=9.5, color=INK, weight="bold", ha="left", va="center")

box(3.0, 6.9, 3.0, 1.35, [T("Snow-17", 11.5, INK, "bold"), T("Tesseract A · snowmelt"),
                          T("in: precip + temperature", 8, MUTED, style="italic")])
box(7.0, 6.9, 3.0, 1.35, [T("SAC-SMA", 11.5, INK, "bold"), T("Tesseract B · soil moisture"),
                          T("in: evapotranspiration", 8, MUTED, style="italic")])

box(5, 4.6, 3.4, 0.7, [T("simulated streamflow", 10, INK)])
box(5, 3.0, 3.9, 0.95, [T("NSE loss", 11.5, INK, "bold"), T("vs. observed USGS streamflow")],
    fc=ORANGE_BG, ec=ORANGE, lw=1.6)

# --- forward pass --------------------------------------------------------
arrow((5, 10.55), (5, 9.975))
arrow((4.4, 9.025), (3.3, 7.575))
arrow((5.6, 9.025), (6.7, 7.575))
text(3.72, 8.62, "θ_A (11)", ha="right", va="center")
text(6.28, 8.62, "θ_B (16)", ha="left", va="center")

arrow((4.5, 6.9), (5.5, 6.9), color=INK, lw=2.4, ms=14)
text(5, 7.2, "RAIM", size=9, color=INK, weight="bold", ha="center", va="center")
text(5, 6.58, "rain +\nmelt", size=7.8, ha="center", va="center", linespacing=0.95)

arrow((6.4, 6.225), (5.55, 4.95))
arrow((5, 4.25), (5, 3.475))

# --- gradient -----------------------------------------------------------
# One path: forward-mode JVPs carry parameter tangents through Snow-17, across
# RAIM, and through SAC-SMA (tesseract-torch chains the two endpoints); the
# resulting d(runoff)/d(theta) meets ordinary reverse-mode backprop from the
# loss into ParamNet. See src/coupling.py.
gradient_path([(6.95, 3.0), (9.5, 3.0), (9.5, 9.5), (6.7, 9.5)], BLUE)
text(9.32, 4.45, "gradient:\nforward-mode JVPs\nthrough both\nTesseracts, then\nbackprop into\nParamNet",
     size=8.3, color=BLUE, weight="bold", ha="right", va="center", multialignment="right")

fig.savefig(OUT, facecolor=SURFACE)
print(f"wrote {OUT}")
