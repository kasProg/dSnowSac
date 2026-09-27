"""README hero figure: simulated vs. observed streamflow for held-out
basins -- basins the model never saw during training -- over two water
years, so a non-hydrologist can see what "NSE 0.8" looks like.

Data: results/runs/model_9yrs_spatial/test_predictions.json (simulated,
from src/infer.py) and CAMELS observed streamflow via
data/camels_loader.py. Before plotting, each basin's NSE is recomputed
over the full run window and asserted equal to the stored value -- a
date misalignment between the two sources would fail loudly here rather
than draw a plausible but shifted hydrograph.

Usage (matplotlib is an optional extra, as for results/compare_runs.py):
    .venv/bin/python -m pip install matplotlib
    .venv/bin/python results/plot_hydrograph.py
Writes results/hydrograph_heldout.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "data"))

from camels_loader import load_basin_timeseries  # noqa: E402

PREDICTIONS = REPO_ROOT / "results/runs/model_9yrs_spatial/test_predictions.json"
OUT_PATH = REPO_ROOT / "results/hydrograph_heldout.png"

# Must match results/runs/model_9yrs_spatial/config.yaml split.window.
WINDOW_START, WINDOW_END = "1990-10-01", "1999-09-30"
# Shown slice: two water years, enough to see two melt pulses.
SHOW_START, SHOW_END = "1995-10-01", "1997-09-30"
# One strong and one median held-out basin -- not only the best one.
BASINS = ["09035900", "13023000"]

COLOR_OBS = "#0b0b0b"
COLOR_SIM = "#2a78d6"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def _nse(sim: np.ndarray, obs: np.ndarray) -> float:
    ok = ~np.isnan(obs)
    s, o = sim[ok], obs[ok]
    return 1.0 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def main() -> None:
    preds = json.loads(PREDICTIONS.read_text())["predictions"]
    window = pd.date_range(WINDOW_START, WINDOW_END, freq="D")

    fig, axes = plt.subplots(len(BASINS), 1, figsize=(10, 2.8 * len(BASINS)),
                             sharex=True, facecolor=SURFACE)
    for ax, gid in zip(np.atleast_1d(axes), BASINS):
        sim = np.asarray(preds[gid]["sim_mm_day"])
        assert len(sim) == len(window), (gid, len(sim), len(window))

        ts = load_basin_timeseries(gid)
        obs = pd.Series(ts.q_obs, index=ts.dates).reindex(window).to_numpy()
        nse = _nse(sim, obs)
        assert abs(nse - preds[gid]["nse"]) < 1e-3, (gid, nse, preds[gid]["nse"])

        show = (window >= SHOW_START) & (window <= SHOW_END)
        ax.set_facecolor(SURFACE)
        ax.plot(window[show], obs[show], color=COLOR_OBS, lw=1.4, label="Observed (USGS gauge)")
        ax.plot(window[show], sim[show], color=COLOR_SIM, lw=1.4, label="Simulated (hybrid model)")
        ax.set_title(f"USGS {gid}, held-out basin, NSE {nse:.2f}",
                     loc="left", fontsize=11, color=COLOR_OBS)
        ax.set_ylabel("Streamflow (mm/day)", color=INK_SECONDARY)
        ax.grid(axis="y", color=GRIDLINE, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_SECONDARY)

    np.atleast_1d(axes)[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, facecolor=SURFACE)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
