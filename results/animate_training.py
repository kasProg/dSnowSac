"""README hero animation: one held-out basin's simulated streamflow as
ParamNet trains, from the untrained initialization (epoch 0) to the
final checkpoint (epoch 150), with the median held-out NSE across all
10 held-out basins drawn as a learning curve underneath.

Every frame is real model output, not interpolation: epoch 0 is the
network exactly as src/train.py initializes it (same torch.manual_seed
-> build_split -> build_model order), and epochs 5..150 are the saved
checkpoints in results/runs/model_9yrs_spatial/checkpoints/, each run
forward through both Fortran Tesseracts. The learning curve is
history.json's median_test_nse, logged during training. Before
rendering, the final frame's NSE is asserted equal to the value stored
in test_predictions.json, so a mismatch in data, config, or model
loading fails loudly instead of animating something plausible but wrong.

Usage (matplotlib/Pillow are optional extras, as for the other plots):
    .venv/bin/python -m pip install matplotlib pillow
    .venv/bin/python results/animate_training.py            # simulate + render
    .venv/bin/python results/animate_training.py --render   # re-render from cache
Writes results/training_heldout.gif (cache: results/.animate_cache.npz).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "data"))

RUN_DIR = REPO_ROOT / "results/runs/model_9yrs_spatial"
CACHE = REPO_ROOT / "results/.animate_cache.npz"
OUT_PATH = REPO_ROOT / "results/training_heldout.gif"

# The held-out basin where learning is most visible: NSE -0.42 untrained
# (worse than predicting the mean) -> +0.82 final, one of the best fits.
# Chosen for legibility, not typicality -- the README caption says so.
BASIN = "13310700"
# Water year 1995: its fit (NSE 0.85) is close to this basin's full-period
# NSE (0.82), so the shown year is not a flattering outlier.
WATER_YEAR = 1995
SHOW_START, SHOW_END = f"{WATER_YEAR - 1}-10-01", f"{WATER_YEAR}-09-30"


def simulate() -> None:
    import torch
    from omegaconf import OmegaConf

    from data_module import build_split, nse_value
    from model_factory import build_model
    from pipeline import CoupledNWSStack

    cfg = OmegaConf.load(RUN_DIR / "config.yaml")
    torch.manual_seed(cfg.seed)  # same order as src/train.py -> same init
    split = build_split(cfg)
    ids = split.train_ids + split.test_ids
    net = build_model(cfg, split.X_static[ids[0]].shape[0], split.X_climate[ids[0]].shape[1])
    init_state = {k: v.clone() for k, v in net.state_dict().items()}

    i = split.test_ids.index(BASIN)
    ex = split.test_examples[i]
    x_static = torch.tensor(split.X_static[BASIN][None], dtype=torch.float64)
    x_climate = torch.tensor(split.X_climate[BASIN][None], dtype=torch.float64)
    stack = CoupledNWSStack()

    ckpts = sorted((RUN_DIR / "checkpoints").glob("epoch_*.pt"))
    frames = [(0, init_state)] + [
        (int(p.stem.split("_")[1]), torch.load(p, map_location="cpu")["model_state_dict"])
        for p in ckpts
    ]

    epochs, sims, nses = [], [], []
    for epoch, state in frames:
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            theta_A, theta_B = net(x_static, x_climate)
            sim = stack.run(theta_A[0], theta_B[0], ex.snow17_forcing, ex.sacsma_forcing)
        epochs.append(epoch)
        sims.append(sim.numpy())
        nses.append(nse_value(sim, ex))
        print(f"epoch {epoch:3d}  NSE {nses[-1]:+.3f}")

    stored = json.loads((RUN_DIR / "test_predictions.json").read_text())["predictions"][BASIN]["nse"]
    assert abs(nses[-1] - stored) < 1e-4, (nses[-1], stored)

    # Untrained medians over the held-out and training basins, so both
    # learning curves have a real epoch-0 point (history.json starts after
    # epoch 1).
    net.load_state_dict(init_state)
    net.eval()

    def untrained_median(gids, examples):
        with torch.no_grad():
            xs = torch.tensor(np.stack([split.X_static[g] for g in gids]), dtype=torch.float64)
            xc = torch.tensor(np.stack([split.X_climate[g] for g in gids]), dtype=torch.float64)
            a, b = net(xs, xc)
            return float(np.median([
                nse_value(stack.run(a[j], b[j], e.snow17_forcing, e.sacsma_forcing), e)
                for j, e in enumerate(examples)
            ]))

    init_median = untrained_median(split.test_ids, split.test_examples)
    init_median_train = untrained_median(split.train_ids, split.train_examples)
    print(f"untrained median NSE: held-out {init_median:+.3f}, train {init_median_train:+.3f}")

    obs = ex.observed.numpy().copy()
    obs[~ex.valid_mask.numpy()] = np.nan
    np.savez(CACHE, epochs=epochs, sims=np.stack(sims), nses=nses, obs=obs, init_median=init_median,
             init_median_train=init_median_train,
             start=cfg.split.window.start, end=cfg.split.window.end)


def render() -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from PIL import Image

    c = np.load(CACHE)
    dates = pd.date_range(str(c["start"]), str(c["end"]), freq="D")
    assert len(dates) == c["sims"].shape[1]
    show = (dates >= SHOW_START) & (dates <= SHOW_END)
    d, obs = dates[show], c["obs"][show]
    sims, nses, epochs = c["sims"][:, show], c["nses"], c["epochs"]

    hist = json.loads((RUN_DIR / "history.json").read_text())["history"]
    curve = [(0, float(c["init_median"]))] + [
        (r["epoch"], r["median_test_nse"]) for r in hist if "median_test_nse" in r
    ]
    ce, cn = np.array(curve).T
    train_curve = [(0, float(c["init_median_train"]))] + [
        (r["epoch"], r["median_train_nse"]) for r in hist
    ]
    te, tn = np.array(train_curve).T

    # Palette: validated categorical slot 1 blue + ink tokens (dataviz skill),
    # same as results/plot_basin_comparison.py.
    SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    BLUE, ORANGE, OBS_FILL = "#2a78d6", "#eb6834", "#c9c8bf"

    ymax = max(np.nanmax(obs), sims.max()) * 1.08
    images, durations = [], []
    for k, (ep, nse) in enumerate(zip(epochs, nses)):
        fig = plt.figure(figsize=(9, 5.4), dpi=100, facecolor=SURFACE)
        gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1], hspace=0.42,
                              left=0.075, right=0.975, top=0.83, bottom=0.09)
        ax, axc = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

        fig.text(0.075, 0.935, "A neural network learning to calibrate NOAA's Fortran models",
                 fontsize=14, weight="bold", color=INK)
        fig.text(0.075, 0.885, f"Streamflow in a basin it never trained on (USGS {BASIN}), water year {WATER_YEAR}",
                 fontsize=10.5, color=INK2)

        # Hydrograph: observed as a quiet filled area, fading trail of the
        # last few epochs, current epoch as the bold blue line.
        ax.set_facecolor(SURFACE)
        ax.fill_between(d, 0, obs, color=OBS_FILL, lw=0, label="Observed (USGS gauge)")
        for j, alpha in zip(range(k - 3, k), (0.10, 0.18, 0.30)):
            if j >= 0:
                ax.plot(d, sims[j], color=BLUE, lw=1.2, alpha=alpha)
        ax.plot(d, sims[k], color=BLUE, lw=2.2, label="Simulated (NOAA Snow-17 + SAC-SMA)")
        ax.set_ylim(0, ymax)
        ax.set_xlim(d[0], d[-1])
        ax.set_ylabel("mm/day", color=INK2, fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK2)

        label = "untrained: worse than predicting the average" if ep == 0 else f"epoch {ep}"
        ax.text(0.985, 0.93, f"NSE {nse:+.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=22, weight="bold", color=ORANGE if nse < 0 else BLUE,
                family="DejaVu Sans Mono",
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.9), zorder=5)
        ax.text(0.985, 0.72, label, transform=ax.transAxes, ha="right", va="top",
                fontsize=11, color=INK2,
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.9), zorder=5)

        # Learning curve: median NSE over all 10 held-out basins, drawn up
        # to the current epoch.
        axc.set_facecolor(SURFACE)
        # Training basins as the lighter line: showing it makes the widening
        # train/held-out gap visible instead of hiding it.
        axc.plot(ce, cn, color=GRID, lw=1.5)
        axc.plot(te, tn, color=GRID, lw=1.2)
        mt, m = te <= ep, ce <= ep
        axc.plot(te[mt], tn[mt], color=MUTED, lw=1.4, label="35 training basins")
        axc.plot(ce[m], cn[m], color=INK, lw=1.8, label="10 held-out basins")
        axc.plot([te[mt][-1]], [tn[mt][-1]], "o", color=MUTED, ms=4)
        axc.plot([ce[m][-1]], [cn[m][-1]], "o", color=INK, ms=5)
        ha = "right" if ep > 130 else "left"
        dx = -6 if ep > 130 else 6
        axc.annotate(f"{tn[mt][-1]:.2f}", (te[mt][-1], tn[mt][-1]), xytext=(dx, 5),
                     textcoords="offset points", fontsize=8.5, color=MUTED, ha=ha, va="bottom")
        axc.annotate(f"{cn[m][-1]:.2f}", (ce[m][-1], cn[m][-1]), xytext=(dx, -5),
                     textcoords="offset points", fontsize=9, weight="bold", color=INK,
                     ha=ha, va="top")
        axc.legend(frameon=False, loc="lower right", ncol=2, fontsize=8.5, labelcolor=INK2)
        axc.set_xlim(0, 150)
        axc.set_ylim(min(0.0, cn.min() - 0.05), 1.0)
        axc.set_yticks([0, 0.5, 1.0])
        axc.set_xlabel("training epoch", color=INK2, fontsize=9)
        axc.set_title("median NSE", loc="left", fontsize=9, color=INK2)
        axc.grid(axis="y", color=GRID, lw=0.6)

        for a in (ax, axc):
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                a.spines[side].set_color(MUTED)
            a.tick_params(colors=INK2, labelsize=8.5)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=SURFACE)
        plt.close(fig)
        images.append(Image.open(buf).convert("RGB"))
        # Hold the untrained frame, linger where learning is fastest, then
        # hold the final frame.
        durations.append(2000 if ep == 0 else 3500 if k == len(epochs) - 1
                         else 380 if ep <= 20 else 90)

    # Shared palette from first + last frames, so both the orange (untrained)
    # and blue (trained) readouts survive quantization.
    w, h = images[0].size
    sheet = Image.new("RGB", (w, 2 * h))
    sheet.paste(images[0], (0, 0))
    sheet.paste(images[-1], (0, h))
    pal = sheet.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in images]
    frames[0].save(OUT_PATH, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="re-render from cache only")
    args = ap.parse_args()
    if not args.render:
        simulate()
    render()
