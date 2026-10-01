"""Unified basin-example loading + train/test split logic, shared by
src/train.py (both models) and src/infer.py. Extracted from what used to
be src/train.py's module-level WINDOW_START/WINDOW_END constants and
inline basin-loading code, so the same logic serves both split modes
below instead of being hardcoded to one.

Two split modes, selected via configs/split/*.yaml's `mode` field (see
those files' own comments for the full reasoning):

  spatial  -- prediction in ungauged basins (PUB). One fixed date
              window; basins partitioned into train/test by
              data/select_basins.py's own "split" column. This is the
              setup behind every results/ JSON so far.
  temporal -- prediction in ungauged period (PUP). The SAME basin set
              for both groups, but a different date window per group --
              the shape needed for the parked full-CAMELS-671/dHBV-period
              comparison (see notes/logs.md). Not yet exercised by any
              saved results/ run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from camels_loader import load_basin_timeseries
from pipeline import SacSmaForcing, Snow17Forcing

# Fixed, not learnable -- Snow17's areal depletion curve. A standard
# generic curve shape (matches what's used throughout this project's
# earlier single-basin work); ADC is kept out of scope for the
# differentiable parameters (see notes/logs.md).
DEFAULT_ADC = np.array(
    [0.05, 0.09, 0.16, 0.31, 0.54, 0.74, 0.84, 0.89, 0.93, 0.97, 1.0], dtype=np.float64
)


class BasinExample:
    """Everything needed to run + score one basin over one date window,
    precomputed once and reused across every epoch (forcing/observations
    don't change during training). window_start/window_end were a fixed
    module-level pair in the pre-Hydra src/train.py; now an explicit
    argument so the same class serves both the spatial split's single
    shared window and the temporal split's per-group windows."""

    def __init__(self, gauge_id: str, window_start: pd.Timestamp, window_end: pd.Timestamp):
        ts = load_basin_timeseries(gauge_id)
        mask = (ts.dates >= window_start) & (ts.dates <= window_end)
        dates = ts.dates[mask]

        self.gauge_id = gauge_id
        self.window_start = window_start
        self.window_end = window_end
        # Unique per basin AND window -- the temporal split has the same
        # gauge in both groups. PhysicsPool looks forcing up by this.
        self.key = f"{gauge_id}/{window_start.date()}/{window_end.date()}"
        self.snow17_forcing = Snow17Forcing(
            idt=24, idts=86400,
            iyr=dates.year.to_numpy().astype(np.int32),
            imn=dates.month.to_numpy().astype(np.int32),
            ida=dates.day.to_numpy().astype(np.int32),
            pcp=ts.prcp[mask], tmp=ts.tmean[mask],
            alat=ts.lat, elev=ts.elev,
            adc=DEFAULT_ADC,
            cs0=np.zeros(19, dtype=np.float64), tprev0=0.0,
        )
        self.sacsma_forcing = SacSmaForcing(
            dtm=86400.0, tmp=ts.tmean[mask], etp=ts.pet[mask],
            state0=np.zeros(6, dtype=np.float64),
        )
        q_obs = ts.q_obs[mask]
        self.observed = torch.tensor(q_obs, dtype=torch.float64)
        self.valid_mask = torch.tensor(~np.isnan(q_obs))
        assert self.valid_mask.sum() > 0, (
            f"{gauge_id}: no valid observed days in window "
            f"{window_start.date()}..{window_end.date()}"
        )


def masked_nse_loss(sim: torch.Tensor, example: BasinExample) -> torch.Tensor:
    """1 - NSE, computed only over non-missing observed days. Runs on
    sim's device; the example's observations are kept on CPU and moved
    per call (one series per basin -- negligible next to the physics)."""
    mask = example.valid_mask.to(sim.device)
    obs = example.observed.to(sim.device)[mask]
    sim_valid = sim[mask]
    denom = torch.clamp(torch.sum((obs - obs.mean()) ** 2), min=1e-6)
    nse = 1.0 - torch.sum((obs - sim_valid) ** 2) / denom
    return 1.0 - nse


def nse_value(sim: torch.Tensor, example: BasinExample) -> float:
    with torch.no_grad():
        return float(1.0 - masked_nse_loss(sim, example).item())


def load_basin_features(data_cfg) -> tuple[pd.DataFrame, dict, dict]:
    """Returns (selected_basins_df, static_attrs_by_gauge_id,
    climatology_by_gauge_id), from paths in configs/data/*.yaml."""
    selected = pd.read_csv(data_cfg.selected_basins_csv, dtype={"gauge_id": str})
    attrs = np.load(data_cfg.attributes_npz, allow_pickle=True)
    gauge_ids = list(attrs["gauge_ids"])
    X_static = {gid: attrs["X"][i] for i, gid in enumerate(gauge_ids)}

    clim = np.load(data_cfg.climatology_npz, allow_pickle=True)
    clim_ids = list(clim["gauge_ids"])
    X_climate = {gid: clim["X"][i] for i, gid in enumerate(clim_ids)}
    assert set(X_static) == set(X_climate), "static/climatology basin sets don't match"

    return selected, X_static, X_climate


@dataclass
class SplitData:
    train_examples: list[BasinExample]
    test_examples: list[BasinExample]
    train_ids: list[str]
    test_ids: list[str]
    X_static: dict
    X_climate: dict


def build_split(cfg) -> SplitData:
    """Dispatches on cfg.split.mode ("spatial" or "temporal") and
    returns train/test BasinExamples + basin ID lists from one code
    path. cfg needs cfg.data and cfg.split (the full composed Hydra
    config, or an equivalent manually built one -- see tests/test_train.py
    for the non-Hydra construction used in tests)."""
    selected, X_static, X_climate = load_basin_features(cfg.data)
    mode = cfg.split.mode

    if mode == "spatial":
        train_ids = selected.loc[selected["split"] == "train", "gauge_id"].tolist()
        test_ids = selected.loc[selected["split"] == "heldout", "gauge_id"].tolist()
        w0 = pd.Timestamp(cfg.split.window.start)
        w1 = pd.Timestamp(cfg.split.window.end)
        train_examples = [BasinExample(g, w0, w1) for g in train_ids]
        test_examples = [BasinExample(g, w0, w1) for g in test_ids]

    elif mode == "temporal":
        train_ids = test_ids = selected["gauge_id"].tolist()
        tw0 = pd.Timestamp(cfg.split.train_window.start)
        tw1 = pd.Timestamp(cfg.split.train_window.end)
        ew0 = pd.Timestamp(cfg.split.test_window.start)
        ew1 = pd.Timestamp(cfg.split.test_window.end)
        train_examples = [BasinExample(g, tw0, tw1) for g in train_ids]
        test_examples = [BasinExample(g, ew0, ew1) for g in test_ids]

    else:
        raise ValueError(f"unknown split.mode: {mode!r} (expected 'spatial' or 'temporal')")

    return SplitData(train_examples, test_examples, train_ids, test_ids, X_static, X_climate)
