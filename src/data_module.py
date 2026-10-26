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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from camels_loader import load_basin_timeseries
from pipeline import SacSmaForcing, Snow17Forcing
from prepare_dataset import ensure_dataset

# configs/data/camels_list.yaml names a dataset after its basin list's
# file stem: name: ${path_stem:${data.basin_list}}.
OmegaConf.register_new_resolver("path_stem", lambda path: Path(path).stem, replace=True)


class NoObservationsError(ValueError):
    """A basin has no valid observed streamflow in the requested window."""

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
    shared window and the temporal split's per-group windows.

    warmup > 0: the forcing starts `warmup` days before window_start, so
    the physics spins up from its cold start on the preceding forcing
    (the end of the training period, when a test window directly
    follows it) and the
    window itself starts from warmed-up states. Observations and scores
    cover the window only -- use scored() to drop the warmup days from a
    simulation."""

    def __init__(
        self, gauge_id: str, window_start: pd.Timestamp, window_end: pd.Timestamp, warmup: int = 0
    ):
        # Every forcing day: a warmup may reach back before the gauge's
        # streamflow record starts (missing flow -> NaN, never scored).
        ts = load_basin_timeseries(gauge_id, all_forcing_days=True)
        run_start = window_start - pd.Timedelta(days=warmup)
        run_mask = (ts.dates >= run_start) & (ts.dates <= window_end)
        mask = (ts.dates >= window_start) & (ts.dates <= window_end)
        n_before = int((run_mask & (ts.dates < window_start)).sum())
        if n_before != warmup:
            raise ValueError(
                f"{gauge_id}: {warmup} warmup days requested before {window_start.date()}, "
                f"but only {n_before} days of data precede it (CAMELS starts 1980-01-01)"
            )
        dates = ts.dates[run_mask]

        self.gauge_id = gauge_id
        self.window_start = window_start
        self.window_end = window_end
        self.warmup = warmup
        # Unique per basin, window AND warmup -- the temporal split has
        # the same gauge in both groups. PhysicsPool looks forcing up by this.
        self.key = f"{gauge_id}/{window_start.date()}/{window_end.date()}" + (
            f"/warmup{warmup}" if warmup else ""
        )
        self.snow17_forcing = Snow17Forcing(
            idt=24, idts=86400,
            iyr=dates.year.to_numpy().astype(np.int32),
            imn=dates.month.to_numpy().astype(np.int32),
            ida=dates.day.to_numpy().astype(np.int32),
            pcp=ts.prcp[run_mask], tmp=ts.tmean[run_mask],
            alat=ts.lat, elev=ts.elev,
            adc=DEFAULT_ADC,
            cs0=np.zeros(19, dtype=np.float64), tprev0=0.0,
        )
        self.sacsma_forcing = SacSmaForcing(
            dtm=86400.0, tmp=ts.tmean[run_mask], etp=ts.pet[run_mask],
            state0=np.zeros(6, dtype=np.float64),
        )
        q_obs = ts.q_obs[mask]
        self.observed = torch.tensor(q_obs, dtype=torch.float64)
        self.valid_mask = torch.tensor(~np.isnan(q_obs))
        if self.valid_mask.sum() == 0:
            raise NoObservationsError(
                f"{gauge_id}: no valid observed days in window "
                f"{window_start.date()}..{window_end.date()}"
            )
        # Observed variance over the WHOLE window -- the fixed scale
        # window_nse_loss normalizes every sampled sub-window by. Same
        # clamp as masked_nse_loss, so the two agree on the full window.
        valid = q_obs[~np.isnan(q_obs)]
        self.obs_variance = max(float(np.sum((valid - valid.mean()) ** 2)), 1e-6) / valid.size

    def scored(self, sim: torch.Tensor) -> torch.Tensor:
        """The window part of a simulation over this example's forcing:
        drops the warmup days, checking the length lines up exactly."""
        n = len(self.valid_mask)
        if sim.shape[0] != self.warmup + n:
            raise ValueError(
                f"{self.key}: simulation has {sim.shape[0]} days, expected "
                f"{self.warmup} warmup + {n} window days"
            )
        return sim[self.warmup :]


def scored_pair(sim: torch.Tensor, example: BasinExample) -> tuple[torch.Tensor, torch.Tensor]:
    """(sim, obs) on the non-missing observed days of the example's
    window (sim covers its warmup too; see BasinExample.scored). Runs on
    sim's device; the example's observations are kept on CPU and moved
    per call (one series per basin -- negligible next to the physics)."""
    sim = example.scored(sim)
    mask = example.valid_mask.to(sim.device)
    return sim[mask], example.observed.to(sim.device)[mask]


def window_pair(
    sim: torch.Tensor, example: BasinExample, start: int, warmup: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(sim, obs) on the scored, non-missing observed days of one
    sampled time window -- see window_nse_loss for what sim covers."""
    # Indexes observations by forcing position: only valid without an
    # example-level warmup prepended to the forcing.
    assert example.warmup == 0, "time-window training needs examples built with warmup=0"
    stop = start + sim.shape[0]
    mask = example.valid_mask[start + warmup : stop].to(sim.device)
    return sim[warmup:][mask], example.observed[start + warmup : stop].to(sim.device)[mask]


def masked_nse_loss(sim: torch.Tensor, example: BasinExample) -> torch.Tensor:
    """1 - NSE, computed only over non-missing observed days of the
    example's window (see scored_pair)."""
    sim_valid, obs = scored_pair(sim, example)
    denom = torch.clamp(torch.sum((obs - obs.mean()) ** 2), min=1e-6)
    nse = 1.0 - torch.sum((obs - sim_valid) ** 2) / denom
    return 1.0 - nse


def nse_value(sim: torch.Tensor, example: BasinExample) -> float:
    with torch.no_grad():
        return float(1.0 - masked_nse_loss(sim, example).item())


def window_nse_loss(sim: torch.Tensor, example: BasinExample, start: int, warmup: int) -> torch.Tensor:
    """Loss for one sampled time window (src/train.py's time-window
    minibatching). `sim` is runoff for days [start, start + len(sim)) of
    the example's window, run from a cold start; its first `warmup` days
    are spin-up and not scored.

    Mean squared error over the scored days' valid observations, divided
    by the observed variance of the basin's WHOLE window, not of this
    sub-window. A per-window NSE would divide by one year's variance:
    a flat dry year would get a tiny denominator and dominate the batch.
    With warmup=0 and the full window this equals masked_nse_loss."""
    sim_valid, obs = window_pair(sim, example, start, warmup)
    return torch.mean((obs - sim_valid) ** 2) / example.obs_variance


def rmse_comb_loss(
    pairs: list[tuple[torch.Tensor, torch.Tensor]], alpha: float = 0.25, beta: float = 1e-6
) -> torch.Tensor:
    """Feng et al. (2022, WRR)'s combined loss for dHBV, written from
    the paper's equation:

        (1 - alpha) * RMSE(q) + alpha * RMSE(log10(sqrt(q + beta) + 0.1))

    The transformed term weights low flows and recessions, which the raw
    RMSE (dominated by peaks) barely sees; beta keeps sqrt's gradient
    finite at q = 0. Both RMSEs are POOLED over every (sim, obs) pair of
    the minibatch -- one per basin, from scored_pair/window_pair -- as
    over their [time, basin] batch tensor, not averaged per basin: unlike
    the NSE losses above, a basin weighs by its flow magnitude and its
    number of observed days. sim >= 0: SAC-SMA clamps TCI at zero
    (external/sac-sma/src/sac/sac1.f90, after the riparian ET E4)."""
    sim = torch.cat([s for s, _ in pairs])
    obs = torch.cat([o for _, o in pairs])

    def transform(q):
        return torch.log10(torch.sqrt(q + beta) + 0.1)

    rmse = torch.sqrt(torch.mean((sim - obs) ** 2))
    rmse_log = torch.sqrt(torch.mean((transform(sim) - transform(obs)) ** 2))
    return (1.0 - alpha) * rmse + alpha * rmse_log


def window_starts(example: BasinExample, warmup: int, rho: int) -> np.ndarray:
    """Every start index whose window -- `warmup` spin-up days then `rho`
    scored days -- fits inside the example's series and has at least one
    observed day among the scored ones (otherwise its loss is undefined)."""
    n_valid = np.concatenate([[0], np.cumsum(example.valid_mask.numpy())])
    starts = np.arange(len(example.valid_mask) - warmup - rho + 1)
    scored = n_valid[starts + warmup + rho] - n_valid[starts + warmup]
    return starts[scored > 0]


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


# Attributes and climatology are z-scored over whichever basin list they
# were built from. A network trained on list A must see list B's basins
# scaled with A's statistics, not B's own -- otherwise the same basin
# gets different inputs depending on what else is in the list. Training
# saves A's statistics next to its checkpoint; inference re-scales to them.
_NORM_KEYS = ("mean", "std", "feature_names")


def save_normalization(data_cfg, path: Path, forcing_stats: tuple | None = None) -> None:
    """forcing_stats: (mean, std) from forcing_normalization, for a
    network reading daily forcing (model=hybrid_daily)."""
    attrs = np.load(data_cfg.attributes_npz, allow_pickle=True)
    clim = np.load(data_cfg.climatology_npz, allow_pickle=True)
    forcing = {}
    if forcing_stats is not None:
        forcing = {
            "forcing_mean": forcing_stats[0],
            "forcing_std": forcing_stats[1],
            "forcing_feature_names": np.array(FORCING_FEATURES),
        }
    np.savez(
        path,
        **{f"static_{k}": attrs[k] for k in _NORM_KEYS},
        **{f"climate_{k}": clim[k] for k in _NORM_KEYS},
        **forcing,
    )


def load_forcing_normalization(norm_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The (mean, std) the network's daily forcing was scaled with in
    training -- inference must reuse them, not recompute its own."""
    saved = np.load(norm_path, allow_pickle=True)
    if "forcing_mean" not in saved:
        raise ValueError(f"{norm_path} has no forcing statistics: not from a model=hybrid_daily run")
    return saved["forcing_mean"], saved["forcing_std"]


def apply_training_normalization(
    data_cfg, X_static: dict, X_climate: dict, norm_path: Path
) -> tuple[dict, dict, bool]:
    """Re-scale features built for data_cfg's basin list to the
    statistics saved at training time. Returns (X_static, X_climate,
    changed); unchanged (bit-for-bit) when the statistics are identical,
    i.e. inference on the training basin list itself."""
    saved = np.load(norm_path, allow_pickle=True)
    changed = False
    out = []
    for prefix, npz_path, X in (
        ("static", data_cfg.attributes_npz, X_static),
        ("climate", data_cfg.climatology_npz, X_climate),
    ):
        own = np.load(npz_path, allow_pickle=True)
        if list(own["feature_names"]) != list(saved[f"{prefix}_feature_names"]):
            raise ValueError(
                f"{prefix} features of {npz_path} differ from the ones the checkpoint was "
                "trained with -- the network can't take these inputs"
            )
        if np.array_equal(own["mean"], saved[f"{prefix}_mean"]) and np.array_equal(
            own["std"], saved[f"{prefix}_std"]
        ):
            out.append(X)
            continue
        changed = True
        mean, std = saved[f"{prefix}_mean"], saved[f"{prefix}_std"]
        out.append({g: (x * own["std"] + own["mean"] - mean) / std for g, x in X.items()})
    return out[0], out[1], changed


@dataclass
class SplitData:
    train_examples: list[BasinExample]
    test_examples: list[BasinExample]
    train_ids: list[str]
    test_ids: list[str]
    X_static: dict
    # The network's sequence input: by gauge_id, the monthly climatology
    # or (model=hybrid_daily) train_period_forcing()'s daily forcing; or
    # (model=hybrid_dhbv) a WindowForcing. The daily ones are swapped in
    # by src/train.py / src/infer.py -- see model_factory.sequence_input.
    X_climate: dict


FORCING_FEATURES = ("prcp", "tmean", "pet")


def train_period_forcing(split: SplitData) -> dict[str, np.ndarray]:
    """(T, 3) raw daily [prcp, tmean, pet] by gauge_id over the split's
    training window -- DailyParamNet's input, the same sequence for every
    window a basin is trained or scored on, so it gets one parameter set.

    Cut from the examples' own forcing rather than reloaded: the
    training examples cover the training window; in a spatial split the
    test basins share that window (their eval_warmup days are dropped),
    in a temporal split they are the training basins."""
    w0, w1 = split.train_examples[0].window_start, split.train_examples[0].window_end
    forcing = {}
    for ex in split.train_examples + split.test_examples:
        if ex.gauge_id in forcing:
            continue
        if (ex.window_start, ex.window_end) != (w0, w1):
            raise ValueError(f"{ex.gauge_id}: no example over the training window {w0.date()}..{w1.date()}")
        forcing[ex.gauge_id] = _run_forcing(ex)[ex.warmup :]
    return forcing


def _run_forcing(example: BasinExample) -> np.ndarray:
    """(warmup + window days, 3) raw [prcp, tmean, pet] the physics runs on."""
    x = np.stack(
        [example.snow17_forcing.pcp, example.snow17_forcing.tmp, example.sacsma_forcing.etp], axis=1
    ).astype(np.float64)
    assert np.isfinite(x).all(), f"{example.key}: non-finite forcing"
    return x


class WindowForcing:
    """SplitData.X_climate for a network reading the SIMULATED window's
    own forcing (DailyParamNet's "last" readout, model=hybrid_dhbv):
    each example's z-scored run forcing, warmup included, sliced to the
    window the physics runs -- the LSTM sees exactly the physics' days.
    Scaled with forcing_normalization's training-basin statistics."""

    def __init__(self, examples: list[BasinExample], mean: np.ndarray, std: np.ndarray):
        self.by_key = {ex.key: (_run_forcing(ex) - mean) / std for ex in examples}

    def sequence(self, example: BasinExample, window=None) -> np.ndarray:
        x = self.by_key[example.key]
        return x if window is None else x[window[0] : window[1]]


def network_inputs(
    basins: list[BasinExample], X_static: dict, X_climate, device, windows=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """(x_static, x_sequence) for one forward pass over `basins`.
    X_climate is a per-gauge dict (climatology, or train_period_forcing:
    the same sequence whatever the window) or a WindowForcing, sliced to
    each basin's `windows` entry ([start, stop) into its run; None = the
    whole run)."""
    if isinstance(X_climate, WindowForcing):
        seqs = [X_climate.sequence(ex, w) for ex, w in zip(basins, windows or [None] * len(basins))]
    else:
        seqs = [X_climate[ex.gauge_id] for ex in basins]
    x_static = torch.tensor(
        np.stack([X_static[ex.gauge_id] for ex in basins]), dtype=torch.float64, device=device
    )
    return x_static, torch.tensor(np.stack(seqs), dtype=torch.float64, device=device)


def forcing_normalization(forcing: dict, gauge_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature (mean, std) over every day of the given basins --
    the training basins, so test basins never shape their own scaling."""
    x = np.concatenate([forcing[g] for g in gauge_ids])
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def normalize_forcing(forcing: dict, mean: np.ndarray, std: np.ndarray) -> dict:
    return {g: (x - mean) / std for g, x in forcing.items()}


def daily_inputs(split: SplitData, mode: str, mean: np.ndarray, std: np.ndarray):
    """SplitData.X_climate for a daily-forcing model, scaled by (mean,
    std): mode "train_period" or "window" -- model_factory.sequence_input."""
    if mode == "window":
        return WindowForcing(split.train_examples + split.test_examples, mean, std)
    if mode == "train_period":
        return normalize_forcing(train_period_forcing(split), mean, std)
    raise ValueError(f"no daily input for sequence_input {mode!r}")


def _build_examples(gauge_ids: list[str], start: str, end: str, warmup: int = 0) -> list[BasinExample]:
    """BasinExamples for one window, reporting every basin without
    observations at once rather than stopping at the first."""
    w0, w1 = pd.Timestamp(start), pd.Timestamp(end)
    examples, no_obs = [], []
    for g in gauge_ids:
        try:
            examples.append(BasinExample(g, w0, w1, warmup))
        except NoObservationsError:
            no_obs.append(g)
    if no_obs:
        raise NoObservationsError(
            f"{len(no_obs)} basin(s) have no valid observed streamflow in "
            f"{w0.date()}..{w1.date()}: {no_obs}. Remove them from the basin list "
            "or choose a different window (CAMELS forcing covers 1980-2014)."
        )
    return examples


def build_split(cfg, warm_train: bool = False) -> SplitData:
    """Dispatches on cfg.split.mode ("spatial" or "temporal") and
    returns train/test BasinExamples + basin ID lists from one code
    path. cfg needs cfg.data and cfg.split (the full composed Hydra
    config, or an equivalent manually built one -- see tests/test_train.py
    for the non-Hydra construction used in tests).

    cfg.eval_warmup (default 0) days of preceding forcing are prepended
    to every test example (see BasinExample). Train examples get it only
    with warm_train=True -- for scoring them (src/infer.py), not for
    training, whose examples must start at their window.

    For a data config with a basin_list (configs/data/camels_list.yaml),
    the dataset is downloaded/built first if needed -- see
    data/prepare_dataset.py."""
    ensure_dataset(cfg.data)
    selected, X_static, X_climate = load_basin_features(cfg.data)
    mode = cfg.split.mode
    test_warmup = cfg.get("eval_warmup", 0)
    train_warmup = test_warmup if warm_train else 0

    if mode == "spatial":
        train_ids = selected.loc[selected["split"] == "train", "gauge_id"].tolist()
        test_ids = selected.loc[selected["split"] == "heldout", "gauge_id"].tolist()
        w = cfg.split.window
        train_examples = _build_examples(train_ids, w.start, w.end, train_warmup)
        test_examples = _build_examples(test_ids, w.start, w.end, test_warmup)

    elif mode == "temporal":
        train_ids = test_ids = selected["gauge_id"].tolist()
        tw, ew = cfg.split.train_window, cfg.split.test_window
        train_examples = _build_examples(train_ids, tw.start, tw.end, train_warmup)
        test_examples = _build_examples(test_ids, ew.start, ew.end, test_warmup)

    else:
        raise ValueError(f"unknown split.mode: {mode!r} (expected 'spatial' or 'temporal')")

    return SplitData(train_examples, test_examples, train_ids, test_ids, X_static, X_climate)
