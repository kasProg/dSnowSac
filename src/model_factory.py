"""Model construction from cfg.model -- the one place src/train.py and
src/infer.py build the net, so infer.py can reconstruct the exact same
architecture a checkpoint was trained with without duplicating the
construction logic.

Used to dispatch on cfg.model.name between this hybrid model and a
pure-LSTM baseline (src/benchmark_lstm.py, removed -- see
notes/logs.md). Left as a plain function rather than collapsed further,
since a future second model (e.g. the parked full-CAMELS-671/temporal
comparison, see notes/logs.md) would need this same shape again.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def resolve_device(name: str) -> torch.device:
    """cfg.device -> torch.device. "auto" picks CUDA when available. Only
    the network and loss move; the Fortran physics always runs on CPU
    (see src/coupling.py's run_physics)."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device={name!r} requested but torch.cuda.is_available() is False")
    return device


MODEL_NAMES = ("hybrid", "hybrid_daily", "hybrid_dhbv")


def uses_daily_forcing(cfg) -> bool:
    """Whether the network reads daily forcing rather than the monthly
    climatology -- see sequence_input() for which forcing."""
    if cfg.model.name not in MODEL_NAMES:
        raise ValueError(f"unknown model.name: {cfg.model.name!r} (expected one of {MODEL_NAMES})")
    return cfg.model.name != "hybrid"


def sequence_input(cfg) -> str:
    """What SplitData.X_climate must hold for this model: "climatology"
    (monthly, per gauge), "train_period" (the training window's daily
    forcing, per gauge: data_module.train_period_forcing) or "window"
    (each simulated window's own daily forcing: data_module.WindowForcing)."""
    if not uses_daily_forcing(cfg):
        return "climatology"
    return "window" if cfg.model.get("readout", "mean") == "last" else "train_period"


def build_model(cfg, n_static: int, n_climate: int) -> nn.Module:
    """n_climate: the width of the network's sequence input -- 3 for
    both models (prcp/tmean/pet: monthly climatology, see
    data/build_climatology.py, or daily forcing)."""
    if uses_daily_forcing(cfg):
        from paramnet import DailyParamNet

        return DailyParamNet(
            n_static_features=n_static,
            n_forcing_features=n_climate,
            hidden=cfg.model.hidden,
            dropout=cfg.model.dropout,
            # .get: hybrid_daily runs saved before these keys used 1 and mean.
            n_components=cfg.model.get("n_components", 1),
            readout=cfg.model.get("readout", "mean"),
        )

    from paramnet import ParamNet

    return ParamNet(
        n_static_features=n_static,
        n_climate_features=n_climate,
        lstm_hidden=cfg.model.lstm_hidden,
        mlp_hidden=cfg.model.mlp_hidden,
        n_hidden_layers=cfg.model.n_hidden_layers,
        dropout=cfg.model.dropout,
    )
