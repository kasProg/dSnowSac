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


def build_model(cfg, n_static: int, n_climate: int) -> nn.Module:
    """n_climate: ParamNet's climatology-sequence width (currently 3:
    prcp/tmean/pet, see data/build_climatology.py)."""
    if cfg.model.name != "hybrid":
        raise ValueError(f"unknown model.name: {cfg.model.name!r} (expected 'hybrid')")

    from paramnet import ParamNet

    return ParamNet(
        n_static_features=n_static,
        n_climate_features=n_climate,
        lstm_hidden=cfg.model.lstm_hidden,
        mlp_hidden=cfg.model.mlp_hidden,
        n_hidden_layers=cfg.model.n_hidden_layers,
        dropout=cfg.model.dropout,
    )
