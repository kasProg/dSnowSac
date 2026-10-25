"""LSTM-encoder + MLP head predicting Snow17 + SAC-SMA parameters from
basin static attributes AND a monthly climatology sequence (ParamNet,
model=hybrid). DailyParamNet (model=hybrid_daily, see its docstring) is
the dHBV-style alternative reading daily forcing instead; everything
below about static parameters and bounding holds for both.

Two inputs:
  x_static:  (batch, 39) z-score-normalized CAMELS static attributes
             (data/build_attributes.py) -- climate indices, topography,
             soil, vegetation, geology.
  x_climate: (batch, 12, 3) z-score-normalized monthly climatology
             (data/build_climatology.py) -- mean [prcp, tmean, pet] per
             calendar month, from each basin's full available record.

An LSTM encodes x_climate into a learned seasonal-pattern embedding
(its final hidden state), concatenated with x_static, then a small MLP
head maps the combination to all 27 parameters. Parameters stay STATIC
per rollout (one value per basin per training window) -- a hard
constraint from this project's FD-gradient-cost design (a genuinely
time-varying-parameter output would reintroduce the "one FD run per
RAIM timestep" cost src/coupling.py exists specifically to avoid, see
notes/NOTES.md). What the LSTM buys instead: richer input signal than
static summary attributes alone -- actual monthly seasonal patterns in
precipitation/temperature/PET -- while the output stays a compact,
FD-tractable parameter vector.

Output: each of the 27 parameters is mapped into its own physically
valid range via a bounded sigmoid (see *_BOUNDS below) -- never left as
raw unconstrained network output. tests/test_pipeline_hhwm8.py already
demonstrated what unconstrained direct optimization does to parameters
spanning wildly different scales (SI ~1500 next to MBASE ~0): one bad
step and the Fortran call gets fed a value it can't handle. Bounding at
the network's own output layer prevents that structurally, for every
training step, not dependent on a manually-tuned learning rate.

Methodological lineage, not code: this LSTM-encoder-plus-physical-model
pattern follows the general differentiable-parameter-learning approach
described in Feng et al. (2022, WRR) and the broader MHPI dPL line of
work, and the architecture patterns here were informed by studying
NeuralHydrology (Kratzert et al., github.com/neuralhydrology/neuralhydrology,
BSD-3-Clause) -- NOT MHPI's own dPLHBVrelease/generic_deltamodel/hydrodl2,
which are PSU Non-Commercial licensed and incompatible with this
project's Apache-2.0 submission; their source was deliberately not read
while building this file, only cited as prior work. No code from either
source is copied here; this
is an original implementation of a well-documented, published modeling
pattern.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pipeline import SACSMA_PARAMS, SNOW17_PARAMS

# (low, high) physically valid range for each parameter.
#
# Snow17 ranges: informed by the state-contract/units notes in
# notes/NOTES.md and the real calibrated HHWM8 values seen throughout
# this project (external/snow17/test_cases/ex1/input/params/snow17_params.HHWM8.txt),
# widened to plausible operational bounds rather than tightly bracketing
# just that one basin.
#
# SAC-SMA ranges: standard NWS/SCE-UA calibration bounds widely used in
# the hydrology literature (e.g. Duan, Sorooshian & Gupta 1992) -- not
# tuned specifically for this project. A reasonable-effort default given
# hackathon scope; revisit if training pushes many basins to a bound.
SNOW17_BOUNDS: dict[str, tuple[float, float]] = {
    "scf": (0.7, 2.0),
    "mfmax": (0.5, 2.0),
    "mfmin": (0.05, 0.6),
    "uadj": (0.02, 0.5),
    "si": (0.0, 2000.0),
    "nmf": (0.05, 0.5),
    "tipm": (0.01, 1.0),
    "mbase": (-1.0, 1.0),
    "pxtemp": (-2.0, 2.0),
    "plwhc": (0.02, 0.3),
    "daygm": (0.0, 0.3),
}

SACSMA_BOUNDS: dict[str, tuple[float, float]] = {
    "uztwm": (1.0, 150.0),
    "uzfwm": (1.0, 150.0),
    "uzk": (0.1, 0.5),
    "pctim": (0.0, 0.1),
    "adimp": (0.0, 0.4),
    "riva": (0.0, 0.3),
    "zperc": (1.0, 250.0),
    "rexp": (1.0, 5.0),
    "lztwm": (1.0, 500.0),
    "lzfsm": (1.0, 400.0),
    "lzfpm": (1.0, 1000.0),
    "lzsk": (0.01, 0.35),
    "lzpk": (0.0001, 0.025),
    "pfree": (0.0, 0.6),
    "side": (0.0, 0.3),
    "rserv": (0.0, 0.4),
}


class _BoundedParams(nn.Module):
    """The output side both networks share: raw (batch, 27) network
    output -> (theta_A, theta_B), (batch, 11) float32 and (batch, 16)
    float64 -- dtypes match what src/pipeline.py's CoupledNWSStack.run()
    (and src/coupling.py's per-leaf dtype handling) expect for Snow17 vs.
    SAC-SMA respectively -- each parameter sigmoid-mapped into its
    *_BOUNDS range."""

    def __init__(self) -> None:
        super().__init__()
        self.snow17_names = SNOW17_PARAMS
        self.sacsma_names = SACSMA_PARAMS
        self.n_params = len(self.snow17_names) + len(self.sacsma_names)

        snow17_bounds = torch.tensor(
            [SNOW17_BOUNDS[n] for n in self.snow17_names], dtype=torch.float64
        )
        sacsma_bounds = torch.tensor(
            [SACSMA_BOUNDS[n] for n in self.sacsma_names], dtype=torch.float64
        )
        # Registered as buffers (not parameters) so they move with
        # .to(device)/.double() etc. and are saved/loaded with the model,
        # without being trained.
        self.register_buffer("snow17_lo", snow17_bounds[:, 0])
        self.register_buffer("snow17_hi", snow17_bounds[:, 1])
        self.register_buffer("sacsma_lo", sacsma_bounds[:, 0])
        self.register_buffer("sacsma_hi", sacsma_bounds[:, 1])

    def _to_params(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = raw.to(torch.float64)
        n_a = len(self.snow17_names)
        raw_a, raw_b = raw[..., :n_a], raw[..., n_a:]
        frac_a = torch.sigmoid(raw_a)
        frac_b = torch.sigmoid(raw_b)
        theta_A = (self.snow17_lo + frac_a * (self.snow17_hi - self.snow17_lo)).to(torch.float32)
        theta_B = self.sacsma_lo + frac_b * (self.sacsma_hi - self.sacsma_lo)
        return theta_A, theta_B


class ParamNet(_BoundedParams):
    """forward(x_static, x_climate) -> (theta_A, theta_B), see
    _BoundedParams. x_climate: the (batch, 12, 3) monthly climatology."""

    def __init__(
        self,
        n_static_features: int,
        n_climate_features: int = 3,
        lstm_hidden: int = 32,
        mlp_hidden: int = 64,
        n_hidden_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_climate_features, hidden_size=lstm_hidden, batch_first=True
        ).double()

        layers: list[nn.Module] = []
        in_dim = n_static_features + lstm_hidden
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = mlp_hidden
        layers += [nn.Linear(in_dim, self.n_params)]
        self.head = nn.Sequential(*layers).double()

    def forward(
        self, x_static: torch.Tensor, x_climate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _c_n) = self.lstm(x_climate.to(torch.float64))
        climate_embed = h_n[-1]  # (batch, lstm_hidden) -- final layer's final hidden state

        combined = torch.cat([x_static.to(torch.float64), climate_embed], dim=-1)
        return self._to_params(self.head(combined))


class DailyParamNet(_BoundedParams):
    """dHBV-style parameter LSTM (Feng et al., 2022), keeping this
    project's ONE parameter set per basin.

    forward(x_static, x_forcing) -> (theta_A, theta_B), see
    _BoundedParams. x_forcing: (batch, T, 3) z-scored daily [prcp,
    tmean, pet] over the split's whole training window
    (data_module.train_period_forcing), the same sequence for every
    window the basin is trained or evaluated on.

    As in dHBV, the static attributes join the forcing at EVERY timestep,
    then Linear + ReLU in, an LSTM, and Linear out. dHBV reads parameters
    off the last timestep of the simulated window, so they shift with
    each window's weather; here the readout is the mean over all T steps
    instead -- every training year counts equally, and the parameters
    stay a property of the basin (the recovered-parameter figures need
    that). The mean is taken over hidden states before linear_out, which
    is the same as averaging per-step outputs (linear_out is affine) at
    1/T the memory. Dropout acts on that pooled state -- not dHBV's
    DropConnect on the recurrent weights.

    float32 inside (a 256-unit LSTM over ~3,300 days is ~60x slower in
    float64 on consumer GPUs); only the bounded parameters are float64.

    Memory scales with batch x T: a 100-basin training step over 3,288
    days peaks at ~5.8 GB. Without gradients (evaluation, inference) the
    forward runs EVAL_CHUNK basins at a time -- all 531 at once does not
    fit on an 11 GB card."""

    EVAL_CHUNK = 100

    def __init__(
        self, n_static_features: int, n_forcing_features: int = 3, hidden: int = 256, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.linear_in = nn.Linear(n_forcing_features + n_static_features, hidden)
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden, self.n_params)

    def forward(
        self, x_static: torch.Tensor, x_forcing: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_grad_enabled() and x_static.shape[0] > self.EVAL_CHUNK:
            parts = [
                self(x_static[i : i + self.EVAL_CHUNK], x_forcing[i : i + self.EVAL_CHUNK])
                for i in range(0, x_static.shape[0], self.EVAL_CHUNK)
            ]
            return torch.cat([a for a, _ in parts]), torch.cat([b for _, b in parts])
        n_steps = x_forcing.shape[1]
        x_static = x_static.to(torch.float32)[:, None, :].expand(-1, n_steps, -1)
        x = torch.cat([x_forcing.to(torch.float32), x_static], dim=-1)
        h, _ = self.lstm(torch.relu(self.linear_in(x)))  # (batch, T, hidden)
        return self._to_params(self.linear_out(self.dropout(h.mean(dim=1))))
