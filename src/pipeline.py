"""Wires the real Snow17 and SAC-SMA Tesseracts into a single
forward-mode-differentiable runoff callable for one HRU.

The composition is the whole point: `apply_tesseract(snow17)` produces
RAIM, which feeds `apply_tesseract(sacsma)` as its `pcp` input, and
tesseract-torch chains their jacobian_vector_product endpoints under
forward-mode AD -- carrying the RAIM tangent between the two containers
without ever forming the dense d(runoff)/d(RAIM) Jacobian. src/coupling.py
drives that forward-mode pass and re-attaches the result to autograd so a
downstream loss trains the upstream network by ordinary reverse mode.

There is no hand-written cross-container gradient code here or in
coupling.py -- the gradient path is Tesseract's own forward-mode
composition. See coupling.py's module docstring for why forward mode is
the natural fit (few parameters, one wide intermediate flux, scalar loss).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tesseract_core import Tesseract
from tesseract_torch import apply_tesseract

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from coupling import run_physics  # noqa: E402

SNOW17_TESSERACT_DIR = _REPO_ROOT / "tesseracts" / "snow17"
SACSMA_TESSERACT_DIR = _REPO_ROOT / "tesseracts" / "sacsma"

# Order theta_A / theta_B vectors are expected in -- matches each
# Tesseract's own DIFFERENTIABLE_PARAMS declaration order.
SNOW17_PARAMS = (
    "scf", "mfmax", "mfmin", "uadj", "si", "nmf", "tipm", "mbase", "pxtemp", "plwhc", "daygm",
)
SACSMA_PARAMS = (
    "uztwm", "uzfwm", "uzk", "pctim", "adimp", "riva", "zperc", "rexp",
    "lztwm", "lzfsm", "lzfpm", "lzsk", "lzpk", "pfree", "side", "rserv",
)


@dataclass
class Snow17Forcing:
    """Everything Snow17's Tesseract needs besides the 11 learnable
    parameters -- static for a given basin/period."""

    idt: int
    idts: int
    iyr: np.ndarray
    imn: np.ndarray
    ida: np.ndarray
    pcp: np.ndarray
    tmp: np.ndarray
    alat: float
    elev: float
    adc: np.ndarray  # fixed, not learnable -- see tesseracts/snow17's docstring
    cs0: np.ndarray
    tprev0: float


@dataclass
class SacSmaForcing:
    """Everything SAC-SMA's Tesseract needs besides the 16 learnable
    parameters and RAIM (which comes from Snow17's stage)."""

    dtm: float
    tmp: np.ndarray
    etp: np.ndarray
    state0: np.ndarray


class CoupledNWSStack:
    """Loads both Tesseracts ONCE (local, no Docker -- see notes/logs.md).
    `.run(theta_A, theta_B, snow17_forcing, sacsma_forcing)` takes forcing
    per call, not per instance -- so one CoupledNWSStack is reused across
    every basin in multi-basin training instead of reloading the Tesseract
    clients (an expensive, basin-independent step) once per basin.
    """

    def __init__(self) -> None:
        self._snow17 = Tesseract.from_tesseract_api(str(SNOW17_TESSERACT_DIR / "tesseract_api.py"))
        self._sacsma = Tesseract.from_tesseract_api(str(SACSMA_TESSERACT_DIR / "tesseract_api.py"))

    def make_physics(self, snow17_forcing: Snow17Forcing, sacsma_forcing: SacSmaForcing):
        """Build the physics(theta_A, theta_B) -> runoff callable that
        chains the two Tesseracts through tesseract-torch. Closes over THIS
        call's forcing (not instance state) so interleaved basin runs can't
        cross-contaminate. Under coupling.run_physics's forward-mode
        dual_level context, the theta tensors arrive as dual numbers and
        their tangents flow through both jacobian_vector_product endpoints;
        the RAIM tangent is handed from Snow17's `raim` output to SAC-SMA's
        `pcp` input automatically by apply_tesseract's chaining."""
        sf, cf = snow17_forcing, sacsma_forcing

        def physics(theta_A: torch.Tensor, theta_B: torch.Tensor) -> torch.Tensor:
            snow17_inputs = dict(
                idt=sf.idt, idts=sf.idts, iyr=sf.iyr, imn=sf.imn, ida=sf.ida,
                pcp=sf.pcp.astype(np.float32), tmp=sf.tmp.astype(np.float32),
                alat=float(sf.alat), elev=float(sf.elev),
                adc=sf.adc.astype(np.float32),
                cs0=sf.cs0.astype(np.float32), tprev0=float(sf.tprev0),
            )
            for name, value in zip(SNOW17_PARAMS, theta_A):
                snow17_inputs[name] = value
            raim = apply_tesseract(self._snow17, snow17_inputs)["raim"]

            sacsma_inputs = dict(
                dtm=float(cf.dtm), pcp=raim.to(torch.float64),
                tmp=cf.tmp.astype(np.float64), etp=cf.etp.astype(np.float64),
                state0=cf.state0.astype(np.float64),
            )
            for name, value in zip(SACSMA_PARAMS, theta_B):
                sacsma_inputs[name] = value
            return apply_tesseract(self._sacsma, sacsma_inputs)["q"]

        return physics

    def run(
        self,
        theta_A: torch.Tensor,
        theta_B: torch.Tensor,
        snow17_forcing: Snow17Forcing,
        sacsma_forcing: SacSmaForcing,
    ) -> torch.Tensor:
        """theta_A: 11 Snow17 params, in SNOW17_PARAMS order (float32).
        theta_B: 16 SAC-SMA params, in SACSMA_PARAMS order (float64).
        Returns: runoff (TCI), float64 torch.Tensor, differentiable w.r.t.
        both theta_A and theta_B by forward-mode AD over the two Tesseracts
        (see src/coupling.py)."""
        physics = self.make_physics(snow17_forcing, sacsma_forcing)
        return run_physics(physics, theta_A, theta_B)
