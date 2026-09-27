# SPDX-License-Identifier: Apache-2.0

"""Tesseract API module for SAC-SMA.

Wraps fortran/sacsma_shim.f90 (via src/sacsma.py's ctypes binding) as a
Tesseract: a full-rollout apply() plus both finite-difference derivative
endpoints -- jacobian_vector_product() (forward mode) and
vector_jacobian_product() (reverse mode). Mirrors
tesseracts/snow17/tesseract_api.py's design and structure closely -- see
that file's docstring for the shared rationale (why full-rollout
granularity, why FD is explicitly permitted by the hackathon rules) --
adapted for SAC-SMA's specifics below.

Differentiable inputs: all 16 named scalar SAC-SMA parameters (UZTWM,
UZFWM, UZK, PCTIM, ADIMP, RIVA, ZPERC, REXP, LZTWM, LZFSM, LZFPM, LZSK,
LZPK, PFREE, SIDE, RSERV) -- unlike Snow17, there's no extra curve
parameter to exclude; the actual ex1 parameter file lists exactly 16.

Real kind: EXSAC/SAC1 use explicit DOUBLE PRECISION (confirmed by reading
the source -- see notes/NOTES.md), not Snow17's default 4-byte REAL. This
schema uses Float64 throughout; do not copy Snow17's Float32 convention
here.

In the coupled pipeline this Tesseract is the DOWNSTREAM stage: Snow17's
JVP produces a tangent on RAIM, which SAC-SMA consumes as the tangent on
its `pcp` input. Because `pcp` is a per-timestep forcing array (not one
of the 16 scalar parameters), its forward-mode sensitivity is carried by
tesseract-torch as a single tangent vector along the chain -- SAC-SMA
never has to materialize the dense d(q)/d(RAIM) Jacobian that a naive
reverse-mode composition would demand. See src/coupling.py for the full
argument. This endpoint also stands alone for any single-Tesseract reuse
(training SAC-SMA parameters directly against observed RAIM).
"""

import os
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType
from tesseract_core.runtime.experimental import finite_difference_jvp, finite_difference_vjp

# Local dev: tesseract_api.py -> tesseracts/sacsma -> tesseracts -> repo root
# (3 parents up). Inside a built container this doesn't hold -- see the
# matching comment in tesseracts/snow17/tesseract_api.py for why, and
# TESSERACT_PROJECT_ROOT's role.
_REPO_ROOT = Path(os.environ.get("TESSERACT_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from sacsma import STATE_SIZE, SacSmaParams, run_sacsma  # noqa: E402

#
# Schemas
#

DIFFERENTIABLE_PARAMS = (
    "uztwm", "uzfwm", "uzk", "pctim", "adimp", "riva", "zperc", "rexp",
    "lztwm", "lzfsm", "lzfpm", "lzsk", "lzpk", "pfree", "side", "rserv",
)


class InputSchema(BaseModel):
    # Fixed timestep config -- seconds, matches EXSAC's own DTM
    # convention (it divides internally by 86400 to get days).
    dtm: Float64

    # Forcing, mm/day, deg C, mm/day. pcp IS differentiable: in the
    # coupled pipeline it receives RAIM from Snow17, and forward-mode
    # composition carries the RAIM tangent through here (that tangent is
    # the whole reason the pipeline avoids a dense d(q)/d(RAIM) Jacobian).
    # tmp/etp are not differentiated -- the LSTM predicts SAC-SMA
    # *parameters*, not forcing perturbations, and tmp currently has zero
    # effect on output anyway (only read behind the disabled IFRZE
    # frozen-ground flag -- see notes/logs.md).
    pcp: Differentiable[Array[(None,), Float64]]
    tmp: Array[(None,), Float64]
    etp: Array[(None,), Float64]

    # Learnable scalar parameters -- differentiable via the FD VJP below.
    uztwm: Differentiable[Float64]
    uzfwm: Differentiable[Float64]
    uzk: Differentiable[Float64]
    pctim: Differentiable[Float64]
    adimp: Differentiable[Float64]
    riva: Differentiable[Float64]
    zperc: Differentiable[Float64]
    rexp: Differentiable[Float64]
    lztwm: Differentiable[Float64]
    lzfsm: Differentiable[Float64]
    lzfpm: Differentiable[Float64]
    lzsk: Differentiable[Float64]
    lzpk: Differentiable[Float64]
    pfree: Differentiable[Float64]
    side: Differentiable[Float64]
    rserv: Differentiable[Float64]

    # Initial carryover state: UZTWC, UZFWC, LZTWC, LZFSC, LZFPC, ADIMC.
    # Cold start is all-zero -- see src/sacsma.py's run_sacsma docstring.
    state0: Array[(STATE_SIZE,), Float64]


class OutputSchema(BaseModel):
    q: Differentiable[Array[(None,), Float64]]      # mm/day, TCI -- "runoff" for the NSE loss
    eta: Differentiable[Array[(None,), Float64]]     # mm/day, actual ET
    qs: Array[(None,), Float64]      # diagnostic: surface flow (not differentiated)
    qg: Array[(None,), Float64]      # diagnostic: groundwater flow
    roimp: Array[(None,), Float64]
    sdro: Array[(None,), Float64]
    ssur: Array[(None,), Float64]
    sif: Array[(None,), Float64]
    bfs: Array[(None,), Float64]
    bfp: Array[(None,), Float64]
    bfncc: Array[(None,), Float64]   # needed for the mass-balance check, see tests/test_sacsma_shim.py
    state_final: Array[(STATE_SIZE,), Float64]


#
# Shared rollout call -- wrapped by apply(), which the finite-difference
# derivative endpoints call repeatedly for their base + perturbed
# evaluations.
#


def _params_from_inputs(inputs: InputSchema) -> SacSmaParams:
    return SacSmaParams(
        uztwm=float(inputs.uztwm), uzfwm=float(inputs.uzfwm), uzk=float(inputs.uzk),
        pctim=float(inputs.pctim), adimp=float(inputs.adimp), riva=float(inputs.riva),
        zperc=float(inputs.zperc), rexp=float(inputs.rexp), lztwm=float(inputs.lztwm),
        lzfsm=float(inputs.lzfsm), lzfpm=float(inputs.lzfpm), lzsk=float(inputs.lzsk),
        lzpk=float(inputs.lzpk), pfree=float(inputs.pfree), side=float(inputs.side),
        rserv=float(inputs.rserv),
    )


def _rollout(inputs: InputSchema) -> dict[str, np.ndarray]:
    """One full call to the shim: forcings + parameters + initial state ->
    q, eta, + diagnostics, final state. This is the single unit apply()
    and vector_jacobian_product()'s finite-difference evaluations both
    run."""
    out = run_sacsma(
        np.asarray(inputs.pcp, dtype=np.float64),
        np.asarray(inputs.tmp, dtype=np.float64),
        np.asarray(inputs.etp, dtype=np.float64),
        _params_from_inputs(inputs),
        state0=np.asarray(inputs.state0, dtype=np.float64),
        dtm=float(inputs.dtm),
    )
    return {
        "q": out.q,
        "eta": out.eta,
        "qs": out.qs,
        "qg": out.qg,
        "roimp": out.roimp,
        "sdro": out.sdro,
        "ssur": out.ssur,
        "sif": out.sif,
        "bfs": out.bfs,
        "bfp": out.bfp,
        "bfncc": out.bfncc,
        "state_final": out.state,
    }


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**_rollout(inputs))


#
# Optional endpoints
#

# All differentiable inputs: the 16 scalar parameters plus the pcp
# forcing (RAIM in the coupled pipeline). pcp is differentiable as a whole
# vector, not element by element -- see the endpoints below for how each
# mode handles it.
DIFFERENTIABLE_INPUTS = (*DIFFERENTIABLE_PARAMS, "pcp")

# Both derivative endpoints delegate to tesseract-core's experimental
# finite-difference helpers (see the sibling snow17 wrapper's comment for
# the shared rationale). We supply per-input absolute step sizes: for the
# scalar parameters, relative-with-floor (SAC-SMA spans UZTWM ~O(100) mm
# next to LZPK ~O(0.01), so one absolute step can't serve both). For the
# pcp forcing, the JVP is a DIRECTIONAL derivative along the incoming
# tangent, and finite_difference_jvp handles that natively -- one central
# pair, O(1) rollouts regardless of series length -- provided we hand it a
# step sized to the perturbation the tangent actually produces (below).
_FD_REL_STEP = 1e-3
_FD_MIN_STEP = 1e-4

# Step for the pcp directional perturbation. Deliberately small AND
# central (the helper default): SAC-SMA's storage-full / percolation
# conditionals are piecewise-linear kinks in pcp, and a large one-sided
# step secants ACROSS a kink at any timestep whose perturbation crosses
# it, giving a slope that is neither the left nor the right derivative and
# drifts with step size. A small central difference converges to the
# honest local slope (verified stable to ~1e-8 across eps from 1e-3 down
# to 1e-6). Same class of hard-threshold issue documented for Snow17's
# PXTEMP; the scalar parameters mostly act smoothly, but the wide pcp
# coupling is where it bites.
_PCP_FD_REL_STEP = 1e-4


def _fd_step(value: float) -> float:
    return max(abs(value) * _FD_REL_STEP, _FD_MIN_STEP)


def _pcp_eps(inputs: InputSchema, tangent: np.typing.ArrayLike) -> float:
    """Absolute step so that finite_difference_jvp's `pcp + eps*tangent`
    perturbation has magnitude ~ _PCP_FD_REL_STEP * ||pcp|| -- i.e. the
    directional step is sized to the pcp series, not the (arbitrary-norm)
    incoming tangent."""
    pcp_norm = float(np.linalg.norm(np.asarray(inputs.pcp, dtype=np.float64))) or 1.0
    tangent_norm = float(np.linalg.norm(np.asarray(tangent, dtype=np.float64))) or 1.0
    return _PCP_FD_REL_STEP * pcp_norm / tangent_norm


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, np.typing.ArrayLike],
) -> dict[str, np.typing.ArrayLike]:
    """Forward-mode: push input tangents (scalar params and/or the pcp
    series) to output tangents, via tesseract-core's finite_difference_jvp.
    This is the endpoint the coupled forward-mode pipeline drives -- it
    arrives with a pcp tangent (Snow17's RAIM tangent) and, when parameters
    are also being learned, tangents on the 16 scalars."""
    unsupported = set(jvp_inputs) - set(DIFFERENTIABLE_INPUTS)
    if unsupported:
        raise ValueError(
            f"jacobian_vector_product only supports {DIFFERENTIABLE_INPUTS}, "
            f"got unsupported input(s): {sorted(unsupported)}"
        )
    # Drop zero-tangent inputs before differencing -- see the identical
    # step in tesseracts/snow17/tesseract_api.py's jacobian_vector_product.
    # Here it also covers pcp: on the 16 SAC-SMA-parameter passes the
    # incoming RAIM tangent is all zeros, since Snow17 had nothing seeded.
    active = {name for name in jvp_inputs if np.any(np.asarray(tangent_vector[name]))}
    if not active:
        n = len(inputs.pcp)
        return {name: np.zeros(n, dtype=np.float64) for name in jvp_outputs}
    eps: dict[str, float] = {
        name: _fd_step(float(getattr(inputs, name)))
        for name in active
        if name != "pcp"
    }
    if "pcp" in active:
        eps["pcp"] = _pcp_eps(inputs, tangent_vector["pcp"])
    return finite_difference_jvp(
        apply, inputs, active, jvp_outputs,
        {name: tangent_vector[name] for name in active},
        algorithm="central", eps=eps,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, np.typing.ArrayLike],
) -> dict[str, np.typing.ArrayLike]:
    """Reverse-mode: pull output cotangents back to input gradients, via
    tesseract-core's finite_difference_vjp. Only the scalar parameters are
    supported here -- a reverse-mode gradient w.r.t. the full pcp series is
    exactly the dense-Jacobian object the coupled pipeline is designed to
    avoid (it would cost one rollout per pcp timestep). Callers needing pcp
    sensitivity use forward mode (jacobian_vector_product) instead."""
    unsupported = set(vjp_inputs) - set(DIFFERENTIABLE_PARAMS)
    if unsupported:
        raise ValueError(
            f"vector_jacobian_product supports only the scalar parameters "
            f"{DIFFERENTIABLE_PARAMS} (not pcp -- use forward mode for that); "
            f"got unsupported input(s): {sorted(unsupported)}"
        )
    eps = {name: _fd_step(float(getattr(inputs, name))) for name in vjp_inputs}
    out = finite_difference_vjp(
        apply, inputs, vjp_inputs, vjp_outputs, cotangent_vector, algorithm="central", eps=eps
    )
    return {name: np.float64(float(np.asarray(out[name]))) for name in vjp_inputs}


def abstract_eval(abstract_inputs) -> dict:
    """Shapes only, no computation. `abstract_inputs` is a
    Tesseract-generated model with ShapeDType values in place of arrays
    (attribute access, not dict-style) -- see the equivalent snow17
    docstring for the same caveat."""
    n = abstract_inputs.pcp.shape[0]
    return {
        "q": ShapeDType(shape=(n,), dtype="float64"),
        "eta": ShapeDType(shape=(n,), dtype="float64"),
        "qs": ShapeDType(shape=(n,), dtype="float64"),
        "qg": ShapeDType(shape=(n,), dtype="float64"),
        "roimp": ShapeDType(shape=(n,), dtype="float64"),
        "sdro": ShapeDType(shape=(n,), dtype="float64"),
        "ssur": ShapeDType(shape=(n,), dtype="float64"),
        "sif": ShapeDType(shape=(n,), dtype="float64"),
        "bfs": ShapeDType(shape=(n,), dtype="float64"),
        "bfp": ShapeDType(shape=(n,), dtype="float64"),
        "bfncc": ShapeDType(shape=(n,), dtype="float64"),
        "state_final": ShapeDType(shape=(STATE_SIZE,), dtype="float64"),
    }
