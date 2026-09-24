# SPDX-License-Identifier: Apache-2.0

"""Tesseract API module for Snow-17.

Wraps fortran/snow17_shim.f90 (via src/snow17.py's ctypes binding) as a
Tesseract: a full-rollout apply() plus both finite-difference derivative
endpoints -- jacobian_vector_product() (forward mode) and
vector_jacobian_product() (reverse mode) -- so PyTorch's autograd can
cross the Fortran boundary in either direction via tesseract-torch's
apply_tesseract().

Differentiable inputs: the 11 named scalar snow17 parameters (SCF, MFMAX,
MFMIN, UADJ, SI, NMF, TIPM, MBASE, PXTEMP, PLWHC, DAYGM). The 11-point ADC
(areal depletion curve) is intentionally NOT differentiable in this
version -- see notes/logs.md for why.

Both derivative endpoints delegate to tesseract-core's experimental
finite-difference helpers (finite_difference_jvp / finite_difference_vjp)
over this module's apply(); we supply per-parameter step sizes (see
_fd_eps). Wrapped at full-rollout granularity (this whole module runs one
call to EXSNOW19 per timestep, not per Tesseract call), per Tesseract's
own guidance that it targets kernels running at least several seconds --
finite-differencing per-timestep would be both wrong (state carries
across timesteps within a rollout) and far too fine-grained.

The coupled Snow17 -> SAC-SMA training pipeline (src/pipeline.py) drives
composition in FORWARD mode: it seeds a tangent on each snow17 parameter
and lets tesseract-torch chain this JVP endpoint into SAC-SMA's, carrying
the RAIM tangent between them without ever forming the dense
d(runoff)/d(RAIM) Jacobian. See src/coupling.py for why forward mode is
the natural fit here (few parameters, one wide intermediate flux, a
scalar loss).
"""

import os
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from tesseract_core.runtime import Array, Differentiable, Float32, Int32, ShapeDType
from tesseract_core.runtime.experimental import finite_difference_jvp, finite_difference_vjp

# Local dev: tesseract_api.py -> tesseracts/snow17 -> tesseracts -> repo root
# (3 parents up). Inside a built container, tesseract_api.py instead lands
# flat at /tesseract/tesseract_api.py (see Dockerfile.base -- COPY places
# it directly at that path, not nested), so the 3-parents-up relationship
# doesn't hold there. tesseract_config.yaml's build_config sets
# TESSERACT_PROJECT_ROOT=/tesseract via `env:` for that case, with
# package_data laying out src/ and fortran/ under it to match. Falls back
# to the local-dev relative path when the env var isn't set.
_REPO_ROOT = Path(os.environ.get("TESSERACT_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from snow17 import CS_SIZE, Snow17Params, run_snow17  # noqa: E402

#
# Schemas
#

# The 11 named scalar parameters snow17_shim.f90 exposes as learnable,
# minus ADC (see module docstring). Order matters only for iteration
# below, not for correctness.
DIFFERENTIABLE_PARAMS = (
    "scf", "mfmax", "mfmin", "uadj", "si",
    "nmf", "tipm", "mbase", "pxtemp", "plwhc", "daygm",
)


class InputSchema(BaseModel):
    # Fixed timestep config -- must match what the shim was built/tested
    # against (fortran/snow17_shim.f90, tests/test_snow17_shim.py use idt=24,
    # idts=86400 throughout, i.e. daily).
    idt: Int32
    idts: Int32

    # Per-timestep calendar (avoids Gregorian/leap-year arithmetic
    # anywhere in this stack -- same reasoning as the shim itself).
    iyr: Array[(None,), Int32]
    imn: Array[(None,), Int32]
    ida: Array[(None,), Int32]

    # Forcing, mm/day and deg C. Not differentiated here: the LSTM this
    # Tesseract feeds into predicts snow17 *parameters* from basin
    # attributes, not forcing perturbations.
    pcp: Array[(None,), Float32]
    tmp: Array[(None,), Float32]

    # Basin attributes -- not learnable.
    alat: Float32
    elev: Float32

    # Learnable scalar parameters -- differentiable via the FD VJP below.
    scf: Differentiable[Float32]
    mfmax: Differentiable[Float32]
    mfmin: Differentiable[Float32]
    uadj: Differentiable[Float32]
    si: Differentiable[Float32]
    nmf: Differentiable[Float32]
    tipm: Differentiable[Float32]
    mbase: Differentiable[Float32]
    pxtemp: Differentiable[Float32]
    plwhc: Differentiable[Float32]
    daygm: Differentiable[Float32]

    # Areal depletion curve -- fixed for v1, see module docstring.
    adc: Array[(11,), Float32]

    # Initial carryover state. Cold start is CS=0, TPREV=0 -- EXSNOW19's
    # own documented convention.
    cs0: Array[(CS_SIZE,), Float32]
    tprev0: Float32


class OutputSchema(BaseModel):
    raim: Differentiable[Array[(None,), Float32]]   # mm/day, rain-plus-melt -> feeds SAC-SMA
    sneqv: Differentiable[Array[(None,), Float32]]   # m, SWE
    snowh: Array[(None,), Float32]                   # m, snow depth (diagnostic; not differentiated)
    cs_final: Array[(CS_SIZE,), Float32]
    tprev_final: Float32


#
# Shared rollout call -- used by apply() and, repeatedly, by
# vector_jacobian_product() for the base + perturbed evaluations.
#


class _ArrayDates:
    """Adapts plain iyr/imn/ida int arrays to the .year/.month/.day
    attribute interface run_snow17() expects (normally a pandas
    DatetimeIndex) -- keeps this module free of a pandas dependency."""

    def __init__(self, iyr: np.ndarray, imn: np.ndarray, ida: np.ndarray) -> None:
        self.year, self.month, self.day = iyr, imn, ida


def _params_from_inputs(inputs: InputSchema) -> Snow17Params:
    return Snow17Params(
        alat=float(inputs.alat), elev=float(inputs.elev),
        scf=float(inputs.scf), mfmax=float(inputs.mfmax), mfmin=float(inputs.mfmin),
        uadj=float(inputs.uadj), si=float(inputs.si), nmf=float(inputs.nmf),
        tipm=float(inputs.tipm), mbase=float(inputs.mbase), pxtemp=float(inputs.pxtemp),
        plwhc=float(inputs.plwhc), daygm=float(inputs.daygm),
        adc=np.asarray(inputs.adc, dtype=np.float32),
    )


def _rollout(inputs: InputSchema) -> dict[str, np.ndarray]:
    """One full call to the shim: forcings + parameters + initial state ->
    raim, sneqv, snowh, final state. This is the single unit apply() and
    vector_jacobian_product()'s finite-difference evaluations both run."""
    dates = _ArrayDates(
        np.asarray(inputs.iyr), np.asarray(inputs.imn), np.asarray(inputs.ida)
    )
    out = run_snow17(
        dates,
        np.asarray(inputs.pcp, dtype=np.float32),
        np.asarray(inputs.tmp, dtype=np.float32),
        _params_from_inputs(inputs),
        cs0=np.asarray(inputs.cs0, dtype=np.float32),
        tprev0=float(inputs.tprev0),
        idt=int(inputs.idt),
        idts=int(inputs.idts),
    )
    return {
        "raim": out.raim,
        "sneqv": out.sneqv,
        "snowh": out.snowh,
        "cs_final": out.cs,
        "tprev_final": np.float32(out.tprev),
    }


#
# Required endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**_rollout(inputs))


#
# Optional endpoints
#

# Both derivative endpoints delegate to tesseract-core's experimental
# finite-difference helpers (finite_difference_jvp / finite_difference_vjp).
# The JVP helper carries a tangent through as a single directional
# derivative (O(1) rollouts, the property the coupled pipeline relies on).
# We supply per-parameter step sizes: the helpers take `eps` as an ABSOLUTE
# perturbation, but Snow17's parameters span very different magnitudes (SI
# ~1500 next to MBASE ~0), so a single absolute step is simultaneously too
# coarse for the small ones and too fine (float32-noise-dominated) for the
# large ones. _fd_eps applies a relative step with a floor. Central
# differencing (the helper default) is used throughout -- it kills the
# leading truncation term a one-sided difference carries, worth the extra
# rollout given float32 noise.
_FD_REL_STEP = 1e-3
_FD_MIN_STEP = 1e-4


def _fd_step(value: float) -> float:
    return max(abs(value) * _FD_REL_STEP, _FD_MIN_STEP)


def _fd_eps(inputs: InputSchema, names: set[str], endpoint: str) -> dict[str, float]:
    """Per-path absolute step for the FD helpers, from each parameter's own
    magnitude. Also the single place unsupported inputs are rejected --
    ADC and forcing/state are not differentiable (see module docstring)."""
    unsupported = set(names) - set(DIFFERENTIABLE_PARAMS)
    if unsupported:
        raise ValueError(
            f"{endpoint} only supports {DIFFERENTIABLE_PARAMS}, "
            f"got unsupported input(s): {sorted(unsupported)}"
        )
    return {name: _fd_step(float(getattr(inputs, name))) for name in names}


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, np.typing.ArrayLike],
) -> dict[str, np.typing.ArrayLike]:
    """Forward-mode: push input parameter tangents to output tangents,
    via tesseract-core's finite_difference_jvp. This is the endpoint the
    coupled forward-mode pipeline drives."""
    eps = _fd_eps(inputs, jvp_inputs, "jacobian_vector_product")
    out = finite_difference_jvp(
        apply, inputs, jvp_inputs, jvp_outputs, tangent_vector, algorithm="central", eps=eps
    )
    return {name: np.asarray(out[name], dtype=np.float32) for name in jvp_outputs}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, np.typing.ArrayLike],
) -> dict[str, np.typing.ArrayLike]:
    """Reverse-mode: pull output cotangents back to input parameter grads,
    via tesseract-core's finite_difference_vjp."""
    eps = _fd_eps(inputs, vjp_inputs, "vector_jacobian_product")
    out = finite_difference_vjp(
        apply, inputs, vjp_inputs, vjp_outputs, cotangent_vector, algorithm="central", eps=eps
    )
    return {name: np.float32(float(np.asarray(out[name]))) for name in vjp_inputs}


def abstract_eval(abstract_inputs) -> dict:
    """Shapes only, no computation -- output array lengths are exactly
    the input series length; state arrays are the fixed CS_SIZE/scalar
    shapes regardless of series length.

    `abstract_inputs` is an instance of a Tesseract-generated model with
    the same fields as InputSchema but ShapeDType values in place of
    arrays (attribute access, not dict-style -- it is not a plain dict
    despite the type hint convention used elsewhere in this file)."""
    n = abstract_inputs.pcp.shape[0]
    return {
        "raim": ShapeDType(shape=(n,), dtype="float32"),
        "sneqv": ShapeDType(shape=(n,), dtype="float32"),
        "snowh": ShapeDType(shape=(n,), dtype="float32"),
        "cs_final": ShapeDType(shape=(CS_SIZE,), dtype="float32"),
        "tprev_final": ShapeDType(shape=(), dtype="float32"),
    }
