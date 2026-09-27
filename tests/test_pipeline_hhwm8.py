"""Single-basin, direct-parameter-optimization proof: Snow17 -> SAC-SMA ->
NSE-style loss -> backward(), chained via src/pipeline.py's
CoupledNWSStack (built on src/coupling.py's forward-mode bridge).
The actual milestone check: loss goes down.

No real observed streamflow ships with either vendored ex1 test case --
checked directly (grep across both test_cases/ trees for obs/flow/
discharge/streamflow file names came up empty), not assumed. So this
uses a synthetic-target parameter-recovery setup instead: run the real
HHWM8 chain once at the basin's actual calibrated parameters to generate
a synthetic "observed" runoff series, then optimize FROM a perturbed
initial guess back toward it. This proves gradients flow end-to-end
through both Fortran models via real autograd and that an optimizer can
actually use them to reduce a loss -- not a claim about real calibration
skill against observed streamflow, which needs real CAMELS data (see
src/train.py).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from pipeline import (  # noqa: E402
    SACSMA_PARAMS,
    SNOW17_PARAMS,
    CoupledNWSStack,
    SacSmaForcing,
    Snow17Forcing,
)
from test_sacsma_shim import _load_ex1_forcing as _load_sacsma_forcing  # noqa: E402
from test_sacsma_shim import _load_ex1_params as _load_sacsma_params  # noqa: E402
from test_snow17_shim import _load_ex1_forcing as _load_snow17_forcing  # noqa: E402
from test_snow17_shim import _load_ex1_params as _load_snow17_params  # noqa: E402

HRU_ID = "HHWM8IL"  # same basin/HRU/period in both vendored ex1 test cases


def _theta_vector(obj, names) -> np.ndarray:
    return np.array([getattr(obj, n) for n in names], dtype=np.float64)


@pytest.fixture(scope="module")
def hhwm8_setup():
    snow_params = _load_snow17_params(HRU_ID)
    dates, pcp, tmp = _load_snow17_forcing(HRU_ID)
    mask = (dates >= "1970-10-01") & (dates <= "1971-09-30")
    dates_wy, pcp_wy, tmp_wy = dates[mask], pcp[mask], tmp[mask]
    assert len(dates_wy) == 365

    sac_params = _load_sacsma_params(HRU_ID)
    sac_dates, _sac_pcp, _sac_tmp, sac_etp = _load_sacsma_forcing(HRU_ID)
    sac_mask = (sac_dates >= "1970-10-01") & (sac_dates <= "1971-09-30")
    etp_wy = sac_etp[sac_mask]
    assert len(etp_wy) == 365

    snow17_forcing = Snow17Forcing(
        idt=24, idts=86400,
        iyr=dates_wy.year.to_numpy().astype(np.int32),
        imn=dates_wy.month.to_numpy().astype(np.int32),
        ida=dates_wy.day.to_numpy().astype(np.int32),
        pcp=pcp_wy.astype(np.float64), tmp=tmp_wy.astype(np.float64),
        alat=snow_params.alat, elev=snow_params.elev,
        adc=snow_params.adc.astype(np.float64),
        cs0=np.zeros(19, dtype=np.float64), tprev0=0.0,
    )
    sacsma_forcing = SacSmaForcing(
        dtm=86400.0, tmp=tmp_wy.astype(np.float64), etp=etp_wy.astype(np.float64),
        state0=np.zeros(6, dtype=np.float64),
    )

    stack = CoupledNWSStack()

    theta_A_true = _theta_vector(snow_params, SNOW17_PARAMS)
    theta_B_true = _theta_vector(sac_params, SACSMA_PARAMS)

    return stack, snow17_forcing, sacsma_forcing, theta_A_true, theta_B_true


def test_loss_decreases_with_synthetic_target_recovery(hhwm8_setup):
    """The actual milestone check: gradients flowing through both real
    Fortran models via CoupledNWSStack let an optimizer reduce a
    streamflow loss over a real HHWM8 water year, starting from a
    perturbed initial guess."""
    stack, snow17_forcing, sacsma_forcing, theta_A_true, theta_B_true = hhwm8_setup

    with torch.no_grad():
        theta_A_ref = torch.tensor(theta_A_true, dtype=torch.float32)
        theta_B_ref = torch.tensor(theta_B_true, dtype=torch.float64)
        observed = stack.run(theta_A_ref, theta_B_ref, snow17_forcing, sacsma_forcing).clone()
    assert torch.all(torch.isfinite(observed)) and float(observed.sum()) > 0.0

    # Perturbed initial guess: 15% off truth, multiplicative (keeps sign
    # and legitimate zeros -- e.g. PCTIM/ADIMP are 0 for this basin --
    # unperturbed rather than accidentally flipping sign).
    rng = np.random.default_rng(0)
    theta_A0 = theta_A_true * (1.0 + 0.15 * rng.uniform(-1, 1, len(theta_A_true)))
    theta_B0 = theta_B_true * (1.0 + 0.15 * rng.uniform(-1, 1, len(theta_B_true)))

    # Reparametrize: optimize a dimensionless z = theta / scale (scale =
    # each parameter's own natural magnitude, floored away from 0) rather
    # than raw physical units directly. Necessary, not cosmetic -- a first
    # attempt with plain Adam directly on raw theta (uniform lr=0.02)
    # produced NaN gradients within one step: Adam's per-step update is
    # ~lr in the PARAMETER'S OWN units regardless of that parameter's
    # sensible range, so the same lr that barely nudges SI (~1500) blew
    # MBASE (~0) and PXTEMP (~0.7, must stay near [0,1]-ish) into a
    # physically invalid region the Fortran call couldn't handle. In
    # z-space every parameter starts at the same O(1) scale, so one
    # global lr is actually meaningful for all of them -- this is the
    # standard fix for optimizing heterogeneous-scale physical parameters
    # directly, not a workaround specific to this pipeline.
    scale_A = torch.tensor(np.maximum(np.abs(theta_A_true), 1e-3), dtype=torch.float32)
    scale_B = torch.tensor(np.maximum(np.abs(theta_B_true), 1e-3), dtype=torch.float64)
    z_A = (torch.tensor(theta_A0, dtype=torch.float32) / scale_A).clone().requires_grad_(True)
    z_B = (torch.tensor(theta_B0, dtype=torch.float64) / scale_B).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z_A, z_B], lr=0.05)

    losses = []
    n_steps = 12
    for _ in range(n_steps):
        optimizer.zero_grad()
        theta_A = z_A * scale_A
        theta_B = z_B * scale_B
        sim = stack.run(theta_A, theta_B, snow17_forcing, sacsma_forcing)
        denom = torch.sum((observed - observed.mean()) ** 2)
        nse = 1.0 - torch.sum((observed - sim) ** 2) / denom
        loss = 1.0 - nse
        loss.backward()

        assert z_A.grad is not None and torch.all(torch.isfinite(z_A.grad)), (
            f"z_A.grad is None or non-finite: {z_A.grad}"
        )
        assert z_B.grad is not None and torch.all(torch.isfinite(z_B.grad)), (
            f"z_B.grad is None or non-finite: {z_B.grad}"
        )

        optimizer.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0] * 0.5, (
        f"loss did not drop substantially over {n_steps} steps: "
        f"{losses[0]:.4f} -> {losses[-1]:.4f} (full trace: {losses})"
    )


def test_coupled_gradient_matches_independent_brute_force_fd(hhwm8_setup):
    """The load-bearing correctness check: the forward-mode gradient that
    flows through both real Fortran models via CoupledNWSStack must agree
    with an INDEPENDENT brute-force finite difference of the same loss --
    computed by running the whole Snow17 -> SAC-SMA chain end to end at
    perturbed parameters, with no reference to the Tesseracts' own
    derivative endpoints. Two independent computations agreeing is what
    rules out a silent wiring bug (wrong sign, wrong parameter order,
    wrong output name, RAIM tangent dropped between containers) that
    comparing FD to itself never could.

    Checked on a handful of parameters per model (not all 27 -- each
    brute-force column is two full chained rollouts, and the point is
    corroboration across both stages and both dtypes, not exhaustive
    coverage). loss = sum(runoff), so the incoming cotangent is all-ones
    and the analytic gradient is J^T @ 1 = column sums of the parameter
    Jacobian.
    """
    stack, snow17_forcing, sacsma_forcing, theta_A_true, theta_B_true = hhwm8_setup

    theta_A = torch.tensor(theta_A_true, dtype=torch.float32, requires_grad=True)
    theta_B = torch.tensor(theta_B_true, dtype=torch.float64, requires_grad=True)
    sim = stack.run(theta_A, theta_B, snow17_forcing, sacsma_forcing)
    sim.sum().backward()  # all-ones cotangent
    grad_A_fwd = theta_A.grad.numpy().copy()
    grad_B_fwd = theta_B.grad.numpy().copy()

    def loss_at(theta_a, theta_b):
        with torch.no_grad():
            out = stack.run(
                torch.tensor(theta_a, dtype=torch.float32),
                torch.tensor(theta_b, dtype=torch.float64),
                snow17_forcing, sacsma_forcing,
            )
        return float(out.sum())

    def brute(theta, other, i, theta_is_a):
        eps = max(abs(theta[i]) * 5e-3, 1e-4)  # unrelated to the endpoints' own steps
        plus = theta.copy(); plus[i] += eps
        minus = theta.copy(); minus[i] -= eps
        if theta_is_a:
            return (loss_at(plus, other) - loss_at(minus, other)) / (2 * eps)
        return (loss_at(other, plus) - loss_at(other, minus)) / (2 * eps)

    # Tolerance: 5% relative. This is a finite-difference-vs-finite-
    # difference comparison, and Snow17 runs in float32 (~7 significant
    # digits), so neither side is an exact derivative -- a brute-force
    # central difference on the theta_A parameters visibly scatters at the
    # several-percent level as its own step size changes, and the
    # endpoints' internal steps differ from this check's. 5% catches every
    # bug this test exists to catch (a sign flip, a wrong parameter, a
    # dropped RAIM tangent between containers all miss by >100%) while not
    # flagging honest float32 FD noise. The SAC-SMA parameters (float64)
    # agree far tighter; the loose bound is set by the Snow17 side.
    for i in (0, 1, 2):  # SCF, MFMAX, MFMIN
        ref = brute(theta_A_true, theta_B_true, i, theta_is_a=True)
        assert abs(grad_A_fwd[i] - ref) <= 0.05 * abs(ref) + 1e-3, (
            f"theta_A[{i}] ({SNOW17_PARAMS[i]}): forward-mode {grad_A_fwd[i]:.6g} "
            f"vs brute-force FD {ref:.6g}"
        )
    for j in (0, 2, 8):  # UZTWM, UZK, LZTWM
        ref = brute(theta_B_true, theta_A_true, j, theta_is_a=False)
        assert abs(grad_B_fwd[j] - ref) <= 0.05 * abs(ref) + 1e-3, (
            f"theta_B[{j}] ({SACSMA_PARAMS[j]}): forward-mode {grad_B_fwd[j]:.6g} "
            f"vs brute-force FD {ref:.6g}"
        )


def test_coupled_gradient_rollout_budget(hhwm8_setup, monkeypatch):
    """Cost regression guard: one forward+backward through the coupled
    chain must stay near 3 rollouts per model per parameter pass (plain
    run + central pair), not one central pair per parameter per pass.

    The failure this catches is silent -- gradients stay correct, only the
    run count explodes. tesseract-torch hands each jacobian_vector_product
    every differentiable input, zero tangents included; without the
    zero-tangent filter in both tesseract_api.py files the HHWM8 water year
    cost 513 Snow17 + 729 SAC-SMA rollouts (vs 49 + 77 with it).
    """
    import sacsma
    import snow17

    _, snow17_forcing, sacsma_forcing, theta_A_true, theta_B_true = hhwm8_setup
    counts = {"snow17": 0, "sacsma": 0}

    def counting(fn, key):
        def wrapped(*args, **kwargs):
            counts[key] += 1
            return fn(*args, **kwargs)
        return wrapped

    # Patch before building the stack: each tesseract_api.py binds
    # run_snow17 / run_sacsma by name when it is loaded.
    monkeypatch.setattr(snow17, "run_snow17", counting(snow17.run_snow17, "snow17"))
    monkeypatch.setattr(sacsma, "run_sacsma", counting(sacsma.run_sacsma, "sacsma"))
    stack = CoupledNWSStack()

    theta_A = torch.tensor(theta_A_true, dtype=torch.float32, requires_grad=True)
    theta_B = torch.tensor(theta_B_true, dtype=torch.float64, requires_grad=True)
    stack.run(theta_A, theta_B, snow17_forcing, sacsma_forcing).sum().backward()

    n_a, n_b = len(SNOW17_PARAMS), len(SACSMA_PARAMS)
    # Snow17: plain + central pair on its own passes, plain only on SAC-SMA's.
    # SAC-SMA: plain + central pair on every pass (fewer when a Snow17
    # parameter has exactly zero effect on RAIM, e.g. SI/PXTEMP here).
    assert 0 < counts["snow17"] <= 3 * n_a + n_b, counts
    assert 0 < counts["sacsma"] <= 3 * (n_a + n_b), counts
