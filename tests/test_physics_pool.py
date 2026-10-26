"""src/physics_pool.py against the serial reference path
(CoupledNWSStack.run -> coupling.run_physics), on the real HHWM8
Snow17 -> SAC-SMA chain.

Parallelizing must change WHERE the passes run, never WHAT they
compute: every check here is bitwise equality, not a tolerance. A
reassembly bug (columns swapped between basins, J_A/J_B split at the
wrong index, primal taken from the wrong job) would still produce
plausible-looking runoff and finite gradients -- only exact comparison
against the serial path catches it.

No CAMELS data needed: two "basins" are the same HHWM8 water year under
two keys, run at two different parameter vectors so a cross-basin mixup
shows up as a mismatch.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import physics_pool  # noqa: E402
from physics_pool import PhysicsPool  # noqa: E402
from test_pipeline_hhwm8 import hhwm8_setup  # noqa: E402, F401 -- pytest fixture


@pytest.fixture(scope="module")
def two_basins(hhwm8_setup):
    stack, snow17_forcing, sacsma_forcing, theta_A_true, theta_B_true = hhwm8_setup
    forcings = {"b0": (snow17_forcing, sacsma_forcing), "b1": (snow17_forcing, sacsma_forcing)}
    theta_A = np.stack([theta_A_true, theta_A_true * 1.1])
    theta_B = np.stack([theta_B_true, theta_B_true * 0.9])
    return stack, forcings, theta_A, theta_B


def _leaves(theta_A, theta_B):
    return (
        torch.tensor(theta_A, dtype=torch.float32, requires_grad=True),
        torch.tensor(theta_B, dtype=torch.float64, requires_grad=True),
    )


def _loss(sims):
    # Different weights per basin so swapped basins change the gradient.
    return sum((k + 1.0) * (sim**2).mean() for k, sim in enumerate(sims))


def _serial_reference(stack, forcings, theta_A, theta_B):
    tA, tB = _leaves(theta_A, theta_B)
    sims = [stack.run(tA[b], tB[b], *forcings[key]) for b, key in enumerate(("b0", "b1"))]
    _loss(sims).backward()
    return [s.detach() for s in sims], tA.grad, tB.grad


@pytest.mark.parametrize("n_workers", [0, 2])
def test_pool_is_bitwise_identical_to_serial_path(two_basins, n_workers):
    stack, forcings, theta_A, theta_B = two_basins
    ref_sims, ref_grad_A, ref_grad_B = _serial_reference(stack, forcings, theta_A, theta_B)

    tA, tB = _leaves(theta_A, theta_B)
    with PhysicsPool(forcings, n_workers=n_workers) as pool:
        sims = pool.run(tA, tB, ["b0", "b1"])
    _loss(sims).backward()

    for sim, ref in zip(sims, ref_sims):
        assert torch.equal(sim.detach(), ref)
    assert not torch.equal(ref_sims[0], ref_sims[1]), "basins identical -- mixup undetectable"
    assert torch.equal(tA.grad, ref_grad_A)
    assert torch.equal(tB.grad, ref_grad_B)
    assert tA.grad.abs().max() > 0 and tB.grad.abs().max() > 0


def test_no_grad_runs_one_primal_job_per_basin(two_basins, monkeypatch):
    stack, forcings, theta_A, theta_B = two_basins
    ref_sims, _, _ = _serial_reference(stack, forcings, theta_A, theta_B)

    jobs = []
    real_run_job = physics_pool._run_job

    def counting_run_job(stack, forcings, job):
        jobs.append(job)
        return real_run_job(stack, forcings, job)

    monkeypatch.setattr(physics_pool, "_run_job", counting_run_job)
    tA, tB = _leaves(theta_A, theta_B)
    with torch.no_grad(), PhysicsPool(forcings, n_workers=0) as pool:
        sims = pool.run(tA, tB, ["b0", "b1"])

    assert len(jobs) == 2 and all(directions == [] for *_, directions in jobs), (
        f"expected one primal-only job per basin, got {len(jobs)} jobs"
    )
    for sim, ref in zip(sims, ref_sims):
        assert sim.grad_fn is None
        assert torch.equal(sim, ref)


def test_unregistered_basin_key_raises(two_basins):
    _stack, forcings, theta_A, theta_B = two_basins
    tA, tB = _leaves(theta_A, theta_B)
    with PhysicsPool(forcings, n_workers=0) as pool, pytest.raises(KeyError, match="not registered"):
        pool.run(tA[:1], tB[:1], ["nope"])


@pytest.mark.parametrize("n_workers", [0, 2])
def test_windowed_job_matches_presliced_forcing(two_basins, n_workers):
    """A job's [start, stop) window must run exactly the forcing it names:
    same runoff and gradients, bitwise, as registering that slice as its
    own basin. Two different windows in one batch so a window applied to
    the wrong basin, or an off-by-one, changes the result."""
    stack, forcings, theta_A, theta_B = two_basins
    windows = [(40, 240), (100, 365)]
    sliced = {
        key: (sf.window(*w), cf.window(*w))
        for (key, (sf, cf)), w in zip(forcings.items(), windows)
    }
    ref_sims, ref_grad_A, ref_grad_B = _serial_reference(stack, sliced, theta_A, theta_B)

    tA, tB = _leaves(theta_A, theta_B)
    with PhysicsPool(forcings, n_workers=n_workers) as pool:
        sims = pool.run(tA, tB, ["b0", "b1"], windows)
    _loss(sims).backward()

    assert [s.shape[0] for s in sims] == [200, 265]
    for sim, ref in zip(sims, ref_sims):
        assert torch.equal(sim.detach(), ref)
    assert torch.equal(tA.grad, ref_grad_A)
    assert torch.equal(tB.grad, ref_grad_B)


def _run_dirs():
    import glob
    import tempfile

    tmp = Path(tempfile.gettempdir())
    # Ours, plus the SDK's own default output dirs, which must not appear.
    return set(glob.glob(str(tmp / "dsnowsac_tesseract_runs_*"))) | set(
        glob.glob(str(tmp / "tesseract_output_*"))
    )


@pytest.mark.parametrize("n_workers", [0, 2])
def test_tesseract_run_logs_do_not_accumulate(two_basins, n_workers):
    """Tesseract writes a run directory per endpoint call; a gradient run
    makes dozens of calls per basin and training millions. Nothing may be
    left behind after a run, and each worker's directory must be gone once
    the pool shuts down -- an earlier version filled the disk this way."""
    _stack, forcings, theta_A, theta_B = two_basins
    before = _run_dirs()
    tA, tB = _leaves(theta_A, theta_B)
    with PhysicsPool(forcings, n_workers=n_workers) as pool:
        pool.run(tA, tB, ["b0", "b1"])
    left = _run_dirs() - before
    if n_workers == 0:
        # The in-process stack keeps its (emptied) directory until exit.
        assert left == {str(pool._stack._run_dir)}
        assert list(pool._stack._run_dir.iterdir()) == []
    else:
        assert not left, f"Tesseract run directories left behind: {left}"


@pytest.mark.parametrize("n_workers", [0, 2])
def test_run_components_averages_independent_runs(two_basins, n_workers):
    """K = 2 components per basin: each basin's runoff must be the mean of
    its own components' serial runs -- and each component's gradient its
    own share of the loss. Basin 0's components are the two reference
    parameter sets; basin 1's are the same pair swapped, so averaging
    across basins or misordering components changes a gradient."""
    stack, forcings, theta_A, theta_B = two_basins
    comp_A = np.stack([theta_A, theta_A[::-1]])  # (basin, component, 11)
    comp_B = np.stack([theta_B, theta_B[::-1]])

    # Serial reference: every (basin, component) run on its own.
    ref_A, ref_B = _leaves(comp_A, comp_B)
    ref_sims = [
        torch.stack([stack.run(ref_A[b, k], ref_B[b, k], *forcings[key]) for k in range(2)]).mean(dim=0)
        for b, key in enumerate(("b0", "b1"))
    ]
    _loss(ref_sims).backward()

    tA, tB = _leaves(comp_A, comp_B)
    with PhysicsPool(forcings, n_workers=n_workers) as pool:
        sims = pool.run_components(tA, tB, ["b0", "b1"])
    _loss(sims).backward()

    for sim, ref in zip(sims, ref_sims):
        torch.testing.assert_close(sim.detach(), ref.detach(), rtol=1e-14, atol=0)
    torch.testing.assert_close(tA.grad, ref_A.grad, rtol=1e-12, atol=0)
    torch.testing.assert_close(tB.grad, ref_B.grad, rtol=1e-12, atol=0)
    assert not torch.equal(tA.grad[0], tA.grad[1]), "per-basin loss weights not reflected"
    assert all(tA.grad[b, k].abs().max() > 0 for b in range(2) for k in range(2))


def test_run_components_with_one_set_is_run(two_basins):
    stack, forcings, theta_A, theta_B = two_basins
    tA, tB = _leaves(theta_A, theta_B)
    with torch.no_grad(), PhysicsPool(forcings, n_workers=0) as pool:
        plain = pool.run(tA, tB, ["b0", "b1"])
        as_components = pool.run_components(tA[:, None], tB[:, None], ["b0", "b1"])
        two_d = pool.run_components(tA, tB, ["b0", "b1"])
    for p, c, d in zip(plain, as_components, two_d):
        assert torch.equal(p, c) and torch.equal(p, d)
