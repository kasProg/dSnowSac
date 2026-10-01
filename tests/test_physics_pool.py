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
