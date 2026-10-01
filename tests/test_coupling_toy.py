"""Validation for src/coupling.py's forward-mode bridge (run_physics),
using cheap torch stand-ins for Snow17/SAC-SMA instead of the real
Fortran -- validates the cross-container gradient composition before
wiring it to the real Tesseracts (see notes/NOTES.md for the full
argument this validates).

Three-way check, matching the discipline used for the real Tesseracts
(verify a gradient against an independent computation, don't just trust
it):

1. run_physics's parameter gradients: forward-mode AD over the physics
   (the actual mechanism the real pipeline uses), contracted against the
   loss by reverse mode.
2. autograd ground truth: the SAME physics as one plain torch graph,
   differentiated entirely by reverse-mode .backward(). run_physics splits
   the differentiation (forward mode over physics, reverse over the loss);
   this checks that split reproduces the undivided reverse-mode answer.
3. an INDEPENDENT brute-force finite difference of the whole pipeline's
   loss, at a step size unrelated to any the Tesseracts use internally.
   Two independent computations agreeing is what rules out a wiring bug
   (wrong sign, wrong parameter, wrong output) that an FD-vs-itself
   comparison cannot.

The toy `physics` is a plain torch callable, exactly as the real
apply_tesseract-based one in src/pipeline.py is: run_physics seeds
forward-mode dual tensors into it and reads the runoff tangent, so the
stand-in must be torch-differentiable the same way the real chain is.
The stand-in deliberately has a wide intermediate flux (RAIM) between two
stages with real timestep memory, so it exercises the exact structure
forward mode exists to handle cheaply.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from coupling import run_physics  # noqa: E402

# ---------------------------------------------------------------------------
# Toy stand-ins. Same shape as the real problem on purpose:
#   - stage A (toy "Snow17"): a temperature-threshold rain/snow partition
#     (sigmoid-relaxed, mirroring Snow17's planned PXTEMP relaxation),
#     3 parameters at different scales.
#   - stage B (toy "SAC-SMA"): a leaky-reservoir recursion with real
#     memory/lag (state carries across timesteps), 3 parameters including
#     one deliberately near zero.
# Written in torch so forward-mode dual tensors flow through, exactly as
# they flow through the real apply_tesseract chain.
# ---------------------------------------------------------------------------

T = 60  # timesteps -- cheap, but long enough for the recursion to matter


def toy_snow(theta_a: torch.Tensor, precip: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
    scale, sharpness, threshold = theta_a[0], theta_a[1], theta_a[2]
    frac = torch.sigmoid(sharpness * (temp - threshold))
    return scale * precip * frac  # RAIM -- the wide intermediate flux


def toy_sac(theta_b: torch.Tensor, raim: torch.Tensor) -> torch.Tensor:
    k, c, q0 = theta_b[0], theta_b[1], theta_b[2]
    state = torch.zeros((), dtype=raim.dtype)
    out = []
    for t in range(raim.shape[0]):
        state = k * state + c * raim[t]
        out.append(state + q0)
    return torch.stack(out)


def nse_loss(sim: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """1 - NSE, so minimizing this maximizes NSE. Nonlinear
    (mean-subtraction + ratio), not just a sum -- exercises that the loss
    differentiates for free downstream of runoff by ordinary reverse mode."""
    denom = torch.sum((obs - obs.mean()) ** 2)
    nse = 1.0 - torch.sum((obs - sim) ** 2) / denom
    return 1.0 - nse


@pytest.fixture(scope="module")
def forcings():
    rng = np.random.default_rng(0)
    precip = torch.tensor(rng.gamma(1.0, 3.0, T), dtype=torch.float64)
    temp = torch.tensor(rng.normal(2.0, 4.0, T), dtype=torch.float64)  # spans the threshold
    return precip, temp


@pytest.fixture(scope="module")
def observed(forcings):
    """A fixed, arbitrary 'observed' target -- just needs to make the loss
    nontrivial and not identically zero at the test parameters."""
    precip, _temp = forcings
    rng = np.random.default_rng(1)
    return torch.tensor(np.abs(rng.normal(1.0, 0.5, T)) + 0.1 * precip.numpy(), dtype=torch.float64)


THETA_A0 = np.array([1.5, 3.0, 2.0])   # scale, sharpness, threshold
THETA_B0 = np.array([0.7, 0.5, 0.02])  # k, c, q0 (q0 deliberately near zero)


def _make_physics(forcings):
    precip, temp = forcings

    def physics(theta_A: torch.Tensor, theta_B: torch.Tensor) -> torch.Tensor:
        raim = toy_snow(theta_A, precip, temp)
        return toy_sac(theta_B, raim)

    return physics


def test_forward_mode_gradients_match_autograd_and_independent_fd(forcings, observed):
    physics = _make_physics(forcings)

    # ---- 1. run_physics (the mechanism under test) ----
    theta_A = torch.tensor(THETA_A0, dtype=torch.float64, requires_grad=True)
    theta_B = torch.tensor(THETA_B0, dtype=torch.float64, requires_grad=True)
    runoff = run_physics(physics, theta_A, theta_B)
    assert runoff.grad_fn is not None  # sanity: a real graph node
    loss = nse_loss(runoff, observed)
    loss.backward()
    grad_A_fwd = theta_A.grad.numpy().copy()
    grad_B_fwd = theta_B.grad.numpy().copy()

    # ---- 2. autograd ground truth (same physics, one undivided reverse graph) ----
    theta_A_ref = torch.tensor(THETA_A0, dtype=torch.float64, requires_grad=True)
    theta_B_ref = torch.tensor(THETA_B0, dtype=torch.float64, requires_grad=True)
    loss_ref = nse_loss(physics(theta_A_ref, theta_B_ref), observed)
    loss_ref.backward()
    np.testing.assert_allclose(grad_A_fwd, theta_A_ref.grad.numpy(), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(grad_B_fwd, theta_B_ref.grad.numpy(), rtol=1e-6, atol=1e-9)

    # ---- 3. independent brute-force FD of the LOSS ----
    obs_np = observed.numpy()
    precip_t, temp_t = forcings

    def loss_at(theta_a, theta_b):
        raim = toy_snow(torch.tensor(theta_a), precip_t, temp_t)
        runoff = toy_sac(torch.tensor(theta_b), raim)
        return float(nse_loss(runoff, torch.tensor(obs_np)))

    def brute_force_grad(theta, other, theta_is_a: bool):
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            eps = max(abs(theta[i]) * 1e-4, 1e-6)
            plus = theta.copy(); plus[i] += eps
            minus = theta.copy(); minus[i] -= eps
            if theta_is_a:
                grad[i] = (loss_at(plus, other) - loss_at(minus, other)) / (2 * eps)
            else:
                grad[i] = (loss_at(other, plus) - loss_at(other, minus)) / (2 * eps)
        return grad

    grad_A_brute = brute_force_grad(THETA_A0, THETA_B0, theta_is_a=True)
    grad_B_brute = brute_force_grad(THETA_B0, THETA_A0, theta_is_a=False)
    np.testing.assert_allclose(grad_A_fwd, grad_A_brute, rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(grad_B_fwd, grad_B_brute, rtol=1e-3, atol=1e-6)


def test_mixed_dtype_thetas_get_matching_gradient_dtypes(forcings, observed):
    """Snow17 predicts/consumes float32, SAC-SMA float64 (see
    notes/NOTES.md). Every gradient must come back in its OWN leaf's
    dtype, not the other block's -- checked here in a cheap toy run rather
    than being caught only when the slow Fortran-backed pipeline is
    exercised. The physics runs in float64 internally (the toy recursion
    upcasts), so this specifically exercises _PhysicsRunoff.backward
    landing each parameter Jacobian's contraction on the right leaf dtype.
    """
    physics = _make_physics(forcings)
    theta_A = torch.tensor(THETA_A0, dtype=torch.float32, requires_grad=True)
    theta_B = torch.tensor(THETA_B0, dtype=torch.float64, requires_grad=True)

    runoff = run_physics(physics, theta_A, theta_B)
    assert runoff.dtype == torch.float64
    nse_loss(runoff, observed).backward()

    assert theta_A.grad.dtype == torch.float32, (
        f"theta_A.grad has dtype {theta_A.grad.dtype}, expected float32 -- "
        "gradients must come back in each leaf's OWN dtype, not the other block's."
    )
    assert theta_B.grad.dtype == torch.float64
    assert torch.all(torch.isfinite(theta_A.grad))
    assert torch.all(torch.isfinite(theta_B.grad))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_thetas_run_physics_on_cpu_and_match_cpu_gradients(forcings, observed):
    """GPU training puts theta (and the loss) on CUDA, but the physics is
    Fortran behind Tesseract and only runs on CPU. run_physics must hand
    the physics CPU tensors, return runoff on theta's device, and land
    each gradient on its CUDA leaf -- numerically identical to the CPU run
    (the Jacobian is computed on CPU either way; only the J^T @ g
    contraction moves)."""
    base_physics = _make_physics(forcings)
    seen_devices = set()

    def physics(theta_A, theta_B):
        seen_devices.update({theta_A.device.type, theta_B.device.type})
        return base_physics(theta_A, theta_B)

    def grads(device):
        theta_A = torch.tensor(THETA_A0, dtype=torch.float32, device=device, requires_grad=True)
        theta_B = torch.tensor(THETA_B0, dtype=torch.float64, device=device, requires_grad=True)
        runoff = run_physics(physics, theta_A, theta_B)
        assert runoff.device.type == device
        nse_loss(runoff, observed.to(device)).backward()
        assert theta_A.grad.device.type == device and theta_B.grad.device.type == device
        return theta_A.grad.cpu().numpy(), theta_B.grad.cpu().numpy()

    grad_A_cpu, grad_B_cpu = grads("cpu")
    grad_A_cuda, grad_B_cuda = grads("cuda")

    assert seen_devices == {"cpu"}, f"physics saw devices {seen_devices}, expected CPU only"
    np.testing.assert_allclose(grad_A_cuda, grad_A_cpu, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(grad_B_cuda, grad_B_cpu, rtol=1e-10, atol=1e-12)


def test_evaluation_count_is_one_pass_per_parameter(forcings, observed):
    """Forward mode's cost is one physics evaluation per parameter
    direction (len A + len B), regardless of the RAIM series length -- the
    dense d(runoff)/d(RAIM) Jacobian is never built. Assert the physics
    callable is invoked exactly that many times for one run_physics call,
    which is the whole quantitative claim the forward-mode choice rests on.
    """
    base_physics = _make_physics(forcings)
    calls = {"n": 0}

    def counting_physics(theta_A, theta_B):
        calls["n"] += 1
        return base_physics(theta_A, theta_B)

    theta_A = torch.tensor(THETA_A0, dtype=torch.float64, requires_grad=True)
    theta_B = torch.tensor(THETA_B0, dtype=torch.float64, requires_grad=True)
    run_physics(counting_physics, theta_A, theta_B)

    assert calls["n"] == len(THETA_A0) + len(THETA_B0), (
        f"physics called {calls['n']} times, expected {len(THETA_A0) + len(THETA_B0)} "
        "(one forward-mode pass per parameter) -- if this grew with T, the wide "
        "d(runoff)/d(RAIM) Jacobian is being materialized, defeating forward mode."
    )
