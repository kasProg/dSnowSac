"""Forward-mode bridge across the Snow17 -> SAC-SMA Tesseract chain.

The pipeline is:

    theta_A, theta_B  (predicted by a PyTorch network)
        -> [Snow17 Tesseract]  --RAIM-->  [SAC-SMA Tesseract]
        -> runoff  ->  scalar loss (NSE) vs. observed streamflow

Both models are compiled Fortran wrapped as Tesseracts whose derivatives
come from finite differences (see each tesseract_api.py). The question is
only how to compose them into PyTorch's autograd graph.

Why forward mode. Snow17's parameters reach runoff only through RAIM, a
full daily time series (~thousands of values). A reverse-mode composition
would ask SAC-SMA for d(runoff)/d(RAIM) -- a dense n x n Jacobian against
that intermediate flux, which finite differences can only build one
column per rollout: thousands of rollouts. Forward mode never forms it.
It seeds a tangent on a parameter, Snow17's jacobian_vector_product turns
it into a RAIM *tangent* (one vector), and SAC-SMA's jacobian_vector_product
consumes that vector and returns a runoff tangent -- a constant number of
rollouts, independent of series length. This is the natural mode here:
few inputs (27 parameters), one wide intermediate, a scalar output.

tesseract-torch already dispatches PyTorch forward-mode AD
(torch.autograd.forward_ad dual tensors) to each Tesseract's
jacobian_vector_product endpoint, and chains the two automatically when
one's differentiable output feeds the other's differentiable input. So
the composition itself is `apply_tesseract(snow17) -> apply_tesseract(sacsma)`
under a dual_level context, with no hand-written cross-container gradient
code. src/pipeline.py builds that chain as a `physics(theta_A, theta_B)
-> runoff` callable.

The one seam this module owns. Forward mode differentiates the physics;
theta is produced by a network that trains by reverse mode. run_physics()
below builds the parameter Jacobian J = d(runoff)/d(theta) one column at a
time by forward-mode AD over the Tesseract chain (27 columns, each one
dual-level pass), then hands runoff back as an ordinary autograd tensor
carrying J. Downstream, `loss(runoff).backward()` contracts J against
d(loss)/d(runoff) by normal reverse mode and flows on into the network.
Forward mode across the physics, reverse mode across the network and the
loss, joined at runoff -- and the loss stays plain PyTorch, outside this
module.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.autograd.forward_ad as fwAD

# A "physics" callable maps the two parameter vectors to a runoff tensor by
# chaining the two Tesseracts. It must be built from apply_tesseract so that,
# under a forward_ad.dual_level() context, dual tensors passed in carry their
# tangents through both jacobian_vector_product endpoints. src/pipeline.py's
# CoupledNWSStack builds the real one; tests pass cheap stand-ins.
# Signature: physics(theta_A, theta_B) -> runoff, all torch tensors.
Physics = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _parameter_jacobian(
    physics: Physics,
    theta_A: torch.Tensor,
    theta_B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runoff and its Jacobian columns w.r.t. each parameter, by
    forward-mode AD over the physics chain.

    One dual-level pass per parameter: seed a unit tangent on that single
    parameter, run physics, and read runoff's output tangent -- exactly
    d(runoff)/d(that parameter), one length-n column. len(theta_A) +
    len(theta_B) passes total, independent of series length; the wide RAIM
    intermediate is only ever carried as a tangent vector between the two
    Tesseracts, never materialized as a Jacobian.

    Returns (runoff_primal, J_A, J_B) with J_A of shape (n, len(theta_A))
    and J_B of shape (n, len(theta_B)). runoff_primal is detached; the
    caller re-attaches autograd through _PhysicsRunoff.
    """
    n_a, n_b = theta_A.numel(), theta_B.numel()
    zero_A = torch.zeros_like(theta_A)
    zero_B = torch.zeros_like(theta_B)

    def unit(theta: torch.Tensor, i: int) -> torch.Tensor:
        e = torch.zeros_like(theta)
        e[i] = 1.0
        return e

    cols_A: list[torch.Tensor] = []
    cols_B: list[torch.Tensor] = []
    runoff_primal: torch.Tensor | None = None

    for i in range(n_a):
        with fwAD.dual_level():
            runoff = physics(fwAD.make_dual(theta_A, unit(theta_A, i)), fwAD.make_dual(theta_B, zero_B))
            primal, tangent = fwAD.unpack_dual(runoff)
            cols_A.append(tangent.detach().clone() if tangent is not None else torch.zeros_like(primal))
            if runoff_primal is None:
                runoff_primal = primal.detach().clone()

    for j in range(n_b):
        with fwAD.dual_level():
            runoff = physics(fwAD.make_dual(theta_A, zero_A), fwAD.make_dual(theta_B, unit(theta_B, j)))
            _primal, tangent = fwAD.unpack_dual(runoff)
            cols_B.append(tangent.detach().clone() if tangent is not None else torch.zeros_like(_primal))

    assert runoff_primal is not None  # theta_A always has >= 1 parameter
    J_A = torch.stack(cols_A, dim=1)  # (n, n_a)
    J_B = torch.stack(cols_B, dim=1)  # (n, n_b)
    return runoff_primal, J_A, J_B


class _PhysicsRunoff(torch.autograd.Function):
    """Re-attaches autograd to a forward-mode-computed runoff.

    forward(theta_A, theta_B, J_A, J_B, runoff_primal) -> runoff

    The primal and its parameter Jacobians J_A = d(runoff)/d(theta_A),
    J_B = d(runoff)/d(theta_B) are computed outside (by _parameter_jacobian's
    forward-mode passes) and passed in. backward() contracts the incoming
    runoff cotangent against those Jacobians -- grad_theta = J^T @ g_runoff
    -- giving each parameter leaf its gradient in its own dtype. J_A/J_B
    are non-tensor-leaf autograd inputs only in the sense that they carry
    no grad themselves; backward returns None for their slots.
    """

    @staticmethod
    def forward(ctx, theta_A, theta_B, J_A, J_B, runoff_primal):
        ctx.save_for_backward(J_A, J_B)
        ctx.theta_A_dtype, ctx.theta_A_device = theta_A.dtype, theta_A.device
        ctx.theta_B_dtype, ctx.theta_B_device = theta_B.dtype, theta_B.device
        return runoff_primal

    @staticmethod
    def backward(ctx, grad_runoff):
        J_A, J_B = ctx.saved_tensors
        # Contract in each Jacobian's own dtype (J_A is float32, J_B float64),
        # then land each gradient on its parameter leaf's dtype/device.
        grad_A = (J_A.transpose(0, 1) @ grad_runoff.to(J_A.dtype)).to(
            dtype=ctx.theta_A_dtype, device=ctx.theta_A_device
        )
        grad_B = (J_B.transpose(0, 1) @ grad_runoff.to(J_B.dtype)).to(
            dtype=ctx.theta_B_dtype, device=ctx.theta_B_device
        )
        return grad_A, grad_B, None, None, None


def run_physics(
    physics: Physics,
    theta_A: torch.Tensor,
    theta_B: torch.Tensor,
) -> torch.Tensor:
    """Run the coupled Snow17 -> SAC-SMA chain and return runoff as an
    autograd tensor differentiable w.r.t. theta_A and theta_B.

    Physics derivatives come from forward-mode AD over the two Tesseracts
    (see _parameter_jacobian); the returned tensor carries them so an
    ordinary downstream `loss(runoff).backward()` reaches both parameter
    leaves and, above them, the network. J is computed once here regardless
    of what loss the caller applies.

    Device: the physics is Fortran behind Tesseract and only ever runs on
    CPU, so theta is moved to CPU here -- the one place the network's
    device meets the physics -- and runoff/J are moved back to theta's
    device below. The network, the loss and the backward contraction
    J^T @ g all stay on whatever device theta lives on (e.g. CUDA).
    """
    runoff_primal, J_A, J_B = _parameter_jacobian(
        physics, theta_A.detach().cpu(), theta_B.detach().cpu()
    )
    J_A = J_A.to(dtype=theta_A.dtype, device=theta_A.device)
    J_B = J_B.to(dtype=theta_B.dtype, device=theta_B.device)
    runoff_primal = runoff_primal.to(dtype=theta_B.dtype, device=theta_B.device)
    return _PhysicsRunoff.apply(theta_A, theta_B, J_A, J_B, runoff_primal)
