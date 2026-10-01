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


# One forward-mode direction: which parameter block ("A" = Snow17's theta_A,
# "B" = SAC-SMA's theta_B) and which index within it gets the unit tangent.
Direction = tuple[str, int]


def all_directions(n_a: int, n_b: int) -> list[Direction]:
    """Every parameter direction, in J_A-then-J_B column order."""
    return [("A", i) for i in range(n_a)] + [("B", j) for j in range(n_b)]


def jacobian_columns(
    physics: Physics,
    theta_A: torch.Tensor,
    theta_B: torch.Tensor,
    directions: list[Direction],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Runoff and its Jacobian columns for the given parameter directions,
    by forward-mode AD over the physics chain.

    One dual-level pass per direction: seed a unit tangent on that single
    parameter, run physics, and read runoff's output tangent -- exactly
    d(runoff)/d(that parameter), one length-n column. The wide RAIM
    intermediate is only ever carried as a tangent vector between the two
    Tesseracts, never materialized as a Jacobian.

    Each pass is independent of the others, which is what lets
    src/physics_pool.py spread one basin's passes across worker processes:
    all_directions() in one call is the full Jacobian; one direction per
    call is one column of it. An empty `directions` runs physics once on
    plain tensors and returns only the primal (no Jacobian, no JVP calls).

    Returns (runoff_primal, columns), both detached, columns in
    `directions` order.
    """
    if not directions:
        with torch.no_grad():
            return physics(theta_A, theta_B).detach().clone(), []

    def unit(theta: torch.Tensor, i: int | None) -> torch.Tensor:
        e = torch.zeros_like(theta)
        if i is not None:
            e[i] = 1.0
        return e

    columns: list[torch.Tensor] = []
    runoff_primal: torch.Tensor | None = None
    for block, i in directions:
        tangent_A = unit(theta_A, i if block == "A" else None)
        tangent_B = unit(theta_B, i if block == "B" else None)
        with fwAD.dual_level():
            runoff = physics(fwAD.make_dual(theta_A, tangent_A), fwAD.make_dual(theta_B, tangent_B))
            primal, tangent = fwAD.unpack_dual(runoff)
            columns.append(tangent.detach().clone() if tangent is not None else torch.zeros_like(primal))
            if runoff_primal is None:
                runoff_primal = primal.detach().clone()

    assert runoff_primal is not None
    return runoff_primal, columns


class _PhysicsRunoff(torch.autograd.Function):
    """Re-attaches autograd to a forward-mode-computed runoff.

    forward(theta_A, theta_B, J_A, J_B, runoff_primal) -> runoff

    The primal and its parameter Jacobians J_A = d(runoff)/d(theta_A),
    J_B = d(runoff)/d(theta_B) are computed outside (by jacobian_columns'
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


def needs_jacobian(theta_A: torch.Tensor, theta_B: torch.Tensor) -> bool:
    """Whether a caller will backpropagate into theta. When not (eval,
    inference, anything under torch.no_grad()), only the primal is needed
    and the len(theta_A) + len(theta_B) forward-mode passes are skipped."""
    return torch.is_grad_enabled() and (theta_A.requires_grad or theta_B.requires_grad)


def attach_jacobian(
    theta_A: torch.Tensor,
    theta_B: torch.Tensor,
    runoff_primal: torch.Tensor,
    J_A: torch.Tensor | None,
    J_B: torch.Tensor | None,
) -> torch.Tensor:
    """Runoff on theta's device, carrying J for backward. With J_A/J_B
    None (needs_jacobian() was False), the plain primal."""
    runoff_primal = runoff_primal.to(dtype=theta_B.dtype, device=theta_B.device)
    if J_A is None or J_B is None:
        return runoff_primal
    J_A = J_A.to(dtype=theta_A.dtype, device=theta_A.device)
    J_B = J_B.to(dtype=theta_B.dtype, device=theta_B.device)
    return _PhysicsRunoff.apply(theta_A, theta_B, J_A, J_B, runoff_primal)


def run_physics(
    physics: Physics,
    theta_A: torch.Tensor,
    theta_B: torch.Tensor,
) -> torch.Tensor:
    """Run the coupled Snow17 -> SAC-SMA chain and return runoff as an
    autograd tensor differentiable w.r.t. theta_A and theta_B.

    Physics derivatives come from forward-mode AD over the two Tesseracts
    (see jacobian_columns); the returned tensor carries them so an
    ordinary downstream `loss(runoff).backward()` reaches both parameter
    leaves and, above them, the network. J is computed once here regardless
    of what loss the caller applies -- and not at all when nothing will
    backpropagate (see needs_jacobian). This is the serial, single-basin
    path; src/physics_pool.py runs the same passes across processes.

    Device: the physics is Fortran behind Tesseract and only ever runs on
    CPU, so theta is moved to CPU here -- the one place the network's
    device meets the physics -- and runoff/J are moved back to theta's
    device by attach_jacobian. The network, the loss and the backward
    contraction J^T @ g all stay on whatever device theta lives on.
    """
    n_a = theta_A.numel()
    directions = all_directions(n_a, theta_B.numel()) if needs_jacobian(theta_A, theta_B) else []
    runoff_primal, columns = jacobian_columns(
        physics, theta_A.detach().cpu(), theta_B.detach().cpu(), directions
    )
    if not columns:
        return attach_jacobian(theta_A, theta_B, runoff_primal, None, None)
    J_A = torch.stack(columns[:n_a], dim=1)  # (n, n_a)
    J_B = torch.stack(columns[n_a:], dim=1)  # (n, n_b)
    return attach_jacobian(theta_A, theta_B, runoff_primal, J_A, J_B)
