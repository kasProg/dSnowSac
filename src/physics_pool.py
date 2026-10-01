"""Runs the coupled Snow17 -> SAC-SMA physics for a batch of basins
across worker processes.

Why processes, and why this split. Each basin's runoff Jacobian is
len(theta_A) + len(theta_B) = 27 independent forward-mode passes
(src/coupling.py's jacobian_columns), and the basins are independent of
each other. Serially, one epoch is 35 basins x 27 passes, one after
another, on one core. Here each (basin, direction) pass is its own job,
so a batch of B basins becomes B x 27 jobs spread over n_workers
processes. Threads would not help: the per-call cost is dominated by
Python-side Tesseract schema validation (GIL-bound), not the Fortran.

What stays in the main process. Workers only ever see numpy parameter
vectors and return numpy runoff + Jacobian columns -- no autograd state
crosses the process boundary. The main process reassembles J_A/J_B per
basin and re-attaches autograd with coupling.attach_jacobian, so a
downstream loss backpropagates into the network exactly as with the
serial CoupledNWSStack.run. Results are bitwise identical to the serial
path (same code, same order of operations per pass; tested in
tests/test_physics_pool.py).

When nothing will backpropagate (eval, inference, torch.no_grad()), each
basin is one primal-only job instead of 27 Jacobian passes.

Forcing for every basin is shipped to each worker once, at pool start,
keyed by a string (BasinExample.key); a job then names its basin by key
instead of re-pickling ~3,000-day forcing series for every pass. It is
sent as each worker's first task, not as an initializer argument:
initializer arguments travel in the process-start message, and a
payload bigger than the pipe buffer (~2 MB for 45 basins) blocks the
parent until that worker finishes starting up -- serializing startup
(32 workers: ~66 s instead of ~6 s).

Worker startup. Workers come from a forkserver that has already
imported this module (torch, tesseract-core, the Fortran shims), so each
worker is a cheap fork rather than a fresh interpreter importing torch
-- with spawn, 32 workers took ~60 s to come up. Not plain fork: the
parent may hold a CUDA context and initialized torch/OpenMP thread
pools, neither of which survives fork; the forkserver process never
touches either. All workers are started and initialized in __init__,
so that cost is paid once, up front, not inside the first epoch.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from coupling import Direction, all_directions, attach_jacobian, jacobian_columns, needs_jacobian
from pipeline import CoupledNWSStack, SacSmaForcing, Snow17Forcing

Forcings = dict[str, tuple[Snow17Forcing, SacSmaForcing]]
# (basin key, theta_A, theta_B, directions) -- directions empty = primal only.
_Job = tuple[str, np.ndarray, np.ndarray, list[Direction]]

# Per-worker state, set once by _init_worker.
_WORKER_STACK: CoupledNWSStack | None = None
_WORKER_FORCINGS: Forcings | None = None
_WORKER_BARRIER: threading.Barrier | None = None

_STARTUP_TIMEOUT_S = 600


def _run_job(stack: CoupledNWSStack, forcings: Forcings, job: _Job):
    key, theta_A, theta_B, directions = job
    snow17_forcing, sacsma_forcing = forcings[key]
    physics = stack.make_physics(snow17_forcing, sacsma_forcing)
    primal, columns = jacobian_columns(
        physics, torch.from_numpy(theta_A), torch.from_numpy(theta_B), directions
    )
    return primal.numpy(), [c.numpy() for c in columns]


def _init_worker(barrier) -> None:
    global _WORKER_STACK, _WORKER_BARRIER
    # One process per core already; intra-op threads would oversubscribe.
    torch.set_num_threads(1)
    _WORKER_STACK = CoupledNWSStack()
    _WORKER_BARRIER = barrier


def _receive_forcings(forcings: Forcings) -> None:
    # Each of these warm-up tasks blocks its worker at the barrier, so
    # n_workers of them can only complete once n_workers distinct workers
    # exist, have initialized, and each holds its own copy of the forcing.
    global _WORKER_FORCINGS
    _WORKER_FORCINGS = forcings
    assert _WORKER_BARRIER is not None
    _WORKER_BARRIER.wait(timeout=_STARTUP_TIMEOUT_S)


def _worker_job(job: _Job):
    assert _WORKER_STACK is not None and _WORKER_FORCINGS is not None
    return _run_job(_WORKER_STACK, _WORKER_FORCINGS, job)


class PhysicsPool:
    """`run(theta_A_batch, theta_B_batch, keys) -> [runoff per basin]`.

    n_workers=0 runs every job in this process (no subprocesses) through
    the same job function -- the reference path for tests and small runs.

    Use as a context manager, or call close(), to shut the workers down.
    """

    def __init__(self, forcings: Forcings, n_workers: int = 0) -> None:
        self._forcings = forcings
        self.n_workers = n_workers
        if n_workers > 0:
            ctx = mp.get_context("forkserver")
            ctx.set_forkserver_preload(["physics_pool"])
            self._executor: ProcessPoolExecutor | None = ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(ctx.Barrier(n_workers),),
            )
            list(self._executor.map(_receive_forcings, [forcings] * n_workers))
            self._stack = None
        else:
            self._executor = None
            self._stack = CoupledNWSStack()

    def run(
        self,
        theta_A_batch: torch.Tensor,
        theta_B_batch: torch.Tensor,
        keys: list[str],
    ) -> list[torch.Tensor]:
        """theta_A_batch: (B, 11), theta_B_batch: (B, 16), keys: B basin
        keys registered at construction. Returns B runoff tensors on the
        thetas' device, each differentiable w.r.t. its own row of the
        batch (unless needs_jacobian() is False -- then plain primals)."""
        assert theta_A_batch.shape[0] == theta_B_batch.shape[0] == len(keys)
        missing = [k for k in keys if k not in self._forcings]
        if missing:
            raise KeyError(f"basins not registered with this PhysicsPool: {missing}")

        n_a, n_b = theta_A_batch.shape[1], theta_B_batch.shape[1]
        directions = all_directions(n_a, n_b) if needs_jacobian(theta_A_batch, theta_B_batch) else []
        theta_A_np = theta_A_batch.detach().cpu().numpy()
        theta_B_np = theta_B_batch.detach().cpu().numpy()

        # One job per (basin, direction); one primal-only job per basin
        # when no Jacobian is needed.
        per_basin = [[d] for d in directions] or [[]]
        jobs: list[_Job] = [
            (key, theta_A_np[b], theta_B_np[b], dirs) for b, key in enumerate(keys) for dirs in per_basin
        ]
        if self._executor is not None:
            results = list(self._executor.map(_worker_job, jobs))
        else:
            results = [_run_job(self._stack, self._forcings, job) for job in jobs]

        runoffs = []
        k = len(per_basin)
        for b in range(len(keys)):
            basin_results = results[b * k : (b + 1) * k]
            primal = torch.from_numpy(basin_results[0][0])
            if not directions:
                runoffs.append(attach_jacobian(theta_A_batch[b], theta_B_batch[b], primal, None, None))
                continue
            columns = [torch.from_numpy(cols[0]) for _primal, cols in basin_results]
            J_A = torch.stack(columns[:n_a], dim=1)
            J_B = torch.stack(columns[n_a:], dim=1)
            runoffs.append(attach_jacobian(theta_A_batch[b], theta_B_batch[b], primal, J_A, J_B))
        return runoffs

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def __enter__(self) -> PhysicsPool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
