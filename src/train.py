"""Config-driven training entrypoint (Hydra) for the hybrid
Snow17+SAC-SMA+ParamNet stack (src/paramnet.py). One script, one CLI --
`split=` switches spatial (prediction in ungauged basins) vs. temporal
(prediction in ungauged period) evaluation. See configs/ and the main
README's "Reproducing experiments" section.

    .venv/bin/python src/train.py                # spatial split (default)
    .venv/bin/python src/train.py split=temporal seed=1

Kept as a plain function (run_training(cfg)) wrapped by a thin
@hydra.main CLI (cli()) at the bottom -- tests and results/README.md's
regeneration snippets call run_training() directly with a manually built
config, no Hydra compose/multirun machinery needed for that path.

One gradient step per minibatch of basins (train.batch_size; null =
all training basins, i.e. one full-batch step per epoch, the setting
behind every saved run). theta_A/theta_B for the minibatch come from one
batched ParamNet forward pass; the coupled Snow17 -> SAC-SMA runs are
single-HRU by construction, so src/physics_pool.py fans each basin's 27
forward-mode Jacobian passes out across n_workers processes. The
minibatch's losses are averaged into ONE scalar before a single
.backward()/optimizer.step() call. Evaluation runs only the primal
physics (no Jacobian), one job per basin.

Time-window minibatching (train.rho set, HydroDL-style): instead of
each basin's full training window, every step samples batch_size basins
and, for each, a random window of train.warmup + train.rho days inside
its training window. The physics runs that window from a cold start and
the loss scores only the last rho days (data_module.window_nse_loss). An
epoch is iterations_per_epoch() such steps.

Evaluation scores the full test window, after eval_warmup days of
spin-up on the forcing just before it (data_module.BasinExample).

A pure data-driven LSTM baseline (src/benchmark_lstm.py) used to live
alongside this as a second `model=` option, quantifying what the
physical constraint bought relative to a black-box model trained on the
same data. Removed: it was a small, from-scratch LSTM that made a weak
baseline, and keeping it around risked being read as "beats an LSTM" in
general rather than "beats this particular small LSTM" -- a distinction
that matters and is easy to lose in a README. See
results/external/neuralhydrology_lstm_pub/ for the honest version of
that comparison (a properly-engineered LSTM, on the exact same held-out
basins) and notes/logs.md for the full removal rationale.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "data"))

from data_module import (  # noqa: E402
    BasinExample,
    build_split,
    masked_nse_loss,
    nse_value,
    save_normalization,
    window_nse_loss,
    window_starts,
)
from model_factory import build_model, resolve_device  # noqa: E402
from physics_pool import PhysicsPool  # noqa: E402


def _network_inputs(basins, X_static: dict, X_climate: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    x_static = torch.tensor(
        np.stack([X_static[b.gauge_id] for b in basins]), dtype=torch.float64, device=device
    )
    x_climate = torch.tensor(
        np.stack([X_climate[b.gauge_id] for b in basins]), dtype=torch.float64, device=device
    )
    return x_static, x_climate


def run_epoch_hybrid(
    net,
    pool: PhysicsPool,
    basins: list[BasinExample],
    X_static: dict,
    X_climate: dict,
    optimizer: torch.optim.Optimizer | None,
    batch_size: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Returns {gauge_id: nse}.

    optimizer=None -> eval: no gradient step, and the physics runs
    primal-only (no Jacobian). Otherwise one optimizer step per minibatch
    of `batch_size` basins, freshly shuffled by `rng` every call; with
    batch_size None (or >= len(basins)) one full-batch step in the given
    basin order. Train NSEs are each basin's NSE at the parameters before
    its minibatch's step.

    Inputs go to whatever device net lives on; the physics always runs on
    CPU (src/coupling.py's run_physics)."""
    device = next(net.parameters()).device

    if optimizer is None:
        net.eval()
        with torch.no_grad():
            theta_A, theta_B = net(*_network_inputs(basins, X_static, X_climate, device))
            sims = pool.run(theta_A, theta_B, [b.key for b in basins])
        return {ex.gauge_id: nse_value(sim, ex) for ex, sim in zip(basins, sims)}

    net.train()
    if batch_size is None or batch_size >= len(basins):
        batch_size = len(basins)
        order = np.arange(len(basins))
    else:
        if rng is None:
            raise ValueError("minibatching (batch_size < number of basins) needs an rng")
        order = rng.permutation(len(basins))

    nses = {}
    for start in range(0, len(basins), batch_size):
        batch = [basins[i] for i in order[start : start + batch_size]]
        theta_A, theta_B = net(*_network_inputs(batch, X_static, X_climate, device))
        sims = pool.run(theta_A, theta_B, [b.key for b in batch])
        losses = [masked_nse_loss(sim, ex) for ex, sim in zip(batch, sims)]
        nses.update({ex.gauge_id: nse_value(sim, ex) for ex, sim in zip(batch, sims)})

        optimizer.zero_grad()
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()

    return nses


def iterations_per_epoch(n_basins: int, batch_size: int, n_days: int, warmup: int, rho: int) -> int:
    """Gradient steps per epoch of time-window minibatching: enough that
    any given (basin, scorable day) lands in at least one sampled window
    with ~99% probability. One step scores batch_size * rho of the
    n_basins * (n_days - warmup) scorable (basin, day) pairs, a fraction
    p; n steps miss a given pair with probability ~(1 - p)^n <= 0.01."""
    p = batch_size * rho / (n_basins * (n_days - warmup))
    if p >= 1.0:
        return 1
    return int(np.ceil(np.log(0.01) / np.log(1.0 - p)))


def run_epoch_windows(
    net,
    pool: PhysicsPool,
    basins: list[BasinExample],
    X_static: dict,
    X_climate: dict,
    optimizer: torch.optim.Optimizer,
    starts: dict[str, np.ndarray],
    *,
    batch_size: int | None,
    warmup: int,
    rho: int,
    n_iters: int,
    rng: np.random.Generator,
) -> list[float]:
    """n_iters gradient steps, each on batch_size distinct basins drawn
    at random, each basin on a random window [s, s + warmup + rho) with s
    drawn from starts[basin.key] (data_module.window_starts). Returns
    1 - window loss for every sampled window, at the parameters before
    its step -- an NSE normalized by the basin's whole-window variance."""
    device = next(net.parameters()).device
    net.train()
    batch_size = min(batch_size or len(basins), len(basins))

    nses = []
    for _ in range(n_iters):
        batch = [basins[i] for i in rng.choice(len(basins), size=batch_size, replace=False)]
        t0s = [int(rng.choice(starts[ex.key])) for ex in batch]
        theta_A, theta_B = net(*_network_inputs(batch, X_static, X_climate, device))
        sims = pool.run(
            theta_A, theta_B, [ex.key for ex in batch], [(t0, t0 + warmup + rho) for t0 in t0s]
        )
        losses = torch.stack([window_nse_loss(sim, ex, t0, warmup) for ex, sim, t0 in zip(batch, sims, t0s)])
        nses += (1.0 - losses.detach().cpu()).tolist()

        optimizer.zero_grad()
        losses.mean().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        optimizer.step()

    return nses


def run_training(cfg: DictConfig) -> dict:
    torch.manual_seed(cfg.seed)  # reproducible network init -- see notes/logs.md

    print(f"Loading basins (data={cfg.data.name}, split={cfg.split.mode})...")
    t0 = time.time()
    split = build_split(cfg)
    print(
        f"  {len(split.train_ids)} train + {len(split.test_ids)} test basins, "
        f"loaded in {time.time()-t0:.1f}s; test scored after "
        f"{cfg.get('eval_warmup', 0)} warmup days"
    )

    n_static = split.X_static[split.train_ids[0]].shape[0]
    n_climate = split.X_climate[split.train_ids[0]].shape[1]
    # .get: manually built configs (tests, results/README.md snippets)
    # predate the device key.
    device = resolve_device(cfg.get("device", "cpu"))
    print(f"  network + loss on {device}; Fortran physics on cpu")
    net = build_model(cfg, n_static, n_climate).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.train.lr)

    n_workers = cfg.get("n_workers", 0)
    batch_size = cfg.train.get("batch_size", None)
    print(
        f"  physics on {n_workers or 'no'} worker processes; "
        f"batch_size={batch_size or 'all'} basins per gradient step"
    )
    pool = PhysicsPool(
        {ex.key: (ex.snow17_forcing, ex.sacsma_forcing)
         for ex in split.train_examples + split.test_examples},
        n_workers=n_workers,
    )
    # Separate stream from torch's: minibatch order doesn't perturb
    # network init or dropout draws.
    rng = np.random.default_rng(cfg.seed)

    def run_epoch(basins, opt):
        return run_epoch_hybrid(
            net, pool, basins, split.X_static, split.X_climate, opt, batch_size=batch_size, rng=rng
        )

    # Time-window minibatching -- see the module docstring. .get: configs
    # built by hand (tests, results/README.md) predate these keys.
    rho = cfg.train.get("rho", None)
    warmup = cfg.train.get("warmup", 0) if rho is not None else 0
    if rho is not None:
        starts = {ex.key: window_starts(ex, warmup, rho) for ex in split.train_examples}
        unusable = [ex.gauge_id for ex in split.train_examples if starts[ex.key].size == 0]
        if unusable:
            raise ValueError(
                f"{len(unusable)} training basin(s) have no {warmup}+{rho}-day window with an "
                f"observed day to score: {unusable}. Lengthen the training window or shorten "
                "train.warmup/train.rho."
            )
        n_days = int(np.mean([len(ex.valid_mask) for ex in split.train_examples]))
        n_iters = cfg.train.get("iters_per_epoch", None) or iterations_per_epoch(
            len(split.train_examples), min(batch_size or len(split.train_examples), len(split.train_examples)),
            n_days, warmup, rho,
        )
        print(f"  time windows: {warmup} warmup + {rho} scored days, {n_iters} steps per epoch")

        def run_train_epoch():
            return run_epoch_windows(
                net, pool, split.train_examples, split.X_static, split.X_climate, optimizer, starts,
                batch_size=batch_size, warmup=warmup, rho=rho, n_iters=n_iters, rng=rng,
            )
    else:
        def run_train_epoch():
            return list(run_epoch(split.train_examples, optimizer).values())

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")
    # The feature scaling this network is trained on -- src/infer.py
    # re-scales other basin lists to it.
    save_normalization(cfg.data, output_dir / "normalization.npz")
    checkpoints_dir = output_dir / "checkpoints"

    history = []
    for epoch in range(1, cfg.train.n_epochs + 1):
        t0 = time.time()
        train_nses = run_train_epoch()
        # Median, not mean, for reporting -- standard practice for
        # cross-basin NSE aggregation (a handful of badly-fit basins
        # shouldn't dominate the headline number the way they would
        # under a mean). The training loss itself (run_epoch_hybrid's
        # torch.stack(losses).mean()) stays a mean -- that's a distinct,
        # gradient-facing computation, not this human-facing metric.
        # With time windows: median over every window sampled this epoch.
        median_train_nse = float(np.median(train_nses))
        dt = time.time() - t0

        row = {"epoch": epoch, "median_train_nse": median_train_nse, "seconds": dt}
        if epoch == 1 or epoch % cfg.train.eval_every == 0 or epoch == cfg.train.n_epochs:
            test_nses = run_epoch(split.test_examples, None)
            row["median_test_nse"] = float(np.median(list(test_nses.values())))
        history.append(row)
        print(
            f"epoch {epoch:3d}  train_nse={median_train_nse:+.4f}"
            + (f"  test_nse={row.get('median_test_nse'):+.4f}" if "median_test_nse" in row else "")
            + f"  ({dt:.2f}s)"
        )

        # Resume-capable periodic checkpoint -- model AND optimizer state,
        # unlike the model-only checkpoint.pt saved below. Every
        # checkpoint_every epochs, and once more on the final epoch
        # regardless of N so there's always one reflecting the true end
        # of training even when n_epochs isn't a multiple of N.
        if epoch % cfg.train.checkpoint_every == 0 or epoch == cfg.train.n_epochs:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoints_dir / f"epoch_{epoch:04d}.pt",
            )

    torch.save(net.state_dict(), output_dir / "checkpoint.pt")

    # Final-epoch predictions on the test (heldout) set -- same shape as
    # src/infer.py's predictions.json, generated here from the
    # just-trained net directly rather than reloading the checkpoint.
    net.eval()
    predictions: dict[str, dict] = {}
    with torch.no_grad():
        theta_A_test, theta_B_test = net(
            *_network_inputs(split.test_examples, split.X_static, split.X_climate, device)
        )
        sims = pool.run(theta_A_test, theta_B_test, [ex.key for ex in split.test_examples])
        for ex, sim in zip(split.test_examples, sims):
            predictions[ex.gauge_id] = {
                "sim_mm_day": ex.scored(sim).cpu().numpy().tolist(),
                "nse": nse_value(sim, ex) if ex.valid_mask.any() else None,
            }
    pool.close()
    valid_nses = [p["nse"] for p in predictions.values() if p["nse"] is not None]
    (output_dir / "test_predictions.json").write_text(json.dumps(
        {
            "model": cfg.model.name,
            "split_mode": cfg.split.mode,
            "epoch": cfg.train.n_epochs,
            "eval_warmup": cfg.get("eval_warmup", 0),
            "basin_ids": split.test_ids,
            "median_nse": float(np.median(valid_nses)) if valid_nses else None,
            "predictions": predictions,
        },
        indent=2,
    ))

    result = {
        "model": cfg.model.name,
        "split_mode": cfg.split.mode,
        "seed": cfg.seed,
        "n_epochs": cfg.train.n_epochs,
        "lr": cfg.train.lr,
        "batch_size": batch_size,
        "rho": rho,
        "warmup": warmup,
        "eval_warmup": cfg.get("eval_warmup", 0),
        "n_train_basins": len(split.train_ids),
        "n_test_basins": len(split.test_ids),
        "train_basin_ids": split.train_ids,
        "test_basin_ids": split.test_ids,
        "history": history,
    }
    (output_dir / "history.json").write_text(json.dumps(result, indent=2))
    print(f"Saved config + checkpoint + history + test_predictions -> {output_dir}")

    return {"net": net, "output_dir": str(output_dir), **result}


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def cli(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    cli()
