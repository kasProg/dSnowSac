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

One gradient step per epoch, not per basin: theta_A/theta_B are computed
for all training basins in a single batched ParamNet forward pass, then
each basin's own (expensive, Fortran-backed) coupled Snow17 -> SAC-SMA
run happens individually via CoupledNWSStack.run -- that part isn't
batchable, SAC-SMA/Snow17 are single-HRU by construction -- and their
losses are averaged into ONE scalar before a single
.backward()/optimizer.step() call. Standard full-batch gradient descent
over basins, not per-basin SGD.

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

from data_module import BasinExample, build_split, masked_nse_loss, nse_value  # noqa: E402
from model_factory import build_model, resolve_device  # noqa: E402
from pipeline import CoupledNWSStack  # noqa: E402


def run_epoch_hybrid(
    net,
    stack: CoupledNWSStack,
    basins: list[BasinExample],
    X_static: dict,
    X_climate: dict,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """optimizer=None -> eval mode, no gradient step. Returns
    {gauge_id: nse}. Inputs go to whatever device net lives on; the
    physics itself always runs on CPU (src/coupling.py's run_physics)."""
    device = next(net.parameters()).device
    x_static_batch = torch.tensor(
        np.stack([X_static[b.gauge_id] for b in basins]), dtype=torch.float64, device=device
    )
    x_climate_batch = torch.tensor(
        np.stack([X_climate[b.gauge_id] for b in basins]), dtype=torch.float64, device=device
    )
    if optimizer is not None:
        net.train()
        theta_A_batch, theta_B_batch = net(x_static_batch, x_climate_batch)
    else:
        net.eval()
        with torch.no_grad():
            theta_A_batch, theta_B_batch = net(x_static_batch, x_climate_batch)
        theta_A_batch = theta_A_batch.detach().requires_grad_(False)
        theta_B_batch = theta_B_batch.detach().requires_grad_(False)

    losses = []
    nses = {}
    for i, ex in enumerate(basins):
        sim = stack.run(theta_A_batch[i], theta_B_batch[i], ex.snow17_forcing, ex.sacsma_forcing)
        loss = masked_nse_loss(sim, ex)
        losses.append(loss)
        nses[ex.gauge_id] = nse_value(sim, ex)

    if optimizer is not None:
        total_loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
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
        f"loaded in {time.time()-t0:.1f}s"
    )

    n_static = split.X_static[split.train_ids[0]].shape[0]
    n_climate = split.X_climate[split.train_ids[0]].shape[1]
    # .get: manually built configs (tests, results/README.md snippets)
    # predate the device key.
    device = resolve_device(cfg.get("device", "cpu"))
    print(f"  network + loss on {device}; Fortran physics on cpu")
    net = build_model(cfg, n_static, n_climate).to(device)
    stack = CoupledNWSStack()
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.train.lr)

    def run_epoch(basins, opt):
        return run_epoch_hybrid(net, stack, basins, split.X_static, split.X_climate, opt)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")
    checkpoints_dir = output_dir / "checkpoints"

    history = []
    for epoch in range(1, cfg.train.n_epochs + 1):
        t0 = time.time()
        train_nses = run_epoch(split.train_examples, optimizer)
        # Median, not mean, for reporting -- standard practice for
        # cross-basin NSE aggregation (a handful of badly-fit basins
        # shouldn't dominate the headline number the way they would
        # under a mean). The training loss itself (run_epoch_hybrid's
        # torch.stack(losses).mean()) stays a mean -- that's a distinct,
        # gradient-facing computation, not this human-facing metric.
        median_train_nse = float(np.median(list(train_nses.values())))
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
    x_static_test = torch.tensor(
        np.stack([split.X_static[g] for g in split.test_ids]), dtype=torch.float64, device=device
    )
    x_climate_test = torch.tensor(
        np.stack([split.X_climate[g] for g in split.test_ids]), dtype=torch.float64, device=device
    )
    predictions: dict[str, dict] = {}
    with torch.no_grad():
        theta_A_test, theta_B_test = net(x_static_test, x_climate_test)
        for i, ex in enumerate(split.test_examples):
            sim = stack.run(theta_A_test[i], theta_B_test[i], ex.snow17_forcing, ex.sacsma_forcing)
            predictions[ex.gauge_id] = {
                "sim_mm_day": sim.cpu().numpy().tolist(),
                "nse": nse_value(sim, ex) if ex.valid_mask.any() else None,
            }
    valid_nses = [p["nse"] for p in predictions.values() if p["nse"] is not None]
    (output_dir / "test_predictions.json").write_text(json.dumps(
        {
            "model": cfg.model.name,
            "split_mode": cfg.split.mode,
            "epoch": cfg.train.n_epochs,
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
