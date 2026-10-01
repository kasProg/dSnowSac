"""Config-driven inference: load a trained checkpoint, run it
forward-only over a (possibly different) data/split, save per-basin
simulated streamflow + NSE (where observations are available).

    .venv/bin/python src/infer.py checkpoint=results/runs/hybrid_spatial_.../checkpoint.pt

Overriding split/data lets you score a trained model on a NEW window or
NEW basin set without retraining -- e.g. checking a spatial-split
checkpoint against a later date range, or (once configs/data/camels_full671.yaml
exists, see notes/logs.md's "parked for later" entry) basins outside the
original 45. The composed model config must match what the checkpoint
was actually trained with -- this script does not store/infer
architecture from the checkpoint file itself, only from the composed
config, since state_dict() alone doesn't carry hyperparameters like
lstm_hidden.

Kept as a plain function (run_inference(cfg)) wrapped by a thin
@hydra.main CLI (cli()), same pattern as src/train.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "data"))

from data_module import apply_training_normalization, build_split, nse_value  # noqa: E402
from model_factory import build_model, resolve_device  # noqa: E402
from physics_pool import PhysicsPool  # noqa: E402


def run_inference(cfg: DictConfig) -> dict:
    ckpt_path = Path(cfg.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt_path} -- pass checkpoint=<path to checkpoint.pt "
            "from a src/train.py run's output_dir>"
        )

    print(f"Loading basins (data={cfg.data.name}, split={cfg.split.mode})...")
    t0 = time.time()
    split = build_split(cfg)
    all_examples = split.train_examples + split.test_examples
    all_ids = split.train_ids + split.test_ids
    print(f"  {len(all_examples)} basins, loaded in {time.time()-t0:.1f}s")

    # Features must be scaled the way the network saw them in training,
    # not by this basin list's own statistics -- see data_module.py.
    norm_path = ckpt_path.parent / "normalization.npz"
    if norm_path.exists():
        split.X_static, split.X_climate, rescaled = apply_training_normalization(
            cfg.data, split.X_static, split.X_climate, norm_path
        )
        if rescaled:
            print("  features re-scaled to the training basin list's normalization")
    else:
        print(
            f"  WARNING: no normalization.npz next to {ckpt_path.name}; using this basin "
            "list's own feature scaling, which is only right if it is the training list"
        )

    n_static = split.X_static[all_ids[0]].shape[0]
    n_climate = split.X_climate[all_ids[0]].shape[1]
    device = resolve_device(cfg.get("device", "cpu"))
    net = build_model(cfg, n_static, n_climate)
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    net.to(device).eval()
    n_workers = cfg.get("n_workers", 0)
    print(f"  network on {device}; physics on {n_workers or 'no'} worker processes")

    x_static = torch.tensor(
        np.stack([split.X_static[g] for g in all_ids]), dtype=torch.float64, device=device
    )
    x_climate = torch.tensor(
        np.stack([split.X_climate[g] for g in all_ids]), dtype=torch.float64, device=device
    )
    # No gradients needed: one primal-only physics run per basin.
    with torch.no_grad(), PhysicsPool(
        {ex.key: (ex.snow17_forcing, ex.sacsma_forcing) for ex in all_examples}, n_workers=n_workers
    ) as pool:
        theta_A, theta_B = net(x_static, x_climate)
        sims = pool.run(theta_A, theta_B, [ex.key for ex in all_examples])

    # Grouped by train/test, not one dict keyed by gauge: in a temporal
    # split every basin appears in both groups (different windows), and
    # a flat dict let the test window silently overwrite the train one.
    groups = {"train": split.train_examples, "test": split.test_examples}
    predictions: dict[str, dict[str, dict]] = {name: {} for name in groups}
    sim_by_key = {ex.key: sim for ex, sim in zip(all_examples, sims)}
    for name, examples in groups.items():
        for ex in examples:
            sim = sim_by_key[ex.key]
            predictions[name][ex.gauge_id] = {
                "window": [str(ex.window_start.date()), str(ex.window_end.date())],
                "sim_mm_day": sim.cpu().numpy().tolist(),
                "nse": nse_value(sim, ex) if ex.valid_mask.any() else None,
            }

    # Median, not mean -- see src/train.py's matching comment; same
    # cross-basin-aggregation reasoning applies here.
    median_nse = {}
    for name, preds in predictions.items():
        valid = [p["nse"] for p in preds.values() if p["nse"] is not None]
        median_nse[name] = float(np.median(valid)) if valid else None
        if valid:
            print(f"Median NSE, {name} ({len(valid)} basins): {median_nse[name]:+.4f}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": cfg.model.name,
        "data": cfg.data.name,
        "split_mode": cfg.split.mode,
        "checkpoint": str(ckpt_path),
        "median_nse": median_nse,
        "predictions": predictions,
    }
    (output_dir / "predictions.json").write_text(json.dumps(result, indent=2))
    print(f"Saved predictions -> {output_dir / 'predictions.json'}")

    return result


@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def cli(cfg: DictConfig) -> None:
    run_inference(cfg)


if __name__ == "__main__":
    cli()
