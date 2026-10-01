"""Builds a monthly-climatology sequence per basin -- the LSTM's dynamic
input, alongside the static attribute matrix from build_attributes.py.

Why climatology instead of feeding the raw daily training-window series
through the LSTM: the training window (WY1991-1993, see src/train.py) is
only 3 years, chosen for coupled-pipeline runtime, not because it's a
representative climatological sample. Computing monthly means from each
basin's FULL available record (up to 35 years) gives a genuine seasonal
signal -- when does precip peak, how cold do winters get, how much does
PET vary through the year -- that's independent of which 3-year window
later gets used for the actual differentiable rollout, and is a much
shorter, more robust sequence (12 steps) than a raw multi-year daily
series would be for an LSTM trained on only 35 basins.

Writes data/camels/basin_climatology.npz: gauge_ids (45,), X (45, 12, 3)
z-score normalized (features: prcp, tmean, pet), mean (3,), std (3,).

build_climatology(gauge_ids, out_path) works for any basin list (see
data/prepare_dataset.py); running this script builds the default
45-basin selection.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camels_loader import load_basin_timeseries  # noqa: E402

CAMELS_DIR = Path(__file__).resolve().parent / "camels"


def monthly_climatology(gauge_id: str) -> np.ndarray:
    """(12, 3) array: mean [prcp, tmean, pet] for each calendar month,
    averaged over the basin's full available record."""
    ts = load_basin_timeseries(gauge_id)
    df = pd.DataFrame({
        "month": ts.dates.month, "prcp": ts.prcp, "tmean": ts.tmean, "pet": ts.pet,
    })
    monthly = df.groupby("month")[["prcp", "tmean", "pet"]].mean()
    monthly = monthly.reindex(range(1, 13))  # ensure Jan..Dec order, even if a month were missing
    assert not monthly.isna().any().any(), f"{gauge_id}: missing a calendar month in climatology"
    return monthly.to_numpy(dtype=np.float64)


def build_climatology(gauge_ids: list[str], out_path: Path) -> None:
    gauge_ids = list(gauge_ids)
    X = np.stack([monthly_climatology(gid) for gid in gauge_ids])  # (n_basins, 12, 3)

    mean = X.reshape(-1, 3).mean(axis=0)
    std = X.reshape(-1, 3).std(axis=0)
    std[std == 0] = 1.0
    X_norm = (X - mean) / std

    np.savez(
        out_path,
        gauge_ids=np.array(gauge_ids),
        feature_names=np.array(["prcp", "tmean", "pet"]),
        X=X_norm, mean=mean, std=std,
    )
    print(f"Built ({len(gauge_ids)}, 12, 3) climatology sequence matrix -> {out_path}")


if __name__ == "__main__":
    selected = pd.read_csv(CAMELS_DIR / "selected_basins.csv", dtype={"gauge_id": str})
    build_climatology(selected["gauge_id"].tolist(), CAMELS_DIR / "basin_climatology.npz")
