"""Precomputes and caches Hargreaves PET for every selected basin's full
forcing record, writing data/camels/pet/{gauge_id}_pet.csv.

Run this before build_climatology.py/src/train.py so a training run
reads PET from the cache rather than triggering computation implicitly
the first time each basin's data is touched -- camels_loader.get_pet()
falls back to computing+caching on first access regardless (this script
just makes that step explicit and up front, and lets the cached values
be inspected/reviewed as their own artifact rather than only living
inside whatever consumes them).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camels_loader import _load_raw_forcing, get_pet  # noqa: E402

CAMELS_DIR = Path(__file__).resolve().parent / "camels"


def build_pet(gauge_ids: list[str], verbose: bool = True) -> None:
    """PET cache for any basin list (see data/prepare_dataset.py). The
    cache is per gauge, shared across basin lists."""
    for gauge_id in gauge_ids:
        dates, _prcp, tmax, tmin, lat, _elev, _area_km2 = _load_raw_forcing(gauge_id)
        pet = get_pet(gauge_id, dates, tmax, tmin, lat)
        if verbose:
            print(f"{gauge_id}: {len(pet)} days, mean PET = {pet.mean():.3f} mm/day")


if __name__ == "__main__":
    selected = pd.read_csv(CAMELS_DIR / "selected_basins.csv", dtype={"gauge_id": str})
    build_pet(selected["gauge_id"].tolist())
