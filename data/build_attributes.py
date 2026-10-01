"""Build the static basin-attribute feature matrix for the selected
basins (data/select_basins.py must run first).

Uses every numeric column across CAMELS' five attribute files (climate
indices, topography, soil, vegetation, geology) -- 39 features total,
not a hand-picked subset matching any particular paper's exact list.
Categorical columns (dom_land_cover, geol_1st_class, etc.) are dropped
rather than one-hot encoded -- reasonable scope for a static attribute
set this small, not a claim that they're uninformative.

Missing values (a handful of basins are missing root_depth_50/99 or
geol_porostiy) are median-imputed using the FULL 671-basin distribution,
not just our 45-basin subset -- more stable than a 45-sample median,
and the "typical" value across all CAMELS basins is still a reasonable
default for the handful of gaps within our selection.

Writes data/camels/basin_attributes.npz: gauge_ids (45,), feature_names
(39,), X (45, 39) z-score normalized, mean (39,), std (39,) -- the last
two saved so a basin can be featurized identically at inference time.

build_attributes(gauge_ids, out_path) works for any basin list; it is
what data/prepare_dataset.py calls for a user-supplied list. Running
this script directly builds the default 45-basin selection.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CAMELS_DIR = Path(__file__).resolve().parent / "camels"

ATTRIBUTE_FILES = [
    "camels_clim.txt",
    "camels_topo.txt",
    "camels_soil.txt",
    "camels_vege.txt",
    "camels_geol.txt",
]


def build_attributes(gauge_ids: list[str], out_path: Path) -> None:
    selected = pd.DataFrame({"gauge_id": list(gauge_ids)})

    full = None
    for fname in ATTRIBUTE_FILES:
        df = pd.read_csv(CAMELS_DIR / fname, sep=";", dtype={"gauge_id": str})
        df["gauge_id"] = df["gauge_id"].str.zfill(8)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        df = df[["gauge_id"] + numeric_cols]
        full = df if full is None else full.merge(df, on="gauge_id", how="outer")

    feature_names = [c for c in full.columns if c != "gauge_id"]

    # Median from the FULL 671-basin set, not just our 45 -- see module docstring.
    medians = full[feature_names].median()
    full[feature_names] = full[feature_names].fillna(medians)

    subset = selected[["gauge_id"]].merge(full, on="gauge_id", how="left")
    assert subset[feature_names].isna().sum().sum() == 0, "unfilled NaNs after median imputation"
    assert len(subset) == len(selected), "lost or duplicated basins during merge"

    X = subset[feature_names].to_numpy(dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0  # guard against a constant column (shouldn't occur here, but cheap)
    X_norm = (X - mean) / std

    np.savez(
        out_path,
        gauge_ids=subset["gauge_id"].to_numpy(),
        feature_names=np.array(feature_names),
        X=X_norm,
        mean=mean,
        std=std,
    )
    print(f"Built ({len(subset)}, {len(feature_names)}) attribute matrix -> {out_path}")
    print(f"Features: {feature_names}")


if __name__ == "__main__":
    selected = pd.read_csv(CAMELS_DIR / "selected_basins.csv", dtype={"gauge_id": str})
    build_attributes(selected["gauge_id"].tolist(), CAMELS_DIR / "basin_attributes.npz")
