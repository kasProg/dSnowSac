"""Prepare a training dataset from any list of CAMELS basins.

    .venv/bin/python data/prepare_dataset.py my_basins.txt
    .venv/bin/python src/train.py data=camels_list data.basin_list=my_basins.txt

The second command calls the same ensure_dataset() itself, so running
this script first is optional -- it just lets you check the list before
training.

The basin list is either
  - a text file with one CAMELS gauge ID per line (leading zeros
    optional; blank lines and lines starting with # are ignored), or
  - a CSV with a `gauge_id` column and, optionally, a `split` column
    (train / heldout) to fix the spatial train/held-out partition.
Without a split column, a held-out fraction is drawn with a fixed seed,
so the same list always gives the same split.

What gets built, under data/camels/datasets/<name>/ (name defaults to
the list's file stem):
  selected_basins.csv     the list, with CAMELS metadata and the split
  basin_attributes.npz    static attributes (data/build_attributes.py)
  basin_climatology.npz   monthly climatology (data/build_climatology.py)
plus each basin's PET cache under data/camels/pet/ (shared across lists).
Attributes and climatology are z-scored over the basins in the list, so
each list gets its own files. They are rebuilt automatically whenever
the list (or its split) changes, and left alone otherwise.

CAMELS itself has no per-basin download: if the archive is missing, the
whole thing (~3.4 GB, all 671 basins) is fetched once by
data/download_camels.sh.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from build_attributes import build_attributes
from build_climatology import build_climatology
from build_pet import build_pet

DATA_DIR = Path(__file__).resolve().parent
CAMELS_DIR = DATA_DIR / "camels"
DATASETS_DIR = CAMELS_DIR / "datasets"

_REQUIRED_RAW = [
    "camels_clim.txt", "camels_topo.txt", "camels_soil.txt", "camels_vege.txt",
    "camels_geol.txt", "camels_name.txt", "basin_dataset_public_v1p2",
]
_SPLITS = {"train", "heldout"}


def ensure_camels_downloaded() -> None:
    missing = [f for f in _REQUIRED_RAW if not (CAMELS_DIR / f).exists()]
    if not missing:
        return
    print(f"CAMELS data missing ({', '.join(missing)}) -- downloading once (~3.4 GB)...")
    subprocess.run(["bash", str(DATA_DIR / "download_camels.sh")], check=True)


def read_basin_list(path: str | Path) -> pd.DataFrame:
    """-> DataFrame with gauge_id (8-digit str) and split (str, or None
    for every row when the list doesn't specify one)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"basin list not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, comment="#")
        if "gauge_id" not in df.columns:
            raise ValueError(f"{path}: CSV basin list needs a 'gauge_id' column (has {list(df.columns)})")
        df = df[["gauge_id"] + (["split"] if "split" in df.columns else [])]
    else:
        ids = []
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                ids.append(line.split()[0].rstrip(","))
        df = pd.DataFrame({"gauge_id": ids}, dtype=str)

    if df.empty:
        raise ValueError(f"{path}: no gauge IDs found")
    df["gauge_id"] = df["gauge_id"].str.strip().str.zfill(8)
    if not df["gauge_id"].str.fullmatch(r"\d{8}").all():
        bad = df.loc[~df["gauge_id"].str.fullmatch(r"\d{8}"), "gauge_id"].tolist()
        raise ValueError(f"{path}: not CAMELS gauge IDs (expected up to 8 digits): {bad}")
    dupes = df.loc[df["gauge_id"].duplicated(), "gauge_id"].tolist()
    if dupes:
        raise ValueError(f"{path}: duplicate gauge IDs: {dupes}")

    if "split" in df.columns:
        df["split"] = df["split"].str.strip().str.lower()
        bad = sorted(set(df["split"].dropna()) - _SPLITS)
        if bad or df["split"].isna().any():
            raise ValueError(
                f"{path}: 'split' must be train or heldout on every row "
                f"(found {bad or 'empty values'})"
            )
    else:
        df["split"] = None
    return df.reset_index(drop=True)


def assign_split(df: pd.DataFrame, heldout_fraction: float, seed: int) -> pd.DataFrame:
    """Fill in train/heldout when the list didn't. Seeded, so the same
    list always gets the same split."""
    if df["split"].notna().all():
        return df
    n = len(df)
    n_heldout = int(round(n * heldout_fraction))
    if heldout_fraction > 0 and n >= 2:
        n_heldout = min(max(n_heldout, 1), n - 1)  # keep at least one basin on each side
    rng = np.random.default_rng(seed)
    heldout_idx = rng.choice(n, size=n_heldout, replace=False)
    df = df.copy()
    df["split"] = "train"
    df.loc[heldout_idx, "split"] = "heldout"
    return df


def _camels_metadata() -> pd.DataFrame:
    tables = [
        pd.read_csv(CAMELS_DIR / f, sep=";", dtype={"gauge_id": str})
        for f in ("camels_name.txt", "camels_clim.txt", "camels_topo.txt")
    ]
    meta = tables[0].merge(tables[1], on="gauge_id").merge(tables[2], on="gauge_id")
    meta["gauge_id"] = meta["gauge_id"].str.zfill(8)
    cols = [
        "gauge_id", "gauge_name", "huc_02", "frac_snow", "pet_mean", "aridity",
        "elev_mean", "gauge_lat", "gauge_lon", "area_gages2",
    ]
    return meta[cols]


def _npz_gauge_ids(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    return [str(g) for g in np.load(path, allow_pickle=True)["gauge_ids"]]


def ensure_dataset(data_cfg) -> None:
    """Build (or confirm up to date) the files data_cfg points at, from
    data_cfg.basin_list. No-op for data configs without a basin_list --
    the curated camels_snow35 set is built by data/select_basins.py and
    the build_*.py scripts instead."""
    if not data_cfg.get("basin_list"):
        return

    ensure_camels_downloaded()
    basins = read_basin_list(data_cfg.basin_list)

    meta = _camels_metadata()
    unknown = sorted(set(basins["gauge_id"]) - set(meta["gauge_id"]))
    if unknown:
        raise ValueError(
            f"{len(unknown)} gauge IDs in {data_cfg.basin_list} are not CAMELS basins: {unknown}"
        )

    basins = assign_split(
        basins, data_cfg.get("heldout_fraction", 0.2), data_cfg.get("split_seed", 0)
    )
    selected = basins.merge(meta, on="gauge_id", how="left")
    selected = selected[[c for c in meta.columns] + ["split"]]

    csv_path = Path(data_cfg.selected_basins_csv)
    attrs_path = Path(data_cfg.attributes_npz)
    clim_path = Path(data_cfg.climatology_npz)
    ids = selected["gauge_id"].tolist()

    if csv_path.exists():
        current = pd.read_csv(csv_path, dtype={"gauge_id": str})
        up_to_date = (
            current["gauge_id"].tolist() == ids
            and current["split"].tolist() == selected["split"].tolist()
            and _npz_gauge_ids(attrs_path) == ids
            and _npz_gauge_ids(clim_path) == ids
        )
        if up_to_date:
            return

    n_train = int((selected["split"] == "train").sum())
    print(
        f"Preparing dataset '{data_cfg.name}': {len(ids)} basins "
        f"({n_train} train, {len(ids) - n_train} heldout) -> {csv_path.parent}"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    build_pet(ids, verbose=False)
    build_attributes(ids, attrs_path)
    build_climatology(ids, clim_path)
    # Written last: its presence + match is what marks the dataset complete.
    selected.to_csv(csv_path, index=False)


def data_cfg_for_list(basin_list: str, name: str | None = None, heldout_fraction: float = 0.2,
                      split_seed: int = 0) -> dict:
    """The same paths configs/data/camels_list.yaml resolves to."""
    name = name or Path(basin_list).stem
    out = DATASETS_DIR / name
    return {
        "name": name,
        "basin_list": str(basin_list),
        "heldout_fraction": heldout_fraction,
        "split_seed": split_seed,
        "selected_basins_csv": str(out / "selected_basins.csv"),
        "attributes_npz": str(out / "basin_attributes.npz"),
        "climatology_npz": str(out / "basin_climatology.npz"),
    }


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("basin_list", help="text file of gauge IDs, or CSV with gauge_id[,split]")
    parser.add_argument("--name", help="dataset name (default: the list's file stem)")
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()

    cfg = OmegaConf.create(
        data_cfg_for_list(args.basin_list, args.name, args.heldout_fraction, args.split_seed)
    )
    ensure_dataset(cfg)
    selected = pd.read_csv(cfg.selected_basins_csv, dtype={"gauge_id": str})
    print(selected[["gauge_id", "gauge_name", "frac_snow", "split"]].to_string(index=False))
    overrides = [f"data.basin_list={args.basin_list}"]
    if args.name:
        overrides.append(f"data.name={args.name}")
    if args.heldout_fraction != 0.2:
        overrides.append(f"data.heldout_fraction={args.heldout_fraction}")
    if args.split_seed != 0:
        overrides.append(f"data.split_seed={args.split_seed}")
    print("\nTrain with:\n  .venv/bin/python src/train.py data=camels_list " + " ".join(overrides))
