"""data/prepare_dataset.py (any list of CAMELS basins -> a training
dataset) and the train-time feature normalization that inference on a
different basin list relies on.

List parsing and split assignment need no data. Everything that builds
files skips without the CAMELS download, like tests/test_train.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "data"))

from prepare_dataset import assign_split, ensure_dataset, read_basin_list  # noqa: E402

CAMELS_DIR = REPO_ROOT / "data" / "camels"
_DATA_AVAILABLE = (CAMELS_DIR / "basin_dataset_public_v1p2").exists() and (
    CAMELS_DIR / "camels_name.txt"
).exists()
needs_camels = pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")

# Three real CAMELS basins, none from the default 45-basin snow selection
# except 06623800 (shared, so normalization can be compared across lists).
SMALL_LIST = ["01013500", "06623800", "12145500"]


def _cfg(tmp_path: Path, basin_list: Path, name: str = "t") -> OmegaConf:
    out = tmp_path / name
    return OmegaConf.create({
        "name": name,
        "basin_list": str(basin_list),
        "heldout_fraction": 0.2,
        "split_seed": 0,
        "selected_basins_csv": str(out / "selected_basins.csv"),
        "attributes_npz": str(out / "basin_attributes.npz"),
        "climatology_npz": str(out / "basin_climatology.npz"),
    })


# ---------------------------------------------------------------- parsing


def test_text_list_ignores_comments_and_pads_ids(tmp_path):
    f = tmp_path / "basins.txt"
    f.write_text("# my basins\n1013500\n\n06623800  # snowy\n12145500,\n")
    df = read_basin_list(f)
    assert df["gauge_id"].tolist() == SMALL_LIST
    assert df["split"].isna().all()


def test_csv_list_keeps_given_split(tmp_path):
    f = tmp_path / "basins.csv"
    f.write_text("gauge_id,split\n1013500,train\n06623800,HELDOUT\n")
    df = read_basin_list(f)
    assert df["gauge_id"].tolist() == ["01013500", "06623800"]
    assert df["split"].tolist() == ["train", "heldout"]


@pytest.mark.parametrize(
    "content, match",
    [
        ("01013500\n01013500\n", "duplicate"),
        ("01013500\nabc\n", "not CAMELS gauge IDs"),
        ("# nothing\n", "no gauge IDs"),
    ],
)
def test_bad_text_lists_are_rejected(tmp_path, content, match):
    f = tmp_path / "basins.txt"
    f.write_text(content)
    with pytest.raises(ValueError, match=match):
        read_basin_list(f)


def test_csv_with_partial_or_unknown_split_is_rejected(tmp_path):
    f = tmp_path / "basins.csv"
    f.write_text("gauge_id,split\n01013500,train\n06623800,validation\n")
    with pytest.raises(ValueError, match="train or heldout"):
        read_basin_list(f)


def test_shipped_basin_lists():
    lists = REPO_ROOT / "data" / "basin_lists"
    ids531 = read_basin_list(lists / "camels_531.txt")["gauge_id"]
    ids671 = read_basin_list(lists / "camels_671.txt")["gauge_id"]
    assert len(ids531) == 531 and len(ids671) == 671
    assert set(ids531) <= set(ids671)


def test_assigned_split_is_seeded_and_keeps_both_sides():
    df = pd.DataFrame({"gauge_id": [f"{i:08d}" for i in range(10)], "split": None})
    a = assign_split(df, 0.2, seed=0)
    assert a["split"].tolist() == assign_split(df, 0.2, seed=0)["split"].tolist()
    assert (a["split"] == "heldout").sum() == 2
    tiny = assign_split(df.head(2), 0.2, seed=0)
    assert sorted(tiny["split"]) == ["heldout", "train"]


# ---------------------------------------------------------------- building


@needs_camels
def test_unknown_gauge_id_is_rejected(tmp_path):
    f = tmp_path / "basins.txt"
    f.write_text("01013500\n99999999\n")
    with pytest.raises(ValueError, match="not CAMELS basins.*99999999"):
        ensure_dataset(_cfg(tmp_path, f))


@needs_camels
def test_dataset_builds_once_and_rebuilds_when_list_changes(tmp_path):
    f = tmp_path / "basins.txt"
    f.write_text("\n".join(SMALL_LIST))
    cfg = _cfg(tmp_path, f)

    ensure_dataset(cfg)
    selected = pd.read_csv(cfg.selected_basins_csv, dtype={"gauge_id": str})
    assert selected["gauge_id"].tolist() == SMALL_LIST
    for npz in (cfg.attributes_npz, cfg.climatology_npz):
        assert list(np.load(npz, allow_pickle=True)["gauge_ids"]) == SMALL_LIST

    mtime = Path(cfg.attributes_npz).stat().st_mtime_ns
    ensure_dataset(cfg)
    assert Path(cfg.attributes_npz).stat().st_mtime_ns == mtime, "rebuilt an up-to-date dataset"

    f.write_text("\n".join(SMALL_LIST[:2]))
    ensure_dataset(cfg)
    assert list(np.load(cfg.attributes_npz, allow_pickle=True)["gauge_ids"]) == SMALL_LIST[:2]


@needs_camels
def test_default_45_basin_list_reproduces_curated_dataset(tmp_path):
    """Feeding the curated selected_basins.csv through the generic path
    must rebuild the same features the saved runs were trained on.
    Climatology to 1e-12, not bitwise: the curated file predates the PET
    CSV cache, whose round trip moves PET by ~1e-16."""
    cfg = _cfg(tmp_path, CAMELS_DIR / "selected_basins.csv")
    ensure_dataset(cfg)
    for built, curated, exact in (
        (cfg.attributes_npz, CAMELS_DIR / "basin_attributes.npz", True),
        (cfg.climatology_npz, CAMELS_DIR / "basin_climatology.npz", False),
    ):
        a, b = np.load(built, allow_pickle=True), np.load(curated, allow_pickle=True)
        assert list(a["gauge_ids"]) == list(b["gauge_ids"])
        if exact:
            assert np.array_equal(a["X"], b["X"])
        else:
            np.testing.assert_allclose(a["X"], b["X"], rtol=0, atol=1e-12)
    curated_split = pd.read_csv(CAMELS_DIR / "selected_basins.csv", dtype={"gauge_id": str})
    built_split = pd.read_csv(cfg.selected_basins_csv, dtype={"gauge_id": str})
    assert built_split["split"].tolist() == curated_split["split"].tolist()


@needs_camels
def test_training_normalization_makes_features_list_independent(tmp_path):
    """The same basin must reach the network with the same features
    whatever list it is in. Train on the 45-basin list, infer on a
    3-basin list sharing one basin: after re-scaling, that basin's
    features match the training list's, though the two lists' own
    z-scores differ."""
    from data_module import apply_training_normalization, load_basin_features, save_normalization

    train_cfg = _cfg(tmp_path, CAMELS_DIR / "selected_basins.csv", name="train")
    ensure_dataset(train_cfg)
    _, Xs_train, Xc_train = load_basin_features(train_cfg)
    norm = tmp_path / "normalization.npz"
    save_normalization(train_cfg, norm)

    f = tmp_path / "small.txt"
    f.write_text("\n".join(SMALL_LIST))
    small_cfg = _cfg(tmp_path, f, name="small")
    ensure_dataset(small_cfg)
    _, Xs_small, Xc_small = load_basin_features(small_cfg)

    shared = "06623800"
    assert not np.allclose(Xs_small[shared], Xs_train[shared]), "lists' own scaling already agrees"
    Xs, Xc, changed = apply_training_normalization(small_cfg, Xs_small, Xc_small, norm)
    assert changed
    np.testing.assert_allclose(Xs[shared], Xs_train[shared], rtol=0, atol=1e-9)
    np.testing.assert_allclose(Xc[shared], Xc_train[shared], rtol=0, atol=1e-9)

    # Same list as training: returned untouched, bit for bit.
    Xs_same, Xc_same, changed = apply_training_normalization(train_cfg, Xs_train, Xc_train, norm)
    assert not changed and Xs_same is Xs_train and Xc_same is Xc_train
