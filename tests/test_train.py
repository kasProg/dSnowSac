"""Regression test for src/train.py's multi-basin hybrid training loop
and its Hydra-config plumbing (src/data_module.py's build_split()).

Skips gracefully (not part of the fast local suite's assumptions) when
CAMELS data hasn't been downloaded -- data/download_camels.sh is a
separate, ~3.4GB, one-time step, deliberately not run by `make test`.
Same pattern as tests/test_tesseract_build.py skipping when Docker/the
built images aren't available.

Small and fast on purpose: 3 train + 2 test basins, 3 epochs, a fixed
3-year window (hardcoded below, independent of configs/split/spatial.yaml's
current default) -- this is a smoke test that the training loop runs and
produces a real gradient signal (loss moves, held-out eval doesn't
crash), not a reproduction of the full 45-basin run in
results/runs/model_9yrs_spatial/ (see results/README.md).

Builds a config with OmegaConf.create() directly rather than going
through Hydra's compose()/initialize() -- avoids that API's global-state
gotchas across repeated calls in one pytest process, and keeps this test
independent of configs/*.yaml's exact file layout (still exercises the
same build_split()/run_epoch_hybrid() code path run_training() uses).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "data"))

CAMELS_DIR = REPO_ROOT / "data" / "camels"
_DATA_AVAILABLE = (
    (CAMELS_DIR / "selected_basins.csv").exists()
    and (CAMELS_DIR / "basin_attributes.npz").exists()
    and (CAMELS_DIR / "basin_climatology.npz").exists()
    and (CAMELS_DIR / "basin_dataset_public_v1p2").exists()
)


def _spatial_cfg() -> OmegaConf:
    return OmegaConf.create({
        "data": {
            "name": "camels_snow35",
            "selected_basins_csv": str(CAMELS_DIR / "selected_basins.csv"),
            "attributes_npz": str(CAMELS_DIR / "basin_attributes.npz"),
            "climatology_npz": str(CAMELS_DIR / "basin_climatology.npz"),
        },
        "split": {
            "mode": "spatial",
            "window": {"start": "1990-10-01", "end": "1993-09-30"},
        },
    })


@pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason="CAMELS data not downloaded -- run data/download_camels.sh, "
    "data/select_basins.py, data/build_attributes.py, data/build_climatology.py first",
)
def test_training_loop_runs_and_loss_improves():
    import torch

    from data_module import build_split
    from paramnet import ParamNet
    from physics_pool import PhysicsPool
    from train import run_epoch_hybrid

    split = build_split(_spatial_cfg())
    train_basins = split.train_examples[:3]
    heldout_basins = split.test_examples[:2]

    net = ParamNet(
        n_static_features=split.X_static[train_basins[0].gauge_id].shape[0],
        n_climate_features=split.X_climate[train_basins[0].gauge_id].shape[1],
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    forcings = {ex.key: (ex.snow17_forcing, ex.sacsma_forcing) for ex in train_basins + heldout_basins}

    # n_workers=2: exercises the real multi-process path end to end.
    with PhysicsPool(forcings, n_workers=2) as pool:
        def epoch(basins, opt):
            return run_epoch_hybrid(net, pool, basins, split.X_static, split.X_climate, opt)

        nse_before = epoch(train_basins, None)
        for _ in range(3):
            epoch(train_basins, optimizer)
        nse_after = epoch(train_basins, None)
        heldout_nses = epoch(heldout_basins, None)

    mean_before = np.mean(list(nse_before.values()))
    mean_after = np.mean(list(nse_after.values()))
    assert mean_after > mean_before, (
        f"mean train NSE did not improve over 3 epochs: {mean_before:.4f} -> {mean_after:.4f}"
    )
    assert all(np.isfinite(v) for v in heldout_nses.values())


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_minibatching_steps_once_per_minibatch_and_covers_every_basin():
    """batch_size=2 over 3 basins: two optimizer steps (2 + 1 basins),
    every basin scored exactly once, and a different order on the next
    epoch (the rng reshuffles)."""
    import torch

    from data_module import build_split
    from paramnet import ParamNet
    from physics_pool import PhysicsPool
    from train import run_epoch_hybrid

    split = build_split(_spatial_cfg())
    basins = split.train_examples[:3]
    net = ParamNet(
        n_static_features=split.X_static[basins[0].gauge_id].shape[0],
        n_climate_features=split.X_climate[basins[0].gauge_id].shape[1],
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    steps, batches = [], []
    real_step = optimizer.step
    optimizer.step = lambda *a, **k: (steps.append(1), real_step(*a, **k))[1]

    forcings = {ex.key: (ex.snow17_forcing, ex.sacsma_forcing) for ex in basins}
    with PhysicsPool(forcings, n_workers=0) as pool:
        real_run = pool.run
        pool.run = lambda tA, tB, keys: (batches.append(list(keys)), real_run(tA, tB, keys))[1]
        rng = np.random.default_rng(0)
        orders = []
        for _ in range(4):
            batches.clear()
            nses = run_epoch_hybrid(
                net, pool, basins, split.X_static, split.X_climate, optimizer, batch_size=2, rng=rng
            )
            assert [len(b) for b in batches] == [2, 1]
            assert sorted(k for b in batches for k in b) == sorted(ex.key for ex in basins)
            assert set(nses) == {ex.gauge_id for ex in basins}
            orders.append(tuple(k for b in batches for k in b))

    assert len(steps) == 4 * 2
    assert len(set(orders)) > 1, "minibatch order never changed across epochs"


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_cuda_network_gradients_match_cpu():
    """GPU training end to end on the real Fortran stack: the same
    ParamNet on CPU and on CUDA must produce the same loss and the same
    network-parameter gradients for a real basin. The physics runs on CPU
    in both cases (src/coupling.py's run_physics), so any mismatch beyond
    float rounding means a device seam is dropping or misrouting gradient.
    dropout=0 so both runs compute the same function (train mode stays
    on -- cuDNN's LSTM backward refuses eval mode, and training never
    uses it)."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    from data_module import build_split, masked_nse_loss
    from paramnet import ParamNet
    from pipeline import CoupledNWSStack

    split = build_split(_spatial_cfg())
    ex = split.train_examples[0]
    x_s = torch.tensor(split.X_static[ex.gauge_id][None], dtype=torch.float64)
    x_c = torch.tensor(split.X_climate[ex.gauge_id][None], dtype=torch.float64)

    torch.manual_seed(0)
    dims = dict(n_static_features=x_s.shape[1], n_climate_features=x_c.shape[2], dropout=0.0)
    net_cpu = ParamNet(**dims)
    net_cuda = ParamNet(**dims)
    net_cuda.load_state_dict(net_cpu.state_dict())
    net_cuda = net_cuda.cuda()
    stack = CoupledNWSStack()

    def loss_and_grads(net, device):
        theta_A, theta_B = net(x_s.to(device), x_c.to(device))
        sim = stack.run(theta_A[0], theta_B[0], ex.snow17_forcing, ex.sacsma_forcing)
        assert sim.device.type == device
        loss = masked_nse_loss(sim, ex)
        loss.backward()
        return loss.item(), {n: p.grad.cpu().numpy() for n, p in net.named_parameters()}

    loss_cpu, g_cpu = loss_and_grads(net_cpu, "cpu")
    loss_cuda, g_cuda = loss_and_grads(net_cuda, "cuda")

    # Measured gap on an RTX 3090 Ti: ~1e-7 of the largest gradient. Not
    # bit-exact (CUDA float64 kernels round differently), and theta_A is
    # float32 + FD-differentiated, so a last-bit flip in theta could move
    # the Snow17 Jacobian -- hence a scale-relative bound, not equality.
    np.testing.assert_allclose(loss_cuda, loss_cpu, rtol=1e-6)
    g_max = max(np.abs(g).max() for g in g_cpu.values())
    assert g_max > 0, "all-zero gradients -- nothing tested"
    for name in g_cpu:
        gap = np.abs(g_cuda[name] - g_cpu[name]).max() / g_max
        assert gap < 1e-5, f"{name}: CUDA vs CPU gradient gap {gap:.2e} of max |grad|"


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_build_split_spatial_matches_selected_basins_csv():
    """The config-driven split must reproduce exactly what
    data/select_basins.py wrote -- this is the thing the whole Hydra
    refactor has to get right, so it gets its own direct check rather
    than only being exercised incidentally by the training smoke test."""
    import pandas as pd

    from data_module import build_split

    split = build_split(_spatial_cfg())
    selected = pd.read_csv(CAMELS_DIR / "selected_basins.csv", dtype={"gauge_id": str})

    expected_train = set(selected.loc[selected["split"] == "train", "gauge_id"])
    expected_heldout = set(selected.loc[selected["split"] == "heldout", "gauge_id"])
    assert set(split.train_ids) == expected_train
    assert set(split.test_ids) == expected_heldout
    assert set(split.train_ids).isdisjoint(split.test_ids)


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_build_split_temporal_uses_same_basins_different_windows():
    """temporal split's defining property: train_ids == test_ids (same
    basins), but each BasinExample's own window differs."""
    from data_module import build_split

    cfg = OmegaConf.create({
        "data": {
            "name": "camels_snow35",
            "selected_basins_csv": str(CAMELS_DIR / "selected_basins.csv"),
            "attributes_npz": str(CAMELS_DIR / "basin_attributes.npz"),
            "climatology_npz": str(CAMELS_DIR / "basin_climatology.npz"),
        },
        "split": {
            "mode": "temporal",
            "train_window": {"start": "1990-10-01", "end": "1991-09-30"},
            "test_window": {"start": "1991-10-01", "end": "1992-09-30"},
        },
    })
    split = build_split(cfg)
    assert split.train_ids == split.test_ids
    assert len(split.train_examples) == len(split.test_examples) > 0
    for train_ex, test_ex in zip(split.train_examples[:3], split.test_examples[:3]):
        assert train_ex.gauge_id == test_ex.gauge_id
        assert train_ex.window_start != test_ex.window_start


def test_iterations_per_epoch():
    from train import iterations_per_epoch

    # Each step scores the whole (basin, day) space: one step is enough.
    assert iterations_per_epoch(n_basins=10, batch_size=10, n_days=730, warmup=365, rho=365) == 1
    # Half of it per step: (1/2)^n <= 0.01 -> n = 7.
    assert iterations_per_epoch(n_basins=10, batch_size=10, n_days=1095, warmup=365, rho=365) == 7
    # Fewer basins per step -> more steps.
    assert iterations_per_epoch(10, 5, 1095, 365, 365) > 7


def test_window_starts_require_a_scored_observation():
    from types import SimpleNamespace

    import torch

    from data_module import window_starts

    # 20 days, observations only on days 12 and 13.
    valid = torch.zeros(20, dtype=torch.bool)
    valid[[12, 13]] = True
    ex = SimpleNamespace(valid_mask=valid)
    # warmup 3 + rho 5: start s scores days [s+3, s+8); last start is 12.
    starts = window_starts(ex, warmup=3, rho=5)
    assert starts.tolist() == [5, 6, 7, 8, 9, 10]
    assert window_starts(ex, warmup=15, rho=6).size == 0  # longer than the series


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_window_loss_on_full_window_equals_nse_loss():
    """With no warmup and the whole window, the time-window loss must be
    exactly the full-window 1 - NSE -- the normalization is the same
    quantity, only computed once up front."""
    import torch

    from data_module import build_split, masked_nse_loss, window_nse_loss

    split = build_split(_spatial_cfg())
    ex = split.train_examples[0]
    sim = ex.observed.nan_to_num(0.0) * 0.8 + 0.3
    torch.testing.assert_close(window_nse_loss(sim, ex, 0, 0), masked_nse_loss(sim, ex), rtol=1e-12, atol=0)

    # Offset window with warmup, against an independent numpy recomputation:
    # sim covers days [100, 300); days [150, 300) are scored.
    obs = ex.observed.numpy()
    full = obs[~np.isnan(obs)]
    expected = np.nanmean((obs[150:300] - sim[150:300].numpy()) ** 2) / np.mean((full - full.mean()) ** 2)
    np.testing.assert_allclose(window_nse_loss(sim[100:300], ex, 100, 50).item(), expected, rtol=1e-10)


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_window_epoch_samples_valid_windows_and_steps_per_iteration():
    """run_epoch_windows: n_iters optimizer steps, batch_size distinct
    basins each, every window warmup + rho days long and starting at an
    allowed start, one train NSE per sampled window."""
    import torch

    from data_module import build_split, window_starts
    from paramnet import ParamNet
    from physics_pool import PhysicsPool
    from train import run_epoch_windows

    split = build_split(_spatial_cfg())
    basins = split.train_examples[:3]
    warmup, rho = 30, 60
    starts = {ex.key: window_starts(ex, warmup, rho) for ex in basins}
    net = ParamNet(
        n_static_features=split.X_static[basins[0].gauge_id].shape[0],
        n_climate_features=split.X_climate[basins[0].gauge_id].shape[1],
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    steps, calls = [], []
    real_step = optimizer.step
    optimizer.step = lambda *a, **k: (steps.append(1), real_step(*a, **k))[1]

    forcings = {ex.key: (ex.snow17_forcing, ex.sacsma_forcing) for ex in basins}
    with PhysicsPool(forcings, n_workers=2) as pool:
        real_run = pool.run

        def recording_run(tA, tB, keys, windows):
            sims = real_run(tA, tB, keys, windows)
            calls.append((list(keys), list(windows), [s.shape[0] for s in sims]))
            return sims

        pool.run = recording_run
        nses = run_epoch_windows(
            net, pool, basins, split.X_static, split.X_climate, optimizer, starts,
            batch_size=2, warmup=warmup, rho=rho, n_iters=3, rng=np.random.default_rng(0),
        )

    assert len(steps) == 3 and len(calls) == 3 and len(nses) == 6
    assert all(np.isfinite(nses))
    for keys, windows, lengths in calls:
        assert len(set(keys)) == 2
        assert lengths == [warmup + rho] * 2
        for key, (t0, t1) in zip(keys, windows):
            assert t0 in starts[key] and t1 - t0 == warmup + rho


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_eval_warmup_spins_up_on_the_training_period():
    """Temporal split, test window right after the train window: each test
    example's forcing must be the last `warmup` days of its train
    example's forcing followed by the test window's own forcing, while
    observations (and so scores) cover the test window only. Train
    examples stay un-warmed unless asked for."""
    import torch

    from data_module import build_split, masked_nse_loss

    warmup = 100
    cfg = OmegaConf.create({
        "data": {
            "name": "camels_snow35",
            "selected_basins_csv": str(CAMELS_DIR / "selected_basins.csv"),
            "attributes_npz": str(CAMELS_DIR / "basin_attributes.npz"),
            "climatology_npz": str(CAMELS_DIR / "basin_climatology.npz"),
        },
        "split": {
            "mode": "temporal",
            "train_window": {"start": "1990-10-01", "end": "1991-09-30"},
            "test_window": {"start": "1991-10-01", "end": "1992-09-30"},
        },
        "eval_warmup": warmup,
    })
    split = build_split(cfg)
    cold = build_split(OmegaConf.merge(cfg, {"eval_warmup": 0}))
    train_ex, test_ex, cold_test = split.train_examples[0], split.test_examples[0], cold.test_examples[0]

    assert train_ex.warmup == 0 and test_ex.warmup == warmup
    n = len(cold_test.valid_mask)
    for name in ("pcp", "tmp", "iyr", "imn", "ida"):
        warm = getattr(test_ex.snow17_forcing, name)
        assert len(warm) == warmup + n
        np.testing.assert_array_equal(warm[:warmup], getattr(train_ex.snow17_forcing, name)[-warmup:])
        np.testing.assert_array_equal(warm[warmup:], getattr(cold_test.snow17_forcing, name))
    for name in ("tmp", "etp"):
        warm = getattr(test_ex.sacsma_forcing, name)
        np.testing.assert_array_equal(warm[:warmup], getattr(train_ex.sacsma_forcing, name)[-warmup:])
        np.testing.assert_array_equal(warm[warmup:], getattr(cold_test.sacsma_forcing, name))
    assert torch.equal(test_ex.observed.nan_to_num(-1), cold_test.observed.nan_to_num(-1))
    assert test_ex.key != cold_test.key

    # Scoring drops exactly the warmup days, and refuses a misaligned series.
    sim = torch.rand(warmup + n, dtype=torch.float64)
    assert torch.equal(masked_nse_loss(sim, test_ex), masked_nse_loss(sim[warmup:], cold_test))
    with pytest.raises(ValueError, match="expected"):
        masked_nse_loss(sim[warmup:], test_ex)

    assert build_split(cfg, warm_train=True).train_examples[0].warmup == warmup
