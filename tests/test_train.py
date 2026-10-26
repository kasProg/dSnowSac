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

import json
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
        pool.run = lambda tA, tB, keys, windows=None: (batches.append(list(keys)), real_run(tA, tB, keys, windows))[1]
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


def test_evaluates_at():
    from train import evaluates_at

    assert [e for e in range(1, 26) if evaluates_at(e, 10, 25)] == [1, 10, 20, 25]
    # 0/None: no periodic evaluation (and no modulo by zero).
    for every in (0, None):
        assert [e for e in range(1, 101) if evaluates_at(e, every, 100)] == [1, 100]


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
    steps, calls, step_losses = [], [], []
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
            on_step=step_losses.append,
        )

    assert len(steps) == 3 and len(calls) == 3 and len(nses) == 6
    # Progress callback: once per optimizer step, with the batch-mean loss.
    assert len(step_losses) == 3 and all(np.isfinite(step_losses))
    np.testing.assert_allclose(step_losses[0], 1.0 - np.mean(nses[:2]), rtol=1e-10)
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


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_eval_warmup_reaches_before_the_streamflow_record():
    """02427250's streamflow record starts 1989-10-01, but its forcing
    goes back to 1980: a 365-day warmup before that date must run on the
    forcing (it needs no streamflow), not fail for lack of flow days."""
    import pandas as pd

    from data_module import BasinExample

    ex = BasinExample("02427250", pd.Timestamp("1989-10-01"), pd.Timestamp("1990-09-30"), warmup=365)
    assert len(ex.snow17_forcing.pcp) == 365 + 365
    assert len(ex.valid_mask) == 365 and bool(ex.valid_mask[0])


def _small_run_cfg(output_dir: Path, n_epochs: int):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs")):
        return compose("config", overrides=[
            "data=camels_snow35", "split=spatial", "device=cpu", "n_workers=0", "eval_warmup=0",
            f"train.n_epochs={n_epochs}", "train.eval_every=2", "train.checkpoint_every=2",
            "train.batch_size=4", "train.iters_per_epoch=2", "train.rho=60", "train.warmup=30",
            f"output_dir={output_dir}",
        ])


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_resume_matches_an_uninterrupted_run(tmp_path, monkeypatch):
    """Stopping after epoch 2 and resuming to epoch 4 must give bitwise
    the same weights and history as running 4 epochs straight -- or the
    resumed half of a run silently differs from what it claims to be."""
    import torch
    from train import resume_cli, run_training

    monkeypatch.chdir(REPO_ROOT)
    run_training(_small_run_cfg(tmp_path / "straight", 4))
    run_training(_small_run_cfg(tmp_path / "stopped", 2))
    resume_cli(str(tmp_path / "stopped"), ["train.n_epochs=4"])

    def weights(name):
        return torch.load(tmp_path / name / "checkpoint.pt", weights_only=True)

    def history(name):
        rows = json.loads((tmp_path / name / "history.json").read_text())["history"]
        return [{k: v for k, v in row.items() if k != "seconds"} for row in rows]

    straight, resumed = weights("straight"), weights("stopped")
    assert all(torch.equal(straight[k], resumed[k]) for k in straight)
    assert history("stopped") == history("straight")
    assert [row["epoch"] for row in history("stopped")] == [1, 2, 3, 4]


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_resume_from_checkpoint_without_rng_state(tmp_path, monkeypatch):
    """Checkpoints written before RNG state and history were saved still
    resume; history.json then starts at the resumed epoch."""
    import torch
    from train import resume_cli, run_training

    monkeypatch.chdir(REPO_ROOT)
    run_training(_small_run_cfg(tmp_path, 2))
    path = tmp_path / "checkpoints" / "epoch_0002.pt"
    ckpt = torch.load(path, weights_only=False)
    torch.save({k: ckpt[k] for k in ("epoch", "model_state_dict", "optimizer_state_dict")}, path)
    resume_cli(str(tmp_path), ["train.n_epochs=3"])
    rows = json.loads((tmp_path / "history.json").read_text())["history"]
    assert [row["epoch"] for row in rows] == [3]


def test_rmse_comb_loss_matches_an_independent_recomputation_and_pools():
    """Against numpy, from Feng et al. (2022)'s equation. Two basins of
    different lengths and magnitudes: the RMSEs pool their days, so the
    loss is NOT the mean of the per-basin losses."""
    import torch

    from data_module import rmse_comb_loss

    rng = np.random.default_rng(0)
    obs = [rng.gamma(2.0, 1.0, 50), rng.gamma(2.0, 5.0, 20)]
    sim = [o * rng.uniform(0.5, 1.5, o.size) for o in obs]
    obs[0][:3] = 0.0  # exact zeros: the transform must stay finite there
    pairs = [(torch.tensor(s), torch.tensor(o)) for s, o in zip(sim, obs)]
    alpha, beta = 0.25, 1e-6

    def transform(q):
        return np.log10(np.sqrt(q + beta) + 0.1)

    s, o = np.concatenate(sim), np.concatenate(obs)
    expected = (1 - alpha) * np.sqrt(np.mean((s - o) ** 2)) + alpha * np.sqrt(
        np.mean((transform(s) - transform(o)) ** 2)
    )
    np.testing.assert_allclose(rmse_comb_loss(pairs, alpha, beta).item(), expected, rtol=1e-12)
    per_basin = np.mean([rmse_comb_loss([p], alpha, beta).item() for p in pairs])
    assert not np.isclose(per_basin, expected)
    # alpha = 0: plain pooled RMSE.
    np.testing.assert_allclose(rmse_comb_loss(pairs, 0.0, beta).item(), np.sqrt(np.mean((s - o) ** 2)), rtol=1e-12)


def test_rmse_comb_gradient_is_finite_at_zero_flow():
    """beta's job: d/dq sqrt(q) is infinite at q = 0, and SAC-SMA's TCI
    is exactly 0 on dry days."""
    import torch

    from data_module import rmse_comb_loss

    sim = torch.tensor([0.0, 0.0, 1.0, 3.0], dtype=torch.float64, requires_grad=True)
    obs = torch.tensor([0.0, 0.2, 1.5, 2.0], dtype=torch.float64)
    rmse_comb_loss([(sim, obs)]).backward()
    assert torch.isfinite(sim.grad).all()
    assert sim.grad[1] < 0  # under-predicting a low flow pulls sim up


def test_training_loss_dispatch():
    import torch

    from train import training_loss

    nse_losses = [torch.tensor(0.2), torch.tensor(0.6)]
    pairs = [(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 2.0]))]
    for cfg in (None, {"name": "nse"}):
        torch.testing.assert_close(training_loss(cfg, nse_losses, pairs), torch.tensor(0.4))
    assert training_loss({"name": "rmse_comb", "alpha": 0.0}, nse_losses, pairs).item() == pytest.approx(
        np.sqrt(0.125)
    )
    with pytest.raises(ValueError, match="unknown train.loss.name"):
        training_loss({"name": "kge"}, nse_losses, pairs)


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_window_epoch_trains_on_rmse_comb_and_still_reports_nse():
    """With loss.name=rmse_comb, each step's loss is rmse_comb_loss over
    the sampled windows' scored days, while the returned per-window
    values stay 1 - window NSE loss."""
    import torch

    from data_module import build_split, rmse_comb_loss, window_nse_loss, window_pair, window_starts
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
    loss_cfg = {"name": "rmse_comb", "alpha": 0.25, "beta": 1e-6}
    calls, step_losses = [], []
    by_key = {ex.key: ex for ex in basins}

    forcings = {ex.key: (ex.snow17_forcing, ex.sacsma_forcing) for ex in basins}
    with PhysicsPool(forcings, n_workers=0) as pool:
        real_run = pool.run

        def recording_run(tA, tB, keys, windows):
            sims = real_run(tA, tB, keys, windows)
            calls.append((list(keys), [w[0] for w in windows], [s.detach().clone() for s in sims]))
            return sims

        pool.run = recording_run
        nses = run_epoch_windows(
            net, pool, basins, split.X_static, split.X_climate, optimizer, starts,
            batch_size=2, warmup=warmup, rho=rho, n_iters=2, rng=np.random.default_rng(0),
            on_step=step_losses.append, loss_cfg=loss_cfg,
        )

    expected_nses = []
    for (keys, t0s, sims), step_loss in zip(calls, step_losses):
        exs = [by_key[k] for k in keys]
        pairs = [window_pair(s, ex, t0, warmup) for ex, s, t0 in zip(exs, sims, t0s)]
        np.testing.assert_allclose(step_loss, rmse_comb_loss(pairs, 0.25, 1e-6).item(), rtol=1e-10)
        expected_nses += [1.0 - window_nse_loss(s, ex, t0, warmup).item() for ex, s, t0 in zip(exs, sims, t0s)]
    np.testing.assert_allclose(nses, expected_nses, rtol=1e-10)


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_train_period_forcing_is_each_basins_training_window():
    """DailyParamNet's input: raw CAMELS [prcp, tmean, pet] over the
    TRAINING window for every basin -- test basins included, warmup days
    dropped, and in a temporal split never the test window."""
    import pandas as pd

    from camels_loader import load_basin_timeseries
    from data_module import build_split, forcing_normalization, normalize_forcing, train_period_forcing

    def expected(gauge_id, start, end):
        ts = load_basin_timeseries(gauge_id, all_forcing_days=True)
        m = (ts.dates >= pd.Timestamp(start)) & (ts.dates <= pd.Timestamp(end))
        return np.stack([ts.prcp[m], ts.tmean[m], ts.pet[m]], axis=1)

    cfg = _spatial_cfg()
    cfg.eval_warmup = 100
    split = build_split(cfg)
    forcing = train_period_forcing(split)
    assert set(forcing) == set(split.train_ids) | set(split.test_ids)
    for g in (split.train_ids[0], split.test_ids[0]):
        np.testing.assert_array_equal(forcing[g], expected(g, "1990-10-01", "1993-09-30"))

    mean, std = forcing_normalization(forcing, split.train_ids)
    train_days = np.concatenate([forcing[g] for g in split.train_ids])
    np.testing.assert_allclose(mean, train_days.mean(axis=0))
    z = np.concatenate(list(normalize_forcing({g: forcing[g] for g in split.train_ids}, mean, std).values()))
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(z.std(axis=0), 1.0, rtol=1e-10)

    cfg.split = {
        "mode": "temporal",
        "train_window": {"start": "1991-10-01", "end": "1992-09-30"},
        "test_window": {"start": "1990-10-01", "end": "1991-09-30"},
    }
    split = build_split(cfg)
    g = split.test_ids[0]
    np.testing.assert_array_equal(train_period_forcing(split)[g], expected(g, "1991-10-01", "1992-09-30"))


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_daily_model_trains_and_infer_reproduces_its_test_predictions(tmp_path, monkeypatch):
    """model=hybrid_daily end to end: a short training run, then
    src/infer.py on its checkpoint must rebuild exactly the same network
    inputs -- same training-window forcing, same saved scaling -- and so
    the same test-basin simulations. A scaling or window mismatch here
    would give plausible but wrong parameters, not an error."""
    from hydra import compose, initialize_config_dir

    from infer import run_inference
    from train import run_training

    monkeypatch.chdir(REPO_ROOT)
    cfg = _small_run_cfg(tmp_path / "run", 1)
    cfg.model = OmegaConf.load(REPO_ROOT / "configs" / "model" / "hybrid_daily.yaml")
    cfg.model.hidden = 16
    run_training(cfg)

    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs")):
        infer_cfg = compose("infer", overrides=[
            "data=camels_snow35", "split=spatial", "model=hybrid_daily", "model.hidden=16",
            "eval_warmup=0", "n_workers=0", "device=cpu",
            f"checkpoint={tmp_path / 'run' / 'checkpoint.pt'}", f"output_dir={tmp_path / 'infer'}",
        ])
    inferred = run_inference(infer_cfg)["predictions"]["test"]
    trained = json.loads((tmp_path / "run" / "test_predictions.json").read_text())["predictions"]
    assert set(inferred) == set(trained)
    # Not bitwise: infer.py batches the test basins with the train ones,
    # and the float32 network's kernels round differently by batch size
    # (measured ~1e-6 relative in the sims). A wrong window or scaling
    # moves the parameters by far more than 1e-4.
    for g in trained:
        np.testing.assert_allclose(
            inferred[g]["sim_mm_day"], trained[g]["sim_mm_day"], rtol=1e-4, atol=1e-6
        )


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_window_forcing_is_the_simulated_days():
    """WindowForcing (model=hybrid_dhbv): the z-scored raw CAMELS forcing
    of exactly the days the physics runs -- a sampled window of a train
    example, or a test example's whole run including its warmup."""
    import pandas as pd

    from camels_loader import load_basin_timeseries
    from data_module import WindowForcing, build_split, forcing_normalization, train_period_forcing

    cfg = _spatial_cfg()
    cfg.eval_warmup = 100
    split = build_split(cfg)
    mean, std = forcing_normalization(train_period_forcing(split), split.train_ids)
    wf = WindowForcing(split.train_examples + split.test_examples, mean, std)

    def expected(ex, start, n_days):
        ts = load_basin_timeseries(ex.gauge_id, all_forcing_days=True)
        i = int(np.searchsorted(ts.dates, start))
        raw = np.stack([ts.prcp, ts.tmean, ts.pet], axis=1)[i : i + n_days]
        return (raw - mean) / std

    train_ex, test_ex = split.train_examples[0], split.test_examples[0]
    np.testing.assert_allclose(
        wf.sequence(train_ex, (200, 290)), expected(train_ex, pd.Timestamp("1990-10-01") + pd.Timedelta(days=200), 90)
    )
    n_run = 100 + len(test_ex.valid_mask)
    np.testing.assert_allclose(wf.sequence(test_ex), expected(test_ex, pd.Timestamp("1990-10-01") - pd.Timedelta(days=100), n_run))


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_dhbv_window_epoch_feeds_each_window_and_runs_every_component():
    """run_epoch_windows with the dHBV setup: the network's forcing input
    for each basin is the window the physics simulates, and the physics
    gets K rows per basin -- that basin's key, window and component k."""
    import torch

    from data_module import WindowForcing, build_split, forcing_normalization, train_period_forcing, window_starts
    from paramnet import DailyParamNet
    from physics_pool import PhysicsPool
    from train import run_epoch_windows

    split = build_split(_spatial_cfg())
    basins = split.train_examples[:3]
    warmup, rho, k = 30, 60, 2
    mean, std = forcing_normalization(train_period_forcing(split), split.train_ids)
    wf = WindowForcing(basins, mean, std)
    starts = {ex.key: window_starts(ex, warmup, rho) for ex in basins}
    torch.manual_seed(0)
    net = DailyParamNet(split.X_static[basins[0].gauge_id].shape[0], hidden=8, n_components=k, readout="last")
    seen_inputs, calls, thetas = [], [], []
    net.register_forward_pre_hook(lambda _m, args: seen_inputs.append(args[1].detach().clone()))
    net.register_forward_hook(lambda _m, _a, out: thetas.append(out[1].detach().clone()))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    with PhysicsPool({ex.key: (ex.snow17_forcing, ex.sacsma_forcing) for ex in basins}, n_workers=2) as pool:
        real_run = pool.run

        def recording_run(tA, tB, keys, windows=None):
            calls.append((list(keys), list(windows), tB.detach().clone()))
            return real_run(tA, tB, keys, windows)

        pool.run = recording_run
        nses = run_epoch_windows(
            net, pool, basins, split.X_static, wf, optimizer, starts,
            batch_size=2, warmup=warmup, rho=rho, n_iters=2, rng=np.random.default_rng(0),
        )

    by_key = {ex.key: ex for ex in basins}
    assert len(calls) == len(seen_inputs) == 2 and len(nses) == 4 and all(np.isfinite(nses))
    for (keys, windows, tB), x_forcing, theta_B in zip(calls, seen_inputs, thetas):
        assert keys[0::k] == keys[1::k] and windows[0::k] == windows[1::k]
        for i, (key, window) in enumerate(zip(keys[0::k], windows[0::k])):
            assert window[1] - window[0] == warmup + rho
            np.testing.assert_array_equal(x_forcing[i].numpy(), wf.sequence(by_key[key], window))
        torch.testing.assert_close(tB, theta_B.reshape(-1, theta_B.shape[-1]))


@pytest.mark.skipif(not _DATA_AVAILABLE, reason="CAMELS data not downloaded")
def test_dhbv_model_trains_and_infer_reproduces_its_test_predictions(tmp_path, monkeypatch):
    """model=hybrid_dhbv (window readout, 2 components) end to end, as
    the hybrid_daily test above: infer.py must rebuild the same window
    inputs and component runs, so the same test simulations."""
    from hydra import compose, initialize_config_dir

    from infer import run_inference
    from train import run_training

    monkeypatch.chdir(REPO_ROOT)
    overrides = ["model.hidden=16", "model.n_components=2"]
    cfg = _small_run_cfg(tmp_path / "run", 1)
    cfg.model = OmegaConf.merge(
        OmegaConf.load(REPO_ROOT / "configs" / "model" / "hybrid_dhbv.yaml"),
        OmegaConf.from_dotlist([o.removeprefix("model.") for o in overrides]),
    )
    run_training(cfg)

    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs")):
        infer_cfg = compose("infer", overrides=[
            "data=camels_snow35", "split=spatial", "model=hybrid_dhbv", *overrides,
            "eval_warmup=0", "n_workers=0", "device=cpu",
            f"checkpoint={tmp_path / 'run' / 'checkpoint.pt'}", f"output_dir={tmp_path / 'infer'}",
        ])
    inferred = run_inference(infer_cfg)["predictions"]["test"]
    trained = json.loads((tmp_path / "run" / "test_predictions.json").read_text())["predictions"]
    assert set(inferred) == set(trained)
    for g in trained:  # tolerance: see the hybrid_daily test
        np.testing.assert_allclose(
            inferred[g]["sim_mm_day"], trained[g]["sim_mm_day"], rtol=1e-4, atol=1e-6
        )

    # A temporal split with warmup: the train and test groups' window
    # sequences differ in length, so infer.py must run them separately.
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs")):
        temporal_cfg = compose("infer", overrides=[
            "data=camels_snow35", "split=temporal", "model=hybrid_dhbv", *overrides,
            "split.train_window.start=1990-10-01", "split.train_window.end=1992-09-30",
            "split.test_window.start=1992-10-01", "split.test_window.end=1995-09-30",
            "eval_warmup=100", "n_workers=2", "device=cpu",
            f"checkpoint={tmp_path / 'run' / 'checkpoint.pt'}", f"output_dir={tmp_path / 'infer_temporal'}",
        ])
    result = run_inference(temporal_cfg)["predictions"]
    for group, n_days in (("train", 731), ("test", 1095)):
        assert all(len(p["sim_mm_day"]) == n_days and np.isfinite(p["nse"]) for p in result[group].values())
