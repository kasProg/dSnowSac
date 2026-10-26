"""Tests for src/paramnet.py -- pure PyTorch, no Fortran/Tesseract
involved, so these stay fast regardless of what the physics-backed
suite costs.
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from paramnet import SACSMA_BOUNDS, SNOW17_BOUNDS, ParamNet  # noqa: E402

N_STATIC = 39
N_CLIMATE = 3
N_MONTHS = 12


def _make_net() -> ParamNet:
    return ParamNet(n_static_features=N_STATIC, n_climate_features=N_CLIMATE)


def _make_inputs(batch: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x_static = torch.tensor(rng.normal(0, 1, (batch, N_STATIC)), dtype=torch.float64)
    x_climate = torch.tensor(rng.normal(0, 1, (batch, N_MONTHS, N_CLIMATE)), dtype=torch.float64)
    return x_static, x_climate


def test_output_shapes_and_dtypes():
    net = _make_net()
    x_static, x_climate = _make_inputs(5)
    theta_A, theta_B = net(x_static, x_climate)

    assert theta_A.shape == (5, 11)
    assert theta_B.shape == (5, 16)
    # Must match src/pipeline.py's CoupledNWSStack.run() expectations --
    # Snow17 float32, SAC-SMA float64 (src/coupling.py tracks each
    # leaf's dtype independently; getting this wrong here would silently
    # break gradient dtype matching downstream).
    assert theta_A.dtype == torch.float32
    assert theta_B.dtype == torch.float64


def test_outputs_always_respect_parameter_bounds():
    """Structural guarantee, not a training outcome: the bounded-sigmoid
    output layer must keep every parameter inside its physical range
    regardless of what the (untrained, random-init) network weights are
    -- this is what prevents the NaN-blowup failure mode
    tests/test_pipeline_hhwm8.py hit with unconstrained direct
    optimization (see that file's docstring)."""
    net = _make_net()
    # A wide spread of inputs, including extreme values -- the bound
    # guarantee must hold regardless of how extreme the raw network
    # output gets, not just for "reasonable" inputs.
    x_static_1, x_climate_1 = _make_inputs(20, seed=0)
    x_static_2, x_climate_2 = _make_inputs(20, seed=1)
    x_static = torch.cat([x_static_1, x_static_2 * 50])
    x_climate = torch.cat([x_climate_1, x_climate_2 * 50])
    theta_A, theta_B = net(x_static, x_climate)

    for i, name in enumerate(net.snow17_names):
        lo, hi = SNOW17_BOUNDS[name]
        vals = theta_A[:, i].detach().numpy()
        assert np.all(vals >= lo) and np.all(vals <= hi), (name, vals.min(), vals.max(), lo, hi)

    for i, name in enumerate(net.sacsma_names):
        lo, hi = SACSMA_BOUNDS[name]
        vals = theta_B[:, i].detach().numpy()
        assert np.all(vals >= lo) and np.all(vals <= hi), (name, vals.min(), vals.max(), lo, hi)


def test_gradient_flows_to_every_network_parameter():
    net = _make_net()
    x_static, x_climate = _make_inputs(3)
    theta_A, theta_B = net(x_static, x_climate)
    loss = theta_A.sum() + theta_B.sum()
    loss.backward()

    params_with_grad = [p for p in net.parameters() if p.grad is not None]
    assert len(params_with_grad) == len(list(net.parameters())), (
        "some network parameter never received a gradient -- a layer "
        "(e.g. the LSTM or the head) is disconnected from the output"
    )
    assert all(torch.all(torch.isfinite(p.grad)) for p in params_with_grad)
    assert any(p.grad.abs().sum() > 0 for p in params_with_grad), (
        "all gradients are exactly zero -- silently disconnected from the "
        "loss despite grad_fn being set"
    )


def test_saved_paramnet_checkpoints_still_load():
    """ParamNet now shares its bounding with DailyParamNet through a base
    class; its state_dict keys must not have moved, or every saved run's
    checkpoint.pt stops loading."""
    import pytest

    path = REPO_ROOT / "results" / "runs" / "model_9yrs_spatial" / "checkpoint.pt"
    if not path.exists():
        pytest.skip("no saved checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=True)
    ParamNet(n_static_features=state["head.0.weight"].shape[1] - 32).load_state_dict(state)


N_DAYS = 400


def _make_daily(batch: int, seed: int = 0, dropout: float = 0.5):
    from paramnet import DailyParamNet

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = DailyParamNet(n_static_features=N_STATIC, hidden=32, dropout=dropout)
    x_static = torch.tensor(rng.normal(0, 1, (batch, N_STATIC)), dtype=torch.float64)
    x_forcing = torch.tensor(rng.normal(0, 1, (batch, N_DAYS, 3)), dtype=torch.float64)
    return net, x_static, x_forcing


def test_daily_paramnet_shapes_dtypes_and_bounds():
    net, x_static, x_forcing = _make_daily(6)
    theta_A, theta_B = net(x_static * 50.0, x_forcing * 50.0)  # extreme inputs
    assert theta_A.shape == (6, 11) and theta_A.dtype == torch.float32
    assert theta_B.shape == (6, 16) and theta_B.dtype == torch.float64
    for theta, names, bounds in (
        (theta_A, net.snow17_names, SNOW17_BOUNDS),
        (theta_B, net.sacsma_names, SACSMA_BOUNDS),
    ):
        lo = torch.tensor([bounds[n][0] for n in names], dtype=theta.dtype)
        hi = torch.tensor([bounds[n][1] for n in names], dtype=theta.dtype)
        assert torch.all(theta >= lo) and torch.all(theta <= hi)


def test_daily_paramnet_basins_do_not_mix():
    """A basin's parameters must not depend on which other basins share
    its batch -- catches pooling over the batch dimension instead of
    time, which would look like training just fine."""
    net, x_static, x_forcing = _make_daily(4)
    net.eval()
    with torch.no_grad():
        theta_A, theta_B = net(x_static, x_forcing)
        alone_A, alone_B = net(x_static[2:3], x_forcing[2:3])
    torch.testing.assert_close(alone_A[0], theta_A[2])
    torch.testing.assert_close(alone_B[0], theta_B[2])


def test_daily_paramnet_reads_every_day_and_the_statics():
    """Time-mean readout: the FIRST day's forcing must move the
    parameters about as much as the last day's (measured ~1.9x here; a
    last-step readout over these 400 days gives exactly 0), and the
    static attributes must move them too."""
    net, x_static, x_forcing = _make_daily(2, dropout=0.0)
    x_static.requires_grad_(True)
    x_forcing.requires_grad_(True)
    theta_A, theta_B = net(x_static, x_forcing)
    (theta_A.double().sum() + theta_B.sum()).backward()
    per_day = x_forcing.grad.abs().sum(dim=(0, 2))
    assert per_day[0] > 0.1 * per_day[-1]
    assert x_static.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_daily_paramnet_chunked_eval_matches_one_batch():
    """No-grad forwards run EVAL_CHUNK basins at a time; the chunk
    boundaries must not reorder, drop or alter any basin's parameters."""
    net, x_static, x_forcing = _make_daily(7)
    net.eval()
    with torch.no_grad():
        whole_A, whole_B = net(x_static, x_forcing)
        net.EVAL_CHUNK = 3  # 3 + 3 + 1
        chunked_A, chunked_B = net(x_static, x_forcing)
    torch.testing.assert_close(chunked_A, whole_A)
    torch.testing.assert_close(chunked_B, whole_B)


def test_daily_paramnet_components_shapes_bounds_and_independence():
    """n_components = K: (batch, K, 11) / (batch, K, 16), all in bounds,
    distinct sets per component, and still no mixing across basins."""
    from paramnet import DailyParamNet

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    net = DailyParamNet(n_static_features=N_STATIC, hidden=32, n_components=3, readout="last").eval()
    x_static = torch.tensor(rng.normal(0, 1, (4, N_STATIC)), dtype=torch.float64)
    x_forcing = torch.tensor(rng.normal(0, 1, (4, N_DAYS, 3)), dtype=torch.float64)
    with torch.no_grad():
        theta_A, theta_B = net(x_static * 50.0, x_forcing * 50.0)
        alone_A, _ = net(x_static[1:2] * 50.0, x_forcing[1:2] * 50.0)
        net.EVAL_CHUNK = 3
        chunked_A, chunked_B = net(x_static * 50.0, x_forcing * 50.0)
    assert theta_A.shape == (4, 3, 11) and theta_A.dtype == torch.float32
    assert theta_B.shape == (4, 3, 16) and theta_B.dtype == torch.float64
    for theta, names, bounds in (
        (theta_A, net.snow17_names, SNOW17_BOUNDS),
        (theta_B, net.sacsma_names, SACSMA_BOUNDS),
    ):
        lo = torch.tensor([bounds[n][0] for n in names], dtype=theta.dtype)
        hi = torch.tensor([bounds[n][1] for n in names], dtype=theta.dtype)
        assert torch.all(theta >= lo) and torch.all(theta <= hi)
    assert not torch.allclose(theta_B[:, 0], theta_B[:, 1]), "components share one parameter set"
    torch.testing.assert_close(alone_A[0], theta_A[1])
    torch.testing.assert_close(chunked_A, theta_A)
    torch.testing.assert_close(chunked_B, theta_B)


def test_daily_paramnet_last_readout_reads_the_window_end():
    """dHBV's readout: the parameters come from the last step, so the
    final day moves them and the first of 400 days does not (the
    opposite of the mean readout's test above)."""
    from paramnet import DailyParamNet

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    net = DailyParamNet(n_static_features=N_STATIC, hidden=32, dropout=0.0, readout="last")
    x_static = torch.tensor(rng.normal(0, 1, (2, N_STATIC)), dtype=torch.float64)
    x_forcing = torch.tensor(rng.normal(0, 1, (2, N_DAYS, 3)), dtype=torch.float64, requires_grad=True)
    theta_A, theta_B = net(x_static, x_forcing)
    (theta_A.double().sum() + theta_B.sum()).backward()
    per_day = x_forcing.grad.abs().sum(dim=(0, 2))
    assert per_day[-1] > 0
    assert per_day[0] < 1e-6 * per_day[-1]
