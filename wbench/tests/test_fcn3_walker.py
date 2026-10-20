"""The FCN3 walker's state format, seeds, grid crop and per-walker noise streams - on CPU.

The noise tests drive makani's REAL ``DiffusionNoiseS2`` on a tiny grid through a stub
model package, so they need the ``fcn3`` env (torch, torch-harmonics, makani) but no GPU
and no checkpoint. The full forward is exercised on the GPU by ``wbench.checks``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wbench import fcn3_walker as FW

LATS = np.linspace(90.0, -90.0, 721)
LONS = np.linspace(0.0, 360.0, 1440, endpoint=False)


def test_segment_seed_is_stable_and_distinct():
    a = FW.segment_seed(1, 2, 3)
    assert a == FW.segment_seed(1, 2, 3)
    assert len({FW.segment_seed(1, w, s) for w in range(20) for s in range(20)}) == 400
    assert 0 <= a < 2 ** 63


def test_conus_crop_matches_the_climatology_grid():
    # CLAUDE.md: the 0.25 deg CONUS climatology and every cube are lat=105, lon=237.
    la, lo = FW.conus_slices(LATS, LONS)
    assert (la.stop - la.start, lo.stop - lo.start) == (105, 237)
    assert LATS[la][0] == 50.0 and LATS[la][-1] == 24.0


@pytest.mark.parametrize("fmt", ["nc", "npz"])
@pytest.mark.parametrize("with_noise", [True, False])
def test_state_round_trip_is_bitwise(tmp_path, fmt, with_noise):
    rng = np.random.default_rng(0)
    x = rng.normal(280, 20, (3, 5, 8)).astype(np.float32)
    x[0, 0, 0] = 2.0e5                              # geopotential-sized: float16 would overflow
    noise = rng.normal(size=(1, 2, 4, 3, 2)).astype(np.float32) if with_noise else None
    t = pd.Timestamp("2021-06-10T00")
    p = FW.write_state(tmp_path / f"s.{fmt}", x, noise, t, ["a", "b", "c"],
                       np.arange(5.0), np.arange(8.0), attrs=dict(walker=3), fmt=fmt)
    r = FW.read_state(p)
    assert r["x"].dtype == np.float32 and np.array_equal(r["x"], x)
    assert r["valid_time"] == t and r["variables"] == ["a", "b", "c"]
    assert int(r["attrs"]["noise_persisted"]) == int(with_noise)
    if with_noise:
        assert np.array_equal(r["noise"], noise)
    else:
        assert r["noise"] is None
    assert not list(tmp_path.glob("*.tmp.*"))       # atomic: no litter


def test_state_rejects_wrong_rank(tmp_path):
    with pytest.raises(ValueError):
        FW.write_state(tmp_path / "s.nc", np.zeros((5, 8), np.float32), None,
                       "2021-06-10", ["a"], np.arange(5.0), np.arange(8.0))


# --------------------------------------------------------------------------- #
# Per-walker noise streams, against makani's real noise process
# --------------------------------------------------------------------------- #
@pytest.fixture
def native():
    torch = pytest.importorskip("torch")
    noise_mod = pytest.importorskip("makani.models.noise")

    class Stub(torch.nn.Module):
        pass

    pre = Stub()
    pre.input_noise = noise_mod.DiffusionNoiseS2(img_shape=(9, 16), batch_size=1,
                                                 num_channels=2, num_time_steps=1,
                                                 sigma=1.0, kT=0.01, lambd=0.1)
    net = Stub()
    net.preprocessor = pre
    pkg = Stub()
    pkg.model = net
    pkg.register_buffer("in_bias", torch.zeros(1, 3, 1, 1))
    pkg.add_zenith, pkg.timestep = True, 6
    pkg.lats, pkg.lons = LATS, LONS
    px = Stub()
    px.model, px.variables = pkg, ["u10m", "t2m", "z500"]
    return torch, FW.NativeFCN3(px), pre.input_noise


def test_init_noise_is_makanis_own_stationary_draw(native):
    """At B=1 the walker's first noise state is exactly what earth2studio's set_rng(seed)
    + _reset_internal_state would draw: makani's update(replace_state=True) from a
    generator seeded with ``seed``."""
    torch, nat, noise = native
    seed = FW.segment_seed(7, 0, 0)
    mine = nat.init_noise(nat.generators([seed]))
    noise.set_rng(seed=seed)
    noise.reset(batch_size=1)
    noise.update(replace_state=True)
    assert torch.equal(mine, noise.state)


def test_noise_streams_do_not_depend_on_the_batch(native):
    torch, nat, _ = native
    seeds = [FW.segment_seed(7, w, 1) for w in range(4)]
    n4 = nat.init_noise(nat.generators(seeds))
    n4 = nat.advance_noise(n4, nat.generators([s + 1 for s in seeds]))
    n1 = nat.init_noise(nat.generators(seeds[2:3]))
    n1 = nat.advance_noise(n1, nat.generators([seeds[2] + 1]))
    assert torch.equal(n4[2:3], n1)


def test_clones_draw_different_noise_from_one_parent(native):
    torch, nat, _ = native
    parent = nat.init_noise(nat.generators([FW.segment_seed(7, 0, 0)]))
    kids = nat.advance_noise(parent.repeat(2, 1, 1, 1, 1, 1),
                             nat.generators([FW.segment_seed(7, w, 1) for w in (10, 11)]))
    assert not torch.equal(kids[0], kids[1])


def test_advance_carries_the_ar1_memory(native):
    """advance_noise is phi * state + eta, not a fresh draw: with phi = exp(-0.1) the new
    state correlates ~0.9 with the old one, where a re-drawn (earth2studio-restart) state
    is uncorrelated. That memory is what a restart loses unless the state is persisted."""
    torch, nat, noise = native
    s0 = nat.init_noise(nat.generators([1]))
    s1 = nat.advance_noise(s0, nat.generators([2]))
    fresh = nat.init_noise(nat.generators([3]))

    def corr(a, b):
        a, b = a.flatten().double(), b.flatten().double()
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm()))

    phi = float(noise.phi.flatten()[0])
    assert corr(s1, s0) > 0.7 and abs(corr(s1, s0) - phi) < 0.15
    assert abs(corr(fresh, s0)) < 0.3


def test_release_restores_a_batch_one_module(native):
    torch, nat, noise = native
    nat.init_noise(nat.generators([1, 2, 3]))
    nat.release()
    assert noise.state.shape[0] == 1 and not noise.state.is_inference()
