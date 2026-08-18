"""Contract tests for the GenCast walker.

The walker's job is to be *restartable from an arbitrary state*, which is exactly what the
original inference path never had to do: ``gencast_s2s/inference.py`` always starts from
ERA5 at a fixed lead before a known event peak. Everything here pins some part of that
generalisation - that a batch built from a state at an arbitrary time has the same shape
as one built from ERA5, that a state survives a disk round trip, and that a restart state
is rejected loudly if it is cropped, mis-spaced or short a variable.

The batch tests use a hand-written task spec rather than the real checkpoint so they stay
fast and hermetic. ``test_task_spec_matches_checkpoint`` pins that spec against the actual
0.25 degree checkpoint, and is the one test that reads it (opt in with
``AIRES_CHECK_CKPT=1``; ~1 min).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aires import aconfig as A
from aires import walker as W

EVENT = "PNW_HeatDome_2021"

# Mirrors the ``<2019`` checkpoints' TaskConfig. Note total_precipitation_12hr is a target
# but NOT an input: GenCast does not condition on its own precip.
TASK_SPEC = dict(
    input_variables=(
        "2m_temperature", "mean_sea_level_pressure", "10m_v_component_of_wind",
        "10m_u_component_of_wind", "sea_surface_temperature", "temperature",
        "geopotential", "u_component_of_wind", "v_component_of_wind",
        "vertical_velocity", "specific_humidity", "year_progress_sin",
        "year_progress_cos", "day_progress_sin", "day_progress_cos",
        "geopotential_at_surface", "land_sea_mask"),
    target_variables=(
        "2m_temperature", "mean_sea_level_pressure", "10m_v_component_of_wind",
        "10m_u_component_of_wind", "total_precipitation_12hr",
        "sea_surface_temperature", "temperature", "geopotential",
        "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
        "specific_humidity"),
    forcing_variables=("year_progress_sin", "year_progress_cos",
                       "day_progress_sin", "day_progress_cos"),
    pressure_levels=tuple(A.P13),
    input_duration="24h",
)


def _need_state():
    p = A.gencast_inputs_path(EVENT)
    if not p.exists():
        pytest.skip(f"missing run data: {p}")
    return p


@pytest.fixture(scope="module")
def state():
    return W.read_state(_need_state())


# --------------------------------------------------------------------------- #
# State contract
# --------------------------------------------------------------------------- #
def test_state_is_global_two_frames(state):
    assert state.sizes["lat"] == A.STATE_NLAT and state.sizes["lon"] == A.STATE_NLON
    assert state.sizes["time"] == 2
    assert W.valid_time(state) == pd.Timestamp("2021-06-07 00:00")


def test_prognostic_static_split(state):
    assert set(W.prognostic(state).data_vars) == set(A.STATE_PROGNOSTIC)
    assert set(W.statics(state).data_vars) == set(A.STATE_STATIC)


def test_cropped_state_is_rejected(state):
    """The cached xres cubes are CONUS-cropped; restarting from one must fail loudly."""
    with pytest.raises(ValueError, match="global"):
        W.check_state(state.isel(lat=slice(0, 105), lon=slice(0, 237)))


def test_one_frame_state_is_rejected(state):
    with pytest.raises(ValueError, match="2 input frames"):
        W.check_state(state.isel(time=[0]))


def test_missing_variable_is_rejected(state):
    with pytest.raises(ValueError, match="specific_humidity"):
        W.check_state(state.drop_vars("specific_humidity"))


def test_wrong_frame_spacing_is_rejected(state):
    """GenCast conditions on two frames 12 h apart; 6 h apart is a different model."""
    t = pd.DatetimeIndex(state["time"].values)
    bad = state.assign_coords(time=("time", np.array([t[0] + pd.Timedelta(hours=6), t[1]])))
    with pytest.raises(ValueError, match="apart"):
        W.check_state(bad)


def test_state_round_trip(tmp_path, state):
    small = state.isel(level=slice(0, 2))          # keep the temp file small
    p = tmp_path / "state.nc"
    W.write_state(small.assign_attrs(marker="abc"), p)
    back = xr.open_dataset(p).load()
    assert back.attrs["marker"] == "abc"
    for v in A.STATE_PROGNOSTIC:
        assert np.array_equal(back[v].values, small[v].values, equal_nan=True)


def test_write_state_rejects_a_cropped_state(tmp_path, state):
    with pytest.raises(ValueError, match="global"):
        W.write_state(state.isel(lat=slice(0, 105), lon=slice(0, 237)), tmp_path / "x.nc")


def test_float16_would_overflow_geopotential(state):
    """Why write_state is float32: aires.md proposed float16 for archived states.

    Geopotential at 50 hPa is ~2e5 m2 s-2 and float16 tops out at 65504, so a float16
    archive stores inf for the top of every column - silently, on write.
    """
    z_top = float(np.nanmax(np.abs(state["geopotential"].sel(level=50).values)))
    assert z_top > np.finfo(np.float16).max, z_top
    assert np.isinf(np.float16(z_top))


# --------------------------------------------------------------------------- #
# Batch construction from an ARBITRARY state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_steps", [1, 2, 6])
def test_build_batch_from_event_init(state, n_steps):
    full, init = W.build_batch(state, n_steps)
    assert init == W.valid_time(state)
    assert full.sizes["time"] == 2 + n_steps
    for v in A.STATE_STATIC:                       # the leak this guards against
        assert full[v].dims == ("lat", "lon")


def test_build_batch_from_a_shifted_state(state):
    """The whole point of the walker: a state at an arbitrary time restarts identically."""
    shift = pd.Timedelta(days=9)
    moved = state.assign_coords(
        time=("time", pd.DatetimeIndex(state["time"].values) + shift))
    full, init = W.build_batch(moved, 4)
    assert init == W.valid_time(state) + shift
    dt = pd.DatetimeIndex(np.asarray(full["datetime"].values).ravel())
    assert dt[-1] == init + pd.Timedelta(hours=4 * W.STEP_H)
    assert (np.diff(dt) == pd.Timedelta(hours=W.STEP_H)).all()


def test_build_batch_targets_are_nan(state):
    full, _ = W.build_batch(state, 3)
    assert np.isnan(full["2m_temperature"].isel(time=slice(2, None)).values).all()
    assert not np.isnan(full["2m_temperature"].isel(time=slice(0, 2)).values).any()


def test_build_batch_does_not_mutate_the_state(state):
    before = float(state["2m_temperature"].isel(batch=0, time=-1).values[0, 0])
    W.build_batch(state, 3)
    assert float(state["2m_temperature"].isel(batch=0, time=-1).values[0, 0]) == before


@pytest.mark.parametrize("n_steps", [1, 5])
def test_extracted_arrays_have_the_expected_shape(state, n_steps):
    from graphcast import data_utils

    full, _ = W.build_batch(state, n_steps)
    i, t, f = data_utils.extract_inputs_targets_forcings(
        full, target_lead_times=slice(f"{W.STEP_H}h", f"{n_steps * W.STEP_H}h"),
        **TASK_SPEC)
    nch = W.check_batch_structure(i, t, f, n_steps)
    # 5 surface + 6 upper-air x 13 levels + 4 forcings, over 2 frames, + 2 statics.
    assert nch["inputs"] == 2 * (5 + 6 * 13 + 4) + 2
    # 6 surface (precip is a target) + 6 upper-air x 13 levels, per step.
    assert nch["targets"] == n_steps * (6 + 6 * 13)
    assert nch["forcings"] == n_steps * 4


def test_leaked_static_time_axis_is_caught(state):
    """The documented failure mode: a static that gained a time axis."""
    from graphcast import data_utils

    full, _ = W.build_batch(state, 2)
    i, t, f = data_utils.extract_inputs_targets_forcings(
        full, target_lead_times=slice("12h", "24h"), **TASK_SPEC)
    broken = i.copy()
    broken["land_sea_mask"] = i["land_sea_mask"].expand_dims(time=i["time"])
    with pytest.raises(RuntimeError, match="land_sea_mask"):
        W.check_batch_structure(broken, t, f, 2)


def test_wrong_step_count_is_caught(state):
    from graphcast import data_utils

    full, _ = W.build_batch(state, 4)
    i, t, f = data_utils.extract_inputs_targets_forcings(
        full, target_lead_times=slice("12h", "48h"), **TASK_SPEC)
    with pytest.raises(RuntimeError, match="targets must span"):
        W.check_batch_structure(i, t, f, 5)


@pytest.mark.skipif(os.environ.get("AIRES_CHECK_CKPT") != "1",
                    reason="set AIRES_CHECK_CKPT=1 to read the 0.25 deg checkpoint (~1 min)")
def test_task_spec_matches_checkpoint():
    import dataclasses

    got = dataclasses.asdict(W._task_config_only())
    for k, v in TASK_SPEC.items():
        assert tuple(got[k]) == tuple(v), k


# --------------------------------------------------------------------------- #
# RNG / cloning
# --------------------------------------------------------------------------- #
def test_segment_key_is_deterministic():
    assert np.array_equal(W.segment_key(3, 2), W.segment_key(3, 2))


def test_segment_key_differs_per_walker_and_step():
    keys = {tuple(np.asarray(W.segment_key(w, k)).tolist())
            for w in range(6) for k in range(4)}
    assert len(keys) == 24, "cloned walkers would share a noise stream"


def test_segment_key_respects_base_seed():
    assert not np.array_equal(W.segment_key(0, 1, base_seed=1),
                              W.segment_key(0, 1, base_seed=2))


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def test_run_tree_layout():
    sp = A.state_path("EV", 7, 3)
    assert sp.name == "state.nc" and sp.parent.name == "step03"
    assert sp.parent.parent.name == "w07"
    assert A.diag_path("EV", 7, 3).parent == sp.parent
    assert A.state_path("EV", 0, 1) != A.state_path("EV", 1, 1)
