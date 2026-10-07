"""``aires/amaps.py`` - the parts that can be tested without a finished run tree.

The figure functions need a real 64-walker genealogy on disk, so what is pinned here is
the one piece that is pure I/O and had a latent trap in it: ``truth_field`` used to
hardcode both the filename stem and the variable name to ``t2m_anom``, which is correct
for every AI+RES production so far and silently wrong for the first one that is not.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from aires import amaps as M
from gencast_s2s import config as C


def _write(path, var):
    lat = np.arange(24.0, 50.25, 0.25)
    lon = np.arange(235.0, 294.25, 0.25)
    da = xr.DataArray(np.zeros((lat.size, lon.size), dtype="float32") + 1.5,
                      dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name=var)
    path.parent.mkdir(parents=True, exist_ok=True)
    da.to_dataset().to_netcdf(path)


def test_truth_field_reads_the_metric_it_is_asked_for(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "OBS_DIR", tmp_path)
    _write(tmp_path / "EventX_verif_t2m_anom.nc", "t2m_anom")
    _write(tmp_path / "EventX_verif_u850_speed.nc", "u850_speed")
    _write(tmp_path / "EventX_hrrr_verif_t2m_anom.nc", "t2m_anom")

    assert M.truth_field("EventX").name == "t2m_anom"          # the default is unchanged
    assert M.truth_field("EventX", "t2m_anom").name == "t2m_anom"
    assert M.truth_field("EventX", "u850_speed").name == "u850_speed"
    assert M.truth_field("EventX", "t2m_anom", "hrrr").name == "t2m_anom"


def test_truth_field_names_the_file_it_wanted_rather_than_loading_another_metric(
        tmp_path, monkeypatch):
    """The trap: with the stem hardcoded, a wind event would have silently returned the
    SAME event's T2m field - right shape, right grid, wrong quantity."""
    monkeypatch.setattr(C, "OBS_DIR", tmp_path)
    _write(tmp_path / "EventX_verif_t2m_anom.nc", "t2m_anom")
    with pytest.raises(SystemExit, match="EventX_verif_u850_speed.nc"):
        M.truth_field("EventX", "u850_speed")


def test_the_symmetric_scale_survives_a_near_median_event():
    """``_sym`` sets every anomaly panel's limits. A near-zero-anomaly event is the case
    where it could return 0 and hand matplotlib ``vmin == vmax``; the ``or 1.0`` guard
    is what stops that, and a genuinely small field must still get a small, non-zero
    scale rather than being snapped to the guard."""
    assert M._sym(np.zeros(9)) == 1.0
    assert M._sym(np.array([0.21, -0.08, 0.13])) == pytest.approx(0.21)
    assert M._sym(np.array([np.nan, -0.4, 0.2])) == pytest.approx(0.4)
