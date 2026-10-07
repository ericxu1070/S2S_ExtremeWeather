"""CFS source for acal.maps: estimator, bias correction, scorer parity."""
import numpy as np
import xarray as xr

from acal import maps


def test_cfs_prob_ties_and_cold_sign():
    # 4 members at one cell; thresholds -2,-3,-4,2,3,4. A tie at the threshold counts (>=).
    F = np.array([2.0, 3.0, -3.0, -4.0]).reshape(4, 1, 1)
    p = maps.cfs_prob(F)[:, 0, 0]
    got = dict(zip(maps.THRESHOLDS, p))
    assert got[-2.0] == 0.5 and got[-3.0] == 0.5 and got[-4.0] == 0.25   # cold: <= a
    assert got[2.0] == 0.5 and got[3.0] == 0.25 and got[4.0] == 0.0      # heat: >= a


def test_corrected_is_raw_minus_bias():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(16, 3, 4)).astype("float32")
    b = rng.normal(size=(3, 4)).astype("float32")
    shifted = maps.cfs_prob(F - b[None])
    manual = np.stack([((F - b[None]) >= a).mean(0) if a > 0 else ((F - b[None]) <= a).mean(0)
                       for a in maps.THRESHOLDS]).astype("float32")
    np.testing.assert_array_equal(shifted, manual)
    assert not np.array_equal(shifted, maps.cfs_prob(F))


def _old_scores(ds, land):
    """The pre-refactor scorer, inlined: reads d.prob directly."""
    a = 3.0
    d = ds.sel(threshold=a).where(ds.family == "heat", drop=True)
    o = (d.truth >= a).astype("float64")
    yes = (d.prob >= maps.YES).astype("float64")
    bs = ((d.prob - o) ** 2).sum("case")
    bsc = ((d.clim - o) ** 2).sum("case")
    return (1 - bs / bsc.where(bsc > 0)), (yes * o).sum("case") / o.sum("case").where(o.sum("case") > 0)


def test_scores_parity_with_prob_var(monkeypatch):
    rng = np.random.default_rng(1)
    nc, nl, nx = 6, 3, 4
    shp = (nc, len(maps.THRESHOLDS), nl, nx)
    ds = xr.Dataset(
        dict(prob=(("case", "threshold", "lat", "lon"), rng.random(shp)),
             clim=(("case", "threshold", "lat", "lon"), rng.random(shp)),
             truth=(("case", "lat", "lon"), rng.normal(3, 2, (nc, nl, nx)))),
        coords=dict(case=list("abcdef"), threshold=list(maps.THRESHOLDS),
                    lat=np.arange(nl), lon=np.arange(nx),
                    family=("case", ["heat"] * 4 + ["cold"] * 2)))
    monkeypatch.setattr(maps, "land_mask", lambda d: xr.DataArray(
        np.ones((nl, nx), bool), dims=("lat", "lon"), coords=dict(lat=d.lat, lon=d.lon)))
    bss_old, pod_old = _old_scores(ds, None)
    sc = maps.scores(ds)
    np.testing.assert_allclose(sc.bss.sel(threshold=3.0), bss_old)
    np.testing.assert_allclose(sc.pod.sel(threshold=3.0), pod_old)
    # a renamed probability scores identically
    ds2 = ds.rename(prob="prob_raw")
    xr.testing.assert_allclose(maps.scores(ds2, "prob_raw"), sc)
