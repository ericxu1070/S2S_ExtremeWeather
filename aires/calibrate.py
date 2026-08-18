"""Fit and validate the three FCN3 channels GenCast cannot supply directly.

Both the fit and the test set are FCN3 initial-condition files already on disk. Those
carry all 72 channels, including the TRUE ``u100m``/``v100m``/``tcwv``, alongside the
inputs the derivation uses - so calibration needs no network access, no GPU and no
second dataset.

    fit  : runs/p90_t2m/ic/*.nc        90 cases, 2021-2025  (the p90 experiment's ICs)
    test : runs/fcn3/week3/ic/*_ic.nc   6 events            (fully held out)

Keeping the two sets disjoint matters: a per-grid-point fit has ~1M free parameters
and would report a flattering in-sample error otherwise.

    PYTHONPATH=. python -m aires.calibrate --fit        # build the calibration
    PYTHONPATH=. python -m aires.calibrate --gate1      # score it on the held-out ICs
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from aires import aconfig as A
from aires import adapter as AD

QVARS = [f"q{lev}" for lev in A.P13]
NEEDED = ["u10m", "v10m", "u100m", "v100m", "tcwv"] + QVARS


def _read(path: str) -> dict[str, np.ndarray]:
    """Pull just the channels the calibration touches out of an FCN3 IC file."""
    with xr.open_dataset(path) as ds:
        ic = ds["ic"].isel(time=0)
        sub = ic.sel(variable=NEEDED).astype("float64").values  # (nvar, lat, lon)
        lat = ds["lat"].values.copy()
    out = {nm: sub[i] for i, nm in enumerate(NEEDED)}
    out["_lat"] = lat
    return out


def _area_weights(lat: np.ndarray, shape) -> np.ndarray:
    w = np.cos(np.deg2rad(lat))
    return np.broadcast_to(w[:, None], shape)


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def _trapezoid_weights() -> np.ndarray:
    """The physical quadrature as a weight vector: ``(1/g) * int q dp`` by trapezoid."""
    p = np.asarray(A.P13, dtype="float64") * 100.0
    w = np.zeros(len(p))
    for i in range(len(p) - 1):
        dp = p[i + 1] - p[i]
        w[i] += 0.5 * dp
        w[i + 1] += 0.5 * dp
    return w / AD.G


def fit(paths: list[str], out_path: Path | None = None, verbose: bool = True,
        ridge: float = 1e-3, chunk: int = 128) -> xr.Dataset:
    """Fit both derived-channel models, streamed over the input frames.

    wind
        ``W100 = c(x) W10`` with c complex - one number carrying both the boundary-layer
        speed-up (``|c|``) and the veering (``arg c``). Least squares through the origin:
        ``c = sum(W100 conj(W10)) / sum(|W10|^2)``.

    tcwv
        ``tcwv = sum_l w_l(x) q_l(x)`` - a per-grid-point quadrature rule. Solved as a
        ridge problem shrunk toward the trapezoid weights, so where the data is
        uninformative (dry columns, few samples) the answer decays to the physics rather
        than to zero. Learning the weights rather than scaling the trapezoid is what
        makes Gate 1 pass: the trapezoid's error is dominated by the missing sub-1000 hPa
        layer and by underground levels over terrain, neither of which a single scale can
        correct.

    Memory is bounded by accumulating the 13x13 normal equations in float32
    (~0.7 GB global) and solving in latitude chunks.
    """
    if not paths:
        raise SystemExit("no calibration files matched")
    paths = sorted(paths)
    nlev = len(A.P13)

    s_ww = s_cross = XtX = XtY = lat = lon = None
    for i, p in enumerate(paths, 1):
        d = _read(p)
        w10 = d["u10m"] + 1j * d["v10m"]
        w100 = d["u100m"] + 1j * d["v100m"]
        q = np.stack([d[k] for k in QVARS]).astype("float32")
        t = d["tcwv"].astype("float32")
        gram = np.einsum("ixy,jxy->ijxy", q, q, dtype="float32")
        rhs = np.einsum("ixy,xy->ixy", q, t, dtype="float32")
        if s_ww is None:
            s_ww = np.abs(w10) ** 2
            s_cross = w100 * np.conj(w10)
            XtX, XtY = gram, rhs
            lat = d["_lat"]
            with xr.open_dataset(p) as ds0:
                lon = ds0["lon"].values.copy()
        else:
            s_ww += np.abs(w10) ** 2
            s_cross += w100 * np.conj(w10)
            XtX += gram
            XtY += rhs
        if verbose and (i % 10 == 0 or i == len(paths)):
            print(f"  [{i}/{len(paths)}] {Path(p).name}", flush=True)

    n = len(paths)

    # ---- wind -------------------------------------------------------------------
    # Grid points becalmed in every sample give a singular normal equation; fall back
    # to the neutral power law there instead of dividing by ~0.
    ww_floor = max(1e-6 * float(np.median(s_ww)), 1e-12)
    wind_ok = s_ww > ww_floor
    coef = np.where(wind_ok, s_cross / np.where(wind_ok, s_ww, 1.0), AD.DEFAULT_WIND_COEF)

    # ---- tcwv quadrature ---------------------------------------------------------
    wtrap = _trapezoid_weights()
    weights = np.empty((lat.size, lon.size, nlev), dtype="float64")
    eye = np.eye(nlev)
    for j0 in range(0, lat.size, chunk):
        j1 = min(j0 + chunk, lat.size)
        gram = np.moveaxis(XtX[:, :, j0:j1], (0, 1), (-2, -1)).astype("float64")
        rhs = np.moveaxis(XtY[:, j0:j1], 0, -1).astype("float64")
        lam = ridge * np.trace(gram, axis1=-2, axis2=-1) / nlev
        # Shrink toward the scaled trapezoid, not toward zero: a dry column should
        # still integrate like an atmosphere.
        den = np.einsum("l,xylm,m->xy", wtrap, gram, wtrap)
        num = np.einsum("l,xyl->xy", wtrap, rhs)
        prior = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)[..., None] * wtrap
        M = gram + lam[..., None, None] * eye
        r = rhs + lam[..., None] * prior
        weights[j0:j1] = np.linalg.solve(M, r[..., None])[..., 0]

    ds = xr.Dataset(
        {
            "wind_coef_real": (("lat", "lon"), coef.real.astype("float32")),
            "wind_coef_imag": (("lat", "lon"), coef.imag.astype("float32")),
            "wind_fallback": (("lat", "lon"), (~wind_ok).astype("int8")),
            "tcwv_weights": (("lat", "lon", "level"), weights.astype("float32")),
        },
        coords={"lat": lat, "lon": lon, "level": list(A.P13)},
    )
    ds["wind_coef_real"].attrs["long_name"] = "real part of W100 = c * W10 (along-wind speed-up)"
    ds["wind_coef_imag"].attrs["long_name"] = "imag part of c (boundary-layer veering)"
    ds["tcwv_weights"].attrs["long_name"] = "per-gridpoint quadrature weights, tcwv = sum_l w_l q_l"
    ds["tcwv_weights"].attrs["ridge"] = ridge
    ds.attrs["provenance"] = f"aires.calibrate fit on {n} ERA5 frames"
    ds.attrs["n_samples"] = n
    ds.attrs["sources"] = "; ".join(Path(x).name for x in paths[:5]) + (" ..." if n > 5 else "")

    if verbose:
        mag, ang = np.abs(coef), np.degrees(np.angle(coef))
        eff = weights @ np.ones(nlev)  # crude scale proxy vs the trapezoid
        print(f"\n  |c|       median {np.median(mag):.3f}   p5 {np.percentile(mag,5):.3f}   p95 {np.percentile(mag,95):.3f}")
        print(f"  arg(c) deg median {np.median(ang):+.2f}   p5 {np.percentile(ang,5):+.2f}   p95 {np.percentile(ang,95):+.2f}")
        print(f"  tcwv weight sum vs trapezoid: {np.median(eff)/wtrap.sum():.3f} (median ratio)")
        print(f"  wind fallback points: {int((~wind_ok).sum())} of {wind_ok.size}")

    out_path = Path(out_path or A.CALIB_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    if verbose:
        print(f"\n  wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    return ds


# --------------------------------------------------------------------------- #
# Evaluate (Gate 1)
# --------------------------------------------------------------------------- #
def evaluate(paths: list[str], calib: xr.Dataset, verbose: bool = True) -> "list[dict]":
    """Per-file skill of the derived channels against the truth in the same file."""
    coef = calib["wind_coef_real"].values + 1j * calib["wind_coef_imag"].values
    tw = calib["tcwv_weights"].sel(level=list(A.P13)).transpose("lat", "lon", "level").values
    rows = []
    for p in sorted(paths):
        d = _read(p)
        w = _area_weights(d["_lat"], d["u10m"].shape)
        u100, v100 = AD.apply_wind_coef(d["u10m"], d["v10m"], coef)
        tcwv = AD.apply_tcwv_weights(np.stack([d[q] for q in QVARS]), tw)

        def wstat(pred, truth):
            err = pred - truth
            rmse = float(np.sqrt((w * err**2).sum() / w.sum()))
            bias = float((w * err).sum() / w.sum())
            return rmse, bias

        ru, bu = wstat(u100, d["u100m"])
        rv, bv = wstat(v100, d["v100m"])
        rt, bt = wstat(tcwv, d["tcwv"])
        tmean = float((w * d["tcwv"]).sum() / w.sum())
        rows.append(
            dict(file=Path(p).name, u100m_rmse=ru, u100m_bias=bu, v100m_rmse=rv, v100m_bias=bv,
                 tcwv_rmse=rt, tcwv_bias=bt, tcwv_mean=tmean,
                 tcwv_rel_rmse=rt / tmean, tcwv_rel_bias=bt / tmean)
        )
        if verbose:
            r = rows[-1]
            print(f"  {r['file']:34s} u100m {ru:6.3f} / {bu:+6.3f}   v100m {rv:6.3f} / {bv:+6.3f}"
                  f"   tcwv {100*r['tcwv_rel_rmse']:5.2f}% / {100*r['tcwv_rel_bias']:+5.2f}%")
    return rows


def gate1(rows: "list[dict]") -> bool:
    """Apply the aires.md Gate 1 criteria to the worst held-out case."""
    g = A.GATE1
    worst = {
        "u100m_rmse": max(max(r["u100m_rmse"], r["v100m_rmse"]) for r in rows),
        "u100m_abs_bias": max(max(abs(r["u100m_bias"]), abs(r["v100m_bias"])) for r in rows),
        "tcwv_rel_rmse": max(r["tcwv_rel_rmse"] for r in rows),
        "tcwv_abs_rel_bias": max(abs(r["tcwv_rel_bias"]) for r in rows),
    }
    checks = [
        ("u100m/v100m RMSE     ", worst["u100m_rmse"], g["u100m_rmse_max"], "m/s"),
        ("u100m/v100m |bias|   ", worst["u100m_abs_bias"], g["u100m_abs_bias_max"], "m/s"),
        ("tcwv relative RMSE   ", worst["tcwv_rel_rmse"], g["tcwv_rel_rmse_max"], ""),
        ("tcwv |relative bias| ", worst["tcwv_abs_rel_bias"], g["tcwv_abs_rel_bias_max"], ""),
    ]
    print("\n  GATE 1 (worst held-out case vs threshold)")
    ok = True
    for label, got, lim, unit in checks:
        good = got <= lim
        ok &= good
        print(f"    {label} {got:8.4f} {unit:4s} <= {lim:6.3f}   {'PASS' if good else 'FAIL'}")
    print(f"\n  GATE 1: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fit", action="store_true", help="fit the calibration and write it")
    ap.add_argument("--gate1", action="store_true", help="evaluate on the held-out ICs")
    ap.add_argument("--fit-glob", default=A.CALIB_IC_GLOB)
    ap.add_argument("--test-glob", default=A.HOLDOUT_IC_GLOB)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if not (a.fit or a.gate1):
        ap.error("pass --fit and/or --gate1")

    if a.fit:
        paths = glob.glob(a.fit_glob)
        print(f"FIT on {len(paths)} frames  ({a.fit_glob})")
        fit(paths, out_path=a.out)

    if a.gate1:
        cal = AD.require_calibration(a.out)
        paths = glob.glob(a.test_glob)
        print(f"\nHELD-OUT EVALUATION on {len(paths)} frames  ({a.test_glob})")
        print(f"  calibration: {cal.attrs.get('provenance')}")
        rows = evaluate(paths, cal)
        return 0 if gate1(rows) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
