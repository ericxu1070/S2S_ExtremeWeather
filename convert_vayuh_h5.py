#!/usr/bin/env python
"""Convert the Vayuh probabilistic forecast HDFStore to a gridded NetCDF cube.

``us_prob_forecast_2026.h5`` is a pandas HDFStore (key ``/forecast``) indexed by
(start_date, lat, lon) with columns ``preds_<L>`` = tmp2m anomaly (degC) at lead L days,
plus exceedance-probability columns. The (lat, lon) set is a masked CONUS subset (~862 of
the 24x57 box), so we reindex onto the full 1 deg grid (missing -> NaN).

Output: runs/heatwave2026/vayuh_forecast.nc with data var ``pred`` (lead, start_date, lat, lon).

Run under an env that has pytables (npl-2025b on Derecho); the plotting env (my-env) only
needs to read the resulting NetCDF.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent
H5 = ROOT / "us_prob_forecast_2026.h5"
OUT = ROOT / "runs" / "heatwave2026" / "vayuh_forecast.nc"
LEADS = list(range(14, 28))                       # preds_14 .. preds_27


def main() -> None:
    if not H5.exists():
        sys.exit(f"missing {H5}")
    with pd.HDFStore(H5, "r") as s:
        df = s.get("/forecast")
    df.index = df.index.set_names(["start_date", "lat", "lon"])

    pred_cols = [f"preds_{L}" for L in LEADS if f"preds_{L}" in df.columns]
    ds = df[pred_cols].to_xarray()                # dims (start_date, lat, lon) per column

    # Reindex to the full 1 deg CONUS box so the grid is regular (missing points -> NaN).
    lat = np.arange(float(ds.lat.min()), float(ds.lat.max()) + 1, 1.0)
    lon = np.arange(float(ds.lon.min()), float(ds.lon.max()) + 1, 1.0)
    ds = ds.reindex(lat=lat, lon=lon)

    leads = [int(c.split("_")[1]) for c in pred_cols]
    pred = xr.concat([ds[c] for c in pred_cols], dim="lead").assign_coords(lead=leads)
    out = pred.to_dataset(name="pred")
    out["pred"].attrs.update(units="degC", long_name="tmp2m anomaly forecast (Vayuh)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".nc.tmp")
    out.to_netcdf(tmp)
    tmp.replace(OUT)
    sd = pd.DatetimeIndex(out.start_date.values)
    print(f"wrote {OUT}")
    print(f"  leads {min(leads)}..{max(leads)}  start_dates {sd.min():%Y-%m-%d}..{sd.max():%Y-%m-%d} "
          f"({out.sizes['start_date']})  grid {out.sizes['lat']}x{out.sizes['lon']}")
    valid = np.isfinite(out['pred'].isel(lead=0)).sum(('lat', 'lon')).max().item()
    print(f"  valid (unmasked) points per field ~ {valid} of {out.sizes['lat'] * out.sizes['lon']}")


if __name__ == "__main__":
    main()
