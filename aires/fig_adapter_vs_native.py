"""Adapter-IC vs native-IC FourCastNet 3: the maps behind Gate 2.

Gate 2 reduced the two ensembles to a scalar (``A_L``, a cos-lat CONUS mean) and to
divergence curves. This script draws the fields those scalars came from, entirely from
data already on disk -- no model is run.

Domain is 0.25 deg CONUS (105 x 237), which is the FULL extent of the cached FCN3
output: ``xres``/``fcn3`` crop to CONUS before the host transfer, so no global forecast
field exists for either ensemble. Both cubes are 24 members from the same ERA5 analysis
at the same init with matched internal-noise seeds, differing only in how the 72 input
channels were assembled.

    PYTHONPATH=. python -m aires.fig_adapter_vs_native
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from aires import aindex as AI
from fcn3 import fevents as F

EVENT = "PNW_HeatDome_2021"
NATIVE_CUBE = Path("runs/fcn3/week3/cache") / f"{EVENT}_cube.nc"
ADAPTER_CUBE = Path("runs/aires/gate2/cache") / f"{EVENT}_adapter_cube.nc"
OUT = Path("figures/aires") / f"adapter_vs_native_{EVENT}.png"


def _event() -> F.Event:
    if EVENT not in F.EVENTS:
        raise SystemExit(f"unknown event {EVENT!r}; known: {', '.join(F.ORDER)}")
    return F.EVENTS[EVENT]


def _mean_anom_field(cube_path: Path, ev: F.Event, where: str) -> xr.DataArray:
    """Ensemble-mean verification-window anomaly field, same code path as Gate 2."""
    cube = xr.open_dataset(cube_path)
    AI.check_window(cube, ev.peak, ev.metric, where=where)
    return AI.field(cube, ev.peak, ev.metric).mean("member")


def _panel(ax, da, *, cmap, vmin, vmax, title):
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_extent([235, 294, 24, 50], crs=ccrs.PlateCarree())
    m = ax.pcolormesh(da["lon"], da["lat"], da.values, transform=ccrs.PlateCarree(),
                      cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
    ax.coastlines(linewidth=0.5, color="0.15")
    ax.add_feature(cfeature.STATES, linewidth=0.25, edgecolor="0.5")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="0.3")
    return m


def main(argv=None) -> int:
    ev = _event()
    for p in (NATIVE_CUBE, ADAPTER_CUBE):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    nat = _mean_anom_field(NATIVE_CUBE, ev, "native cube")
    ada = _mean_anom_field(ADAPTER_CUBE, ev, "adapter cube")
    dif = ada - nat

    fmax = float(max(abs(nat).max(), abs(ada).max()))
    dmax = float(abs(dif).max())
    w = np.cos(np.deg2rad(dif["lat"]))
    rmse = float(np.sqrt((dif ** 2).weighted(w).mean()))

    proj = ccrs.LambertConformal(central_longitude=-98, central_latitude=39,
                                 standard_parallels=(33, 45))
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4), layout="constrained",
                           subplot_kw={"projection": proj})

    m0 = _panel(ax[0], nat, cmap="RdBu_r", vmin=-fmax, vmax=fmax,
                title="FCN3 from truth IC\n(ERA5 $\\rightarrow$ FCN3, 24 members)")
    _panel(ax[1], ada, cmap="RdBu_r", vmin=-fmax, vmax=fmax,
           title="FCN3 from adapter IC\n(GenCast state $\\rightarrow$ FCN3, 24 members)")
    m2 = _panel(ax[2], dif, cmap="PuOr_r", vmin=-dmax, vmax=dmax,
                title=f"difference  (adapter $-$ truth)\nmax $|\\Delta|$ = {dmax:.2f} K")

    cb0 = fig.colorbar(m0, ax=ax[:2], orientation="horizontal",
                       fraction=0.055, pad=0.04, aspect=45)
    cb0.set_label("ensemble-mean T2m anomaly over the verification week (K)", fontsize=10)
    cb2 = fig.colorbar(m2, ax=ax[2], orientation="horizontal",
                       fraction=0.055, pad=0.04, aspect=22)
    cb2.set_label("$\\Delta$ T2m (K)", fontsize=10)

    fig.suptitle(
        f"{EVENT} - FourCastNet 3 initialized from the adapter vs from truth   "
        f"(Gate 2, M=24, init {ev.init:%Y-%m-%d}, week ending {ev.peak})\n"
        f"0.25$^\\circ$ CONUS - the full extent of the cached cubes   |   "
        f"field RMSE {rmse:.3f} K   |   "
        f"$\\Delta\\overline{{A_L}}$ = {float(AI.area_mean(dif)):+.3f} K",
        fontsize=12, y=0.995, va="top")

    # constrained layout reserves height for a ONE-line suptitle; this title is
    # two lines, so give the layout engine an explicit rect that stops below it.
    fig.get_layout_engine().set(rect=(0.005, 0.005, 0.99, 0.84))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"truth  IC: mean {float(AI.area_mean(nat)):+.4f} K, "
          f"range [{float(nat.min()):+.2f}, {float(nat.max()):+.2f}]")
    print(f"adapter IC: mean {float(AI.area_mean(ada)):+.4f} K, "
          f"range [{float(ada.min()):+.2f}, {float(ada.max()):+.2f}]")
    print(f"difference: cos-lat RMSE {rmse:.4f} K, max |diff| {dmax:.4f} K, "
          f"area-mean {float(AI.area_mean(dif)):+.4f} K")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
