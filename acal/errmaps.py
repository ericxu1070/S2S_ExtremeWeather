#!/usr/bin/env python
"""Where the models differ: ensemble-mean error maps and skill maps side by side. CPU ONLY.

The board scores one number per case (the CONUS-mean A_L). This module draws the fields
behind it, one column per model, so the place a model goes wrong is visible:

    truth | AI+RES - truth | CFSv2 - truth | ECMWF IFS (EC46) - truth | GEFSv12 - truth |
    ECCC GEPS - truth

What a panel is
---------------
F_c(x) is a model's ensemble-mean 7-day T2m anomaly of case c (raw anomaly against the
1990-2019 ERA5 climatology, the board's headline variant, no bias correction), O_c(x) the
truth on the SAME window, so every model is verified on its own window:

  * instantaneous sources (AI+RES, CFSv2 = the board's 13-frame `cfs13`, GEFSv12): the 13
    00/12Z frames peak-6 d .. peak; truth on those 13 frames.
  * daily-mean sources (ECCC GEPS, EC46): UTC days peak-6 .. peak-1 (`s2sbase.reduce_daily`);
    truth on the matching 12 frames (peak-6 d 00Z .. peak-1 d 12Z, ruling C9).

AI+RES is the importance-weighted (self-normalized w_i = exp(-V_K,i)) mean of its 32 final
walkers, read from `runs/acal/analysis/closest_members.nc` (panel `res_mean`, built by
`acal.sidebyside --stage members` through `maps.walker_fields`); its CONUS mean is asserted
against sum_i w_i A_L,i of `aires_al_windows.csv`. Every other source is the equal-weight
mean of its native members (CFSv2 16, GEFSv12 31, ECCC GEPS 21, EC46 its own), each member
reduced by `s2sbase.reduce` and, where the source json exists, asserted against its `al`.

Composites (`--stage composite`, figures/acal/overall/errmap_<truth>.png): rows are the
equal-weight case mean of F_c - O_c over the 31 heat and the 11 cold cases, and the mean
absolute error mean_c |F_c - O_c| over all 42. The truth column shows the 13-frame truth
composite (mean |O_c| on the MAE row: the MAE a zero-anomaly forecast would score). A
source missing a case drops that case from its own composite (n printed when short).
Statistics are cos(lat)-weighted over the scored cells (land, and the HRRR mask for an
HRRR truth): bias = mean error, RMSE of the error field, and r = centred pattern
correlation of the model composite with the truth composite; on the MAE row, RMSE pools
every case and r is the mean per-case pattern correlation.

Skill maps (`--stage skill`, figures/acal/overall/skillmaps_{csi,bss}_<truth>.png): CSI and
Brier skill score at +2 K (heat cases) and -2 K (cold cases), `maps.scores` unchanged on
each source's maps file (AI+RES `maps.truth_base(truth)`, every other source
runs/acal/analysis/s2s/<truth>/maps_fields_<source>.nc from `s2sbase --stage maps`).

Per case (`--stage cases`): figures/acal/s2s/errmaps/<truth>/<case>.png (42 per truth) and
runs/acal/analysis/s2s/errmaps_<truth>.csv (case x model: land-mean error, RMSE, r).

A source with no cube yet (EC46 before its download) is absent; rerunning picks it up.

    python -m acal.errmaps --truth era5 --stage all     # ~5 min first time (ens-mean cache)
    python -m acal.errmaps --truth hrrr --stage all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import analyze as AN
from acal import aprep, ccfg
from acal import maps as M
from acal import s2sbase as S2
from acal import truth as TR
from aires import aindex as AI
from matplotlib.ticker import MaxNLocator

OVERALL_DIR = ccfg.FIG_ROOT / "overall"
CASE_FIG_DIR = S2.FIG_DIR / "errmaps"
CACHE_DIR = S2.ANALYSIS / "errmaps"
RES_MEAN_NC = AN.OUT / "closest_members.nc"      # published, read only
CONSISTENCY_TOL = 1e-3                            # K
SKILL_THRESHOLDS = (2.0, -2.0)
BOARD = ("aires",) + tuple(getattr(S2, "BOARD_SOURCES", ("cfs13", "ec46", "gefs", "geps")))

_STYLE_FALLBACK = {
    "aires": dict(label="AI+RES", color="#D55E00"),
    "cfs13": dict(label="CFSv2", color="#0072B2", deg="~0.94 deg"),
    "ec46": dict(label="ECMWF IFS (EC46)", color="#009E73", deg="1.5 deg"),
    "gefs": dict(label="GEFSv12", color="#CC79A7", deg="0.5 deg"),
    "geps": dict(label="ECCC GEPS", color="#E69F00", deg="1 deg"),
}
NATIVE = {"aires": "0.25 deg GenCast"}


def style(name: str) -> dict:
    """Label / colour / native-resolution note of a board model (s2sbase.MODEL_STYLE)."""
    st = dict(_STYLE_FALLBACK.get(name, {}))
    st.update(getattr(S2, "MODEL_STYLE", {}).get(name, {}))
    st["deg"] = NATIVE.get(name, st.get("deg", ""))
    return st


# --------------------------------------------------------------------------- #
# Statistics (pure numpy, unit-tested on synthetic fields)
# --------------------------------------------------------------------------- #
def wstats(F: np.ndarray, O: np.ndarray, valid: np.ndarray, coslat: np.ndarray):
    """(bias, rmse, r) of a (lat, lon) field F against O, cos(lat)-weighted over the cells
    that are `valid` and finite in both; r is the centred pattern correlation."""
    ok = valid & np.isfinite(F) & np.isfinite(O)
    w = np.where(ok, coslat[:, None], 0.0)
    if w.sum() <= 0:
        return np.nan, np.nan, np.nan
    W = w / w.sum()
    f, o = np.where(ok, F, 0.0), np.where(ok, O, 0.0)
    d = f - o
    bias = float((W * d).sum())
    rmse = float(np.sqrt((W * d * d).sum()))
    fa, oa = f - (W * f).sum(), o - (W * o).sum()
    den = np.sqrt((W * fa * fa).sum() * (W * oa * oa).sum())
    r = float((W * fa * oa).sum() / den) if den > 0 else np.nan
    return bias, rmse, r


def wmean(X: np.ndarray, valid: np.ndarray, coslat: np.ndarray) -> float:
    """cos(lat)-weighted mean of a (lat, lon) field over valid, finite cells."""
    ok = valid & np.isfinite(X)
    w = np.where(ok, coslat[:, None], 0.0)
    return float((w * np.where(ok, X, 0.0)).sum() / w.sum()) if w.sum() > 0 else np.nan


def have_cases(F: np.ndarray) -> np.ndarray:
    """(case,) True where the model has a forecast (a missing case is all-NaN)."""
    return np.isfinite(F).any(axis=(1, 2))


def composite(F: np.ndarray, O: np.ndarray, sel: np.ndarray) -> dict:
    """Equal-weight case composites over `sel` & the cases F covers.

    Returns dict(n, err = mean(F - O), fc = mean F, obs = mean O, mae = mean |F - O|),
    each a (lat, lon) array; NaN cells (outside a mask) stay NaN.
    """
    use = np.asarray(sel, bool) & have_cases(F)
    n = int(use.sum())
    if n == 0:
        nan = np.full(F.shape[1:], np.nan)
        return dict(n=0, err=nan, fc=nan, obs=nan, mae=nan)
    f, o = F[use].astype("float64"), O[use].astype("float64")
    return dict(n=n, err=(f - o).mean(0), fc=f.mean(0), obs=o.mean(0),
                mae=np.abs(f - o).mean(0))


def pooled_stats(F: np.ndarray, O: np.ndarray, valid: np.ndarray, coslat: np.ndarray,
                 sel: np.ndarray) -> tuple[float, float, float]:
    """(MAE, RMSE, mean r) over every selected case the model covers: MAE and RMSE pool
    all (case, cell) samples with cos(lat) weights; r is the mean per-case pattern r."""
    use = np.asarray(sel, bool) & have_cases(F)
    if not use.any():
        return np.nan, np.nan, np.nan
    ae, se, rs = [], [], []
    for f, o in zip(F[use], O[use]):
        ae.append(wmean(np.abs(f - o), valid, coslat))
        b, r_, r = wstats(f, o, valid, coslat)
        se.append(r_ ** 2)
        rs.append(r)
    return float(np.mean(ae)), float(np.sqrt(np.mean(se))), float(np.nanmean(rs))


# --------------------------------------------------------------------------- #
# Fields: truth on both windows + one ensemble-mean field per (model, case)
# --------------------------------------------------------------------------- #
def slate() -> pd.DataFrame:
    e = aprep.episodes()[["episode_id", "family", "peak"]].copy()
    e["peak"] = pd.to_datetime(e["peak"])
    return e.reset_index(drop=True)


@dataclass
class Model:
    name: str
    label: str
    color: str
    deg: str
    window: str                      # truth window it is verified on: '13f' | '12f'
    F: np.ndarray                    # (case, lat, lon) ensemble mean, NaN = no forecast
    n_members: np.ndarray            # (case,)
    note: str = ""


@dataclass
class Board:
    truth: str
    label: str
    cases: pd.DataFrame
    lat: np.ndarray
    lon: np.ndarray
    valid: np.ndarray                # (lat, lon) scored cells: land AND the truth's mask
    obs: dict = field(default_factory=dict)       # window -> (case, lat, lon)
    models: list = field(default_factory=list)

    @property
    def coslat(self) -> np.ndarray:
        return np.cos(np.deg2rad(self.lat))

    def O(self, m: Model) -> np.ndarray:
        return self.obs[m.window]


def _grid_like(da: xr.DataArray, lat: np.ndarray, lon: np.ndarray, where: str) -> np.ndarray:
    if not (np.allclose(da["lat"].values, lat, atol=1e-6)
            and np.allclose(da["lon"].values, lon, atol=1e-6)):
        raise SystemExit(f"[errmaps] {where}: grid differs from the maps grid")
    return da.transpose(..., "lat", "lon").values


def aires_ensmean(cases: pd.DataFrame) -> xr.Dataset:
    """AI+RES importance-weighted ensemble mean per case (13 frames), from the published
    closest_members.nc, its CONUS mean checked against the walker table."""
    if not RES_MEAN_NC.exists():
        raise SystemExit(f"[errmaps] {RES_MEAN_NC} missing: run "
                         "`python -m acal.sidebyside --stage members` first")
    with xr.open_dataset(RES_MEAN_NC) as d:
        res = d["field"].sel(panel="res_mean").load()
    if list(map(str, res["case"].values)) != list(cases.episode_id):
        raise SystemExit("[errmaps] closest_members.nc case order != slate")
    tab = pd.read_csv(S2.AIRES_WINDOWS_CSV)
    want = tab.assign(x=tab.weight_sn * tab.al13).groupby("eid").x.sum()
    got = AI.area_mean(res).values
    bad = np.abs(got - want.reindex(cases.episode_id).values)
    if not np.all(bad < CONSISTENCY_TOL):
        raise SystemExit(f"[errmaps] AI+RES ensemble-mean CONUS mean != sum w A_L "
                         f"(max |d| {np.nanmax(bad):.4f} K)")
    n = tab.groupby("eid").size().reindex(cases.episode_id).values
    return xr.Dataset(dict(ensmean=res.astype("float32"),
                           n_members=("case", n.astype("int32"))),
                      attrs=dict(source="aires", window="13f"))


def _source_case(args) -> tuple[str, np.ndarray | None, int, float]:
    """(eid, (lat, lon) member mean, n_members, CONUS mean) of one source case."""
    name, eid, peak = args
    src = S2.get(name)
    with xr.open_dataset(src.cube_path(eid)) as cube:
        cube = cube.load()
    f = AI._squeeze(S2.reduce(cube, pd.Timestamp(peak), src.window))
    S2.check_grid(f)
    jp = src.json_path(eid)
    if jp.exists():
        al = np.asarray(json.loads(jp.read_text())["al"], dtype="float64")
        got = np.asarray(AI.area_mean(f).values, dtype="float64")
        if got.shape != al.shape or np.nanmax(np.abs(got - al)) > CONSISTENCY_TOL:
            raise SystemExit(f"[errmaps] {name} {eid}: member CONUS means != json al")
    mean = f.mean("member")
    return eid, mean.transpose("lat", "lon").values.astype("float32"), \
        int(f.sizes["member"]), float(AI.area_mean(mean))


def source_ensmean(name: str, cases: pd.DataFrame, lat, lon, workers: int = 8,
                   force: bool = False) -> xr.Dataset | None:
    """Equal-weight member mean per case on the source's window, cached under
    runs/acal/analysis/s2s/errmaps/ensmean_<source>.nc; None when the source has no cube.
    The cache is rebuilt when a cube is newer than it or the covered cases changed."""
    try:
        src = S2.get(name)
    except KeyError:
        return None
    paths = {e: src.cube_path(e) for e in cases.episode_id}
    have = [e for e, p in paths.items() if p.exists()]
    if not have:
        return None
    cache = CACHE_DIR / f"ensmean_{name}.nc"
    newest = max(paths[e].stat().st_mtime for e in have)
    if cache.exists() and not force and cache.stat().st_mtime >= newest:
        with xr.open_dataset(cache) as d:
            d = d.load()
        if sorted(map(str, d["case"].values[d["n_members"].values > 0])) == sorted(have):
            return d
    print(f"[errmaps] {name}: ensemble means of {len(have)} cases ({workers} workers)",
          flush=True)
    jobs = [(name, e, cases.set_index("episode_id").peak[e]) for e in have]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_source_case, jobs))
    else:
        res = [_source_case(j) for j in jobs]
    out = np.full((len(cases), lat.size, lon.size), np.nan, "float32")
    nm = np.zeros(len(cases), "int32")
    pos = {e: i for i, e in enumerate(cases.episode_id)}
    for eid, f, n, al in res:
        out[pos[eid]], nm[pos[eid]] = f, n
    ds = xr.Dataset(dict(ensmean=(("case", "lat", "lon"), out), n_members=("case", nm)),
                    coords=dict(case=list(cases.episode_id), lat=lat, lon=lon),
                    attrs=dict(source=name, window=src.window,
                               note="equal-weight member mean of the raw 7-day T2m anomaly "
                                    "(s2sbase.reduce on the source window); NaN = no cube"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp)
    os.replace(tmp, cache)
    print(f"[errmaps] wrote {cache}", flush=True)
    return ds


def build_board(truth: str, workers: int = 8, force: bool = False) -> Board:
    """Truth on both windows, the scored-cell mask and every board model with data."""
    tr = TR.get_truth(truth)
    cases = slate()
    with xr.open_dataset(M.FIELDS) as base:
        lat, lon = base["lat"].values, base["lon"].values
        land = M.land_mask(base).transpose("lat", "lon").values.astype(bool)
    mask = tr.mask()
    valid = land if mask is None else land & _grid_like(mask, lat, lon, "mask").astype(bool)
    obs = {}
    for win in ("13f", "12f"):
        arr = np.stack([_grid_like(TR.frames_on(tr, e, win).mean("time"), lat, lon,
                                   f"{truth} {e}") for e in cases.episode_id])
        obs[win] = arr.astype("float64")
    for i, e in enumerate(cases.episode_id):            # truth field == the scored A_L
        got = float(TR.area_mean(xr.DataArray(obs["13f"][i], dims=("lat", "lon"),
                                              coords=dict(lat=lat, lon=lon)), mask))
        if abs(got - tr.obs_al(e, "13f")) > CONSISTENCY_TOL:
            raise SystemExit(f"[errmaps] {truth} {e}: field mean {got:+.4f} != obs_al")
    b = Board(truth=truth, label=TR.label_of(tr), cases=cases, lat=lat, lon=lon,
              valid=valid, obs=obs)
    for name in BOARD:
        ds = aires_ensmean(cases) if name == "aires" else \
            source_ensmean(name, cases, lat, lon, workers, force)
        if ds is None:
            print(f"[errmaps] {name}: no data yet - absent from the board")
            continue
        st = style(name)
        win = "13f" if ds.attrs["window"] in ("13f", "all") else "12f"
        F = _grid_like(ds["ensmean"], lat, lon, name).astype("float64")
        b.models.append(Model(name=name, label=st["label"], color=st["color"],
                              deg=st["deg"], window=win, F=F,
                              n_members=ds["n_members"].values))
    return b


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def case_table(b: Board) -> pd.DataFrame:
    rows = []
    for i, c in b.cases.iterrows():
        for m in b.models:
            f, o = m.F[i], b.O(m)[i]
            if not np.isfinite(f).any():
                continue
            bias, rmse, r = wstats(f, o, b.valid, b.coslat)
            rows.append(dict(episode_id=c.episode_id, family=c.family,
                             peak=c.peak.date(), model=m.name, label=m.label,
                             window=m.window, n_members=int(m.n_members[i]),
                             truth_land=wmean(o, b.valid, b.coslat),
                             fc_land=wmean(f, b.valid, b.coslat),
                             err_land=bias, rmse=rmse, pattern_r=r))
    return pd.DataFrame(rows)


def composite_table(b: Board) -> pd.DataFrame:
    rows = []
    fam = b.cases.family.values
    for row, sel in (("heat", fam == "heat"), ("cold", fam == "cold"),
                     ("all", np.ones(len(fam), bool))):
        for m in b.models:
            O = b.O(m)
            cp = composite(m.F, O, sel)
            bias, rmse, r = wstats(cp["fc"], cp["obs"], b.valid, b.coslat)
            mae, prmse, mr = pooled_stats(m.F, O, b.valid, b.coslat, sel)
            rows.append(dict(cases=row, model=m.name, label=m.label, window=m.window,
                             n=cp["n"], comp_bias=bias, comp_rmse=rmse, comp_r=r,
                             mae=mae, rmse_pooled=prmse, mean_case_r=mr))
    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False, float_format="%.6g")
    os.replace(tmp, p)
    print(f"[errmaps] wrote {p}")
    return p


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    AN._style(plt)
    proj = ccrs.LambertConformal(central_longitude=-96, standard_parallels=(33, 45))
    return plt, ccrs, cfeature, proj


def _da(b: Board, X: np.ndarray) -> xr.DataArray:
    """X on the scored cells only (sea and off-mask cells NaN = white)."""
    return xr.DataArray(np.where(b.valid, X, np.nan), dims=("lat", "lon"),
                        coords=dict(lat=b.lat, lon=b.lon))


def _sym(arrs, valid, q=98, step=0.5, floor=1.0) -> float:
    v = np.concatenate([np.abs(a[valid & np.isfinite(a)]) for a in arrs])
    return max(float(np.ceil(np.percentile(v, q) / step) * step), floor) if v.size else floor


def _header(ax, color: str | None):
    """Model identity as a colour strip above the map; the header text stays in ink."""
    from matplotlib.patches import Rectangle
    if color:
        ax.add_patch(Rectangle((0.0, 1.012), 1.0, 0.03, transform=ax.transAxes,
                               facecolor=color, edgecolor="none", clip_on=False, zorder=5))


def _native(ax, text: str):
    if text:
        ax.text(0.012, 0.025, f"native {text}", transform=ax.transAxes, fontsize=6.5,
                color="0.25", ha="left", va="bottom", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.2))


def _savefig(fig, p: Path) -> Path:
    import matplotlib.pyplot as plt
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.png")
    fig.savefig(tmp, dpi=140, bbox_inches="tight")
    os.replace(tmp, p)
    plt.close(fig)
    return p


def _footnote(b: Board) -> str:
    inst = ", ".join(f"{m.label} ({_nmem(m)})" for m in b.models if m.window == "13f")
    dly = ", ".join(f"{m.label} ({_nmem(m)})" for m in b.models if m.window == "12f")
    s = (f"Raw ensemble-mean 7-day T2m anomaly (vs 1990-2019 ERA5 clim) minus {b.label}, each "
         f"on its own window. 13 00/12Z frames peak-6 d..peak: {inst}.")
    if dly:
        s += f"\nUTC days peak-6..peak-1, truth on the matching 12 frames: {dly}."
    s += ("\nAI+RES mean importance-weighted; others equal-weight. Sea"
          + (" and cells outside the HRRR domain" if b.truth.startswith("hrrr") else "")
          + " masked. Stats cos(lat)-weighted over the coloured cells.")
    return s


def _lead_text(b: Board) -> str:
    """'AI+RES lead 21 d (baselines 21-27 d)': lagged and weekly-init baselines do not all
    start 21 d before the peak (per-case member leads from the source json records)."""
    lo, hi = [], []
    for m in b.models:
        if m.name == "aires":
            continue
        try:
            r = S2.ens_ranges(m.name)["lead"]
        except Exception:                      # pragma: no cover - registry without json
            r = None
        if r:
            lo.append(r[0])
            hi.append(r[1])
    s = "AI+RES lead 21 d"
    if lo and (min(lo), max(hi)) != (21.0, 21.0):
        s += f" (baselines {min(lo):g}-{max(hi):g} d)"
    return s


def _signed(x: float, nd: int = 2) -> str:
    """`x` with an explicit sign, never '-0.00' (round first, then drop the signed zero)."""
    if not np.isfinite(x):
        return "n/a"
    return f"{round(float(x), nd) + 0.0:+.{nd}f}"


def _nmem(m: Model) -> str:
    n = np.unique(m.n_members[m.n_members > 0])
    return f"{n[0]} members" if n.size == 1 else f"{n.min()}-{n.max()} members"


def composite_figure(b: Board) -> Path:
    plt, ccrs, cfeature, proj = _plt()
    fam = b.cases.family.values
    O13 = b.obs["13f"]
    rows = [("heat", fam == "heat"), ("cold", fam == "cold"), ("all", np.ones(len(fam), bool))]
    ncol = 1 + len(b.models)
    fig = plt.figure(figsize=(2.75 * ncol + 1.0, 7.3))
    gs = fig.add_gridspec(3, ncol + 1, width_ratios=[1] * ncol + [0.045], wspace=0.06,
                          hspace=0.30, left=0.035, right=0.95, top=0.875, bottom=0.14)
    for r, (tag, sel) in enumerate(rows):
        cps = {m.name: composite(m.F, b.O(m), sel) for m in b.models}
        tcp = composite(O13, O13, sel)
        nsel = int(sel.sum())
        if tag == "all":
            panels = [np.abs(O13[sel]).mean(0)] + [cps[m.name]["mae"] for m in b.models]
            vmax = _sym(panels, b.valid, q=99, step=1.0)
            cmap, vmin = plt.get_cmap("Purples").copy(), 0.0
        else:
            panels = [tcp["obs"]] + [cps[m.name]["err"] for m in b.models]
            vmax = _sym(panels, b.valid, q=99, step=1.0)
            cmap, vmin = plt.get_cmap("RdBu_r").copy(), -vmax
        cmap.set_bad("white")
        for k in range(ncol):
            ax = fig.add_subplot(gs[r, k], projection=proj)
            mp = M._map_panel(ax, _da(b, panels[k]), cmap, vmin, vmax, ccrs, cfeature)
            if k == 0:
                if tag == "all":
                    sub = f"mean |anomaly| {wmean(panels[0], b.valid, b.coslat):.2f} K\n" \
                          f"(= MAE of a 0 K forecast)"
                else:
                    sub = f"land mean {wmean(panels[0], b.valid, b.coslat):+.2f} K " \
                          f"({nsel} cases)"
                head = f"{b.label} truth"
                _header(ax, "#000000" if b.truth == "era5" else "#555555")
            else:
                m = b.models[k - 1]
                cp = cps[m.name]
                if tag == "all":
                    mae, prmse, mr = pooled_stats(m.F, b.O(m), b.valid, b.coslat, sel)
                    sub = f"MAE {mae:.2f}  RMSE {prmse:.2f}  mean r {mr:.2f}"
                else:
                    bias, rmse, rr = wstats(cp["fc"], cp["obs"], b.valid, b.coslat)
                    sub = f"bias {bias:+.2f}  RMSE {rmse:.2f}  r {rr:.2f}"
                if cp["n"] < nsel:
                    sub += f"\n({cp['n']} of {nsel} cases)"
                head = f"{m.label} - {b.label.split(' ')[0]}"
                _header(ax, m.color)
                _native(ax, m.deg + ("" if m.window == "13f" else ", daily"))
            title = (head + "\n" + sub) if r == 0 else sub
            ax.set_title(title, fontsize=8.3, pad=7)
        cb = fig.colorbar(mp, cax=fig.add_subplot(gs[r, ncol]),
                          extend="max" if tag == "all" else "both")
        cb.locator = MaxNLocator(nbins=6, integer=True, symmetric=tag != "all")
        cb.update_ticks()
        cb.set_label("mean |error| (K)" if tag == "all" else "anomaly / error (K)", fontsize=8)
        cb.ax.tick_params(labelsize=7.5)
        lab = {"heat": f"Heat composite\n({nsel} cases)", "cold": f"Cold composite\n({nsel} cases)",
               "all": f"Mean absolute error\n(all {nsel} cases)"}[tag]
        fig.text(0.012, (gs[r, 0].get_position(fig).y0 + gs[r, 0].get_position(fig).y1) / 2,
                 lab, rotation=90, ha="center", va="center", fontsize=9.5, weight="bold")
    fig.suptitle(f"Where the models differ: ensemble-mean 7-day T2m error vs {b.label}\n"
                 f"week ending at the peak, {_lead_text(b)}; {S2.SELECTION_NOTE}",
                 fontsize=11.5, y=0.985)
    fig.text(0.035, 0.1, _footnote(b) + "\nRows 1-2: bias = mean error, RMSE of the "
             "composite error map, r = pattern correlation of the model composite with the "
             "truth composite. Row 3: MAE and RMSE pooled over cases, mean r = mean per-case "
             "pattern correlation.\n" + S2.TILT_NOTE, ha="left", va="top", fontsize=7.5,
             color="0.25")
    p = _savefig(fig, OVERALL_DIR / f"errmap_{b.truth}.png")
    print(f"[errmaps] wrote {p}")
    return p


def case_figure(b: Board, i: int) -> Path:
    plt, ccrs, cfeature, proj = _plt()
    c = b.cases.iloc[i]
    O13 = b.obs["13f"][i]
    ncol = 1 + len(b.models)
    panels = [O13] + [m.F[i] - b.O(m)[i] for m in b.models]
    vmax = max(_sym([p for p in panels if np.isfinite(p).any()], b.valid, q=99,
                    step=1.0, floor=3.0), 3.0)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    fig = plt.figure(figsize=(2.75 * ncol + 0.9, 2.9))
    gs = fig.add_gridspec(1, ncol + 1, width_ratios=[1] * ncol + [0.045], wspace=0.06,
                          left=0.01, right=0.95, top=0.77, bottom=0.25)
    for k in range(ncol):
        ax = fig.add_subplot(gs[0, k], projection=proj)
        if k == 0:
            head = f"{b.label} truth"
            sub = f"land mean {wmean(O13, b.valid, b.coslat):+.2f} K (13 frames)"
            _header(ax, "#000000" if b.truth == "era5" else "#555555")
            mp = M._map_panel(ax, _da(b, O13), cmap, -vmax, vmax, ccrs, cfeature)
        else:
            m = b.models[k - 1]
            head = f"{m.label} - {b.label.split(' ')[0]}"
            _header(ax, m.color)
            if np.isfinite(m.F[i]).any():
                bias, rmse, r = wstats(m.F[i], b.O(m)[i], b.valid, b.coslat)
                sub = f"bias {bias:+.2f}  RMSE {rmse:.2f}  r {r:.2f}"
                mp = M._map_panel(ax, _da(b, panels[k]), cmap, -vmax, vmax, ccrs, cfeature)
                _native(ax, m.deg + ("" if m.window == "13f" else ", daily"))
            else:
                sub = "no forecast for this case"
                M._map_panel(ax, _da(b, np.full_like(O13, np.nan)), cmap, -vmax, vmax,
                             ccrs, cfeature)
        ax.set_title(head + "\n" + sub, fontsize=8.3, pad=7)
    cb = fig.colorbar(mp, cax=fig.add_subplot(gs[0, ncol]), extend="both")
    cb.locator = MaxNLocator(nbins=6, integer=True, symmetric=True)
    cb.update_ticks()
    cb.set_label("anomaly / error (K)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    fig.suptitle(f"{c.episode_id}  ({c.family}, week ending {c.peak:%Y-%m-%d}, 21 d lead): "
                 f"ensemble mean minus {b.label}", fontsize=10.5, y=0.99)
    note = ("Raw ensemble-mean 7-day T2m anomaly minus truth on each model's window "
            "(13 00/12Z frames; GEPS/EC46: UTC days peak-6..peak-1 vs 12 frames). "
            "AI+RES importance-weighted. Stats cos(lat)-weighted over the coloured cells; "
            "red = model too warm.")
    fig.text(0.01, 0.17, note, ha="left", va="top", fontsize=7.3, color="0.25")
    return _savefig(fig, CASE_FIG_DIR / b.truth / f"{c.episode_id}.png")


_B: Board | None = None


def _case_job(i: int) -> str:
    return str(case_figure(_B, i))


def case_figures(b: Board, workers: int = 8) -> list[Path]:
    global _B
    _B = b
    idx = list(range(len(b.cases)))
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            out = pool.map(_case_job, idx)
    else:
        out = [_case_job(i) for i in idx]
    print(f"[errmaps] wrote {len(out)} case pages under {CASE_FIG_DIR / b.truth}")
    return [Path(p) for p in out]


# --------------------------------------------------------------------------- #
# Skill maps
# --------------------------------------------------------------------------- #
def skill_files(truth: str, names) -> list[tuple[str, Path, str]]:
    """(model, maps_fields file, prob var) per board model that has a maps file."""
    tr = TR.get_truth(truth)
    out = []
    for name in names:
        if name == "aires":
            out.append((name, M.truth_base(tr, False), "prob"))
            continue
        p = S2.ANALYSIS / truth / f"maps_fields_{name}.nc"
        if p.exists():
            out.append((name, p, "prob_raw"))
        else:
            print(f"[errmaps] skill: no {p} - {name} absent "
                  f"(`python -m acal.s2sbase --source {name} --stage maps --truth {truth}`)")
    return out


def skill_scores(truth: str, names) -> dict:
    out = {}
    for name, p, var in skill_files(truth, names):
        with xr.open_dataset(p) as d:
            ds = d.load()
        sc = M.scores(ds, var)
        n = {}
        for a in SKILL_THRESHOLDS:
            fam = "heat" if a > 0 else "cold"
            pr = ds[var].sel(threshold=a).where(ds.family == fam, drop=True)
            n[a] = int(pr.notnull().any(("lat", "lon")).sum())
        out[name] = (sc, n, p)
    return out


def skill_figures(truth: str, b: Board | None = None) -> list[Path]:
    plt, ccrs, cfeature, proj = _plt()
    names = [m.name for m in b.models] if b is not None else \
        [n for n in BOARD if n == "aires" or S2.has_data(n)]
    sc = skill_scores(truth, names)
    names = [n for n in names if n in sc]
    label = TR.label_of(TR.get_truth(truth))
    paths = []
    for metric, title, cb_lab in (
            ("csi", "Critical success index", "CSI"),
            ("bss", "Brier skill score vs climatology (grey = BSS < 0)", "Brier skill score")):
        ncol = len(names)
        fig = plt.figure(figsize=(2.75 * max(ncol, 4) + 1.0, 4.9))
        gs = fig.add_gridspec(2, ncol + 1, width_ratios=[1] * ncol + [0.045], wspace=0.06,
                              hspace=0.30, left=0.045, right=0.94, top=0.83, bottom=0.15)
        for r, a in enumerate(SKILL_THRESHOLDS):
            fam = "heat" if a > 0 else "cold"
            cmap = plt.get_cmap("Reds" if a > 0 else "Blues").copy()
            cmap.set_bad("white")
            if metric == "bss":
                cmap.set_under("0.82")
            for k, name in enumerate(names):
                s, n, _ = sc[name]
                st = style(name)
                da = s[metric].sel(threshold=a)
                ax = fig.add_subplot(gs[r, k], projection=proj)
                mp = M._map_panel(ax, da, cmap, 0.0, 1.0, ccrs, cfeature)
                v = da.values
                ok = np.isfinite(v)
                cl = np.cos(np.deg2rad(da["lat"].values))[:, None] * np.ones_like(v)
                lm = float((v[ok] * cl[ok]).sum() / cl[ok].sum()) if ok.any() else np.nan
                # BSS: land MEDIAN (as maps.land_means bss_med and the per-source maps); the
                # cos-lat mean is dominated by the few cells with p_clim near 0
                med = float(np.median(v[ok])) if ok.any() else np.nan
                sub = f"land mean {lm:.2f}" if metric == "csi" else \
                    f"land median {_signed(med)}  (> 0 on {float((cl[ok] * (v[ok] > 0)).sum() / cl[ok].sum()):.0%})"
                n_fam = int((b.cases.family == fam).sum()) if b is not None else n[a]
                if n[a] < n_fam:
                    sub += f"\n({n[a]} of {n_fam} cases)"
                head = st["label"]
                _header(ax, st["color"])
                _native(ax, st["deg"] + ("" if S2.get(name).time_kind == "instant"
                                         else ", daily") if name != "aires" else st["deg"])
                ax.set_title((head + "\n" + sub) if r == 0 else sub, fontsize=8.3, pad=7)
            cb = fig.colorbar(mp, cax=fig.add_subplot(gs[r, ncol]),
                              extend="min" if metric == "bss" else "neither")
            cb.set_label(cb_lab, fontsize=8)
            cb.ax.tick_params(labelsize=7.5)
            pos = gs[r, 0].get_position(fig)
            fig.text(0.014, (pos.y0 + pos.y1) / 2, f"{a:+.0f} K\n({fam}, {n_fam} cases)",
                     rotation=90,
                     ha="center", va="center", fontsize=9.5, weight="bold")
        fig.suptitle(f"{title} per cell, 7-day-mean T2m beyond +/-2 K, truth {label}\n"
                     f"{S2.SELECTION_NOTE}", fontsize=11.5, y=0.975)
        fig.text(0.045, 0.115,
                 "Family-matched (heat cases score +2 K, cold cases -2 K). P = AI+RES "
                 "importance-weighted walker fraction, others raw member fraction.\n"
                 + ("yes = P >= 0.5, CSI = hits / (hits + misses + false alarms)."
                    if metric == "csi" else
                    "BSS = 1 - sum (P - o)^2 / sum (p_clim - o)^2 over cases, p_clim = per-cell "
                    "climatological frequency.")
                 + "\nEach model on its own window (13 00/12Z frames; daily-mean sources UTC "
                 "days peak-6..peak-1 vs 12 truth frames).\nSea"
                 + (" and cells outside the HRRR domain" if truth.startswith("hrrr") else "")
                 + (" masked; land mean cos(lat)-weighted over cells where the score is defined."
                    if metric == "csi" else
                    " masked; land median over cells where BSS is defined (a land mean would be "
                    "dominated by cells with p_clim near 0).") + "\n" + S2.TILT_NOTE,
                 ha="left", va="top", fontsize=7.5, color="0.25")
        p = _savefig(fig, OVERALL_DIR / f"skillmaps_{metric}_{truth}.png")
        print(f"[errmaps] wrote {p}")
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--truth", default="era5", choices=("era5", "hrrr", "hrrr_raw"))
    p.add_argument("--stage", default="all",
                   choices=("fields", "composite", "skill", "cases", "all"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true", help="rebuild the ensemble-mean caches")
    a = p.parse_args(argv)
    b = build_board(a.truth, a.workers, a.force)
    print(f"[errmaps] truth {a.truth}: models " + ", ".join(m.name for m in b.models))
    if a.stage in ("composite", "all"):
        _write_csv(composite_table(b), S2.ANALYSIS / f"errmaps_composite_{a.truth}.csv")
        composite_figure(b)
    if a.stage in ("skill", "all"):
        skill_figures(a.truth, b)
    if a.stage in ("cases", "all"):
        _write_csv(case_table(b), S2.ANALYSIS / f"errmaps_{a.truth}.csv")
        case_figures(b, a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
