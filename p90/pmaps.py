"""CONUS heat maps of the p90 T2m-anomaly events — three ensemble collapses vs ERA5.

Companion to ``pcompare`` (which draws PDFs and skill curves): this module draws the
SPATIAL fields behind those scores, in the ``figures/original/allbestmaps.png`` /
``xres.xmapgrid`` style. Only the 45 ``kind == "event"`` cases are plotted — the 45
matched controls exist to calibrate probabilities, not to look at.

Each case's 50-member week-mean T2m anomaly field (the xres recipe, via
``pmetrics.week_anom_members``) is collapsed three ways:

    pooled mean      the ensemble-MEAN field — all members pooled, averaged
    pQ extreme       the member at the ``--extreme-q`` PERCENTILE (default 80th) of
                     cos(lat)-weighted CONUS-mean anomaly. Not the single warmest member
                     (``xres.xcombined``'s ``extreme`` mode); a quantile member is the
                     ensemble's "quite extreme but not the outlier" realization, which is
                     the relevant one for a p90 event and far less sample-noisy than
                     argmax. At the default q=0.80 with 50 members that is rank 40/50
                     (11th warmest).
    best member      the member with the lowest cos(lat)-weighted RMSE vs ERA5 — the
                     same criterion as ``xres.xcombined._best_member``.

The extreme tail is the WARM side for every case by construction (a p90 event is a
90th-percentile *warm* T2m anomaly), so unlike xres there is no cold-event sign flip.
Note the two 90s are unrelated: "p90" names the EVENT population (90th-percentile CONUS
heat days), ``--extreme-q`` selects a MEMBER within a case's ensemble.

Two figures come out, both tagged with the member quantile so runs at different q coexist:

    p90_maps_composite_q80.png     the 45-event average of each field, plus error maps
                                   (forecast composite - ERA5 composite)
    p90_maps_events_q80_pN.png     per-event rows, 9 events per page (5 pages), columns
                                   ERA5 | pooled mean | p80 extreme | best member

Reduced (lat, lon) fields are cached to
``runs/p90_t2m/<model>/maps/<case>_mapfields_q80.nc`` so re-plotting never re-reads the
418 MB cubes. ``--force`` rebuilds the cache.

Usage (cwd = repo root, env ``moe``; login-node CPU only — no GPU, no network):

    python -m p90.pmaps                          # composite + all per-event pages
    python -m p90.pmaps --only composite         # just the composite
    python -m p90.pmaps --cases event_01_20220113,event_44_20251224
    python -m p90.pmaps --extreme-q 0.90         # the old 90th-percentile member
    python -m p90.pmaps --force                  # re-derive fields from the cubes
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.gridspec import GridSpec     # noqa: E402
import numpy as np                           # noqa: E402

from . import pconfig as P                   # noqa: E402
# Draw helpers are shared with the xres map grid so the two projects' CONUS panels are
# pixel-for-pixel the same style (extent, coast/border/state features, badge boxes).
from xres.xmapgrid import _badge, _draw, _geo_ax, _wmean, _wrmse   # noqa: E402

TAG = "[p90 maps]"
DPI = 130
FIELD_CMAP = "RdBu_r"          # diverging, centered on 0 — the xres t2m_anom convention
ERR_CMAP = "RdBu_r"
EXTREME_Q = 0.80               # quantile of the member ranking used for the extreme column
PER_PAGE = 9                   # events per per-event page


def _qtag(q: float) -> str:
    """Filename-safe tag for a member quantile: 0.8 -> 'q80', 0.875 -> 'q87p5'."""
    return "q" + f"{q * 100:g}".replace(".", "p")


def _col_labels(q: float) -> tuple[str, str, str, str]:
    return ("ERA5 (truth)", "pooled mean", f"p{q * 100:g} extreme member", "best member")


# --------------------------------------------------------------------------- #
# Member selection
# --------------------------------------------------------------------------- #
def _conus_mean_by_member(mem):
    """cos(lat)-weighted CONUS mean anomaly per member -> (member,) ndarray."""
    w = np.cos(np.deg2rad(mem["lat"]))
    return np.asarray(mem.weighted(w).mean(("lat", "lon")).values, dtype=float)


def extreme_member(mem, q: float = EXTREME_Q):
    """Member at the ``q`` quantile of the warm-tail ranking.

    Members are sorted ascending by CONUS-mean anomaly and the one at position
    ``round(q * (n - 1))`` is returned, i.e. the nearest-rank quantile member. With the
    50-member p90 ensembles and q=0.80 that is rank 39 (0-based) — the 40th coldest /
    11th warmest. Returns ``(field, member_index, rank_from_cold, n)``.
    """
    ma = _conus_mean_by_member(mem)
    order = np.argsort(ma)                       # coldest -> warmest
    n = ma.size
    pos = int(round(q * (n - 1)))
    idx = int(order[pos])
    return mem.isel(member=idx), idx, pos + 1, n


def best_member(mem, truth):
    """Member with the lowest cos(lat)-weighted RMSE vs ERA5 (xres criterion).

    Also returns where that member falls in the SAME cold->warm ranking used for the
    extreme column, so the two badges are on one scale: rk1 = coldest member, rk n =
    warmest. (Its RMSE rank would be 1 by definition and says nothing.) The heat rank is
    the interesting diagnostic — it says whether the truth-closest member is a warm one or
    a middling one. Returns ``(field, member_index, rmse, warm_rank)``.
    """
    w = np.cos(np.deg2rad(mem["lat"]))
    rmse = np.sqrt(((mem - truth) ** 2).weighted(w).mean(("lat", "lon")))
    idx = int(rmse.argmin("member"))
    ranks = np.argsort(np.argsort(_conus_mean_by_member(mem)))   # 0-based rank per member
    return mem.isel(member=idx), idx, float(rmse.isel(member=idx)), int(ranks[idx]) + 1


# --------------------------------------------------------------------------- #
# Per-case reduced fields (cached)
# --------------------------------------------------------------------------- #
def maps_dir(model: str) -> Path:
    return P.model_dir(model) / "maps"


def fields_path(model: str, case: str, q: float = EXTREME_Q) -> Path:
    """Cache path for one case. The member quantile is in the NAME so a run at a new q
    can never silently reuse fields whose ``extreme`` column came from a different one."""
    return maps_dir(model) / f"{case}_mapfields_{_qtag(q)}.nc"


def case_fields(model: str, case: str, q: float = EXTREME_Q, force: bool = False):
    """The four (lat, lon) fields for one case, from cache or derived from the cube.

    Returns an ``xarray.Dataset`` with ``truth`` / ``ens_mean`` / ``extreme`` / ``best``
    and the member-choice diagnostics in ``.attrs``, or None when the cube or the ERA5
    truth for this case is missing (a partially-complete run still plots).
    """
    import xarray as xr

    out = fields_path(model, case, q)
    if out.exists() and not force:
        cached = xr.open_dataset(out).load()
        # Caches written before the best-member heat rank existed have every field but
        # not the attr the badge needs — re-derive those rather than KeyError at draw time.
        if "best_rank" in cached.attrs:
            return cached
        cached.close()
        print(f"{TAG} {case}: cache predates best_rank; re-deriving")

    cube_p, truth_p = P.cube_path(model, case), P.truth_path(case)
    if not cube_p.exists():
        print(f"{TAG} {case}: no cube; skipping")
        return None
    if not truth_p.exists():
        print(f"{TAG} {case}: no ERA5 truth; skipping")
        return None

    from . import pmetrics as PM
    row = P.case_row(case)
    with xr.open_dataset(cube_p) as cube:
        mem = PM.week_anom_members(cube, row["peak"]).load()
    with xr.open_dataset(truth_p) as ds:
        # Truth is written float32 on the same 0.25deg CONUS grid; take the cube's
        # coords so tiny float32/float64 coord differences cannot misalign the subtraction.
        truth = xr.DataArray(np.asarray(ds["t2m_anom"].values, dtype=np.float32),
                             dims=("lat", "lon"),
                             coords={"lat": mem["lat"], "lon": mem["lon"]})

    ens_mean = mem.mean("member")
    ext, ext_idx, ext_rank, n = extreme_member(mem, q)
    bst, bst_idx, bst_rmse, bst_rank = best_member(mem, truth)

    ds_out = xr.Dataset(
        {"truth": truth, "ens_mean": ens_mean,
         "extreme": ext.drop_vars("member"), "best": bst.drop_vars("member"),
         # The whole cold->warm ranking, so any future member question is answerable from
         # the 400 kB cache instead of a 418 MB cube re-read.
         "member_conus_mean": ("member", _conus_mean_by_member(mem))},
        attrs=dict(
            case=case, model=model, peak=str(row["peak"]), init=str(row["init"]),
            n_members=n, extreme_q=q,
            extreme_member=ext_idx, extreme_rank=ext_rank,
            best_member=bst_idx, best_rmse=bst_rmse, best_rank=bst_rank,
            mean_rmse=_wrmse(ens_mean, truth),
            extreme_rmse=_wrmse(ext, truth),
            truth_conus_mean=_wmean(truth),
            peak_anom_K=float(row["peak_anom_K"]),
        ),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out)
    print(f"{TAG} {case}: fields cached -> {out.name} "
          f"(extreme #{ext_idx} rank {ext_rank}/{n}, best #{bst_idx} "
          f"rank {bst_rank}/{n} rmse {bst_rmse:.2f} K)")
    return ds_out


def load_all(model: str, cases: list[str], q: float = EXTREME_Q, force: bool = False):
    """[(case, fields)] for every case that has both a cube and ERA5 truth."""
    got = []
    for c in cases:
        f = case_fields(model, c, q, force=force)
        if f is not None:
            got.append((c, f))
    return got


# --------------------------------------------------------------------------- #
# Shared drawing pieces
# --------------------------------------------------------------------------- #
def _sym_range(fields, pct: float = 99.0) -> float:
    """Symmetric limit for a diverging scale: the ``pct`` percentile of |values|."""
    vals = np.concatenate([np.abs(np.asarray(f.values).ravel()) for f in fields])
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, pct)) or 1.0


def _fc_columns(f):
    """The three forecast columns of a fields dataset, with their per-panel badges.

    ``rk`` on both member columns is the same cold->warm heat rank, so the best member's
    position in the ensemble's temperature spread reads straight off against the extreme
    member's fixed rank.
    """
    n = f.attrs["n_members"]
    return [
        (f["ens_mean"], f"RMSE {f.attrs['mean_rmse']:.2f}"),
        (f["extreme"], f"#{f.attrs['extreme_member']} rk{f.attrs['extreme_rank']}/{n}"),
        (f["best"], f"#{f.attrs['best_member']} rk{f.attrs['best_rank']}/{n} "
                    f"RMSE {f.attrs['best_rmse']:.2f}"),
    ]


# --------------------------------------------------------------------------- #
# Figure 1 — the 45-event composite
# --------------------------------------------------------------------------- #
def fig_composite(model: str, got, outdirs, q: float = EXTREME_Q) -> None:
    """Event-average of each field (row 1) over its error vs the ERA5 composite (row 2)."""
    if not got:
        print(f"{TAG} composite: nothing to plot")
        return
    import xarray as xr

    n_ev = len(got)
    col_labels = _col_labels(q)
    n_mem = got[0][1].attrs["n_members"]
    best_rank_mean = float(np.mean([f.attrs["best_rank"] for _, f in got]))
    comp = {k: xr.concat([f[k] for _, f in got], dim="case").mean("case")
            for k in ("truth", "ens_mean", "extreme", "best")}
    errs = [(comp[k] - comp["truth"], lbl)
            for k, lbl in (("ens_mean", col_labels[1]),
                           ("extreme", col_labels[2]),
                           ("best", col_labels[3]))]

    fig = plt.figure(figsize=(4.3 * 4 + 1.0, 2.35 * 2))
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.05],
                  hspace=0.22, wspace=0.06)

    # --- row 0: composite fields, one shared diverging scale ---
    lim = _sym_range(list(comp.values()))
    im = None
    for c, (key, lbl) in enumerate(zip(("truth", "ens_mean", "extreme", "best"),
                                       col_labels)):
        ax = _geo_ax(fig, gs[0, c])
        im = _draw(ax, comp[key], -lim, lim, FIELD_CMAP)
        ax.set_title(lbl, fontsize=11, pad=4)
        badge = f"CONUS {_wmean(comp[key]):+.2f} K"
        if key != "truth":
            badge += f" | RMSE {_wrmse(comp[key], comp['truth']):.2f}"
        if key == "best":
            # The composite averages 45 DIFFERENT best members, so the per-case rank
            # becomes a mean: where the truth-closest member typically sits in the spread.
            badge += f" | mean rk {best_rank_mean:.0f}/{n_mem}"
        _badge(ax, badge, dark=True)
        if c == 0:
            ax.text(-0.02, 0.5, f"composite ({n_ev} events)", transform=ax.transAxes,
                    rotation=90, va="center", ha="right", fontsize=11)
    fig.colorbar(im, cax=fig.add_subplot(gs[0, 4]), label="T2m anomaly (K)")

    # --- row 1: errors vs the ERA5 composite, one shared symmetric scale ---
    elim = _sym_range([e for e, _ in errs])
    ime = None
    fig.add_subplot(gs[1, 0]).axis("off")           # no error panel under the truth column
    for c, (err, lbl) in enumerate(errs):
        axe = _geo_ax(fig, gs[1, 1 + c])
        ime = _draw(axe, err, -elim, elim, ERR_CMAP)
        axe.set_title(f"error: {lbl} - ERA5", fontsize=10, pad=4)
        _badge(axe, f"bias {_wmean(err):+.2f} K", dark=False)
    fig.colorbar(ime, cax=fig.add_subplot(gs[1, 4]), label="error (K)")

    fig.suptitle(
        f"p90 T2m-anomaly events — {n_ev}-event composite of the verification-week "
        f"anomaly, {model.upper()} {got[0][1].attrs['n_members']} members, "
        f"{P.LEAD_DAYS}-day lead",
        fontsize=13, y=1.02)
    _save(fig, outdirs, f"p90_maps_composite_{_qtag(q)}.png")


# --------------------------------------------------------------------------- #
# Figure 2 — per-event pages
# --------------------------------------------------------------------------- #
def fig_events_page(model: str, page, page_no: int, n_pages: int, outdirs,
                    q: float = EXTREME_Q) -> None:
    """One page: a row per event, columns ERA5 | pooled mean | pQ extreme | best.

    Each row gets its own shared color scale — event magnitudes span +2.4 K to +5.0 K in
    the CONUS mean and much more locally, so a single figure-wide scale would flatten the
    weaker events into noise (the allbestmaps.png convention).
    """
    nrows = len(page)
    col_labels = _col_labels(q)
    fig = plt.figure(figsize=(4.3 * 4 + 1.0, 2.35 * nrows))
    gs = GridSpec(nrows, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.05],
                  hspace=0.18, wspace=0.06)

    for r, (case, f) in enumerate(page):
        cols = [(f["truth"], f"CONUS {f.attrs['truth_conus_mean']:+.2f} K")] + _fc_columns(f)
        lim = _sym_range([da for da, _ in cols])
        im = None
        for c, (da, badge) in enumerate(cols):
            ax = _geo_ax(fig, gs[r, c])
            im = _draw(ax, da, -lim, lim, FIELD_CMAP)
            if r == 0:
                ax.set_title(col_labels[c], fontsize=11, pad=4)
            if c == 0:
                ax.text(-0.02, 0.5, f"{case}\npeak {f.attrs['peak_anom_K']:+.2f} K",
                        transform=ax.transAxes, rotation=90, va="center", ha="right",
                        fontsize=8.5)
            _badge(ax, badge, dark=True)
        fig.colorbar(im, cax=fig.add_subplot(gs[r, 4]), label="K")

    fig.suptitle(
        f"p90 T2m-anomaly events (page {page_no}/{n_pages}) — verification-week anomaly, "
        f"{model.upper()} {page[0][1].attrs['n_members']} members, {P.LEAD_DAYS}-day lead",
        fontsize=13, y=1.0 + 0.22 / (2.35 * nrows))
    _save(fig, outdirs, f"p90_maps_events_{_qtag(q)}_p{page_no}.png")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _save(fig, outdirs, name: str) -> None:
    for d in outdirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"{TAG} wrote {name} -> {', '.join(str(d) for d in outdirs)}")


def event_cases() -> list[str]:
    """The 45 ``kind == "event"`` case names, in cases.csv order (env filters ignored,
    like ``pconfig.all_cases`` — a stray P90_KIND export must not silently drop rows)."""
    df = P.all_cases()
    return list(df[df["kind"] == "event"]["case"])


def make_all(model: str = "fcn3", cases: list[str] | None = None, force: bool = False,
             only: str = "all", per_page: int = PER_PAGE,
             extreme_q: float = EXTREME_Q) -> None:
    if not 0.0 <= extreme_q <= 1.0:
        raise SystemExit(f"{TAG} --extreme-q must be in [0, 1], got {extreme_q}")
    cases = cases or event_cases()
    got = load_all(model, cases, extreme_q, force=force)
    if not got:
        raise SystemExit(f"{TAG} no cases have both a cube and ERA5 truth — nothing to plot")
    print(f"{TAG} {len(got)}/{len(cases)} cases available "
          f"(extreme column = p{extreme_q * 100:g} member)")

    outdirs = [P.fig_dir(model) / "maps"]
    if only in ("all", "composite"):
        fig_composite(model, got, outdirs, extreme_q)
    if only in ("all", "events"):
        pages = [got[i:i + per_page] for i in range(0, len(got), per_page)]
        for i, page in enumerate(pages, start=1):
            fig_events_page(model, page, i, len(pages), outdirs, extreme_q)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=list(P.MODELS), default="fcn3")
    ap.add_argument("--cases", default=None,
                    help="comma-separated case names (default: all 45 events)")
    ap.add_argument("--only", choices=("all", "composite", "events"), default="all")
    ap.add_argument("--per-page", type=int, default=PER_PAGE)
    ap.add_argument("--extreme-q", type=float, default=EXTREME_Q,
                    help=f"quantile of the member ranking for the extreme column "
                         f"(default {EXTREME_Q}); tags the cache and figure names")
    ap.add_argument("--force", action="store_true",
                    help="re-derive the cached fields from the cubes")
    a = ap.parse_args()
    sel = [s.strip() for s in a.cases.split(",") if s.strip()] if a.cases else None
    make_all(a.model, cases=sel, force=a.force, only=a.only, per_page=a.per_page,
             extreme_q=a.extreme_q)


if __name__ == "__main__":
    main()
