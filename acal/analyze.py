#!/usr/bin/env python
"""Cross-case analysis of the calibration campaign: the tail scorecard. CPU ONLY.

Each of the 42 cases left its own AI+RES estimate under `runs/aires/<case>/res/acal/`.
Nothing there combines them. This module does, and it answers the campaign's question
in the only form the slate allows.

Why a scorecard and not a reliability diagram
---------------------------------------------
Every case was SELECTED ON ITS OUTCOME: it is in the slate because the CONUS-wide 7-day
anomaly crossed 2 K. A reliability diagram bins forecasts by probability and asks how
often the event happened, and here it always did, so the diagram has one populated
column. What can be scored is how much probability AI+RES gave to what happened, and
whether that beats what climatology alone would have given. The selection biases both
numbers downward (and the PIT toward 1), so a pooled statistic here is NOT a calibration
statement - it is a statement about the conditional tail, with that caveat attached.

The estimator
-------------
    P_RES(A >= a) = Z * mean_i( w_i * 1[s A_i >= s a] ),   w_i = exp(-V_K,i)

with `s` the tail sign (`+1` heat, `-1` cold: a cold case cloned its most NEGATIVE
walkers, so "more extreme" is `s*A` larger). This is `aires/run_aires.py::stage_compare`'s
`res_p` exactly, evaluated through `dmc.DMCResult.expectation`, which is the same code
the runs used. Note the `>=`: `DMCResult.exceedance` uses a strict `>`, which only
differs at a tie and would be wrong at the observed value if a walker landed on it.

The raw estimate is the headline. `Z * mean(w)` (the run's `normalization_check`)
estimates 1 but reads anywhere from ~0.15 to ~3.2 across these 42 runs, so the
self-normalized `P_sn = P_raw / normalization_check` is reported alongside as the
sensitivity check. `P_sn` is a proper probability (it is 1 at the population minimum);
`P_raw` is not, and can exceed 1.

The climatology
---------------
`p_clim(a)` is the empirical fraction of days whose CONUS-wide A_L is at least as extreme
as `a`, over `runs/acal/catalog/conus_daily_2021_2025.csv` (built by
`runs/acal/catalog/build_catalog.py` with the same 13-frame cos(lat) definition the slate
uses), within +/-30 calendar days of the peak's anniversary in any year, minus the case's
own +/-10 days. Two caveats travel with it: it is five years, not thirty, and the anomaly
is taken against a 1990-2019 climatology, so the 2021-2025 warming sits inside it (a heat
day is commoner in this pool than the anomaly's name suggests, which LOWERS the heat lift).

    python -m acal.analyze --stage collect     # runs/acal/analysis/cases.csv
    python -m acal.analyze --stage scorecard   # runs/acal/analysis/scorecard.csv (+ summary.json)
    python -m acal.analyze --stage figures     # figures/acal/acal_*.png
    python -m acal.analyze --stage rungs       # rungs_cases.csv, rungs_summary.json,
                                               #   figures/acal/acal_rungs_*.png
    python -m acal.analyze --stage all
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acal import aprep, ccfg
from aires import aceiling, dmc
from aires import aconfig as A

TAG = "acal"
OUT = ccfg.ACAL_ROOT / "analysis"
CASES_OUT = OUT / "cases.csv"
SCORE_OUT = OUT / "scorecard.csv"
SUMMARY_OUT = OUT / "summary.json"
DAILY = ccfg.ACAL_ROOT / "catalog" / "conus_daily_2021_2025.csv"

N_WALKERS = ccfg.RES_N_WALKERS            # 32; anything else is a different experiment
CLIM_WINDOW_DAYS = 30                     # +/- calendar days around the peak's anniversary
CLIM_EXCLUDE_DAYS = 10                    # the case's own +/- days, left out of the pool
CATALOG_TOL = 1e-3                        # K; the slate stores a_l_conus to 3 decimals

# Health thresholds. normalization_check is right-skewed with mean ~1 and median ~0.7
# (aires/dmc.py), so only a reading well outside that is flagged.
NORM_LO, NORM_HI = 0.25, 2.0
# Known anomalies the plan asks the health figure to name, with the reason.
KNOWN_OUTLIERS = {
    "e24_c3_20240119": "~31 min wall vs ~4-5 h typical",
    "e25_h3_20240131": "~31 min wall vs ~4-5 h typical",
    "e33_h4_20250101": "15 GB scores/ vs 2-4 GB elsewhere",
}


# --------------------------------------------------------------------------- #
# One case, loaded and checked
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    """Everything the scorecard needs about one run, read from disk and cross-checked."""
    episode_id: str
    sign: float                 # +1 heat, -1 cold
    obs: float                  # ERA5 CONUS A_L at the peak, from the slate
    al: np.ndarray              # (32,) final-walker CONUS A_L, physical units
    result: dmc.DMCResult
    cmp: dict
    rec: dict
    run: dict

    @property
    def weights(self) -> np.ndarray:
        return self.result.weights

    @property
    def norm(self) -> float:
        return self.result.normalization_check()

    def p_raw(self, a: float) -> float:
        """`Z * mean(w * 1[s A >= s a])` - stage_compare's res_p, via DMCResult."""
        return float(self.result.expectation(aceiling.beyond(self.al, a, self.sign)))

    def p_sn(self, a: float) -> float:
        return self.p_raw(a) / self.norm

    def n_beyond(self, a: float) -> int:
        return int(aceiling.beyond(self.al, a, self.sign).sum())

    def cdf_raw(self, a: float) -> float:
        """Weighted forecast CDF at `a` in tail-signed units: `Z * mean(w * 1[s A < s a])`.

        The PIT. Strict `<`, so `cdf_raw + p_raw == normalization_check` exactly and the
        self-normalized PIT is `1 - p_sn`.
        """
        return float(self.result.expectation(~aceiling.beyond(self.al, a, self.sign)))


def case_dir(eid: str) -> Path:
    return A.res_dir(eid, TAG)


def _read(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def load_case(row) -> Case:
    """Load one case and HARD-FAIL on anything that would make its numbers meaningless."""
    eid = row.episode_id
    d = case_dir(eid)
    for name in ("res_result.json", "compare.json", "run.json"):
        if not (d / name).exists():
            raise SystemExit(f"[analyze] {eid}: missing {d / name}")
    rec, cmp_, run = _read(d / "res_result.json"), _read(d / "compare.json"), _read(d / "run.json")
    result = dmc.DMCResult.from_dict(rec["dmc"])
    al = np.asarray(cmp_["realized"]["box"], dtype="float64")

    if result.n_walkers != N_WALKERS or al.size != N_WALKERS:
        raise SystemExit(f"[analyze] {eid}: n_walkers={result.n_walkers}, "
                         f"realized={al.size}, expected {N_WALKERS}")
    if run.get("box") != "CONUS":
        raise SystemExit(f"[analyze] {eid}: scored on box {run.get('box')!r}, not CONUS")
    # compare.json's weights and log_Z must be the DMCResult's own - otherwise the curve on
    # disk and the numbers below would describe two different estimators.
    if not np.allclose(result.weights, cmp_["weights"], rtol=1e-12, atol=0):
        raise SystemExit(f"[analyze] {eid}: compare.json weights != exp(-final_V)")
    if abs(result.log_Z - float(cmp_["log_Z"])) > 1e-12:
        raise SystemExit(f"[analyze] {eid}: compare.json log_Z != dmc log_Z")

    # The tail sign: one value in the config, repeated on every scored leg. A leg that
    # disagrees would mean the run cloned the wrong tail for part of its schedule.
    sign = float(run["tail_sign"])
    legs = {float(t["tail_sign"]) for t in rec.get("theta", []) if "tail_sign" in t}
    if legs and legs != {sign}:
        raise SystemExit(f"[analyze] {eid}: theta tail_sign {legs} != config {sign}")
    want = 1.0 if row.family == "heat" else -1.0
    if sign != want or np.sign(row.a_l_conus) != want:
        raise SystemExit(f"[analyze] {eid}: family={row.family} but tail_sign={sign}, "
                         f"a_l_conus={row.a_l_conus}")
    return Case(eid, sign, float(row.a_l_conus), al, result, cmp_, rec, run)


def load_all() -> tuple[pd.DataFrame, list[Case]]:
    df = aprep.episodes()
    cases = [load_case(r) for r in df.itertuples()]
    return df, cases


# --------------------------------------------------------------------------- #
# Climatology
# --------------------------------------------------------------------------- #
def load_daily() -> pd.Series:
    d = pd.read_csv(DAILY, parse_dates=["date"])
    return pd.Series(d.a_l_conus.values, index=pd.DatetimeIndex(d.date), name="a_l_conus")


def _anniversary(peak: pd.Timestamp, year: int) -> pd.Timestamp:
    """`peak`'s month/day in `year`; Feb 29 falls back to Feb 28 in a common year."""
    try:
        return peak.replace(year=year)
    except ValueError:
        return peak.replace(year=year, day=28)


def clim_pool(daily: pd.Series, peak, window: int = CLIM_WINDOW_DAYS,
              exclude: int = CLIM_EXCLUDE_DAYS) -> pd.Series:
    """Days within +/-`window` of any anniversary of `peak`, minus `peak` +/- `exclude`.

    Anniversaries run one year past each end of the series so a January peak also picks
    up the previous December (and a December peak the next January).
    """
    peak = pd.Timestamp(peak)
    t = daily.index
    years = range(t.min().year - 1, t.max().year + 2)
    dist = np.min([np.abs((t - _anniversary(peak, y)).days) for y in years], axis=0)
    own = np.abs((t - peak).days) <= exclude
    return daily[(dist <= window) & ~own]


def p_clim(pool: pd.Series, a: float, sign: float) -> tuple[float, int, int]:
    """Empirical `P(s A >= s a)` over the pool: (p, k, n)."""
    k = int(aceiling.beyond(pool.values, a, sign).sum())
    n = int(pool.size)
    return (k / n if n else np.nan), k, n


def _lift(p: float, pc: float) -> float:
    """`p / p_clim`. Undefined (NaN) when climatology never reached the value."""
    return float(p / pc) if pc > 0 else np.nan


# --------------------------------------------------------------------------- #
# Stage: collect
# --------------------------------------------------------------------------- #
def collect() -> pd.DataFrame:
    df, cases = load_all()
    rows = []
    for r, c in zip(df.itertuples(), cases):
        w = c.weights
        ess = np.asarray(c.result.ess_by_step, dtype=float)
        rows.append(dict(
            **r._asdict(),
            tail_sign=c.sign,
            n_walkers=c.result.n_walkers,
            log_Z=c.result.log_Z,
            normalization_check=c.norm,
            ess_weights=aceiling.weight_ess(w),       # Kish ESS of the final weights
            ess_last_step=float(ess[-1]),
            ess_min_step=float(ess.min()),
            max_multiplicity=int(np.max(c.result.max_multiplicity_by_step)),
            n_founders=c.result.n_founders,
            al_mean=float(c.al.mean()),
            al_wmean=float(np.sum(w * c.al) / np.sum(w)),
            al_min=float(c.al.min()),
            al_max=float(c.al.max()),
            al_sd=float(c.al.std(ddof=1)),
            w_min=float(w.min()), w_max=float(w.max()),
            wall_h=float(c.rec.get("wall_seconds", np.nan)) / 3600.0,
            host=c.rec.get("host"), job=c.rec.get("job"),
            finished=c.rec.get("finished"),
        ))
    out = pd.DataFrame(rows).drop(columns="Index")
    ccfg.ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(CASES_OUT, index=False, float_format="%.6g")
    print(f"[collect] {len(out)} cases -> {CASES_OUT}")
    return out


# --------------------------------------------------------------------------- #
# Stage: scorecard
# --------------------------------------------------------------------------- #
def score_case(c: Case, rung: float, peak, daily: pd.Series) -> dict:
    """One scorecard row. Pure function of the case and the daily series (tested)."""
    pool = clim_pool(daily, peak)
    a_rung = c.sign * float(rung)               # the rung as a physical threshold
    out = dict(episode_id=c.episode_id, obs=c.obs, tail_sign=c.sign,
               normalization_check=c.norm)

    out["p_obs_raw"] = c.p_raw(c.obs)
    out["p_obs_sn"] = out["p_obs_raw"] / c.norm
    out["n_beyond_obs"] = c.n_beyond(c.obs)
    # Two ways to have no resolution at obs, both kept as rows: no walker reached it
    # (P == 0, an upper bound), or every walker is beyond it (P_raw IS the normalization
    # check and P_sn == 1 identically, a statement about Z, not about the tail).
    out["unresolved_obs"] = out["n_beyond_obs"] == 0
    out["saturated_obs"] = out["n_beyond_obs"] == N_WALKERS
    out["resolution_obs"] = aceiling.resolution(out["n_beyond_obs"], N_WALKERS)
    out["p_clim_obs"], out["k_clim_obs"], out["n_clim"] = p_clim(pool, c.obs, c.sign)
    out["lift_obs_raw"] = _lift(out["p_obs_raw"], out["p_clim_obs"])
    out["lift_obs_sn"] = _lift(out["p_obs_sn"], out["p_clim_obs"])
    # Defined for every case: a climatological zero is counted as one day, which can only
    # LOWER the lift, so a pooled median over this column is conservative.
    out["lift_obs_raw_cons"] = out["p_obs_raw"] / (max(out["k_clim_obs"], 1) / out["n_clim"])

    out["rung_threshold"] = a_rung
    out["p_rung_raw"] = c.p_raw(a_rung)
    out["p_rung_sn"] = out["p_rung_raw"] / c.norm
    out["n_beyond_rung"] = c.n_beyond(a_rung)
    out["resolution_rung"] = aceiling.resolution(out["n_beyond_rung"], N_WALKERS)
    out["p_clim_rung"], out["k_clim_rung"], _ = p_clim(pool, a_rung, c.sign)
    out["lift_rung_raw"] = _lift(out["p_rung_raw"], out["p_clim_rung"])
    out["lift_rung_sn"] = _lift(out["p_rung_sn"], out["p_clim_rung"])

    out["pit_raw"] = c.cdf_raw(c.obs)
    out["pit_sn"] = out["pit_raw"] / c.norm
    # Where the observation sits in the unweighted population, for reference.
    out["rank_unweighted"] = int(np.sum(c.sign * c.al < c.sign * c.obs))
    return out


def curve_check(c: Case) -> float:
    """Max |res(a) - compare_curve.csv| over the run's own 40 thresholds.

    The acal runs had no `observed` value at compare time, so the curve does NOT contain
    the observed point; agreement at every knot it does contain pins this module's
    estimator to the one the run published.
    """
    cur = pd.read_csv(case_dir(c.episode_id) / "compare_curve.csv")
    mine = np.array([c.p_raw(a) for a in cur.threshold])
    return float(np.max(np.abs(mine - cur.res.values)))


def _q(x) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return dict(n=0)
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return dict(n=int(x.size), median=float(q2), q25=float(q1), q75=float(q3))


def summarize(sc: pd.DataFrame) -> dict:
    keys = ("p_obs_raw", "p_obs_sn", "p_clim_obs", "lift_obs_raw", "lift_obs_sn",
            "lift_obs_raw_cons", "p_rung_raw", "p_clim_rung", "lift_rung_raw")
    s = dict(n_cases=int(len(sc)),
             overall={k: _q(sc[k]) for k in keys},
             by_rung={int(g): {k: _q(d[k]) for k in keys} for g, d in sc.groupby("rung")},
             by_family={f: {k: _q(d[k]) for k in keys} for f, d in sc.groupby("family")},
             n_unresolved_obs=int(sc.unresolved_obs.sum()),
             unresolved_obs=sc.episode_id[sc.unresolved_obs].tolist(),
             n_saturated_obs=int(sc.saturated_obs.sum()),
             saturated_obs=sc.episode_id[sc.saturated_obs].tolist(),
             n_clim_zero_obs=int((sc.k_clim_obs == 0).sum()),
             clim_zero_obs=sc.episode_id[sc.k_clim_obs == 0].tolist(),
             n_lift_gt1_raw=int((sc.lift_obs_raw > 1).sum()),
             n_lift_gt1_sn=int((sc.lift_obs_sn > 1).sum()),
             n_lift_defined=int(np.isfinite(sc.lift_obs_raw).sum()),
             n_lift_cons_gt1_raw=int((sc.lift_obs_raw_cons > 1).sum()),
             n_lift_rung_gt1_raw=int((sc.lift_rung_raw > 1).sum()),
             pit_raw=_q(sc.pit_raw), pit_sn=_q(sc.pit_sn),
             n_pit_sn_ge_0p9=int((sc.pit_sn >= 0.9).sum()),
             n_pit_raw_gt1=int((sc.pit_raw > 1).sum()),
             catalog_match=int(sc.catalog_match.sum()),
             curve_maxdiff=float(sc.curve_maxdiff.max()))
    return s


def truth_inputs(truth=None):
    """(df, cases, published_cases, daily, out_dir) scored against `truth`.

    None = the published ERA5 run (slate obs, catalog pool, runs/acal/analysis/). A truth
    (name or `acal.truth` provider) re-reduces the walkers on its mask
    (`s2sbase.aires_cases`), takes obs from the truth (ERA5 keeps the slate's), pools its
    own daily series clipped to the published span (`truth.pool_series`) and writes under
    runs/acal/analysis/s2s/<truth>/. For 'era5' every number equals the published one.
    """
    df, pub = load_all()
    if truth is None:
        return df, pub, pub, load_daily(), OUT
    from acal import s2sbase as S2  # noqa: PLC0415 - s2sbase imports this module
    from acal import truth as TR  # noqa: PLC0415
    tr = TR.get_truth(truth) if isinstance(truth, str) else truth
    byid = S2.aires_cases(tr, "13f")
    return df, [byid[e] for e in df.episode_id], pub, TR.pool_series(tr), TR.analysis_dir(tr)


def scorecard(truth=None) -> pd.DataFrame:
    df, cases, pub, daily, out = truth_inputs(truth)
    rows = []
    for r, c, c0 in zip(df.itertuples(), cases, pub):
        row = dict(family=r.family, rung=int(r.rung), peak=r.peak)
        row.update(score_case(c, r.rung, r.peak, daily))
        # Verification: the daily series at the peak IS the observed value, so the
        # climatology and the observation are the same statistic (for ERA5, c.obs is the
        # slate's a_l_conus).
        v = daily.get(pd.Timestamp(r.peak), np.nan)
        row["a_l_daily_at_peak"] = float(v)
        row["catalog_match"] = bool(abs(v - c.obs) <= CATALOG_TOL)
        # the estimator check is on the published (unmasked) walkers, whatever the truth
        row["curve_maxdiff"] = curve_check(c0)
        rows.append(row)
    sc = pd.DataFrame(rows)
    cols = ["episode_id", "family", "rung", "peak"]
    sc = sc[cols + [k for k in sc.columns if k not in cols]]
    out.mkdir(parents=True, exist_ok=True)
    score_out, summary_out = out / SCORE_OUT.name, out / SUMMARY_OUT.name
    sc.to_csv(score_out, index=False, float_format="%.6g")
    s = summarize(sc)
    with open(summary_out, "w") as f:
        json.dump(s, f, indent=2)
    print(f"[scorecard] {len(sc)} cases -> {score_out}")
    print(f"  catalog match (|daily - a_l_conus| <= {CATALOG_TOL}): "
          f"{s['catalog_match']}/{len(sc)}")
    print(f"  max |res - compare_curve.csv| over all knots: {s['curve_maxdiff']:.2e}")
    for k in ("p_obs_raw", "p_obs_sn", "p_clim_obs", "lift_obs_raw", "lift_obs_sn",
              "lift_obs_raw_cons"):
        q = s["overall"][k]
        print(f"  {k:13s} median {q['median']:.3g}  IQR [{q['q25']:.3g}, {q['q75']:.3g}]"
              f"  (n={q['n']})")
    print(f"  unresolved at obs (no walker reached it): {s['n_unresolved_obs']}  "
          f"every walker beyond obs: {s['n_saturated_obs']}  "
          f"climatology never reached obs: {s['n_clim_zero_obs']}")
    print(f"  lift > 1: raw {s['n_lift_gt1_raw']}/{s['n_lift_defined']}, "
          f"sn {s['n_lift_gt1_sn']}/{s['n_lift_defined']}")
    print(f"  wrote {summary_out}")
    return sc


# --------------------------------------------------------------------------- #
# Stage: figures
# --------------------------------------------------------------------------- #
# Heat/cold are the diverging pair's two poles (red/blue), never a status color. Rung is
# carried by the marker as well, so identity is never color alone.
FAMILY_COLOR = {"heat": "#e34948", "cold": "#2a78d6"}
RUNG_MARKER = {2: "o", 3: "s", 4: "^"}
CLIM_COLOR = "#52514e"
OBS_COLOR = "#0b0b0b"
FLAG_COLOR = "#eda100"


def _save(fig, name: str) -> Path:
    ccfg.FIG_ROOT.mkdir(parents=True, exist_ok=True)
    p = ccfg.FIG_ROOT / name
    fig.savefig(p, dpi=140, bbox_inches="tight")       # aires/aplots.py's screen dpi
    print(f"  wrote {p} ({p.stat().st_size / 1e3:.0f} kB)")
    return p


def _style(plt) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "0.9", "grid.linewidth": 0.6,
                         "axes.axisbelow": True})


def fig_scorecard(sc: pd.DataFrame, plt) -> Path:
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharex=True, sharey=True)
    # Zeros are drawn on a shaded band below the data, with a "0" tick, not dropped.
    pos = np.concatenate([sc[c][sc[c] > 0].values
                          for c in ("p_clim_obs", "p_obs_sn", "p_obs_raw")])
    floor = pos.min() / 4
    hi = max(1.5, sc.p_obs_raw.max() * 1.3)
    for ax, col, title in ((axes[0], "p_obs_raw", "raw:  Z mean(w 1[A beyond obs])"),
                           (axes[1], "p_obs_sn", "self-normalized:  raw / norm. check")):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.axhspan(floor / 1.7, floor * 1.7, color="0.93", zorder=0)
        ax.axvspan(floor / 1.7, floor * 1.7, color="0.93", zorder=0)
        ax.plot([floor * 1.7, hi], [floor * 1.7, hi], color="0.6", lw=1, ls="--", zorder=1)
        ax.text(hi / 1.15, hi / 1.6, "1:1", ha="right", va="top", color="0.45", fontsize=8)
        for r in sc.itertuples():
            x = r.p_clim_obs if r.p_clim_obs > 0 else floor
            y = getattr(r, col) if getattr(r, col) > 0 else floor
            ax.scatter(x, y, s=46, marker=RUNG_MARKER[r.rung],
                       color=FAMILY_COLOR[r.family], edgecolor="white", linewidth=0.8,
                       alpha=0.9, zorder=3)
            if r.saturated_obs:          # every walker beyond obs: no tail resolution
                ax.scatter(x, y, s=130, marker="o", facecolor="none", edgecolor="0.15",
                           linewidth=1.0, zorder=4)
        ax.set_xlim(floor / 1.7, hi); ax.set_ylim(floor / 1.7, hi)
        ticks = [t for t in (1e-3, 1e-2, 1e-1, 1) if t > floor * 2]
        lab = ["0"] + [f"$10^{{{int(np.log10(t))}}}$" for t in ticks]
        ax.set_xticks([floor] + ticks, lab); ax.set_yticks([floor] + ticks, lab)
        ax.minorticks_off()
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("P_clim(obs): ERA5 2021-2025, +/-30 d of peak")
        n_c = int((sc.p_clim_obs > 0).sum())
        above = int(((sc[col] > sc.p_clim_obs) & (sc.p_clim_obs > 0)).sum())
        ax.text(0.97, 0.13, f"{above} of {n_c} above 1:1 (P_clim > 0)\n"
                f"{int((sc[col] == 0).sum())} at P_RES = 0 (no walker reached obs)\n"
                f"{int((sc.p_clim_obs == 0).sum())} at P_clim = 0 (never in pool)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.25",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    axes[0].set_ylabel("P_RES(A_L at least as extreme as observed)")
    handles = [Line2D([], [], ls="none", marker="o", ms=7, color=FAMILY_COLOR[f], label=f)
               for f in ("heat", "cold")]
    handles += [Line2D([], [], ls="none", marker=RUNG_MARKER[g], ms=7, color="0.45",
                       label=f"rung {g} K") for g in (2, 3, 4)]
    handles.append(Line2D([], [], ls="none", marker="o", ms=10, mfc="none", mec="0.15",
                          label="all 32 walkers beyond obs"))
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("AI+RES probability of the observed outcome vs climatology, 42 cases "
                 "(selected on outcome)", y=1.07, fontsize=10.5)
    p = _save(fig, "acal_scorecard.png")
    plt.close(fig)
    return p


def fig_lift(sc: pd.DataFrame, plt) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    groups = [(f, g) for f in ("heat", "cold") for g in (2, 3, 4)]
    rng = np.random.default_rng(0)
    finite = np.concatenate([sc.lift_obs_raw.values, sc.lift_rung_raw.values])
    finite = finite[np.isfinite(finite) & (finite > 0)]
    floor = 10 ** np.floor(np.log10(finite.min())) / 3
    top = 10 ** np.ceil(np.log10(finite.max()))
    for ax, col, title in ((axes[0], "lift_obs_raw", "lift at the observed value"),
                           (axes[1], "lift_rung_raw", "lift at the case's rung (2/3/4 K)")):
        ax.set_yscale("log")
        ax.axhline(1, color="0.5", lw=1, ls="--", zorder=1)
        ax.axhspan(floor / 1.5, floor * 1.5, color="0.94", zorder=0)
        labels = []
        for i, (f, g) in enumerate(groups):
            d = sc[(sc.family == f) & (sc.rung == g)]
            v = d[col].values
            ok = np.isfinite(v) & (v > 0)
            x = i + rng.uniform(-0.15, 0.15, size=v.size)
            y = np.where(ok, v, floor)
            ax.scatter(x, y, s=34, marker=RUNG_MARKER[g], color=FAMILY_COLOR[f],
                       edgecolor="white", linewidth=0.7, alpha=0.9, zorder=3)
            # The median counts the zeros as zero, not as missing.
            vv = np.where(np.isfinite(v), v, np.nan)
            if np.isfinite(vv).any():
                med = np.nanmedian(vv)
                if med > 0:
                    ax.plot([i - 0.3, i + 0.3], [med, med], color="0.15", lw=2, zorder=4)
            nan = int((~np.isfinite(v)).sum())
            labels.append(f"{f} {g} K\nn={len(d)}" + (f"\n({nan} undef.)" if nan else ""))
        ax.set_xticks(range(len(groups)), labels, fontsize=8)
        ax.set_ylim(floor / 1.5, top)
        ax.set_title(title, fontsize=9.5)
        ticks = [10.0 ** k for k in range(int(np.ceil(np.log10(floor * 2))),
                                           int(np.log10(top)) + 1)]
        ax.set_yticks([floor] + ticks, ["0"] + [f"$10^{{{int(np.log10(t))}}}$" for t in ticks])
        ax.minorticks_off()
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("lift = P_RES (raw) / P_clim   (0: no walker reached it)")
    fig.suptitle("Lift over climatology by family and rung (bar = group median; "
                 "undef. = climatology never reached the value)", fontsize=10)
    fig.tight_layout()
    p = _save(fig, "acal_lift_by_rung.png")
    plt.close(fig)
    return p


def fig_pit(sc: pd.DataFrame, plt) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    bins = np.linspace(0, 1, 11)
    n = len(sc)
    for ax, col, title in ((axes[0], "pit_raw", "raw weighted CDF at obs"),
                           (axes[1], "pit_sn", "self-normalized weighted CDF at obs")):
        v = sc[col].values
        over = int((v > 1).sum())
        heat = np.clip(v[sc.family == "heat"], 0, 1 - 1e-9)
        cold = np.clip(v[sc.family == "cold"], 0, 1 - 1e-9)
        ax.hist([heat, cold], bins=bins, stacked=True,
                color=[FAMILY_COLOR["heat"], FAMILY_COLOR["cold"]],
                edgecolor="white", linewidth=1.2, label=["heat", "cold"])
        ax.axhline(n / 10, color="0.4", ls="--", lw=1)
        ax.text(0.65, n / 10, "uniform", color="0.35", fontsize=8, va="bottom",
                ha="center")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        sat = int(sc.saturated_obs.sum())
        if col == "pit_sn" and sat:
            ax.annotate(f"{sat} cases: all 32 walkers\nbeyond obs, PIT = 0 exactly",
                        xy=(0.05, sat), xytext=(0.13, sat + 5), fontsize=8, color="0.25",
                        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.8))
        ax.set_xlim(0, 1)
        ax.set_xlabel("F(obs) = P_RES(A_L less extreme than observed)")
        ax.set_title(title + (f"\n({over} values > 1 put in the last bin)" if over else ""),
                     fontsize=9.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("cases")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    fig.text(0.5, -0.06, "Every case was selected because the observed anomaly crossed "
             "2 K, so the observation sits in the tail by construction and the PIT is "
             "biased toward 1.\nA pile-up at 1 here is the selection, not by itself "
             "miscalibration; only its size relative to what climatology gives is "
             "informative.", ha="center", fontsize=8.2, color="0.25")
    fig.suptitle("PIT of the observed CONUS A_L under the weighted AI+RES forecast, 42 "
                 "cases", fontsize=10.5)
    fig.tight_layout()
    p = _save(fig, "acal_pit.png")
    plt.close(fig)
    return p


def health_flags(cs: pd.DataFrame) -> dict[str, str]:
    """`episode_id -> reason` for every case the health figure calls out."""
    flags = {}
    for r in cs.itertuples():
        why = []
        if r.normalization_check > NORM_HI or r.normalization_check < NORM_LO:
            why.append(f"norm. check {r.normalization_check:.2f}")
        if r.episode_id in KNOWN_OUTLIERS:
            why.append(KNOWN_OUTLIERS[r.episode_id])
        if why:
            flags[r.episode_id] = "; ".join(why)
    return flags


def fig_health(cs: pd.DataFrame, plt) -> Path:
    flags = health_flags(cs)
    x = np.arange(len(cs))
    cols = [FAMILY_COLOR[f] for f in cs.family]
    panels = [("normalization_check", "norm. check\nZ mean(w)", True),
              ("ess_weights", "Kish ESS of\nfinal weights", False),
              ("n_founders", "founders\n(of 32)", False),
              ("log_Z", "log Z", False),
              ("wall_h", "wall (h)", False)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11.5, 10.0), sharex=True)
    for ax, (col, lab, logy) in zip(axes, panels):
        v = cs[col].values
        ax.scatter(x, v, s=26, c=cols, edgecolor="white", linewidth=0.6, zorder=3)
        if logy:
            ax.set_yscale("log")
            ax.axhspan(NORM_LO, NORM_HI, color="0.95", zorder=0)
            ax.axhline(1, color="0.5", lw=0.8, ls="--")
            ax.set_yticks([0.1, 0.25, 0.5, 1, 2, 4], ["0.1", "0.25", "0.5", "1", "2", "4"])
        if col == "ess_weights":
            ax.scatter(x, cs.ess_min_step, s=14, marker="_", color="0.35", zorder=3,
                       label="min ESS over\nresampling steps")
            ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0),
                      handlelength=1)
        ax.set_ylabel(lab, fontsize=8.5)
        for i, eid in enumerate(cs.episode_id):
            if eid in flags:
                ax.axvspan(i - 0.45, i + 0.45, color=FLAG_COLOR, alpha=0.18, zorder=0)
        ax.grid(axis="x", visible=False)
    axes[-1].set_xticks(x, cs.episode_id, rotation=90, fontsize=7)
    axes[-1].set_xlim(-0.8, len(cs) - 0.2)
    txt = "Flagged (shaded): " + "   ".join(f"{k}: {v}" for k, v in flags.items())
    fig.text(0.01, -0.02, "\n".join(textwrap.wrap(txt, 190)), fontsize=7.8, va="top",
             color="0.2")
    fig.suptitle("Run health per case (red heat, blue cold; band in top panel = "
                 f"{NORM_LO}-{NORM_HI})", fontsize=10.5)
    fig.tight_layout()
    p = _save(fig, "acal_health.png")
    plt.close(fig)
    return p


def fig_curves(sc: pd.DataFrame, cases: list[Case], daily: pd.Series, plt) -> Path:
    ncol = 7
    nrow = int(np.ceil(len(cases) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15.5, 2.15 * nrow), sharey=True)
    floor = 1e-3
    srow = sc.set_index("episode_id")
    for ax, c in zip(axes.flat, cases):
        r = srow.loc[c.episode_id]
        # The run's own 40-knot curve stops at the population's range, so an observation
        # outside it (all walkers beyond it, or none) would float free of the curve. The
        # curve is therefore re-evaluated on a grid that spans obs too; `curve_check`
        # pins it to compare_curve.csv at every knot (max diff ~5e-6, the CSV rounding).
        cur = pd.read_csv(case_dir(c.episode_id) / "compare_curve.csv")
        xs = c.sign * cur.threshold.values
        pool = clim_pool(daily, r.peak)
        grid = np.linspace(min(xs.min(), c.sign * c.obs) - 0.3,
                           max(xs.max(), c.sign * c.obs) + 0.3, 400)
        pc = np.array([p_clim(pool, c.sign * g, c.sign)[0] for g in grid])
        ax.plot(grid, np.maximum(pc, floor), color=CLIM_COLOR, lw=1.1, ls=(0, (3, 2)))
        res = np.array([c.p_raw(c.sign * g) for g in grid])
        ax.plot(grid, np.maximum(res, floor), color=FAMILY_COLOR[r.family], lw=1.6)
        ax.axvline(c.sign * c.obs, color=OBS_COLOR, lw=1.0)
        ax.plot(c.sign * c.obs, max(r.p_obs_raw, floor), "o", ms=5,
                color=FAMILY_COLOR[r.family], mec=OBS_COLOR, mew=0.8, zorder=5)
        ax.set_yscale("log")
        ax.set_ylim(floor / 1.5, 4)
        ax.tick_params(labelsize=7)
        ttl = f"{c.episode_id}"
        if r.unresolved_obs:
            ttl += "  [P=0]"
        elif r.saturated_obs:
            ttl += "  [32/32 beyond]"
        ax.set_title(ttl, fontsize=7.5, pad=2)
        ax.grid(axis="x", visible=False)
    for ax in axes.flat[len(cases):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(beyond)", fontsize=7.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("s * A_L (K)", fontsize=7.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=FAMILY_COLOR["heat"], lw=1.6, label="P_RES raw, heat"),
         Line2D([], [], color=FAMILY_COLOR["cold"], lw=1.6, label="P_RES raw, cold"),
         Line2D([], [], color=CLIM_COLOR, lw=1.1, ls=(0, (3, 2)),
                label="climatology (2021-2025, +/-30 d)"),
         Line2D([], [], color=OBS_COLOR, lw=1.0, label="ERA5 observed")]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.985), fontsize=9)
    fig.suptitle("Exceedance curves per case (x tail-signed: s = -1 for cold; values "
                 f"below {floor:g} drawn at the floor)", y=1.0, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    p = _save(fig, "acal_curves_grid.png")
    plt.close(fig)
    return p


def figures() -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    for p in (CASES_OUT, SCORE_OUT):
        if not p.exists():
            raise SystemExit(f"[figures] no {p}; run --stage collect and --stage scorecard")
    cs = pd.read_csv(CASES_OUT)
    sc = pd.read_csv(SCORE_OUT)
    df, cases = load_all()
    daily = load_daily()
    print(f"[figures] -> {ccfg.FIG_ROOT}")
    return [fig_scorecard(sc, plt), fig_lift(sc, plt), fig_pit(sc, plt),
            fig_health(cs, plt), fig_curves(sc, cases, daily, plt)]


# --------------------------------------------------------------------------- #
# Stage: rungs - conditional reliability at 2 / 3 / 4 K
# --------------------------------------------------------------------------- #
# Why conditional. Every case is in the slate because s*obs >= 2 K, so an unconditional
# reliability test at rung `a` is biased (o = 1 at 2 K for all 42). But if a case's
# forecast distribution F_i is calibrated, then for any a >= b
#
#     P(s A >= a | s A >= b, F_i) = F_i(a) / F_i(b)
#
# exactly, and the selection {s*obs >= 2} is a function of the outcome at b = 2 alone, so
# conditioning on it leaves q_i(a) = F_i(a) / F_i(2) testable against o_i = 1[s*obs >= a]
# with no selection bias. `Z = exp(log_Z)` multiplies numerator and denominator and
# cancels, so q is a ratio of weighted walker sums and is free of the normalization_check
# noise. The assumption is that selection depends only on the outcome at 2 K; two minor
# deviations are the slate's 10-day declustering and that the valid date is the observed
# peak (an argmax inside an episode), which can only push outcomes upward.
RUNG_SPECS = (("3|2", 3.0, 2.0), ("4|2", 4.0, 2.0), ("4|3", 4.0, 3.0))
RUNGS_CASES_OUT = OUT / "rungs_cases.csv"
RUNGS_SUMMARY_OUT = OUT / "rungs_summary.json"
N_BOOT = 2000
BOOT_SEED = 20261001
LOGLOSS_EPS = 0.01          # q clipped to [eps, 1 - eps] for the log-loss only
CLIM_MIN_COND = 5           # pool days at the conditioning rung below which -> all-season
INTERVAL = 0.90


def cond_q(al, w, sign: float, a: float, b: float) -> float:
    """`sum(w 1[sA >= a]) / sum(w 1[sA >= b])`, rungs `a`, `b` in tail-signed K.

    NaN when no weighted walker reaches `b` (the forecast said `< b` was certain).
    """
    al = np.asarray(al, float)
    w = np.asarray(w, float)
    den = float(np.sum(w * aceiling.beyond(al, sign * b, sign)))
    num = float(np.sum(w * aceiling.beyond(al, sign * a, sign)))
    return num / den if den > 0 else np.nan


def boot_q(al, w, sign: float, a: float, b: float, n_boot: int = N_BOOT,
           rng: np.random.Generator | None = None) -> np.ndarray:
    """Bootstrap of `cond_q` over the 32 (A, w) final walkers. NaN where a resample has
    no walker at `b`."""
    rng = np.random.default_rng(0) if rng is None else rng
    al = np.asarray(al, float)
    w = np.asarray(w, float)
    idx = rng.integers(0, al.size, size=(n_boot, al.size))
    A_, W_ = al[idx], w[idx]
    den = np.sum(W_ * (sign * (A_ - sign * b) >= 0), axis=1)
    num = np.sum(W_ * (sign * (A_ - sign * a) >= 0), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def poisson_binomial_pmf(p) -> np.ndarray:
    """Exact pmf of the number of successes of independent Bernoulli(p_i), k = 0..n."""
    pmf = np.array([1.0])
    for pi in np.asarray(p, float):
        pmf = np.convolve(pmf, [1.0 - pi, pi])
    return pmf


def pb_test(p, k: int, level: float = INTERVAL) -> dict:
    """Two-sided p-value (2 x the smaller tail, capped at 1) and the central `level`
    range of the count under the Poisson-binomial with success probabilities `p`."""
    pmf = poisson_binomial_pmf(p)
    cdf = np.cumsum(pmf)
    sf = pmf[::-1].cumsum()[::-1]                     # P(X >= k)
    lo_t = (1 - level) / 2
    lo = int(np.searchsorted(cdf, lo_t - 1e-12))
    hi = int(np.searchsorted(cdf, 1 - lo_t - 1e-12))
    pval = float(min(1.0, 2 * min(cdf[k], sf[k]))) if len(p) else np.nan
    return dict(expected=float(np.sum(p)), observed=int(k), range_lo=lo, range_hi=hi,
                p_value=pval, p_le=float(cdf[k]), p_ge=float(sf[k]))


def _logloss(q, o, eps: float = LOGLOSS_EPS) -> float:
    q = np.clip(np.asarray(q, float), eps, 1 - eps)
    o = np.asarray(o, float)
    return float(-np.mean(o * np.log(q) + (1 - o) * np.log(1 - q)))


def clim_cond(daily: pd.Series, peak, a: float, b: float, sign: float) -> dict:
    """Climatological `P(>=a) / P(>=b)` from the seasonal pool, falling back to the
    all-season pool (minus the case's own +/-10 d) when the pool has < CLIM_MIN_COND
    days at `b`."""
    pool = clim_pool(daily, peak)
    _, ka, n = p_clim(pool, sign * a, sign)
    _, kb, _ = p_clim(pool, sign * b, sign)
    fallback = kb < CLIM_MIN_COND
    if fallback:
        pk = pd.Timestamp(peak)
        allp = daily[np.abs((daily.index - pk).days) > CLIM_EXCLUDE_DAYS]
        _, ka, n = p_clim(allp, sign * a, sign)
        _, kb, _ = p_clim(allp, sign * b, sign)
    return dict(q_clim=(ka / kb) if kb > 0 else np.nan, k_clim_a=ka, k_clim_b=kb,
                n_clim_pool=n, clim_fallback=bool(fallback))


def rungs_rows(df: pd.DataFrame, cases: list[Case], daily: pd.Series):
    """(rows DataFrame, {(spec, episode_id): bootstrap array})."""
    rows, boots = [], {}
    for i, (r, c) in enumerate(zip(df.itertuples(), cases)):
        so = c.sign * c.obs
        for spec, a, b in RUNG_SPECS:
            if so < b:                                 # not in this conditional sample
                continue
            q = cond_q(c.al, c.weights, c.sign, a, b)
            bq = boot_q(c.al, c.weights, c.sign, a, b,
                        rng=np.random.default_rng([BOOT_SEED, i, int(10 * a + b)]))
            boots[(spec, c.episode_id)] = bq
            ok = np.isfinite(bq)
            row = dict(episode_id=c.episode_id, family=r.family, slate_rung=int(r.rung),
                       peak=r.peak, obs=c.obs, s_obs=so, spec=spec, a=a, b=b,
                       q=q, defined=bool(np.isfinite(q)),
                       q_lo=float(np.percentile(bq[ok], 5)) if ok.any() else np.nan,
                       q_hi=float(np.percentile(bq[ok], 95)) if ok.any() else np.nan,
                       boot_undefined_frac=float(1 - ok.mean()),
                       outcome=int(so >= a),
                       F_a_raw=c.p_raw(c.sign * a), F_b_raw=c.p_raw(c.sign * b),
                       n_beyond_a=c.n_beyond(c.sign * a), n_beyond_b=c.n_beyond(c.sign * b),
                       normalization_check=c.norm)
            row["F_a_sn"] = row["F_a_raw"] / c.norm
            row["F_b_sn"] = row["F_b_raw"] / c.norm
            row.update(clim_cond(daily, r.peak, a, b, c.sign))
            rows.append(row)
    return pd.DataFrame(rows), boots


def rung_group_summary(d: pd.DataFrame, boots: dict, spec: str) -> dict:
    """Expected vs observed hits, Brier / BSS / log-loss, for one (spec, group) slice."""
    undef = d[~d.defined]
    g = d[d.defined]
    out = dict(n=int(len(d)), n_defined=int(len(g)), n_undefined=int(len(undef)),
               undefined=undef.episode_id.tolist())
    if not len(g):
        return out
    q, o = g.q.values, g.outcome.values
    out.update(pb_test(q, int(o.sum())))
    # Expected-count interval from the per-case bootstraps, drawn jointly by replicate.
    B = np.stack([boots[(spec, e)] for e in g.episode_id])         # (n, N_BOOT)
    B = np.where(np.isfinite(B), B, q[:, None])
    tot = B.sum(axis=0)
    out["expected_boot_lo"], out["expected_boot_hi"] = (float(v) for v in
                                                       np.percentile(tot, [5, 95]))
    out["brier"] = float(np.mean((q - o) ** 2))
    out["logloss"] = _logloss(q, o)
    qc = g.q_clim.values
    okc = np.isfinite(qc)
    out["n_clim_fallback"] = int(g.clim_fallback.sum())
    out["clim_fallback"] = g.episode_id[g.clim_fallback].tolist()
    out["n_clim_undefined"] = int((~okc).sum())
    if okc.any():
        out["expected_clim"] = float(np.sum(qc[okc]))
        out["clim_test"] = pb_test(qc[okc], int(o[okc].sum()))
        out["brier_clim"] = float(np.mean((qc[okc] - o[okc]) ** 2))
        out["brier_on_clim_cases"] = float(np.mean((q[okc] - o[okc]) ** 2))
        out["bss"] = (1 - out["brier_on_clim_cases"] / out["brier_clim"]
                      if out["brier_clim"] > 0 else np.nan)
        out["logloss_clim"] = _logloss(qc[okc], o[okc])
    out["n_q_zero"] = int((q == 0).sum())
    out["n_q_one"] = int((q == 1).sum())
    # The cases that drove it: every hit, and the misses with the largest q.
    gg = g.assign(sq=(q - o) ** 2)
    out["hits"] = [dict(episode_id=x.episode_id, q=float(x.q), q_clim=float(x.q_clim),
                        s_obs=float(x.s_obs)) for x in gg[gg.outcome == 1].itertuples()]
    out["worst"] = [dict(episode_id=x.episode_id, q=float(x.q), outcome=int(x.outcome),
                         sq=float(x.sq)) for x in gg.nlargest(5, "sq").itertuples()]
    return out


def sharpness_2k(df: pd.DataFrame, cases: list[Case], daily: pd.Series) -> pd.DataFrame:
    """Rung 2 is not testable (o = 1 for all 42): what each forecast gave it."""
    rows = []
    for r, c in zip(df.itertuples(), cases):
        a = c.sign * 2.0
        pc, kc, nc = p_clim(clim_pool(daily, r.peak), a, c.sign)
        fr = c.p_raw(a)
        rows.append(dict(episode_id=c.episode_id, family=r.family, F2_raw=fr,
                         F2_sn=fr / c.norm, n_beyond_2=c.n_beyond(a), p_clim_2=pc,
                         k_clim_2=kc, n_clim=nc, lift2_raw=_lift(fr, pc),
                         lift2_sn=_lift(fr / c.norm, pc)))
    return pd.DataFrame(rows)


def rungs(truth=None) -> tuple[pd.DataFrame, dict]:
    df, cases, _, daily, out = truth_inputs(truth)
    rc, boots = rungs_rows(df, cases, daily)
    sh = sharpness_2k(df, cases, daily)
    s = dict(n_cases=len(cases), n_boot=N_BOOT, interval=INTERVAL,
             logloss_eps=LOGLOSS_EPS, clim_min_cond=CLIM_MIN_COND,
             p_value="two-sided, 2 x min(P(X<=k), P(X>=k)), capped at 1",
             specs={})
    for spec, a, b in RUNG_SPECS:
        d = rc[rc.spec == spec]
        s["specs"][spec] = {grp: rung_group_summary(dd, boots, spec) for grp, dd in
                            (("all", d), ("heat", d[d.family == "heat"]),
                             ("cold", d[d.family == "cold"]))}
    s["rung2_sharpness"] = {
        grp: dict(n=int(len(dd)), **{k: _q(dd[k]) for k in
                  ("F2_raw", "F2_sn", "p_clim_2", "lift2_raw", "lift2_sn")},
                  n_lift2_sn_gt1=int((dd.lift2_sn > 1).sum()),
                  n_lift2_raw_gt1=int((dd.lift2_raw > 1).sum()),
                  n_F2_sn_ge_half=int((dd.F2_sn >= 0.5).sum()),
                  n_F2_zero=int((dd.F2_raw == 0).sum()),
                  F2_zero=dd.episode_id[dd.F2_raw == 0].tolist())
        for grp, dd in (("all", sh), ("heat", sh[sh.family == "heat"]),
                        ("cold", sh[sh.family == "cold"]))}
    # The forecasts that said "< 2 K was certain": a miss at the 2 K rung.
    s["undefined_q_2k"] = sh.episode_id[sh.F2_raw == 0].tolist()
    below = [c.episode_id for c in cases if c.sign * c.obs < 2.0]
    if below:       # only under a non-ERA5 truth: rung 2 is then not certain for these
        s["below_2k"] = below
    out.mkdir(parents=True, exist_ok=True)
    cases_out, summ_out = out / RUNGS_CASES_OUT.name, out / RUNGS_SUMMARY_OUT.name
    rc.to_csv(cases_out, index=False, float_format="%.6g")
    sh.to_csv(out / "rungs_sharpness_2k.csv", index=False, float_format="%.6g")
    with open(summ_out, "w") as f:
        json.dump(s, f, indent=2, default=float)
    print(f"[rungs] {len(rc)} case x rung rows -> {cases_out}")
    for spec, _, _ in RUNG_SPECS:
        for grp in ("all", "heat", "cold"):
            x = s["specs"][spec][grp]
            if "expected" not in x:
                continue
            print(f"  {spec} {grp:4s} n={x['n_defined']:2d} (undef {x['n_undefined']})  "
                  f"exp {x['expected']:.2f} [{x['expected_boot_lo']:.2f},"
                  f"{x['expected_boot_hi']:.2f}]  obs {x['observed']}  "
                  f"90% [{x['range_lo']},{x['range_hi']}]  p={x['p_value']:.3f}  "
                  f"BS {x['brier']:.3f} vs clim {x.get('brier_clim', np.nan):.3f} "
                  f"BSS {x.get('bss', np.nan):+.2f}  clim exp "
                  f"{x.get('expected_clim', np.nan):.2f} (fallback {x['n_clim_fallback']})")
    a2 = s["rung2_sharpness"]["all"]
    print(f"  rung 2: F2_sn median {a2['F2_sn']['median']:.3f}  F2_raw median "
          f"{a2['F2_raw']['median']:.3f}  P_clim(2) median {a2['p_clim_2']['median']:.3f}"
          f"  lift_sn>1 {a2['n_lift2_sn_gt1']}/{a2['n']}  F2 == 0: {a2['F2_zero']}")
    print(f"  wrote {summ_out}")
    return rc, s


# --- rungs figures ---------------------------------------------------------- #
GROUP_COLOR = {"all": "#2b2a28", "heat": FAMILY_COLOR["heat"], "cold": FAMILY_COLOR["cold"]}


def reliability_bins(q, o, n_bins: int = 4) -> list[dict]:
    """Equal-count bins of q (sorted, stable), each with the Poisson-binomial range."""
    q, o = np.asarray(q, float), np.asarray(o, int)
    order = np.argsort(q, kind="stable")
    out = []
    for idx in np.array_split(order, n_bins):
        t = pb_test(q[idx], int(o[idx].sum()))
        n = idx.size
        out.append(dict(n=n, q_mean=float(q[idx].mean()), q_min=float(q[idx].min()),
                        q_max=float(q[idx].max()), freq=float(o[idx].mean()),
                        band_lo=t["range_lo"] / n, band_hi=t["range_hi"] / n,
                        hits=int(o[idx].sum())))
    return out


def fig_rungs_reliability(rc: pd.DataFrame, plt) -> Path:
    from matplotlib.lines import Line2D
    d = rc[rc.spec.isin(["3|2", "4|2"]) & rc.defined].reset_index(drop=True)
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)
    for ax, col, title in ((axes[0], "q", "AI+RES:  q = F(a) / F(2)"),
                           (axes[1], "q_clim", "climatology:  P_clim(a) / P_clim(2)")):
        dd = d[np.isfinite(d[col])]
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", zorder=1)
        for b in reliability_bins(dd[col].values, dd.outcome.values):
            ax.plot([b["q_mean"]] * 2, [b["band_lo"], b["band_hi"]], color="0.78",
                    lw=7, solid_capstyle="butt", zorder=2)
            ax.plot([b["q_min"], b["q_max"]], [b["freq"]] * 2, color="0.35", lw=0.9,
                    zorder=3)
            ax.scatter(b["q_mean"], b["freq"], s=70, color="#2b2a28", edgecolor="white",
                       linewidth=1.2, zorder=4)
            ax.annotate(f"{b['hits']}/{b['n']}", (b["q_mean"], max(b["band_hi"], b["freq"])),
                        xytext=(0, 9), textcoords="offset points", fontsize=8,
                        color="0.2", ha="center")
        for r in dd.itertuples():
            y = (1.07 if r.outcome else -0.07) + rng.uniform(-0.025, 0.025)
            ax.scatter(getattr(r, col), y, s=26, marker=RUNG_MARKER[int(r.a)],
                       color=FAMILY_COLOR[r.family], edgecolor="white", linewidth=0.6,
                       alpha=0.85, zorder=3, clip_on=False)
        ax.axhspan(-0.11, -0.03, color="0.96", zorder=0)
        ax.axhspan(1.03, 1.11, color="0.96", zorder=0)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.12, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel(f"forecast conditional probability ({col})")
        ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel("observed frequency  (strips: individual misses / hits)")
    h = [Line2D([], [], ls="none", marker="o", ms=8, color="#2b2a28",
                label="bin mean (4 equal-count bins; line = q span)"),
         Line2D([], [], color="0.78", lw=7, label="90% range if calibrated"),
         Line2D([], [], ls="none", marker="s", ms=6, color="0.45", label="rung 3|2"),
         Line2D([], [], ls="none", marker="^", ms=6, color="0.45", label="rung 4|2"),
         Line2D([], [], ls="none", marker="o", ms=6, color=FAMILY_COLOR["heat"],
                label="heat"),
         Line2D([], [], ls="none", marker="o", ms=6, color=FAMILY_COLOR["cold"],
                label="cold")]
    fig.legend(handles=h, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.suptitle(f"Conditional reliability, rungs 3 and 4 given s*obs >= 2 K "
                 f"({len(d)} case x rung pairs, 42 cases; pairs share cases)",
                 y=1.05, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = _save(fig, "acal_rungs_reliability.png")
    plt.close(fig)
    return p


def fig_rungs_counts(s: dict, plt) -> Path:
    from matplotlib.lines import Line2D
    rows = [(spec, grp) for spec, _, _ in RUNG_SPECS for grp in ("all", "heat", "cold")]
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    y = 0.0
    yt, yl = [], []
    for i, (spec, grp) in enumerate(rows):
        if i and grp == "all":
            y += 0.7
        x = s["specs"][spec][grp]
        if "expected" in x:
            ax.plot([x["range_lo"], x["range_hi"]], [y, y], color="0.82", lw=9,
                    solid_capstyle="butt", zorder=1)
            ax.plot([x["expected_boot_lo"], x["expected_boot_hi"]], [y, y], color="0.3",
                    lw=1.2, zorder=2)
            ax.plot([x["expected"]] * 2, [y - 0.22, y + 0.22], color="0.1", lw=2, zorder=3)
            if "expected_clim" in x:
                ax.scatter(x["expected_clim"], y, s=55, marker="D", facecolor="none",
                           edgecolor="0.25", linewidth=1.2, zorder=3)
            ax.scatter(x["observed"], y, s=75, color=GROUP_COLOR[grp],
                       edgecolor="white", linewidth=1.2, zorder=4)
            bss = x.get("bss", np.nan)
            ax.text(1.01, y, f"obs {x['observed']}  exp {x['expected']:.1f}  "
                    f"p={x['p_value']:.2f}  BSS {bss:+.2f}",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=8,
                    color="0.2")
        yt.append(y)
        yl.append(f"{spec} K  {grp}  (n={x['n_defined']})")
        y += 1
    ax.set_yticks(yt, yl, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(left=-0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("number of cases reaching the rung (hits)")
    ax.grid(axis="y", visible=False)
    h = [Line2D([], [], ls="none", marker="o", ms=8, color="#2b2a28", label="observed"),
         Line2D([], [], color="0.1", lw=2, label="expected = sum q"),
         Line2D([], [], color="0.3", lw=1.2, label="90% bootstrap of expected"),
         Line2D([], [], color="0.82", lw=8, label="90% range if calibrated"),
         Line2D([], [], ls="none", marker="D", ms=6, mfc="none", mec="0.25",
                label="climatology expected")]
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, fontsize=8,
               bbox_to_anchor=(0.45, 1.0))
    fig.suptitle("Expected vs observed hits per rung (a|b: reach a K given b K)",
                 y=1.04, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = _save(fig, "acal_rungs_counts.png")
    plt.close(fig)
    return p


def fig_rungs_sharpness(sh: pd.DataFrame, plt) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    rng = np.random.default_rng(5)
    cols = (("F2_raw", "raw"), ("F2_sn", "self-\nnorm."), ("p_clim_2", "clim."))
    ax = axes[0]
    labels = []
    for i, (fam, (col, lab)) in enumerate([(f, c) for f in ("heat", "cold") for c in cols]):
        v = sh[sh.family == fam][col].values
        ax.scatter(i + rng.uniform(-0.15, 0.15, v.size), v, s=26,
                   color=FAMILY_COLOR[fam] if col != "p_clim_2" else CLIM_COLOR,
                   edgecolor="white", linewidth=0.6, alpha=0.9, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="0.1", lw=2, zorder=4)
        labels.append(f"{fam}\n{lab}")
    ax.set_xticks(range(len(labels)), labels, fontsize=7.5)
    ax.axhline(1, color="0.6", lw=0.8, ls="--")
    ax.set_ylabel("P(s A >= 2 K)")
    ax.set_title("probability given to the 2 K rung (every case reached it)", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    ax.set_yscale("log")
    ax.axhline(1, color="0.5", lw=1, ls="--")
    for i, fam in enumerate(("heat", "cold")):
        for j, (col, lab) in enumerate((("lift2_raw", "raw"), ("lift2_sn", "self-norm."))):
            v = sh[sh.family == fam][col].values
            v = v[np.isfinite(v) & (v > 0)]
            xx = 2 * i + j
            ax.scatter(xx + rng.uniform(-0.15, 0.15, v.size), v, s=26,
                       color=FAMILY_COLOR[fam], edgecolor="white", linewidth=0.6,
                       alpha=0.9, zorder=3)
            ax.plot([xx - 0.3, xx + 0.3], [np.median(v)] * 2, color="0.1", lw=2, zorder=4)
    ax.set_xticks(range(4), ["heat\nraw", "heat\nself-norm.", "cold\nraw",
                             "cold\nself-norm."], fontsize=7.5)
    ax.set_ylabel("lift = P_RES(2 K) / P_clim(2 K)")
    ax.set_title("lift at 2 K (bar = median; P_clim = 0 cases omitted)", fontsize=9.5)
    ax.grid(axis="x", visible=False)
    fig.suptitle("Rung 2: sharpness, not reliability (42 cases, all hits by selection)",
                 fontsize=10.5)
    fig.tight_layout()
    p = _save(fig, "acal_rungs_sharpness.png")
    plt.close(fig)
    return p


def rungs_figures() -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    for p in (RUNGS_CASES_OUT, RUNGS_SUMMARY_OUT):
        if not p.exists():
            raise SystemExit(f"[rungs] no {p}; run --stage rungs")
    rc = pd.read_csv(RUNGS_CASES_OUT)
    sh = pd.read_csv(OUT / "rungs_sharpness_2k.csv")
    s = _read(RUNGS_SUMMARY_OUT)
    return [fig_rungs_reliability(rc, plt), fig_rungs_counts(s, plt),
            fig_rungs_sharpness(sh, plt)]


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", required=True,
                    choices=("collect", "scorecard", "figures", "rungs", "all"))
    ap.add_argument("--truth", default=None,
                    help="score against this truth (era5 | hrrr | hrrr_raw); tables go to "
                         "runs/acal/analysis/s2s/<truth>/ (scorecard and rungs only). "
                         "Default: the published ERA5 run")
    a = ap.parse_args(argv)
    if a.truth is not None:
        if a.stage not in ("scorecard", "rungs"):
            raise SystemExit("[analyze] --truth applies to --stage scorecard / rungs only")
        (scorecard if a.stage == "scorecard" else rungs)(a.truth)
        return 0
    if a.stage in ("collect", "all"):
        collect()
    if a.stage in ("scorecard", "all"):
        scorecard()
    if a.stage in ("figures", "all"):
        figures()
    if a.stage in ("rungs", "all"):
        rungs()
        rungs_figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
