"""Event registry for the FourCastNet 3 vs GenCast head-to-head comparison.

Single source of truth for WHICH events the comparison covers, WHAT metric scores each
one, and WHERE both models' cubes live. Imported by ``fcn3/run_fcn3.py`` (fcn3 env, no
gencast_s2s/JAX imports allowed) and by ``fcn3/compare_fcn3_gencast.py`` (moe/my-env), so
this module must stay dependency-light: stdlib + numpy/pandas only.

Design
------
All six events run at ONE lead -- **week 3 (21 d)** -- so differences in the figures are
differences between event types, not between lead times. That lead was chosen because it
is simultaneously:

  * ``xres`` week3 (``init = peak - 21 d``), so the GenCast side of the three xres events
    is already cached at ``runs/xres/0p25/week3/cache/<event>_cube.nc`` -- no GenCast
    rerun, no new GPU cost; and
  * the frozen p90 lead (``p90/cases_meta.json`` ``lead_days = 21``), so the three p90
    cases keep the exact inits committed in ``p90/cases.csv``.

The three p90 cases are NOT in ``xres.xconfig.XRES_EVENTS``, so GenCast has never been run
for them. They are injected into the xres event registry at run time through the additive
``XRES_EXTRA_EVENTS`` hook (see ``xres_extra_events_spec`` below and the hook in
``xres/xconfig.py``); after that the ordinary xres prep/infer/compare pipeline builds their
truth, init frames and cubes into the same tree as the other nine, and nothing about the
original 12 events changes.

Event choice
------------
xres side: one heat (the original PNW Heat Dome), one hurricane (Ian -- the strongest and
most-studied of the three CONUS landfalls, scored on u850 wind speed), one cold (Uri --
the Texas freeze; already in ``xres.xconfig.COLD_EVENTS`` so the extreme-tail sign flips
automatically).

p90 side: three cases spanning year, season and magnitude, so the moderate-extreme panel
is not three samples of the same regime:

  p90_20231107  2023-11-07  +3.10 K  median magnitude   (autumn 2023)
  p90_20240802  2024-08-02  +2.39 K  marginal -- just over the +2.3662 K p90 threshold,
                                     and the only warm-season case (summer 2024)
  p90_20251224  2025-12-24  +4.96 K  strong, 12-day event (winter 2025)

All six are post-2019, so all are out of sample for GenCast's ``<2019`` checkpoints; FCN3
is trained through 2017, so they are out of sample for it too.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Experiment constants
# --------------------------------------------------------------------------- #
WEEKS = int(os.environ.get("FCN3_WEEKS", 3))          # xres week index -> lead days
WEEKS_TO_LEAD_DAYS = {2: 14, 3: 21, 4: 28}            # mirrors gencast_s2s.config
LEAD_DAYS = WEEKS_TO_LEAD_DAYS[WEEKS]

STEP_H = 6                                            # FCN3 native step
N_MEMBERS = int(os.environ.get("FCN3_MEMBERS", 24))   # matches the cached GenCast cubes
BASE_SEED = 333
RES = "0p25"

# 21 d x 24 h / 6 h = 84 steps; the rollout ends exactly on the peak (00Z).
NSTEPS = int(os.environ.get("FCN3_NSTEPS", LEAD_DAYS * 24 // STEP_H))

# CONUS verification box (gencast_s2s.config: LAT=slice(24,50), LON=slice(235,294)).
LAT_LO, LAT_HI = 24.0, 50.0
LON_LO, LON_HI = 235.0, 294.0

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("GENCAST_RUNS", ROOT / "runs"))


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Event:
    name: str
    peak: str            # event peak (verification week = [peak-6d, peak])
    metric: str          # xres metric key: t2m_anom | u850_speed
    family: str          # heat | cold | hurricane | p90
    label: str           # short human label for figure panels
    source: str          # "xres"  -> GenCast cube already cached
                         # "p90"   -> injected into xres via XRES_EXTRA_EVENTS, must be run
    note: str = ""

    @property
    def init(self) -> pd.Timestamp:
        return pd.Timestamp(self.peak) - pd.Timedelta(days=LEAD_DAYS)

    @property
    def cold(self) -> bool:
        return self.family == "cold"


EVENTS: dict[str, Event] = {
    # --- already-cached GenCast side (xres week3 cubes exist) --------------- #
    "PNW_HeatDome_2021": Event(
        "PNW_HeatDome_2021", "2021-06-28", "t2m_anom", "heat",
        "PNW Heat Dome 2021", "xres",
        "the original head-to-head event"),
    "HurricaneIan_2022": Event(
        "HurricaneIan_2022", "2022-09-28", "u850_speed", "hurricane",
        "Hurricane Ian 2022", "xres",
        "SW Florida Cat-4/5 landfall; scored on 850 hPa wind speed, not T2m"),
    "WinterStorm_Uri_2021": Event(
        "WinterStorm_Uri_2021", "2021-02-15", "t2m_anom", "cold",
        "Winter Storm Uri 2021", "xres",
        "Texas freeze; cold event -> extreme tail is the COLD side"),
    # --- p90 moderate-extreme side (GenCast must be run) ------------------- #
    "p90_20231107": Event(
        "p90_20231107", "2023-11-07", "t2m_anom", "p90",
        "p90 2023-11-07 (+3.1 K)", "p90",
        "p90 case event_15_20231107; median magnitude"),
    "p90_20240802": Event(
        "p90_20240802", "2024-08-02", "t2m_anom", "p90",
        "p90 2024-08-02 (+2.4 K)", "p90",
        "p90 case event_27_20240802; marginal (threshold +2.3662 K), warm season"),
    "p90_20251224": Event(
        "p90_20251224", "2025-12-24", "t2m_anom", "p90",
        "p90 2025-12-24 (+5.0 K)", "p90",
        "p90 case event_45_20251224; strongest, 12-day event"),
}

ORDER = tuple(EVENTS)                       # stable panel order in the figures

# The p90 case each injected event was taken from (provenance; p90/cases.csv is frozen).
P90_SOURCE_CASE = {
    "p90_20231107": "event_15_20231107",
    "p90_20240802": "event_27_20240802",
    "p90_20251224": "event_45_20251224",
}


def selected() -> list[Event]:
    """Events for this run. ``FCN3_EVENTS`` (comma-separated names) subsets them."""
    sel = os.environ.get("FCN3_EVENTS")
    if not sel:
        return [EVENTS[n] for n in ORDER]
    want = [s.strip() for s in sel.split(",") if s.strip()]
    unknown = [w for w in want if w not in EVENTS]
    if unknown:
        raise SystemExit(f"unknown event(s) in FCN3_EVENTS: {', '.join(unknown)}\n"
                         f"known: {', '.join(ORDER)}")
    return [EVENTS[w] for w in want]


def seed_for(ev: Event) -> int:
    """Deterministic per-event base seed (offset by position in the frozen order)."""
    return BASE_SEED + 1000 * ORDER.index(ev.name)


# --------------------------------------------------------------------------- #
# FCN3 output variables per metric. t2m is always stored (one extra 2-D field per step)
# so every event can also be inspected in T2m regardless of its headline metric.
# --------------------------------------------------------------------------- #
FCN3_VARS = {
    "t2m_anom":   ["t2m"],
    "u850_speed": ["t2m", "u850", "v850"],
}

# FCN3 variable -> (GenCast cube variable name, pressure level or None). The cube written
# by run_fcn3.py uses GenCast's names so xres.xmetrics.forecast_field scores both models'
# cubes with the exact same code path.
FCN3_TO_GENCAST = {
    "t2m":  ("2m_temperature", None),
    "u850": ("u_component_of_wind", 850),
    "v850": ("v_component_of_wind", 850),
}


def fcn3_vars(ev: Event) -> list[str]:
    return list(FCN3_VARS[ev.metric])


# --------------------------------------------------------------------------- #
# Paths. FCN3 gets its own week-namespaced tree; GenCast is read from the xres tree.
# (The pre-existing runs/fcn3/PNW_HeatDome_2021/ files are the abandoned week-2 smoke
# run and are deliberately not reused -- a week-2 IC is the wrong init for week 3.)
# --------------------------------------------------------------------------- #
FCN3_ROOT = RUNS / "fcn3"
FCN3_CACHE = FCN3_ROOT / ".cache"                     # earth2studio + HF caches (shared)


def week_dir() -> Path:
    return FCN3_ROOT / f"week{WEEKS}"


def ic_path(ev: Event) -> Path:
    return week_dir() / "ic" / f"{ev.name}_ic.nc"


def cube_path(ev: Event) -> Path:
    return week_dir() / "cache" / f"{ev.name}_cube.nc"


def timing_path(ev: Event) -> Path:
    return week_dir() / "timing" / f"{ev.name}_timing.json"


def shard_dir() -> Path:
    return week_dir() / "shards"


def shard_tag(shard: int, nshards: int) -> str:
    return f"shard{shard:02d}of{nshards:02d}"


def shard_cube_path(ev: Event, shard: int, nshards: int) -> Path:
    return shard_dir() / f"{ev.name}_{shard_tag(shard, nshards)}_cube.nc"


def shard_timing_path(ev: Event, shard: int, nshards: int) -> Path:
    return shard_dir() / f"{ev.name}_{shard_tag(shard, nshards)}_timing.json"


def shard_zarr_path(ev: Event, shard: int, nshards: int) -> Path:
    return shard_dir() / f"{ev.name}_{shard_tag(shard, nshards)}.zarr"


def gencast_cube_path(ev: Event) -> Path:
    """GenCast 0.25deg cube for this event at the comparison lead. Cached already for the
    xres events; produced by the injected run (see xres_extra_events_spec) for the p90 ones."""
    return RUNS / "xres" / RES / f"week{WEEKS}" / "cache" / f"{ev.name}_cube.nc"


def gencast_inputs_path(ev: Event) -> Path:
    return RUNS / "xres" / RES / f"week{WEEKS}" / "inputs" / f"{ev.name}_inputs.nc"


def era5_truth_path(ev: Event) -> Path:
    """ERA5 observed truth for this event's headline metric (shared xres/obs tree)."""
    return RUNS / "observations" / f"{ev.name}_verif_{ev.metric}.nc"


def model_cache_dir() -> Path:
    """Where a pre-built FCN3 model is pickled so shards/jobs load it instead of each
    rebuilding the (CPU-heavy) DISCO geometry tensors. Lives beside the checkpoint cache,
    NOT under week{N}, because the built model is lead-independent."""
    return FCN3_CACHE / "model"


def model_cache_path(tag: str) -> Path:
    """Pickled built model, filenamed by a version tag so an env upgrade (torch /
    torch-harmonics / earth2studio) never loads a stale, incompatible pickle -- a mismatch
    just misses the cache and rebuilds. ``tag`` is computed at runtime in run_fcn3.py."""
    return model_cache_dir() / f"fcn3_model_{tag}.pt"


def model_build_lock() -> Path:
    """Node-local lock so only ONE shard builds the model at a time (concurrent DISCO
    precomputes contend on memory bandwidth -- a3mega job 723: 4 builds, 17+ min, GPUs
    idle). PID-based; a dead holder's lock is stealable."""
    return model_cache_dir() / "build.lock"


def fig_dir() -> Path:
    return ROOT / "figures" / "fcn3" / f"week{WEEKS}"


def scores_path() -> Path:
    return week_dir() / "scores" / "fcn3_vs_gencast_scores.csv"


def gencast_pool_timing_path() -> Path:
    """Node-wall timing for the GenCast run over the p90 events, written by the launcher
    in ``slurm/xres_pool_0p25_week3_p90.slurm`` (xres/xinference.py is not instrumented,
    and this experiment must not modify it)."""
    return week_dir() / "timing" / "gencast_pool_timing.json"


GENCAST_STEP_H = 12
GENCAST_NSTEPS = LEAD_DAYS * 24 // GENCAST_STEP_H       # 42 at week 3

# --- GenCast reference timings ------------------------------------------------------- #
# PRIMARY (hardware-matched): Slurm job 223, `slurm/xres_pool_0p25_week3.slurm`, ONE
# a3mega 8xH100-80GB node -- the same node type, lead, member count and code path FCN3
# runs on here. logs/xres_0p25pool_wk3-223.out: start 2026-07-15 06:57:36, end 19:38:23
# => 760.8 min node wall for 12 events x 24 members, i.e. ~63.4 min/event; the worker logs
# show ~28 s per 12-hourly step (~19.6 min/member) plus a one-time ~5 min XLA compile.
# This is what the three ALREADY-CACHED xres events cost on this cluster, so the runtime
# figure can quote measured H100 numbers for all six events.
GENCAST_A3MEGA_JOB = "223"
GENCAST_A3MEGA_MINUTES_TOTAL = 760.78
GENCAST_A3MEGA_N_EVENTS = 12
GENCAST_A3MEGA_S_PER_STEP = 28.0
GENCAST_A3MEGA_GPUS = 8

# SECONDARY (cross-machine, for context only): Derecho A100-40GB, logs/6665322[0].desched1.OU
# -- ~80 s per 12-hourly step plus a ~5 min compile, i.e. ~2.9x slower per step than H100.
GENCAST_A100_S_PER_STEP = 80.0
GENCAST_A100_COMPILE_MIN = 5.0


def gencast_a3mega_minutes_per_event() -> float:
    """Measured node wall time per event for a 24-member GenCast ensemble at this lead,
    on one 8xH100 a3mega node (job 223 aggregate)."""
    return GENCAST_A3MEGA_MINUTES_TOTAL / GENCAST_A3MEGA_N_EVENTS


def gencast_a100_minutes(members: int = N_MEMBERS) -> float:
    """Archival single-A100 wall time to generate ``members`` members at this lead."""
    return (GENCAST_A100_COMPILE_MIN
            + members * GENCAST_NSTEPS * GENCAST_A100_S_PER_STEP / 60.0)


def ensure_dirs() -> None:
    for d in (week_dir() / "ic", week_dir() / "cache", week_dir() / "timing",
              shard_dir(), week_dir() / "scores", fig_dir(), model_cache_dir()):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# GenCast injection: the p90 cases are not xres events, so xres has to be told about
# them. ``xres/xconfig.py`` reads XRES_EXTRA_EVENTS and merges the spec into XRES_EVENTS
# before any filtering; the value is either a path to a JSON mapping or this inline form.
# --------------------------------------------------------------------------- #
def xres_extra_events_spec(events: list[Event] | None = None) -> str:
    """Inline ``name=peak:metric`` spec for XRES_EXTRA_EVENTS (p90 events only -- the
    xres-native events must NOT be re-declared, they already exist in XRES_EVENTS)."""
    evs = events if events is not None else [EVENTS[n] for n in ORDER]
    return ",".join(f"{e.name}={e.peak}:{e.metric}" for e in evs if e.source == "p90")


def gencast_events_to_run(events: list[Event] | None = None) -> list[Event]:
    """Events whose GenCast cube does not exist yet and must be inferred."""
    evs = events if events is not None else [EVENTS[n] for n in ORDER]
    return [e for e in evs if not gencast_cube_path(e).exists()]


def describe() -> str:
    lines = [f"FCN3 vs GenCast comparison -- week{WEEKS} ({LEAD_DAYS} d lead), "
             f"{RES}, {N_MEMBERS} members, {NSTEPS} x {STEP_H} h steps",
             ""]
    for n in ORDER:
        e = EVENTS[n]
        gc_state = "cached" if gencast_cube_path(e).exists() else "MUST RUN"
        lines.append(f"  {e.name:22s} {e.family:9s} peak {e.peak}  init "
                     f"{e.init.date()}  {e.metric:10s}  GenCast: {gc_state}")
    return "\n".join(lines)


if __name__ == "__main__":  # `python fcn3/fevents.py` prints the plan; --spec for bash
    import sys
    if "--spec" in sys.argv:
        print(xres_extra_events_spec())
    else:
        print(describe())
