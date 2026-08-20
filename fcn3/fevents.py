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

Null controls
-------------
``CONTROLS`` adds four quiet days -- CONUS-mean T2m anomaly within +-0.2 K of
climatology -- for the adapter test only. See the block above ``CONTROLS`` for why they
are a separate dict and why they never enter ``selected()`` by default.
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
    family: str          # heat | cold | hurricane | p90 | null
    label: str           # short human label for figure panels
    source: str          # "xres"   -> GenCast cube already cached
                         # "p90"    -> injected into xres via XRES_EXTRA_EVENTS, must be run
                         # "p90ctl" -> null control; injected the same way, but GenCast is
                         #             never run for it (the adapter test needs the INIT
                         #             frames, not a rollout)
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


# --------------------------------------------------------------------------- #
# Null controls -- the adapter test only.
#
# The six events above are all, by construction, atmospheric extremes. That makes them a
# biased sample for the question "does the adapter cost skill?": every one of them is a
# case where the forecast is hard, the ensemble spread is large, and a small IC
# perturbation has an unusually good chance of growing. If the adapter penalty is a
# generic property of the added IC error it must show up on quiet days too; if it only
# appears on extremes, that is a different (and more interesting) finding. Neither can be
# said without cases where nothing is happening.
#
# These are drawn from the 45 month-matched non-event days already frozen in
# ``p90/cases.csv`` (kind == "control"), taking the four whose CONUS-mean daily T2m
# anomaly is closest to zero subject to two constraints. "No anomaly" is meant literally:
# every one is inside +-0.33 K of climatology on the peak day, against the +2.3662 K p90
# threshold that defines an event in that set.
#
#   name           peak        CONUS-mean anom   season       p90 case
#   null_20230324  2023-03-24   +0.129 K         spring       control_38_20230324
#   null_20240329  2024-03-29   +0.033 K         spring       control_04_20240329
#   null_20241121  2024-11-21   +0.323 K         autumn       control_16_20241121
#   null_20250526  2025-05-26   +0.198 K         late spring  control_05_20250526
#
# CONSTRAINT 1 -- both the init and the verification week must fall after
# ``gencast_s2s.data.WB2_END`` (2023-01-10). Before that date ERA5 is read from the
# WeatherBench2 zarr, and a single ``.sel(time=t)`` on that store makes dask build a task
# graph big enough to OOM this 15 GB login node (measured: 7.2 GB and still climbing on
# ONE event, killed twice). After it the reader falls through to ``arco_read``, which is
# plain numpy per frame and costs a few hundred MB -- the same path the three p90 cases
# took when they were prepped on this machine. This is a property of where the data is
# read from, not of the case, and it is why the flattest control in the whole set
# (2022-11-23, +0.001 K) is NOT used: its init is 2022-11-02, three months the wrong side
# of the cutoff. On a box with more RAM the pre-2023 controls are perfectly usable.
#
# CONSTRAINT 2 -- no two controls within 30 days of each other, so the four are four
# independent draws for the sign test rather than two pairs. (control_06_20241122 has an
# almost identical anomaly to control_16_20241121 and is the very next day; taking both
# would have been double-counting one quiet week.)
#
# WHAT "NO ANOMALY" DOES AND DOES NOT MEAN. The p90 index is the cos(lat)-weighted CONUS
# MEAN daily T2m anomaly, so a control is a day whose CONUS mean is near zero -- NOT a day
# with no weather. null_20240329 is the clearest example: its CONUS mean is +0.033 K, but
# the field is a strong dipole, about -8 K over the northern plains against widespread
# warmth elsewhere (see figures/aires/adaptest_maps_null_20240329.png). That is the right
# control for this test -- it is a day nobody would have selected as an extreme, initialised
# and scored exactly like one -- but it is not a quiescent atmosphere, and the pooled
# statistic must not be described as one.
#
# What this costs: there is no SUMMER control. Every summer non-event day in the frozen
# p90 control set falls in 2022, before the cutoff. Two of the six extreme events are
# warm-season (the PNW dome in June, p90_20240802 in August), so the extreme-vs-null
# contrast is NOT season-matched and must not be read as one.
#
# They are a SEPARATE dict, not extra entries in EVENTS, because ``F.selected()`` is what
# ``fcn3/run_fcn3.py`` and ``fcn3/compare_fcn3_gencast.py`` iterate over and the FCN3
# vs GenCast comparison must keep meaning exactly what it meant: six extremes, each with a
# GenCast cube to be compared against. A control has no GenCast cube and never needs one
# (the adapter test reads GenCast INIT FRAMES, not rollouts), so putting it in EVENTS
# would silently add a broken seventh panel to that experiment's figures.
#
# ``source = "p90ctl"`` marks them as needing the same additive XRES_EXTRA_EVENTS
# injection the p90 events get, while keeping them out of the p90 GenCast launcher, which
# selects on ``source == "p90"`` exactly.
# --------------------------------------------------------------------------- #
CONTROLS: dict[str, Event] = {
    "null_20230324": Event(
        "null_20230324", "2023-03-24", "t2m_anom", "null",
        "null 2023-03-24 (+0.1 K)", "p90ctl",
        "p90 control_38_20230324; CONUS-mean anomaly +0.129 K, spring"),
    "null_20240329": Event(
        "null_20240329", "2024-03-29", "t2m_anom", "null",
        "null 2024-03-29 (+0.0 K)", "p90ctl",
        "p90 control_04_20240329; CONUS-mean anomaly +0.033 K -- the flattest quiet day "
        "available on the post-WB2 read path"),
    "null_20241121": Event(
        "null_20241121", "2024-11-21", "t2m_anom", "null",
        "null 2024-11-21 (+0.3 K)", "p90ctl",
        "p90 control_16_20241121; CONUS-mean anomaly +0.323 K, autumn"),
    "null_20250526": Event(
        "null_20250526", "2025-05-26", "t2m_anom", "null",
        "null 2025-05-26 (+0.2 K)", "p90ctl",
        "p90 control_05_20250526; CONUS-mean anomaly +0.198 K, late spring"),
}

CONTROL_ORDER = tuple(CONTROLS)

# The p90 control case each null was taken from (provenance; p90/cases.csv is frozen).
P90_CONTROL_CASE = {
    "null_20230324": "control_38_20230324",
    "null_20240329": "control_04_20240329",
    "null_20241121": "control_16_20241121",
    "null_20250526": "control_05_20250526",
}

# Everything the adapter test may score. Controls are APPENDED, never interleaved: seeds
# are ``BASE_SEED + 1000 * ALL_ORDER.index(name)``, so inserting a control ahead of an
# existing event would renumber it and stop its adapter members from pairing with the
# native cube already on disk. Indices 0..5 are frozen.
ALL_EVENTS: dict[str, Event] = {**EVENTS, **CONTROLS}
ALL_ORDER = ORDER + CONTROL_ORDER


def selected() -> list[Event]:
    """Events for this run. ``FCN3_EVENTS`` (comma-separated names) subsets them.

    The DEFAULT is the six head-to-head events and nothing else, because this is what the
    FCN3-vs-GenCast experiment iterates over and a control has no GenCast cube to be
    compared against. A control still has to be *reachable*, though -- the adapter test
    rolls its native FCN3 ensemble with this same driver -- so an explicitly named control
    in ``FCN3_EVENTS`` resolves. Naming one is opt-in; getting one by accident is not
    possible.
    """
    sel = os.environ.get("FCN3_EVENTS")
    if not sel:
        return [EVENTS[n] for n in ORDER]
    want = [s.strip() for s in sel.split(",") if s.strip()]
    unknown = [w for w in want if w not in ALL_EVENTS]
    if unknown:
        raise SystemExit(f"unknown event(s) in FCN3_EVENTS: {', '.join(unknown)}\n"
                         f"known: {', '.join(ALL_ORDER)}")
    return [ALL_EVENTS[w] for w in want]


def seed_for(ev: Event) -> int:
    """Deterministic per-event base seed (offset by position in the frozen order).

    Indexed over ``ALL_ORDER``, whose first six entries ARE ``ORDER`` -- so every seed the
    six native cubes on disk were rolled with is unchanged, and the adapter ensembles keep
    pairing with them member for member. Controls occupy indices 6+.
    """
    return BASE_SEED + 1000 * ALL_ORDER.index(ev.name)


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
    """Inline ``name=peak:metric`` spec for XRES_EXTRA_EVENTS.

    Emits every event in ``events`` that xres does not already know about -- i.e. anything
    whose source is not ``"xres"``. The xres-native events must NOT be re-declared, they
    already exist in XRES_EVENTS.

    The default is still the six head-to-head events, so ``python fcn3/fevents.py --spec``
    prints exactly the three p90 entries it always printed and the frozen p90 GenCast
    launcher is unaffected. Pass ``controls()`` to get the null controls' spec instead --
    that is what ``scripts/adaptest_prep.sh`` does to teach the xres prep pipeline about
    them.
    """
    evs = events if events is not None else [EVENTS[n] for n in ORDER]
    return ",".join(f"{e.name}={e.peak}:{e.metric}" for e in evs if e.source != "xres")


def controls() -> list[Event]:
    """The null controls, in registry order."""
    return [CONTROLS[n] for n in CONTROL_ORDER]


def all_events() -> list[Event]:
    """Six extremes then the null controls -- everything the adapter test scores."""
    return [ALL_EVENTS[n] for n in ALL_ORDER]


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
    lines += ["", f"  null controls (adapter test only; no GenCast cube is needed):"]
    for n in CONTROL_ORDER:
        e = CONTROLS[n]
        lines.append(f"  {e.name:22s} {e.family:9s} peak {e.peak}  init "
                     f"{e.init.date()}  {e.metric:10s}  {e.note}")
    return "\n".join(lines)


if __name__ == "__main__":  # `python fcn3/fevents.py` prints the plan; --spec for bash
    import sys
    if "--spec" in sys.argv:
        print(xres_extra_events_spec())
    elif "--controls-spec" in sys.argv:
        print(xres_extra_events_spec(controls()))
    elif "--controls" in sys.argv:
        print(",".join(CONTROL_ORDER))
    else:
        print(describe())
