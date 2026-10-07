"""Static contracts and paths for the AI+RES stack.

Two models with two different state representations meet here, so this module is the
single place that states how they line up. Everything downstream imports these names
rather than re-deriving them.

    GenCast (JAX/GraphCast)                FourCastNet3 (PyTorch/earth2studio)
    ---------------------------            -----------------------------------
    xr.Dataset, named variables            xr.DataArray, 72 stacked channels
    lat ASCENDING  -90 -> +90              lat DESCENDING +90 -> -90
    lon 0 -> 359.75                        lon 0 -> 359.75            (same)
    13 levels, 50..1000 hPa                13 levels, 50..1000 hPa    (same)
    K / Pa / m s-1 / m2 s-2 / kg kg-1      identical units            (same)

So the conversion is a rename plus a latitude flip plus three derived channels
(u100m, v100m, tcwv) that GenCast does not carry. FCN3 needs exactly ONE time frame,
which is why GenCast's 12 h step versus FCN3's 6 h step is not an issue.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from gencast_s2s import config as C

# --------------------------------------------------------------------------- #
# FCN3's channel contract.
#
# The authoritative list lives in earth2studio (``models/px/fcn3.py::VARIABLES``),
# which is only importable in the ``fcn3`` conda env. Calibration and Gate 1 run in
# the GenCast env, where it is not, so the list is mirrored here and checked against
# the package whenever the package IS importable (``verify_against_package``) and
# against any on-disk IC file (``verify_against_ic``). Do not edit by hand.
# --------------------------------------------------------------------------- #
P13 = tuple(C.P13)                      # (50, 100, ..., 1000) - shared by both models
LEVEL_PREFIXES = ("u", "v", "z", "t", "q")
SURFACE_VARS = ("u10m", "v10m", "u100m", "v100m", "t2m", "msl", "tcwv")

FCN3_VARIABLES = tuple(SURFACE_VARS) + tuple(
    f"{pfx}{lev}" for pfx in LEVEL_PREFIXES for lev in P13
)
assert len(FCN3_VARIABLES) == 72, len(FCN3_VARIABLES)

# FCN3 grid. earth2studio asserts these exactly; a flipped or rolled axis is silent
# garbage rather than an error, so the adapter checks them on every conversion.
FCN3_NLAT, FCN3_NLON = 721, 1440

# --------------------------------------------------------------------------- #
# GenCast -> FCN3 name maps. Direct, 1:1, no unit conversion anywhere.
# --------------------------------------------------------------------------- #
SURFACE_FROM_GENCAST = {
    "u10m": "10m_u_component_of_wind",
    "v10m": "10m_v_component_of_wind",
    "t2m":  "2m_temperature",
    "msl":  "mean_sea_level_pressure",
}
LEVEL_FROM_GENCAST = {
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "z": "geopotential",
    "t": "temperature",
    "q": "specific_humidity",
}
# The only channels GenCast cannot supply directly; see aires/adapter.py.
DERIVED_VARS = ("u100m", "v100m", "tcwv")

# GenCast state variables the adapter actually reads (the rest of GenCast's target
# set - vertical_velocity, sea_surface_temperature, total_precipitation_12hr - is
# carried by the walker but never consumed by FCN3).
GENCAST_NEEDED = tuple(SURFACE_FROM_GENCAST.values()) + tuple(LEVEL_FROM_GENCAST.values())

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = C.ROOT
RUNS = Path(os.environ.get("GENCAST_RUNS", ROOT / "runs"))
AIRES_ROOT = RUNS / "aires"

# Calibration for the three derived channels, fitted once from ERA5 (see calibrate.py).
CALIB_PATH = Path(os.environ.get("AIRES_CALIB", AIRES_ROOT / "calib" / "derived_calib.nc"))

# ERA5 sources for calibration/validation. Both are FCN3 IC files (all 72 channels,
# including the true u100m/v100m/tcwv), so no network access is needed.
CALIB_IC_GLOB = str(RUNS / "p90_t2m" / "ic" / "*.nc")            # 90 cases, 2021-2025: FIT
HOLDOUT_IC_GLOB = str(RUNS / "fcn3" / "week3" / "ic" / "*_ic.nc")  # the 6 events: TEST

# Gate 1 pass criteria (aires.md). Evaluated on HOLDOUT_IC_GLOB only.
GATE1 = {
    "u100m_rmse_max": 1.0,      # m/s
    "u100m_abs_bias_max": 0.2,  # m/s
    "tcwv_rel_rmse_max": 0.05,  # fraction
    "tcwv_abs_rel_bias_max": 0.02,
}


# --------------------------------------------------------------------------- #
# Walker configuration and the per-event run tree.
#
# A walker is a GenCast trajectory that can be STOPPED, checkpointed and restarted, so
# unlike the xres cubes its checkpoint must be GLOBAL: FCN3 consumes the whole sphere, and
# a CONUS-cropped state cannot initialise it (xres/xinference.py crops before the host
# transfer, which is why those cached cubes are useless here).
#
#     runs/aires/<event>/walkers/w<ii>/step<k>/state.nc   global 2-frame restart state
#                                             /diag.nc    CONUS-cropped, every step
#                       /scores/                          FCN3 scores theta(t_k)
#                       /gate3/                           Gate 3 artifacts
# --------------------------------------------------------------------------- #
WALKER_RES = os.environ.get("AIRES_RES", "0p25")   # GenCast checkpoint resolution


def walker_model_kwargs(res: str | None = None) -> dict:
    """The checkpoint arguments for the walker's GenCast. **Never build these by hand.**

    ``gencast_s2s.config.model_cfg`` uses its ``res`` argument ONLY for the ERA5
    coarsening stride. The checkpoint it loads is ``params_file or GENCAST_PARAMS``, and
    ``GENCAST_PARAMS`` is the **1.0 degree Mini** model. So

        M.load_gencast(n_members=1, res=0.25)          # WRONG - silently loads Mini

    loads Mini and rolls it on 0.25 degree fields. It does not raise: GraphCast builds its
    mesh from whatever grid the inputs carry, so the run looks healthy and the output is
    physically wrong - in the first Gate 3 attempt (job 1155) the CONUS-mean T2m lost its
    diurnal cycle within one 3-day segment and drifted 26 K cold over 21 days, and the
    only reason it was caught is that the realized A_L came out at -22 K.

    ``xres`` gets this right by passing ``params_file=rs["params"]`` from its resolution
    registry (``xres/xconfig.py::RES_SPECS``). This function reads the SAME registry, so
    the walker and the xres cubes it is compared against cannot diverge.
    """
    from xres import xconfig as X

    spec = X.res_spec(res or WALKER_RES)
    return dict(params_file=spec["params"], res=spec["res"])
WALKER_STEP_H = 12                                 # GenCast native step
WALKER_WEEKS = int(os.environ.get("AIRES_WEEKS", 3))  # lead of the pilot init (21 d)

# Base seed for walker segment RNGs. A segment's key is fold_in(fold_in(base, step),
# walker), so a cloned walker gets a stream no sibling has ever used -- which IS the
# clone-perturbation mechanism (the paper has to hand-tune spherical-harmonic noise
# because its walker, PlaSim, is deterministic; GenCast's diffusion noise does it for us).
WALKER_BASE_SEED = int(os.environ.get("AIRES_BASE_SEED", 20260101))

# Global state grid, asserted on every read/write. Same numbers as FCN3's, deliberately:
# the walker state feeds the adapter directly.
STATE_NLAT, STATE_NLON = FCN3_NLAT, FCN3_NLON

# The 12 time-varying variables GenCast predicts, plus the 2 statics. A restart state must
# carry all 14: the prognostics are what the rollout advances, the statics are what
# data_utils needs to rebuild the input batch.
STATE_PROGNOSTIC = (
    "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature",
    "geopotential", "mean_sea_level_pressure", "sea_surface_temperature",
    "specific_humidity", "temperature", "total_precipitation_12hr",
    "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
)
STATE_STATIC = ("geopotential_at_surface", "land_sea_mask")

# Gate 3 (aires.md): free-running walkers checkpointed at these leads, each scored by FCN3;
# the score must rank the eventual outcome.
GATE3_LEAD_DAYS = (3, 6, 9, 12, 15)
GATE3 = {
    "spearman_min": 0.5,          # required at t_k = 9 d AND 12 d
    "required_leads": (9, 12),
}

# Walker segmentation for Gate 3. Every segment is the SAME number of steps on purpose:
# GenCast's single-step function is jitted once per process, and keeping the segment shape
# constant guarantees one XLA compile (~5 min) per worker rather than one per distinct
# segment length. 6 steps x 12 h = 3 d, so the checkpoint leads are 3, 6, ... and
# GATE3_LEAD_DAYS is a subset of them; the extra states at 18 d and 21 d are free.
GATE3_SEG_STEPS = 6
GATE3_HORIZON_DAYS = WALKER_WEEKS * 7                 # 21 d - the event peak
GATE3_SEG_DAYS = GATE3_SEG_STEPS * WALKER_STEP_H / 24  # 3.0

GATE3_N_WALKERS = int(os.environ.get("AIRES_GATE3_WALKERS", 16))
GATE3_M_MEMBERS = int(os.environ.get("AIRES_GATE3_MEMBERS", 6))


def gate3_segments(horizon_days: int | None = None) -> list[tuple[int, float, int]]:
    """``[(step, lead_days at the END of the segment, n_steps), ...]`` for one walker.

    Step numbering starts at 1 and matches ``segment_dir``/``state_path``, so
    ``state_path(ev, w, k)`` is the walker's state at lead ``k * GATE3_SEG_DAYS``.
    """
    horizon = GATE3_HORIZON_DAYS if horizon_days is None else horizon_days
    n = int(round(horizon / GATE3_SEG_DAYS))
    if abs(n * GATE3_SEG_DAYS - horizon) > 1e-9:
        raise ValueError(f"horizon {horizon} d is not a whole number of "
                         f"{GATE3_SEG_DAYS} d segments")
    return [(k, k * GATE3_SEG_DAYS, GATE3_SEG_STEPS) for k in range(1, n + 1)]


def gate3_step_for_lead(lead_days: float) -> int:
    """Which segment index ends at ``lead_days``. Raises if the lead is not a checkpoint."""
    k = lead_days / GATE3_SEG_DAYS
    if abs(k - round(k)) > 1e-9 or round(k) < 1:
        raise ValueError(f"lead {lead_days} d is not a walker checkpoint "
                         f"(segments are {GATE3_SEG_DAYS} d)")
    return int(round(k))


# --------------------------------------------------------------------------- #
# The production AI+RES run (Phase 3+).
#
# Settled by Phase 2 and NOT to be re-litigated (aires/ctune.py replayed the schedule on
# the measured Gate 3 skill curve): C_k = (0, 1.0, 1.4, 1.8, 2.0) at t_k = 3/6/9/12/15 d,
# N = 64, M = 6.
#
# The run tree is DISJOINT from Gate 3's on purpose:
#
#     runs/aires/<event>/walkers/w03/step03/     Gate 3: 16 walkers, free-running
#     runs/aires/<event>/res/<tag>/walkers/...   AI+RES: N slots, RESAMPLED
#
# Both index walkers by slot, but only Gate 3's slots are lineages. Under resampling a
# slot is a container that different ancestries pass through - w03 at segment 3 may be a
# clone of w41 - so writing an AI+RES population into the Gate 3 paths would overwrite the
# gate's artifacts with something that is not a free-running walker, and
# `gate3 --stage reduce` (the Phase 1 regression test) would silently start reporting a
# tilted population. The genealogy that says which slot held which ancestry lives in
# `steps/leg<kk>/walk.json`, and nothing may be read from the tree without it.
# --------------------------------------------------------------------------- #
RES_TAG = os.environ.get("AIRES_RES_TAG", "pilot")
RES_N_WALKERS = int(os.environ.get("AIRES_N_WALKERS", 64))
RES_M_MEMBERS = int(os.environ.get("AIRES_M_MEMBERS", 6))
RES_C = tuple(float(x) for x in
              os.environ.get("AIRES_C", "0,1.0,1.4,1.8,2.0").split(","))
RES_LEAD_DAYS = tuple(float(x) for x in
                      os.environ.get("AIRES_RES_LEADS", "3,6,9,12,15").split(","))
RES_HORIZON_DAYS = float(os.environ.get("AIRES_HORIZON_DAYS", GATE3_HORIZON_DAYS))
# 21 d - the event peak. Overridable only so a smoke run can stop short; a production
# run whose horizon is not the peak cannot compute A_L at all (the verification window
# is the 7 days ENDING at the peak), and --stage compare will say so.
RES_SEG_STEPS = GATE3_SEG_STEPS                # 6 x 12 h = 3 d, one XLA compile per worker

# The DMC draw (pivotal sampling) is seeded once and replayed. A resumed coordinator
# re-runs the whole loop; every score it reads is already on disk, so the same seed plus
# the same artifacts reproduce the same clone/kill decisions exactly. That is what makes
# "resubmit the job" a no-op for finished work rather than a different experiment.
RES_DMC_SEED = int(os.environ.get("AIRES_DMC_SEED", 20260819))

# How many trailing segments of walker states to keep on disk. Each is 477 MB and there
# are N of them per segment, so N=64 x 7 segments is 213 GB - against ~600 GB free on a
# 95%-full shared NFS (measured 2026-08-19). Rolling segment s needs only segment s-1, so
# 2 is sufficient and costs 61 GB; <= 0 keeps everything. Diagnostics cubes are NEVER
# pruned - they are 35 MB each (measured on the pilot; the design estimate said 6) and
# every observable in the experiment is derived from them.
RES_KEEP_STATES = int(os.environ.get("AIRES_KEEP_STATES", 2))

if len(RES_C) != len(RES_LEAD_DAYS):
    raise ValueError(f"AIRES_C has {len(RES_C)} coefficients but there are "
                     f"{len(RES_LEAD_DAYS)} resampling times {RES_LEAD_DAYS}")


@dataclass(frozen=True)
class Leg:
    """One propagation leg: what the walkers do between two resampling times.

    ``dmc.run`` calls ``propagate`` once per leg and there is ALWAYS one more leg than
    there are resampling times - the last one carries the survivors of the final
    resampling from ``t_K`` to the verification horizon ``t_f``, and it is not optional
    (see ``aires/dmc.py``). ``C`` is ``None`` on exactly that leg.

    A leg holds one or more 3-day SEGMENTS. Segments, not legs, are the unit on disk and
    the unit GenCast is compiled for: every segment is ``RES_SEG_STEPS`` steps so a worker
    pays one XLA compile rather than one per distinct leg length. The 15 -> 21 d carry is
    therefore two segments, not a 12-step leg.
    """
    k: int
    segments: tuple[int, ...]
    lead_start_days: float
    lead_end_days: float
    C: float | None

    @property
    def resampling(self) -> bool:
        return self.C is not None

    @property
    def first_segment(self) -> int:
        return self.segments[0]

    @property
    def last_segment(self) -> int:
        return self.segments[-1]

    def lead_at(self, segment: int) -> float:
        """Lead in days at the END of ``segment`` (segments are RES_SEG_DAYS apart)."""
        return segment * GATE3_SEG_DAYS


def res_legs(leads=None, horizon_days: float | None = None, C=None) -> list[Leg]:
    """The full leg schedule, resampling legs first and the carry leg last."""
    leads = tuple(float(x) for x in (RES_LEAD_DAYS if leads is None else leads))
    C = tuple(float(x) for x in (RES_C if C is None else C))
    horizon = RES_HORIZON_DAYS if horizon_days is None else float(horizon_days)
    if len(C) != len(leads):
        raise ValueError(f"{len(C)} coefficients for {len(leads)} resampling times")
    if list(leads) != sorted(leads) or len(set(leads)) != len(leads):
        raise ValueError(f"resampling times must be strictly increasing: {leads}")
    if leads and leads[-1] >= horizon:
        raise ValueError(f"the last resampling time {leads[-1]} d must precede the "
                         f"horizon {horizon} d; a run whose last resampling sits ON the "
                         f"horizon has no leg left to carry the survivors")
    bounds = [0.0, *leads, horizon]
    out = []
    for k in range(1, len(bounds)):
        lo, hi = bounds[k - 1], bounds[k]
        s0, s1 = gate3_step_for_lead(lo) + 1 if lo else 1, gate3_step_for_lead(hi)
        out.append(Leg(k=k, segments=tuple(range(s0, s1 + 1)),
                       lead_start_days=lo, lead_end_days=hi,
                       C=C[k - 1] if k - 1 < len(C) else None))
    return out


# --------------------------------------------------------------------------- #
# The run tree
# --------------------------------------------------------------------------- #
def res_dir(event: str, tag: str | None = None) -> Path:
    return event_dir(event) / "res" / (tag or RES_TAG)


def res_walker_dir(event: str, walker: int, tag: str | None = None) -> Path:
    return res_dir(event, tag) / "walkers" / f"w{walker:02d}"


def res_segment_dir(event: str, walker: int, segment: int, tag: str | None = None) -> Path:
    return res_walker_dir(event, walker, tag) / f"step{segment:02d}"


def res_state_path(event: str, walker: int, segment: int, tag: str | None = None) -> Path:
    return res_segment_dir(event, walker, segment, tag) / "state.nc"


def res_diag_path(event: str, walker: int, segment: int, tag: str | None = None) -> Path:
    return res_segment_dir(event, walker, segment, tag) / "diag.nc"


def res_score_cube_path(event: str, walker: int, lead_days: float,
                        tag: str | None = None) -> Path:
    return res_dir(event, tag) / "scores" / f"w{walker:02d}_lead{int(lead_days):02d}_cube.nc"


def res_score_zarr_path(event: str, walker: int, lead_days: float,
                        tag: str | None = None) -> Path:
    return res_dir(event, tag) / "scores" / f"w{walker:02d}_lead{int(lead_days):02d}.zarr"


def res_leg_dir(event: str, leg: int, tag: str | None = None) -> Path:
    return res_dir(event, tag) / "steps" / f"leg{leg:02d}"


def res_log_dir(event: str, tag: str | None = None) -> Path:
    return res_dir(event, tag) / "logs"


def res_ensure_dirs(event: str, tag: str | None = None) -> None:
    ensure_dirs(event)
    for d in (res_dir(event, tag), res_dir(event, tag) / "walkers",
              res_dir(event, tag) / "scores", res_dir(event, tag) / "steps",
              res_log_dir(event, tag)):
        d.mkdir(parents=True, exist_ok=True)


# The segment index is a pure function of the lead (segments are all RES_SEG_DAYS long),
# so Gate 3 and AI+RES agree on it by construction. The neutral name is what non-gate code
# should use.
segment_for_lead = gate3_step_for_lead


def score_cube_path(event: str, walker: int, lead_days: float) -> Path:
    """The FCN3 score forecast launched from walker ``walker``'s state at ``lead_days``."""
    return event_dir(event) / "scores" / f"w{walker:02d}_lead{int(lead_days):02d}_cube.nc"


def score_zarr_path(event: str, walker: int, lead_days: float) -> Path:
    return event_dir(event) / "scores" / f"w{walker:02d}_lead{int(lead_days):02d}.zarr"


def gate3_dir(event: str) -> Path:
    return event_dir(event) / "gate3"


def fig_dir(event: str | None = None) -> Path:
    """Figures live one folder per event; pass the event (or adapter-test case)
    name. Only cross-event aggregates - which belong to no single event - are
    written to the top level, i.e. with `event=None`."""
    base = ROOT / "figures" / "aires"
    return base if event is None else base / event


def event_dir(event: str) -> Path:
    return AIRES_ROOT / event


def walker_dir(event: str, walker: int) -> Path:
    return event_dir(event) / "walkers" / f"w{walker:02d}"


def segment_dir(event: str, walker: int, step: int) -> Path:
    return walker_dir(event, walker) / f"step{step:02d}"


def state_path(event: str, walker: int, step: int) -> Path:
    """Global 2-frame restart state at the END of segment ``step``."""
    return segment_dir(event, walker, step) / "state.nc"


def diag_path(event: str, walker: int, step: int) -> Path:
    """CONUS-cropped every-step diagnostics for segment ``step``."""
    return segment_dir(event, walker, step) / "diag.nc"


def gencast_inputs_path(event: str, res: str | None = None, weeks: int | None = None) -> Path:
    """The event's ERA5 init frames, in GenCast schema -- a walker's step-0 state.

    Built by the xres pipeline; global and 2-frame already, which is exactly the walker
    checkpoint format, so step 0 needs no conversion.
    """
    res = res or WALKER_RES
    weeks = WALKER_WEEKS if weeks is None else weeks
    return RUNS / "xres" / res / f"week{weeks}" / "inputs" / f"{event}_inputs.nc"


def ensure_dirs(event: str | None = None) -> None:
    dirs = [AIRES_ROOT, CALIB_PATH.parent, fig_dir()]
    if event:
        dirs += [event_dir(event), event_dir(event) / "walkers",
                 event_dir(event) / "scores", event_dir(event) / "gate3"]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Contract checks
# --------------------------------------------------------------------------- #
def verify_against_package() -> None:
    """Assert our mirrored channel list matches earth2studio's. No-op if not importable."""
    try:
        from earth2studio.models.px.fcn3 import VARIABLES as PKG
    except Exception:
        return
    if tuple(PKG) != FCN3_VARIABLES:
        raise AssertionError(
            "FCN3_VARIABLES drifted from earth2studio.\n"
            f"  package: {list(PKG)}\n  aires  : {list(FCN3_VARIABLES)}"
        )


def verify_against_ic(path) -> None:
    """Assert an on-disk FCN3 IC file has exactly our channels, in order."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        got = tuple(str(s) for s in ds["variable"].values)
        if got != FCN3_VARIABLES:
            raise AssertionError(f"channel mismatch vs {path}:\n  file : {got}\n  aires: {FCN3_VARIABLES}")
        if ds.sizes["lat"] != FCN3_NLAT or ds.sizes["lon"] != FCN3_NLON:
            raise AssertionError(f"grid mismatch vs {path}: {dict(ds.sizes)}")
        if ds["lat"].values[0] < ds["lat"].values[-1]:
            raise AssertionError(f"{path} has ascending lat; FCN3 expects 90 -> -90")


# --------------------------------------------------------------------------- #
# The CFSv2 operational baseline (aires/cfs.py).
#
# Lives beside the run rather than inside a tag: the cube is a property of the EVENT and
# its lead, not of an AI+RES schedule, so every tag of an event reads the same file.
# --------------------------------------------------------------------------- #
def cfs_dir(event: str) -> Path:
    return event_dir(event) / "cfs"


def cfs_cube_path(event: str, lead_days: float, metric: str) -> Path:
    return cfs_dir(event) / f"{event}_cfs_lead{int(round(lead_days)):02d}_{metric}.nc"


def cfs_json_path(event: str, lead_days: float, metric: str) -> Path:
    return cfs_dir(event) / f"{event}_cfs_lead{int(round(lead_days)):02d}_{metric}.json"
