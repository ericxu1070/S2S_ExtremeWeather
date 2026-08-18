# AI+RES - live state

Read `aires.md` (repo root) for the experiment design. This file is the state of the
work, updated as phases complete. **Trust the commands in "Verify state" over any prose
here** - doc claims rot, `ls` does not.

Branch: `aires`. Machine split: Derecho for CPU work, a3mega H100 for anything touching
FCN3 or GenCast inference.

Visual summary of the plan, the adapter and Gate 1 (design, derivations, open problems):
https://claude.ai/code/artifact/081bd2f9-7725-4e46-8af8-ec57f54ee2df

## Where things stand

| Phase | State |
|---|---|
| 0 - move a3mega FCN3 week-3 results to Derecho | **done**, verified: figures regenerate pixel-identical from the transferred cubes |
| 1 - adapter + calibration + Gate 1 | **done on Derecho** (CPU only) |
| 1 - Gate 2 (forecast equivalence) | **not started - needs a3mega**, FCN3 does not fit an A100-40GB |
| 1 - Gate 3 (score skill, the go/no-go) | **not started - needs a3mega + `aires/walker.py`** |
| 2-5 | not started |

## Verify state (run these, don't trust the table)

```bash
module load conda && conda activate my-env && cd /glade/derecho/scratch/exu/S2S_ExtremeWeather

ls -l runs/aires/calib/derived_calib.nc          # calibration built? (~48 MB)
PYTHONPATH=. python -m pytest aires/tests/ -q    # 25 tests, all should pass
PYTHONPATH=. python -m aires.calibrate --gate1   # re-scores Gate 1 on the held-out ICs
```

## What Phase 1 established

**The adapter is exact where it can be.** Converting a GenCast-format ERA5 frame and
comparing against the independently built FCN3 IC for the same timestamp:
**69 of 72 channels are bit-identical** (max absolute difference 0.0). The two models
share the grid, the 13 pressure levels and every unit; the only structural difference is
that GenCast's latitude ascends (-90 -> +90) and FCN3's descends. `total_precipitation_12hr`
and `sea_surface_temperature` are carried by GenCast and simply not consumed by FCN3.
FCN3 takes **one** time frame, so GenCast's 12 h step vs FCN3's 6 h step is a non-issue.

**Gate 1 PASSED** on the 6 held-out event ICs (fit on the 90 p90 ICs, disjoint):

| criterion | worst held-out | threshold |
|---|---|---|
| u100m/v100m RMSE | 0.634 m/s | 1.0 |
| u100m/v100m abs bias | 0.066 m/s | 0.2 |
| tcwv relative RMSE | 4.57% | 5% |
| tcwv abs relative bias | 0.60% | 2% |

**How the 3 derived channels work.** Wind: `W100 = c(x) * W10` with `W = u + iv` and `c`
complex, so one number carries both the boundary-layer speed-up (`|c|`, median 1.25) and
the veering (`arg c`, +-7 degrees, median ~0 globally because NH and SH veer opposite
ways). tcwv: a per-grid-point quadrature `tcwv = sum_l w_l q_l`, ridge-shrunk toward the
physical trapezoid. Learning the weights rather than scaling the trapezoid is what makes
the gate: the trapezoid's error is dominated by the missing sub-1000 hPa layer and by
underground levels over terrain, and no single scale corrects both (5.17% -> 4.57%).

## Open problems

1. **Gates 2 and 3 must run on a3mega.** Derecho A100-40GB cannot load FCN3 (~50 GiB).
   Nothing in Phase 1 beyond Gate 1 can be finished on this machine.
2. **Gate 3 needs `aires/walker.py`, which is Phase 3 work.** Scoring a walker requires a
   *global* GenCast state, and the cached xres cubes are CONUS-cropped
   (`xres/xinference.py` crops before the host transfer). The plan's phase order should be
   read as: write `walker.py` (global 2-frame checkpointing) **before** Gate 3, not after.
3. **The calibration was fitted on ERA5, but production inputs are GenCast forecasts** at
   3-15 day lead, which are smoother than analysis. Gate 3 is what measures whether that
   matters; Gate 1 cannot see it.
4. **The tcwv residual (~4%) is close to the floor** for any linear function of 13 q
   levels - the sub-1000 hPa moisture is not in the state. Whether 4% matters at all is
   Gate 2's question, not something to optimise further blind.
5. **Calibration seasonal coverage is thin in summer**: the 90 p90 ICs have 0 June and 2
   July frames. The held-out June 7 PNW case passed anyway, so it generalises, but a
   summer-heavy fit would be a cheap improvement if Gate 2 disappoints.

## Continuing on a3mega

The calibration is **deterministic given its inputs and its inputs live on a3mega too**
(the p90 ICs originated there), so rebuild rather than transfer:

```bash
# on a3mega, in the GenCast env (moe)
cd .../S2S_ExtremeWeather && git fetch && git checkout aires
PYTHONPATH=. python -m aires.calibrate --fit --gate1     # ~2 min, CPU, no network
PYTHONPATH=. python -m pytest aires/tests/ -q
```

Expect the identical Gate 1 numbers above. If they differ, the two machines' p90 IC sets
differ - check `ls runs/p90_t2m/ic/*.nc | wc -l` equals 90 on both.

Alternatively copy `runs/aires/calib/derived_calib.nc` (48 MB) by rsync **from a3mega**
(pull), the same direction convention as `scripts/sync_a3mega_to_derecho.sh`.

## Files

| Path | Role |
|---|---|
| `aires.md` | experiment design, agreed configuration, phase plan |
| `aires/aconfig.py` | the two models' contracts: 72 channels, name maps, grids, gate thresholds |
| `aires/adapter.py` | GenCast state -> FCN3 IC; `GenCastICSource` is the earth2studio DataSource |
| `aires/calibrate.py` | fits + scores the 3 derived channels; `--fit`, `--gate1` |
| `aires/tests/test_adapter.py` | 25 tests incl. the 69-channel bit-exactness check |
| `runs/aires/calib/derived_calib.nc` | fitted calibration (gitignored, rebuildable in ~2 min) |

Not yet written: `aires/walker.py`, `aires/score.py`, `aires/dmc.py`, `aires/aindex.py`,
`aires/run_aires.py`, `slurm/aires_*.slurm`.

## Notes

- `pytest` was not installed in Derecho's `my-env`; added with `pip install pytest`
  (9.1.1). The GenCast stack still imports cleanly.
- Every command above assumes `PYTHONPATH=.` from the repo root.
