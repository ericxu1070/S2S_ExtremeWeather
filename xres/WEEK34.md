# Running the week-3 and week-4 cross-resolution experiments

These are the **21-day** and **28-day** lead versions of the `xres` experiment (the
existing one is week-2, 14-day lead). Same 12 events, same two resolutions
(0.25° vs 1.0°), same verification week (the 7 days ending on each event's peak) —
only the **initialization moves earlier** (`init = peak − 21d / − 28d`) and the rollout
gets longer: **42 steps** at week 3 and **56 steps** at week 4, vs 28 at week 2.

Nothing in the Python changed for the horizon: the whole pipeline is already
parameterized by `--weeks {2,3,4}`. All outputs are week-namespaced
(`runs/xres/<res>/week{3,4}/…`, figures in `runs/xres/figures/week{3,4}/` and
`figures/xres/week{3,4}/`), so week-2/3/4 results coexist and never overwrite.

Run **week 3 and week 4 independently** — they are separate output trees. Below, set
`W=3` or `W=4` and go.

---

## Cost (why the two clusters are sized differently)

| | steps | 0.25° per member-hour | 1.0° |
|---|---|---|---|
| week 2 | 28 | baseline | baseline |
| week 3 | 42 | ~1.5× week 2 | ~1.5× |
| week 4 | 56 | ~2× week 2 | ~2× |

0.25° is the cost driver (~50 GiB VRAM/member, ~80 s/step serial). At week 4 that is
~15 h for one member across all 12 events — longer than Derecho's 12 h cap, hence the
resume note below. On the H100s it runs the fast pmap path and there is no walltime cap.

---

## A. a3mega — 8×H100 (PRIMARY, this cluster, env `moe`)

The H100-80GB fits the 0.25° model in the **pmap path** (one member/GPU, 24 members =
3 waves of 8), so both resolutions use the same 8-GPU slurm job — much faster than the
Derecho member-array path.

**Step 0 — prep on the LOGIN node** (has internet; compute nodes may not). This builds
the week-`$W` init frames for both resolutions. The ERA5/HRRR verification truth is
horizon-independent and already cached, so this only reads the 2 init frames per event.

```bash
cd /home/ubuntu/Vayuh/data/eric/S2S_ExtremeWeather
conda activate moe
python run_xres.py --stage prep --weeks 3      # week-3 init frames (both resolutions)
python run_xres.py --stage prep --weeks 4      # week-4 init frames
# verify: 12 input files per (res, week)
ls runs/xres/0p25/week3/inputs/*_inputs.nc | wc -l    # -> 12
ls runs/xres/1p0/week3/inputs/*_inputs.nc  | wc -l    # -> 12
```

**Step 1 — infer, one 8-GPU node per resolution.** Do the cheap 1.0° first to validate,
then the 0.25° cost driver. The slurm scripts re-run prep in-job (cache-aware, instant).

```bash
sbatch slurm/xres_res1p0_week3.slurm      # ~9 h  budget 12 h
sbatch slurm/xres_res0p25_week3.slurm     # cost driver, budget 36 h
sbatch slurm/xres_res1p0_week4.slurm      # ~12 h budget 18 h
sbatch slurm/xres_res0p25_week4.slurm     # cost driver, budget 48 h
squeue -u $USER
tail -f logs/xres_0p25_wk3-<jobid>.out    # per-event, per-wave progress
```

Events cube-cache one at a time, so a walltime kill (or a lost H100 reservation)
preserves finished events — just `sbatch` the same file again to resume from the next
event. Cubes land in `runs/xres/<res>/week$W/cache/`, verif in `.../verif/`.

**Step 2 — compare on the LOGIN node**, once BOTH resolutions of a week are done:

```bash
python run_xres.py --stage compare --weeks 3
python run_xres.py --stage compare --weeks 4
# -> runs/xres/figures/week{3,4}/  and  figures/xres/week{3,4}/
#    (maps/, xres_combined_*.png, xres_brier/crps/rank_hist*.png, xres_scores.csv)
```

---

## B. Derecho — 4×A100-40GB (FALLBACK, PBS, env `my-env`)

Not reachable from the a3mega box — run these in a Derecho session
(`/glade/derecho/scratch/exu/S2S_ExtremeWeather`). The 0.25° model does NOT fit the pmap
path on A100-40GB, so both resolutions run as **24-subjob single-GPU member arrays**
(subjob N rolls member N of all 12 events, cached to
`cache/<event>_memberNN.nc`; the last finisher assembles each cube). Account walltime cap
is 12 h.

**Step 0 — prep** (CPU develop node, internet):

```bash
cd /glade/derecho/scratch/exu/S2S_ExtremeWeather
qsub -v XRES_WEEKS=3 pbs/xres_prep.pbs
qsub -v XRES_WEEKS=4 pbs/xres_prep.pbs
```

**Step 1 — week 3 infer + compare** (0.25° fits in ~11.5 h, 1.0° in ~4.5 h):

```bash
A025=$(qsub pbs/xres_member_0p25_week3.pbs)
A10=$(qsub pbs/xres_member_1p0_week3.pbs)
qsub -W depend=afterok:$A025:$A10 -v XRES_WEEKS=3 pbs/xres_compare_dev.pbs
```

**Step 2 — week 4 infer.** 0.25° week 4 (~15 h) EXCEEDS the 12 h cap on purpose: each
subjob caches every finished `(event, member)`, gets killed after ~9/12 events, and
**resubmitting the array resumes** (cached members skip in ~2 min without loading the
model). Budget ~2 array submissions for 0.25° week 4. 1.0° week 4 (~6 h) fits in one.

```bash
qsub pbs/xres_member_1p0_week4.pbs
qsub pbs/xres_member_0p25_week4.pbs
# after the 0.25° array walltime-kills partway, resubmit to finish the rest:
qsub pbs/xres_member_0p25_week4.pbs
# when all 12 cubes exist (check below), run compare manually (dependency chains don't
# survive a resubmit, so don't afterok-chain week-4 0.25°):
qsub -v XRES_WEEKS=4 pbs/xres_compare_dev.pbs
```

Resubmitting any array is always cheap and safe — completed members and cubes are skipped.

---

## Completion / monitoring checks (either cluster)

```bash
W=3                       # or 4
for R in 0p25 1p0; do
  echo "$R week$W: cubes=$(ls runs/xres/$R/week$W/cache/*_cube.nc 2>/dev/null | wc -l)/12" \
       "verif=$(ls runs/xres/$R/week$W/verif/*_members.nc 2>/dev/null | wc -l)/18" \
       "pending-members=$(ls runs/xres/$R/week$W/cache/*_member[0-9]*.nc 2>/dev/null | wc -l)"
done
```

Done = **12 cubes and 18 verif files** per resolution (18 = 6 T2m + 3×2 u850 + 3×2 tp).
`pending-members` > 0 means an event is still assembling (some member arrays/waves
outstanding) — resubmit the relevant infer job to finish it.
