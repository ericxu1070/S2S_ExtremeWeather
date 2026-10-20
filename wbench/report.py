#!/usr/bin/env python
"""Aggregate a walker-benchmark run into REPORT.md + results.json + results.csv.

Pure stdlib over the JSON records the workers wrote, so it runs in either env (or with
any python3) on the VM or after copying ``results/`` home.

Layer 3, the case projection, prices one AI+RES case from measured rates:

    walk   = N x horizon_days x (s per walker-simulated-day, segment wall, node8 phase)
    start  = (load + compile/first step) x 8 GPUs x legs      [relaunch per leg, as today]
             (load + compile/first step) x 8 GPUs             [persistent workers]
    score  = N x M x sum over scored leads (C_k != 0) of FCN3 steps to the horizon
             x (s per FCN3 member-step)

The score phase is the same whichever model walks (FCN3 scores either way), so it is
reported once and added to both totals. Fed the a3mega rates, the walk + score terms
reproduce the pilot's ~36 + ~25 H100-h (``tests/test_report.py`` pins that), which is the
check that the formula prices the run the project actually does.

    python -m wbench.report --run runs/wbench/<id>
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wbench import timing as T
from wbench import wconfig as W

PHASE_SOLO, PHASE_NODE = "solo", "node8"


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _rng(xs):
    xs = [x for x in xs if x is not None]
    return (min(xs), max(xs)) if xs else (None, None)


def segment_records(recs, config=None, phase=None, batch=None, io=None):
    out = [r for r in recs if str(r.get("kind", "")).startswith("segments")]
    if config:
        out = [r for r in out if r["config"] == config]
    if phase:
        out = [r for r in out if r["phase"] == phase]
    if batch is not None:
        out = [r for r in out if int(r.get("batch", 1)) == int(batch)]
    if io:
        out = [r for r in out if r.get("io") == io]
    return out


def sweep_records(recs, config=None):
    out = [r for r in recs if r.get("kind") == "sweep"]
    return [r for r in out if r["config"] == config] if config else out


def checks_records(recs):
    return [r for r in recs if r.get("kind") == "checks"]


def startup_s(r: dict) -> float | None:
    """Per-process startup a leg pays before its first useful step."""
    s = r.get("startup") or {}
    if r.get("model") == "gencast":
        if s.get("load_s") is None or s.get("first_forward_s") is None:
            return None
        return s["load_s"] + s["first_forward_s"]
    if s.get("load_s") is None:
        return None
    return s["load_s"] + (s.get("first_step_s") or 0.0)


def _phase_per_walker(rs: list[dict]) -> dict:
    """Median over workers of the mean per-walker phase time over timed segments."""
    per = {}
    for r in rs:
        timed = [s for s in r["segments"] if not s["warm"]]
        if not timed:
            continue
        b = int(r.get("batch", 1))
        for k in timed[0]["phases"]:
            per.setdefault(k, []).append(
                statistics.fmean(s["phases"].get(k, 0.0) for s in timed) / b)
    return {k: round(_med(v), 3) for k, v in per.items()}


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #
def layer1(recs) -> list[dict]:
    """Model resident: s per simulated day of pure compute, and the walker's roll cost."""
    rows = []
    for cfg in W.ORDER:
        c = W.CONFIGS[cfg]
        solo = segment_records(recs, cfg, PHASE_SOLO, batch=1)
        sw = sweep_records(recs, cfg)
        b1 = [row for r in sw for row in r["rows"] if row["batch"] == 1 and not row["oom"]]
        if c.env == "gencast":
            comp = _med(r["compute_s_per_step"].get("median") for r in solo)
        else:
            comp = _med(row["step_s"]["median"] for row in b1) or \
                _med(r["compute_s_per_step"].get("median") for r in solo)
        roll = _med(r["roll_s_per_step"].get("median") for r in solo)
        peak = _med(r.get("peak_gpu_gb") or r.get("peak_alloc_gb") for r in solo) or \
            _med(row.get("peak_alloc_gb") for row in b1)
        if comp is None and roll is None:
            rows.append(dict(config=cfg, label=c.label, measured=False))
            continue
        rows.append(dict(
            config=cfg, label=c.label, measured=True, precision=c.precision,
            step_h=c.step_h, n_runs=len(solo) + len(b1),
            compute_s_per_step=_r(comp), roll_s_per_step=_r(roll),
            s_per_sim_day_compute=_r(comp * c.steps_per_day if comp else None),
            s_per_sim_day_roll=_r(roll * c.steps_per_day if roll else None),
            peak_gpu_gb=_r(peak, 1),
            startup_s=_r(_med(startup_s(r) for r in solo)),
            compile_s=_r(_med((r.get("startup") or {}).get("compile_s") for r in solo))))
    base = next((r for r in rows if r["config"] == W.BASELINE and r["measured"]), None)
    for r in rows:
        if base and r["measured"] and r.get("s_per_sim_day_compute"):
            r["speedup_compute"] = _r(base["s_per_sim_day_compute"]
                                      / r["s_per_sim_day_compute"], 2)
    return rows


def sweep_table(recs) -> list[dict]:
    out = []
    for r in sweep_records(recs):
        b1 = next((x for x in r["rows"] if x["batch"] == 1 and not x["oom"]), None)
        for row in r["rows"]:
            d = dict(config=r["config"], worker=r["worker"], phase=r["phase"],
                     batch=row["batch"], oom=row["oom"],
                     peak_alloc_gb=row.get("peak_alloc_gb"))
            if not row["oom"]:
                d.update(step_s=row["step_s"]["median"],
                         walker_steps_per_s=row["walker_steps_per_s"],
                         s_per_walker_sim_day=row["s_per_walker_sim_day"],
                         gain_vs_b1=_r(row["walker_steps_per_s"] / b1["walker_steps_per_s"],
                                       3) if b1 else None)
            out.append(d)
    return out


def best_batch(recs, config="fcn3_native") -> int:
    """The batch size phase 2 should run at: the best walker throughput in the sweep, but
    only if it beats B=1 by ``BATCH_GAIN_MIN``; else 1."""
    gains = {}
    for r in sweep_records(recs, config):
        b1 = next((x for x in r["rows"] if x["batch"] == 1 and not x["oom"]), None)
        for x in r["rows"]:
            if b1 and not x["oom"]:
                gains.setdefault(x["batch"], []).append(
                    x["walker_steps_per_s"] / b1["walker_steps_per_s"])
    if not gains:
        return 1
    b, g = max(((b, _med(v)) for b, v in gains.items()), key=lambda t: t[1])
    return b if g >= W.BATCH_GAIN_MIN else 1


def layer2(recs) -> list[dict]:
    """Walker segments, per (config, phase, batch, io)."""
    groups = {}
    for r in segment_records(recs):
        k = (r["config"], r["phase"], int(r.get("batch", 1)), r.get("io"))
        groups.setdefault(k, []).append(r)
    order = {c: i for i, c in enumerate(W.ORDER)}
    rows = []
    for (cfg, phase, b, io), rs in sorted(groups.items(),
                                          key=lambda kv: (order.get(kv[0][0], 99),
                                                          kv[0][1] != PHASE_SOLO, kv[0][2],
                                                          str(kv[0][3]))):
        spd = [r["s_per_sim_day"].get("segment") for r in rs]
        lo, hi = _rng(spd)
        st_b = _med(s["state_bytes"] if "state_bytes" in s else s.get("state_bytes_per_walker")
                    for r in rs for s in r["segments"])
        rows.append(dict(
            config=cfg, phase=phase, batch=b, io=io, n_workers=len(rs),
            concurrency=_med(r.get("concurrency", 1) for r in rs),
            s_per_sim_day_segment=_r(_med(spd)), range=[_r(lo), _r(hi)],
            s_per_sim_day_roll=_r(_med(r["s_per_sim_day"].get("roll") for r in rs)),
            phases_per_walker_segment_s=_phase_per_walker(rs),
            state_mb=_r(st_b / 1e6, 1) if st_b else None,
            startup_s=_r(_med(startup_s(r) for r in rs)),
            noise_persisted=rs[0].get("noise_persisted"),
            all_finite=all(r.get("all_finite", False) for r in rs)))
    base = next((r for r in rows if r["config"] == W.BASELINE and r["phase"] == PHASE_NODE),
                None) or next((r for r in rows if r["config"] == W.BASELINE), None)
    for r in rows:
        if base and r["s_per_sim_day_segment"]:
            same = [x for x in rows if x["config"] == W.BASELINE and x["phase"] == r["phase"]]
            ref = same[0] if same else base
            r["speedup_vs_gencast_0p25_same_phase"] = _r(
                ref["s_per_sim_day_segment"] / r["s_per_sim_day_segment"], 2)
    return rows


# --------------------------------------------------------------------------- #
# Layer 3: the case
# --------------------------------------------------------------------------- #
def score_member_steps(n_walkers: int) -> int:
    """FCN3 member-steps one case buys: N x M x steps-to-horizon over the scored leads."""
    per = 24 // 6
    steps = sum(round((W.CASE_HORIZON_DAYS - lead) * per)
                for lead, c in zip(W.CASE_LEAD_DAYS, W.CASE_C) if c != 0.0)
    return n_walkers * W.CASE_M_MEMBERS * steps


def project(n_walkers: int, s_per_walker_day: float, startup: float | None,
            s_per_score_step: float) -> dict:
    walk_h = n_walkers * W.CASE_HORIZON_DAYS * s_per_walker_day / 3600
    score_h = score_member_steps(n_walkers) * s_per_score_step / 3600
    st = startup or 0.0
    start_relaunch = st * W.GPUS_PER_NODE * W.CASE_LEGS / 3600
    start_persist = st * W.GPUS_PER_NODE / 3600
    return dict(walk_h=_r(walk_h, 2), score_h=_r(score_h, 2),
                startup_relaunch_h=_r(start_relaunch, 2),
                startup_persistent_h=_r(start_persist, 2),
                total_relaunch_h=_r(walk_h + score_h + start_relaunch, 2),
                total_persistent_h=_r(walk_h + score_h + start_persist, 2),
                node_wall_relaunch_h=_r((walk_h + score_h + start_relaunch)
                                        / W.GPUS_PER_NODE, 2))


def layer3(recs, l2: list[dict], l1: list[dict]) -> dict:
    # Score rate: earth2studio FCN3, B=1, as the scorer runs it - node8 if measured.
    def l2row(cfg, phase, b=1, io="nc-zlib4"):
        return next((r for r in l2 if r["config"] == cfg and r["phase"] == phase
                     and r["batch"] == b and (io is None or r["io"] == io)), None)

    e2s_node = segment_records(recs, "fcn3_e2s", PHASE_NODE, batch=1)
    score_rate = _med(r["compute_s_per_step"].get("median") for r in e2s_node) or next(
        (r["compute_s_per_step"] for r in l1 if r["config"] == "fcn3_e2s"
         and r.get("measured")), None)
    score_src = "measured (fcn3_e2s, B=1)" if score_rate else "a3mega job 1161"
    score_rate = score_rate or W.REF_A3MEGA["fcn3_s_per_member_step"]

    options = []
    for cfg in W.ORDER:
        for b in sorted({r["batch"] for r in l2 if r["config"] == cfg}):
            row = l2row(cfg, PHASE_NODE, b) or l2row(cfg, PHASE_SOLO, b)
            if row and row["s_per_sim_day_segment"]:
                options.append(dict(walker=cfg, batch=b, phase=row["phase"],
                                    s_per_walker_day=row["s_per_sim_day_segment"],
                                    startup_s=row["startup_s"]))
    ref = W.REF_A3MEGA
    options.append(dict(walker="gencast_0p25 @ a3mega rates", batch=1, phase="reference",
                        s_per_walker_day=ref["gencast_0p25_prod_s_per_step"] * 2,
                        startup_s=ref["gencast_0p25_compile_s_8way"]))
    cases = []
    for n in W.CASE_N_WALKERS:
        for o in options:
            rate = score_rate if o["phase"] != "reference" else ref["fcn3_s_per_member_step"]
            cases.append(dict(n_walkers=n, **o,
                              **project(n, o["s_per_walker_day"], o["startup_s"], rate)))
    return dict(score_s_per_member_step=_r(score_rate), score_rate_source=score_src,
                score_member_steps={n: score_member_steps(n) for n in W.CASE_N_WALKERS},
                horizon_days=W.CASE_HORIZON_DAYS, legs=W.CASE_LEGS,
                m_members=W.CASE_M_MEMBERS, lead_days=list(W.CASE_LEAD_DAYS),
                c_k=list(W.CASE_C), cases=cases)


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return str(x)


def _table(head, rows) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def headline(l2, l3) -> list[str]:
    lines = []
    node = {(r["config"], r["batch"]): r for r in l2
            if r["phase"] == PHASE_NODE and r["io"] == "nc-zlib4"}
    base = node.get((W.BASELINE, 1))
    if base:
        for (cfg, b), r in sorted(node.items()):
            if cfg == W.BASELINE:
                continue
            lines.append(f"- **{W.CONFIGS[cfg].label}, B={b}:** "
                         f"{_fmt(r['s_per_sim_day_segment'], 1)} s per walker-day vs "
                         f"{_fmt(base['s_per_sim_day_segment'], 1)} for GenCast 0.25, i.e. "
                         f"**{_fmt(r.get('speedup_vs_gencast_0p25_same_phase'), 1)}x** at node "
                         f"scale (8 workers, checkpoint I/O included).")
    n0 = W.CASE_N_WALKERS[0]
    cs = [c for c in l3["cases"] if c["n_walkers"] == n0]
    g = next((c for c in cs if c["walker"] == W.BASELINE), None)
    f = min((c for c in cs if c["walker"].startswith("fcn3")),
            key=lambda c: c["total_relaunch_h"], default=None)
    if g and f:
        lines.append(f"- **One AI+RES case at N={n0}:** {_fmt(g['total_relaunch_h'], 1)} H100-h "
                     f"with GenCast walkers vs {_fmt(f['total_relaunch_h'], 1)} with "
                     f"{f['walker']} (B={f['batch']}). The FCN3 score phase is "
                     f"{_fmt(g['score_h'], 1)} H100-h of both totals.")
    return lines or ["- Node-scale (phase 2) results missing; see the tables below."]


def render(run: Path, recs, l1, sw, l2, l3, chk) -> str:
    host = next((r["host"] for r in recs if r.get("host")), {})
    gpus = host.get("gpus") or []
    md = ["# Walker benchmark: FCN3 vs GenCast", "",
          f"Run `{run.name}` on `{host.get('hostname', '?')}`: "
          f"{len(gpus)}x {gpus[0]['name'] if gpus else '?'} "
          f"(driver {gpus[0]['driver'] if gpus else '?'}), {host.get('cpu_model', '?')}, "
          f"{host.get('mem_total_gb', '?')} GB RAM. {len(recs)} result records.", "",
          "## Answer", "", *headline(l2, l3), ""]

    md += ["## FCN3 walker checks", ""]
    if chk:
        for c in chk:
            md.append(_table(["check", "verdict", "key numbers"],
                             [[x["name"], x["verdict"], _keynums(x)] for x in c["checks"]]))
    else:
        md.append("Not run.")
    md += ["", "## Layer 1: raw step speed (one worker per model, model resident)", "",
           "GenCast steps 12 h and FCN3 6 h, so compare the per-day columns. Compute is the "
           "model call alone; roll adds what the walker loop does per step (host copy, "
           "CONUS crop).", ""]
    md.append(_table(
        ["config", "precision", "s/step compute", "s/step roll", "s/sim-day compute",
         "s/sim-day roll", "speedup (compute)", "peak GPU GB", "startup s"],
        [[r["label"], r.get("precision"), r.get("compute_s_per_step"),
          r.get("roll_s_per_step"), r.get("s_per_sim_day_compute"),
          r.get("s_per_sim_day_roll"), r.get("speedup_compute"), r.get("peak_gpu_gb"),
          r.get("startup_s")] if r["measured"] else [r["label"], "not measured"] + [None] * 7
         for r in l1]))
    if sw:
        md += ["", "### FCN3 batch sweep", ""]
        md.append(_table(["config", "worker", "B", "s/step", "walker-steps/s",
                          "s per walker-day", "gain vs B=1", "peak GB", "OOM"],
                         [[r["config"], r["worker"], r["batch"], r.get("step_s"),
                           r.get("walker_steps_per_s"), r.get("s_per_walker_sim_day"),
                           r.get("gain_vs_b1"), r.get("peak_alloc_gb"),
                           "yes" if r["oom"] else ""] for r in sw]))
    md += ["", "## Layer 2: walker segments (3 d, checkpoint I/O included)", "",
           "`solo` = one worker per model (other configs on the other GPUs); `node8` = 8 "
           "identical workers on one node, the production layout. Per-walker phase times "
           "are seconds per 3-day segment.", ""]
    md.append(_table(
        ["config", "phase", "B", "io", "workers", "s/walker-day", "range", "speedup",
         "read", "roll", "write state", "write diag", "state MB", "noise kept"],
        [[r["config"], r["phase"], r["batch"], r["io"], r["n_workers"],
          r["s_per_sim_day_segment"], f"{_fmt(r['range'][0])}..{_fmt(r['range'][1])}",
          r.get("speedup_vs_gencast_0p25_same_phase"),
          r["phases_per_walker_segment_s"].get("read_state"),
          r["phases_per_walker_segment_s"].get("roll"),
          r["phases_per_walker_segment_s"].get("write_state"),
          r["phases_per_walker_segment_s"].get("write_diag"), r["state_mb"],
          {True: "yes", False: "no", None: "-"}[r["noise_persisted"]]
          if r["config"].startswith("fcn3") else "n/a"] for r in l2]))
    md += ["", "## Layer 3: one AI+RES case", "",
           f"Horizon {l3['horizon_days']:g} d, {l3['legs']} legs, resampling at "
           f"{', '.join(f'{x:g}' for x in l3['lead_days'])} d with C_k = "
           f"{', '.join(f'{x:g}' for x in l3['c_k'])}; score M = {l3['m_members']} FCN3 members "
           f"at {l3['score_s_per_member_step']} s/member-step ({l3['score_rate_source']}). "
           "`relaunch` = every leg restarts its workers (today's runner); `persistent` = "
           "workers live across legs (the fix proposed in aires/HANDOFF.md).", ""]
    md.append(_table(
        ["N", "walker", "B", "rate from", "walk H100-h", "score H100-h", "startup (relaunch)",
         "total (relaunch)", "total (persistent)", "node wall h"],
        [[c["n_walkers"], c["walker"], c["batch"], c["phase"], c["walk_h"], c["score_h"],
          c["startup_relaunch_h"], c["total_relaunch_h"], c["total_persistent_h"],
          c["node_wall_relaunch_h"]] for c in l3["cases"]]))
    md += ["", "## What this does not measure", "",
           "- Whether FCN3 walkers reach the tail or keep the importance weights honest. "
           "That needs Gate 3 and a production case with FCN3 walking.",
           "- FCN3's clone separation versus GenCast's. `clone_divergence` shows that clones "
           "separate, not whether they separate as fast as the resampling needs.",
           "- Shared-filesystem I/O. The VM writes states to its own disk, while a3mega wrote "
           "them to NFS. The a3mega reference row shows what NFS contention cost there.",
           "- torch.compile, CUDA graphs and multi-GPU spatial parallelism for FCN3. All of "
           "them would only make FCN3 faster.", ""]
    return "\n".join(md)


def _keynums(c: dict) -> str:
    keys = ("t2m_rms_k", "max_abs_diff", "bitwise", "t2m_rms_k_after_segment",
            "t2m_rms_k_per_step", "problems", "batch", "state_bytes", "error", "note")
    parts = []
    for k in keys:
        if k in c and c[k] not in (None, [], ""):
            v = c[k]
            if isinstance(v, list) and len(v) > 4:
                v = [v[0], "...", v[-1]]
            parts.append(f"{k}={v}")
    return "; ".join(parts)[:400]


def write_csv(path: Path, l1, l2) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "config", "phase", "batch", "io", "s_per_walker_sim_day",
                    "s_per_step_compute", "s_per_step_roll", "n"])
        for r in l1:
            if r["measured"]:
                w.writerow([1, r["config"], PHASE_SOLO, 1, "", r["s_per_sim_day_compute"],
                            r["compute_s_per_step"], r["roll_s_per_step"], r["n_runs"]])
        for r in l2:
            w.writerow([2, r["config"], r["phase"], r["batch"], r["io"],
                        r["s_per_sim_day_segment"], "", "", r["n_workers"]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--best-batch", action="store_true",
                    help="print the phase-2 batch size for fcn3_native and exit")
    a = ap.parse_args(argv)
    recs = T.read_results(W.results_dir(a.run))
    if a.best_batch:
        print(best_batch(recs))
        return 0
    if not recs:
        print(f"[report] no results under {W.results_dir(a.run)}")
        return 1
    l1, sw, l2 = layer1(recs), sweep_table(recs), layer2(recs)
    l3 = layer3(recs, l2, l1)
    chk = checks_records(recs)
    out = dict(run=a.run.name, layer1=l1, sweep=sw, layer2=l2, layer3=l3,
               checks=[c["checks"] for c in chk], best_batch=best_batch(recs),
               written=T.now_utc())
    (a.run / "results.json").write_text(json.dumps(out, indent=2, default=str))
    write_csv(a.run / "results.csv", l1, l2)
    md = render(a.run, recs, l1, sw, l2, l3, chk)
    (a.run / "REPORT.md").write_text(md)
    print(md)
    print(f"\n[report] wrote {a.run / 'REPORT.md'}, results.json, results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
