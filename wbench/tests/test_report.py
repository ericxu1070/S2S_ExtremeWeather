"""The report's projection and its handling of the records the workers write."""
from __future__ import annotations

import json

import pytest

from wbench import report as RP
from wbench import timing as T
from wbench import wconfig as W


def test_projection_reproduces_the_pilot_at_a3mega_rates():
    """At a3mega's measured rates, N=64 must price like the pilot: ~36 walk + ~25 score
    H100-h (aires/HANDOFF.md, job 1172). The walk term here is the full 7 segments; the
    pilot's 36 is after Gate 3 seeded its first two, so the bound is loose on that side."""
    ref = W.REF_A3MEGA
    p = RP.project(64, ref["gencast_0p25_prod_s_per_step"] * 2, None,
                   ref["fcn3_s_per_member_step"])
    assert 35.0 <= p["walk_h"] <= 40.0
    assert 24.0 <= p["score_h"] <= 26.5


def test_score_steps_skip_the_null_coefficient():
    # C_1 = 0 buys no score; 6/9/12/15 d -> 15+12+9+6 = 42 d of FCN3 = 168 steps.
    assert RP.score_member_steps(1) == W.CASE_M_MEMBERS * 168


def _seg(config, phase, worker, batch=1, s_per_day=100.0, io="nc-zlib4", model="gencast"):
    segs = [dict(seg=s, warm=(s == 0), steps=6, phases=dict(read_state=1.0, roll=50.0,
                                                           write_state=5.0, write_diag=1.0),
                 wall_s=57.0, state_bytes=477_000_000, finite=True) for s in range(3)]
    return dict(config=config, phase=phase, worker=worker, kind=f"segments_b{batch}",
                model=model, batch=batch, io=io, concurrency=8 if phase == "node8" else 1,
                compute_s_per_step=dict(median=s_per_day / 4),
                roll_s_per_step=dict(median=s_per_day / 3.5),
                s_per_sim_day=dict(compute=s_per_day / 2, roll=s_per_day / 1.5,
                                   segment=s_per_day),
                startup=dict(load_s=60.0, first_forward_s=300.0, first_step_s=2.0,
                             compile_s=290.0),
                peak_gpu_gb=64.0, all_finite=True, segments=segs, noise_persisted=True)


def _sweep(gains=(1.0, 1.3, 1.5), oom_at=None):
    rows = []
    for b, g in zip((1, 2, 4), gains):
        if oom_at == b:
            rows.append(dict(batch=b, oom=True, peak_alloc_gb=79.0))
            break
        rows.append(dict(batch=b, oom=False, step_s=dict(median=1.4 * b / g),
                         walker_steps_per_s=g / 1.4, s_per_walker_sim_day=4 * 1.4 / g,
                         peak_alloc_gb=50.0 + 10 * b, finite=True))
    return dict(config="fcn3_native", phase="solo", worker=0, kind="sweep", model="fcn3",
                rows=rows, startup=dict(load_s=20.0, first_step_s=3.0))


@pytest.fixture
def recs(tmp_path):
    rs = [_seg("gencast_0p25", "solo", 0, s_per_day=60.0),
          _seg("gencast_0p25", "node8", 0, s_per_day=100.0),
          _seg("gencast_0p25", "node8", 1, s_per_day=104.0),
          _seg("fcn3_native", "node8", 0, s_per_day=8.0, model="fcn3"),
          _seg("fcn3_e2s", "node8", 0, s_per_day=9.0, model="fcn3"),
          _sweep()]
    d = W.results_dir(tmp_path)
    for r in rs:
        T.write_result(d, r)
    return T.read_results(d)


def test_results_round_trip_through_disk(recs):
    assert len(recs) == 6
    assert all(r["schema"] == T.SCHEMA for r in recs)


def test_layer2_speedup_is_against_the_same_phase(recs):
    l2 = RP.layer2(recs)
    f = next(r for r in l2 if r["config"] == "fcn3_native")
    assert f["s_per_sim_day_segment"] == 8.0
    assert f["speedup_vs_gencast_0p25_same_phase"] == pytest.approx(102.0 / 8.0, abs=0.01)


def test_best_batch_requires_a_real_gain(recs):
    assert RP.best_batch(recs) == 4
    flat = [_sweep(gains=(1.0, 1.02, 1.05))]
    assert RP.best_batch(flat) == 1
    assert RP.best_batch([_sweep(oom_at=2)]) == 1


def test_render_survives_partial_runs(recs, tmp_path):
    l1, sw, l2 = RP.layer1(recs), RP.sweep_table(recs), RP.layer2(recs)
    l3 = RP.layer3(recs, l2, l1)
    md = RP.render(tmp_path, recs, l1, sw, l2, l3, [])
    assert "Layer 3" in md and "not measured" in md          # gencast_1p0 never ran
    assert "\u2014" not in md                                 # no em dashes in output
    json.dumps(dict(l1=l1, l2=l2, l3=l3), default=str)


def test_summarize_quantiles():
    s = T.summarize([1, 2, 3, 4, 5])
    assert s["median"] == 3 and s["p10"] == pytest.approx(1.4) and s["n"] == 5
    assert T.summarize([]) == dict(n=0)
