"""Tests for the local JSONL metrics sink (the no-W&B logging path)."""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.metrics_logger import JsonlMetricsLogger


def _read(path):
    # parse_constant fires only on NaN/Infinity — strict JSON must never contain them
    return [
        json.loads(line, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
        for line in open(path)
    ]


def test_log_and_finish(tmp_path):
    p = str(tmp_path / "m.jsonl")
    lg = JsonlMetricsLogger(p, meta={"world_size": 8})
    lg.log({"train/loss": 1.25, "train/lr": 1e-4}, step=10)
    lg.log({"val/loss": 1.5}, step=10)
    lg.finish()

    rows = _read(p)
    assert rows[0]["event"] == "run_start" and rows[0]["world_size"] == 8
    assert rows[1]["step"] == 10 and rows[1]["train/loss"] == 1.25
    assert rows[2]["val/loss"] == 1.5
    assert all("time" in r for r in rows)


def test_nan_and_inf_become_null(tmp_path):
    """sigma_std of a batch of 1 is NaN; strict JSON has no NaN/Infinity."""
    p = str(tmp_path / "m.jsonl")
    lg = JsonlMetricsLogger(p)
    lg.log({"train/sigma_std": float("nan"), "x": float("inf"), "ok": 2.0}, step=1)
    lg.finish()

    row = _read(p)[1]  # _read raises on a bare NaN/Infinity token
    assert row["train/sigma_std"] is None
    assert row["x"] is None
    assert row["ok"] == 2.0


def test_append_across_resume(tmp_path):
    """A resumed run appends to the same file with a fresh run_start header."""
    p = str(tmp_path / "m.jsonl")
    lg1 = JsonlMetricsLogger(p, meta={"resumed_at_step": 0})
    lg1.log({"train/loss": 1.0}, step=1)
    lg1.finish()
    lg2 = JsonlMetricsLogger(p, meta={"resumed_at_step": 1})
    lg2.log({"train/loss": 0.9}, step=2)
    lg2.finish()

    rows = _read(p)
    starts = [r for r in rows if r.get("event") == "run_start"]
    assert len(starts) == 2
    assert starts[1]["resumed_at_step"] == 1
    assert rows[-1]["step"] == 2


def test_io_failure_never_raises(tmp_path):
    """Logging must never kill a training run — a dead sink swallows writes."""
    lg = JsonlMetricsLogger(str(tmp_path / "nodir" / "x" / "m.jsonl"))
    # makedirs handles the path above; simulate death instead
    lg._dead = True
    lg.log({"train/loss": 1.0}, step=1)  # must not raise
    lg.finish()

    unwritable = "/proc/definitely/not/writable/m.jsonl"
    lg2 = JsonlMetricsLogger(unwritable)  # open fails -> disabled, no raise
    lg2.log({"train/loss": 1.0}, step=1)
    lg2.finish()


def test_plot_metrics_dedupes_resumed_steps(tmp_path):
    """A resume re-runs up to save_every_n_steps — plotting keeps the LAST value."""
    p = str(tmp_path / "m.jsonl")
    lg = JsonlMetricsLogger(p)
    lg.log({"train/loss": 1.0}, step=10)
    lg.log({"train/loss": 0.7}, step=10)  # same step re-logged after a resume
    lg.finish()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from plot_metrics import load_series

    series, _ = load_series(p)
    assert series["train/loss"] == {10: 0.7}


def test_plot_metrics_splits_epoch_means_from_step_samples(tmp_path):
    """The two train/loss row kinds must not be mixed.

    A per-step row is ONE sample at ONE random sigma (sigma drives ~74% of the log-loss
    variance, so that series is mostly noise); an epoch row carries train/epoch and holds
    the epoch mean. The epoch mean is the curve worth plotting, so it comes back separately
    and must not land in the per-step series.
    """
    p = str(tmp_path / "m.jsonl")
    lg = JsonlMetricsLogger(p)
    lg.log({"train/loss": 0.9}, step=10)                     # per-step sample
    lg.log({"train/loss": 0.2}, step=20)                     # per-step sample
    lg.log({"train/loss": 0.5, "train/epoch": 1}, step=20)   # epoch summary at the same step
    lg.finish()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from plot_metrics import load_series

    series, epoch_loss = load_series(p)
    assert epoch_loss == {20: 0.5}
    assert series["train/loss"] == {10: 0.9, 20: 0.2}  # epoch mean did NOT overwrite step 20


def test_window_averages_reduces_and_stratifies():
    """The logged loss is the window sum / micro-batch count, and empty sigma buckets drop out."""
    from training.trainer import _window_averages, SIGMA_LOG_BUCKET_NAMES

    assert len(SIGMA_LOG_BUCKET_NAMES) == 4
    win_sum = [0.8, 0.6, 0.2]   # sums over the window
    win_n = 4                   # 4 micro-batches
    bkt_sum = [0.9, 0.0, 0.0, 0.05]
    bkt_n = [3, 0, 0, 1]        # buckets 'mlo'/'mhi' never sampled this window

    out = _window_averages(win_sum, win_n, bkt_sum, bkt_n)

    assert abs(out["train/loss"] - 0.2) < 1e-9       # 0.8 / 4
    assert abs(out["train/loss_prog"] - 0.15) < 1e-9
    assert abs(out["train/loss_diag"] - 0.05) < 1e-9
    assert abs(out["train/loss_sigma_lo"] - 0.3) < 1e-9   # 0.9 / 3
    assert abs(out["train/loss_sigma_hi"] - 0.05) < 1e-9  # 0.05 / 1
    assert "train/loss_sigma_mlo" not in out  # empty bucket omitted, not logged as NaN
    assert "train/loss_sigma_mhi" not in out
