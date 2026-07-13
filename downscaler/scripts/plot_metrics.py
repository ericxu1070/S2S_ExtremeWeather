"""Plot training curves from the local JSONL metrics log (the no-W&B path).

Reads the file written by training/metrics_logger.py (logging=local, the default)
and renders loss / learning-rate / grad-norm curves to a PNG.

    python scripts/plot_metrics.py
    python scripts/plot_metrics.py --path metrics/train_metrics.jsonl --out metrics/curves.png

The log is append-only across stop/resume cycles and a resume re-runs up to
`save_every_n_steps` optimizer steps, so a step can appear twice — this keeps
the LAST occurrence of each step.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-3 of the validated reference palette (fixed order).
C_TRAIN = "#2a78d6"   # blue
C_VAL = "#1baf7a"     # aqua
C_AUX = "#eda100"     # yellow
INK = "#42525f"       # text/axis ink — text never wears the series color


def load_series(path: str) -> dict[str, dict[int, float]]:
    """{metric_key: {step: last_value_seen}} from the JSONL log."""
    series: dict[str, dict[int, float]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial last line from a live run
            if row.get("event") == "run_start" or "step" not in row:
                continue
            step = int(row["step"])
            for k, v in row.items():
                if k in ("step", "time", "event") or not isinstance(v, (int, float)):
                    continue
                series.setdefault(k, {})[step] = float(v)
    return series


def _panel(ax, title, ylabel, log_y=False):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(colors=INK, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9cfd4")
    if log_y:
        ax.set_yscale("log")


def _plot(ax, data: dict[int, float], color, label=None, marker=None):
    steps = sorted(data)
    ax.plot(
        steps, [data[s] for s in steps],
        color=color, linewidth=1.8, label=label,
        marker=marker, markersize=4 if marker else 0,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "metrics")
    p.add_argument("--path", default=os.path.join(default_dir, "train_metrics.jsonl"))
    p.add_argument("--out", default=os.path.join(default_dir, "curves.png"))
    args = p.parse_args()

    if not os.path.exists(args.path):
        raise SystemExit(f"no metrics log at {args.path} — has training run with logging=local?")
    series = load_series(args.path)
    if not series.get("train/loss"):
        raise SystemExit(f"{args.path} has no train/loss entries yet")

    panels = [("loss", True), ("lr", False), ("grad_norm", False)]
    have = ["loss"] + [k for k, _ in panels[1:] if series.get(f"train/{k}")]
    fig, axes = plt.subplots(len(have), 1, figsize=(9, 2.6 * len(have)), sharex=True)
    axes = [axes] if len(have) == 1 else list(axes)

    ax = axes[0]
    _panel(ax, "Loss", "EDM loss", log_y=True)
    _plot(ax, series["train/loss"], C_TRAIN, "train")
    if series.get("val/loss"):
        _plot(ax, series["val/loss"], C_VAL, "val (EMA)", marker="o")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    last_step = max(series["train/loss"])
    ax.annotate(
        f"{series['train/loss'][last_step]:.4f}",
        (last_step, series["train/loss"][last_step]),
        textcoords="offset points", xytext=(6, 0),
        fontsize=8, color=INK, va="center",
    )

    i = 1
    if "lr" in have:
        _panel(axes[i], "Learning rate", "lr", log_y=True)
        _plot(axes[i], series["train/lr"], C_AUX)
        i += 1
    if "grad_norm" in have:
        _panel(axes[i], "Gradient norm (pre-clip)", "‖g‖")
        _plot(axes[i], series["train/grad_norm"], C_AUX)

    axes[-1].set_xlabel("optimizer step", fontsize=9, color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    n = len(series["train/loss"])
    print(f"wrote {args.out}  ({n} logged steps, latest step {last_step}, "
          f"latest train/loss {series['train/loss'][last_step]:.4f})")


if __name__ == "__main__":
    main()
