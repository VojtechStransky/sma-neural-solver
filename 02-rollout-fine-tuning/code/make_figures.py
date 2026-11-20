"""Figures of step 02: the baseline (v10) against the fine-tuned network (v17).

  ../figures/loops.png                  stress-strain loops at three temperatures
  ../figures/drift_removed.png          xi along two cold cycles
  ../figures/error_vs_temperature.png   free-run error of every cycle against temperature
  ../figures/error_histogram.png        distribution of the free-run error over the cycles
  ../figures/fine_tuning.png            held-out free-run error during fine-tuning

Needs ../results/v17_per_cycle.csv and ../../01-baseline-network/results/v10_per_cycle.csv
(written by `python -m common.benchmark`, see the README).

Run:  python make_figures.py
"""
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.data import MODELS, REPO, load_cycles
from common.plotstyle import BLUE, INK, MUTED, ORANGE, SURFACE, apply_style
from common.rollout import OnnxStep, free_run
from common.stress import stress

STEP = pathlib.Path(__file__).resolve().parents[1]
FIGURES = STEP / "figures"
V10 = dict(color=ORANGE, ls=(0, (1, 1)), lw=1.9, label="v10 (baseline), free run")
V17 = dict(color=BLUE, ls=(0, (4, 2)), lw=1.7, label="v17 (fine-tuned), free run")
REF = dict(color=INK, lw=1.5, label="reference solver")
LOOP_TEMPERATURES = [-40.0, 0.0, 100.0]
COLD_SHOWN = [161, 213]
BASELINE_VALIDATION_ERROR = 0.0457   # v10 on the same validation windows (printed by finetune_rollout.py)


def per_cycle(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("loading_temperature", "free_run_mae_xi")}


def simple_cycle_near(cycles, temperature):
    """The load-unload cycle with a full transformation that is loaded closest to `temperature`."""
    def suitable(c):
        xi = c.Y[:, 0]
        return (xi.max() - xi.min() > 0.55 and c.strain.max() - c.strain[-1] > 0.02
                and 120 < len(c.X) < 500 and c.reversals <= 2)
    return min(filter(suitable, cycles), key=lambda c: abs(c.loading_temperature - temperature))


def loops(cycles, models):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, temperature in zip(axes, LOOP_TEMPERATURES):
        c = simple_cycle_near(cycles, temperature)
        ax.plot(c.strain, stress(c.strain, c.Y[:, 0], c.Y[:, 1], c.temperature), **REF)
        for model, style in models:
            run = free_run(model, c)
            ax.plot(c.strain, stress(c.strain, run[:, 0], run[:, 1], c.temperature), **style)
        ax.set(xlabel="strain ε", title=f"cycle {c.id}, loaded at {c.loading_temperature:.0f} °C")
    axes[0].set_ylabel("stress σ [MPa]")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "loops.png")
    plt.close(fig)


def drift_removed(cycles, models):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, cid in zip(axes, COLD_SHOWN):
        c = next(x for x in cycles if x.id == cid)
        ax.plot(c.Y[:, 0], **REF)
        for model, style in models:
            ax.plot(free_run(model, c)[:, 0], **style)
        ax.set(xlabel="step", title=f"cycle {cid}, loaded at {c.loading_temperature:.0f} °C")
    axes[0].set_ylabel("martensite fraction ξ")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "drift_removed.png")
    plt.close(fig)


def error_vs_temperature(v10, v17):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    edges = np.arange(-55, 106, 10)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for data, color, marker, label in [(v10, ORANGE, "o", "v10 (baseline)"), (v17, BLUE, "s", "v17 (fine-tuned)")]:
        T, mae = data["loading_temperature"], data["free_run_mae_xi"]
        ax.plot(T, mae, marker, color=color, ms=3.5, alpha=0.3, mec="none")
        which = np.digitize(T, edges) - 1
        mean = [mae[which == k].mean() if (which == k).any() else np.nan for k in range(len(centres))]
        ax.plot(centres, mean, marker=marker, color=color, ms=7, mec=SURFACE, mew=1.5,
                label=f"{label}: mean per 10 °C")
    ax.set(yscale="log", xlabel="temperature during loading [°C]", ylabel="free-run error of ξ (mean absolute)",
           title="Free-run error of each cycle (small marks) and its mean per temperature bin")
    ax.legend(loc="upper right")
    fig.savefig(FIGURES / "error_vs_temperature.png")
    plt.close(fig)


def error_histogram(v10, v17):
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    bins = np.logspace(np.log10(2e-4), np.log10(7e-2), 32)
    for data, color, label in [(v10, ORANGE, "v10 (baseline)"), (v17, BLUE, "v17 (fine-tuned)")]:
        mae = data["free_run_mae_xi"]
        ax.hist(mae, bins=bins, color=color, alpha=0.55, linewidth=0,
                label=f"{label}, median {np.median(mae):.4f}")
        ax.axvline(np.median(mae), color=color, lw=1.4, ls="--")
    ax.set(xscale="log", xlabel="free-run error of ξ per cycle (mean absolute)", ylabel="number of cycles",
           title="Distribution over the 250 cycles (dashed: medians)")
    ax.legend()
    fig.savefig(FIGURES / "error_histogram.png")
    plt.close(fig)


def fine_tuning():
    h = np.load(STEP / "results" / "v17_finetuning_history.npz")
    best = int(np.argmin(h["cold_roll"]))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(h["epoch"], h["cold_roll"], color=BLUE, marker="o", ms=6, mec=SURFACE, mew=1.5)
    ax.plot(h["epoch"][best], h["cold_roll"][best], "o", color=BLUE, ms=11, mfc="none", mew=1.8)
    ax.annotate("kept", (h["epoch"][best], h["cold_roll"][best]), textcoords="offset points",
                xytext=(-8, 12), ha="right")
    ax.axhline(BASELINE_VALIDATION_ERROR, color=ORANGE, lw=1.4, ls="--")
    ax.text(14, BASELINE_VALIDATION_ERROR + 0.003, "v10 before fine-tuning", color=INK, ha="right")
    ax.set(xlabel="fine-tuning epoch", ylabel="free-run error on the 37 validation cycles\n(cold cycles count three times)",
           title="Free-run error on the validation cycles during fine-tuning", ylim=(0, None))
    fig.savefig(FIGURES / "fine_tuning.png")
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    cycles = load_cycles()
    models = [(OnnxStep(MODELS / "v10.onnx"), V10), (OnnxStep(MODELS / "v17.onnx"), V17)]
    v10 = per_cycle(REPO / "01-baseline-network" / "results" / "v10_per_cycle.csv")
    v17 = per_cycle(STEP / "results" / "v17_per_cycle.csv")
    loops(cycles, models)
    drift_removed(cycles, models)
    error_vs_temperature(v10, v17)
    error_histogram(v10, v17)
    fine_tuning()
    print("figures written to", FIGURES)
