"""Figures of step 03, from the recorded numbers in ../results. Needs no solver.

  ../figures/size_vs_error.png     one-step error against the number of weights
  ../figures/rounds.png            four active-learning rounds per variant
  ../figures/committee.png         error against committee size
  ../figures/coverage.png          how fast a dataset covers the reachable states

Run:  python make_figures.py
"""
import csv
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common.plotstyle import AQUA, BLUE, INK, MUTED, ORANGE, SURFACE, YELLOW, apply_style

from committee import saturation

RESULTS, FIGURES = HERE / ".." / "results", HERE / ".." / "figures"
VARIANTS = ["A", "B1", "B2", "C32", "C64", "C128"]


def rounds():
    table = {}
    for row in csv.DictReader(open(RESULTS / "rounds.csv")):
        table.setdefault(row["variant"], []).append({k: float(v) for k, v in row.items() if k != "variant"})
    return table


def size_vs_error():
    table = rounds()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    c = [v for v in VARIANTS if v.startswith("C")]
    ax.plot([table[v][0]["n_parameters"] for v in c], [table[v][0]["holdout_ksi_mae"] for v in c], color=BLUE, lw=1.5, zorder=1)
    offsets = {"A": (10, 6), "B1": (10, -12), "B2": (10, 6), "C32": (8, 8), "C64": (8, 8), "C128": (-10, 10)}
    for v in VARIANTS:
        first = table[v][0]
        color = BLUE if v.startswith("C") else ORANGE
        ax.plot(first["n_parameters"], first["holdout_ksi_mae"], "o", color=color, ms=9, mec=SURFACE, mew=2)
        ax.annotate(f"{v}  {first['holdout_ksi_mae']:.4f}", (first["n_parameters"], first["holdout_ksi_mae"]),
                    xytext=offsets[v], textcoords="offset points", fontsize=9, ha="right" if v == "C128" else "left")
    ax.plot([], [], "o", color=ORANGE, label="dense 64-64-32 (A, B1, B2)")
    ax.plot([], [], "o", color=BLUE, label="residual blocks + physics features (C)")
    ax.set(xscale="log", xlabel="number of weights", ylabel="one-step error of ξ (MAE)\non 12,369 held-out single steps",
           ylim=(0, 0.0072), xlim=(5000, 220000), title="Round 1, 49,476 training rows, one seed per variant")
    ax.set_xticks([6690, 9218, 35330, 138242], ["6.7k", "9.2k", "35k", "138k"])
    ax.minorticks_off()
    ax.legend(loc="lower left")
    fig.savefig(FIGURES / "size_vs_error.png")
    plt.close(fig)


def rounds_figure():
    table = rounds()
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4), sharex=True, sharey=True)
    for ax, v in zip(axes.ravel(), VARIANTS):
        r = [x["round"] for x in table[v]]
        ax.plot(r, [1e3 * x["holdout_ksi_mae"] for x in table[v]], "o-", color=ORANGE, ms=6, mec=SURFACE,
                label="12,369 held-out single steps, from the true previous state")
        ax.plot(r, [1e3 * x["rollout_ksi_mae"] for x in table[v]], "s-", color=BLUE, ms=6, mec=SURFACE,
                label="along the 24 loading paths, from the network's own previous state")
        ax.set_title(f"{v}  ({int(table[v][0]['n_parameters']):,} weights)", loc="left")
        ax.set_xticks([1, 2, 3, 4])
    for ax in axes[1]:
        ax.set_xlabel("active-learning round")
    for ax in axes[:, 0]:
        ax.set_ylabel("error of ξ (MAE) × 1000")
    axes[0, 0].set_ylim(0, 9.6)
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.42), ncol=2)
    fig.savefig(FIGURES / "rounds.png")
    plt.close(fig)


def committee():
    results = json.load(open(RESULTS / "committee.json"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.0))
    for name, color, marker, label in [("A", ORANGE, "o", "A (dense, 6,690 weights)"),
                                       ("C128", BLUE, "s", "C128 (residual, 138,242 weights)")]:
        curve = results[name]["curve"]
        k = np.array([p["k"] for p in curve])
        median = np.array([p["holdout_median"] for p in curve])
        sd = np.array([p["holdout_sd"] for p in curve])
        floor, _ = saturation(k, median)
        a.fill_between(k, median - sd, median + sd, color=color, alpha=0.15, lw=0)
        a.plot(k, median, marker=marker, color=color, ms=5, mec=SURFACE, label=label)
        a.axhline(floor, color=color, lw=1, ls="--")
        a.text(11.5, floor - 0.00018, f"floor {floor:.4f}", color=INK, va="top", fontsize=8.5)
        free = [(p["k"], p["free_run_ksi_mae"]) for p in curve if p["free_run_ksi_mae"] is not None]
        b.plot(*zip(*free), marker=marker, color=color, ms=6, mec=SURFACE, label=label)
    a.set(xlabel="members in the committee", ylabel="one-step error of ξ (MAE)\non 12,369 held-out single steps", ylim=(0, 0.0105),
          xlim=(0.5, 16.6), title="One step: averaging removes variance, down to a floor (dashed)")
    b.set(xlabel="members in the committee", ylabel="free-run error of ξ (MAE)\non 24 loading paths", ylim=(0, 0.0175),
          title="Free run: the ranking flips between 1 and 4 members")
    a.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "committee.png")
    plt.close(fig)


def coverage():
    cov = json.load(open(RESULTS / "coverage.json"))
    fig, a = plt.subplots(figsize=(6.4, 4.0))
    rows, cells = zip(*cov["cells_vs_rows_8pow5"])
    a.plot(rows, cells, "o-", color=BLUE, ms=6, mec=SURFACE)
    a.axhline(1101, color=MUTED, lw=1, ls="--")
    a.text(520, 1125, "all 61,845 rows: 1,101 of 32,768 cells (3.4 %)", color=MUTED, fontsize=9)
    for r, c in ((2000, 660), (10000, 983)):
        a.annotate(f"{r:,} rows: {100 * c / 1101:.0f} %", (r, c), xytext=(10, -14), textcoords="offset points", fontsize=9)
    a.set(xscale="log", xlabel="rows in a random subsample", ylabel="occupied cells of an 8⁵ grid", ylim=(0, 1250),
          title="Coverage saturates long before the row count")
    fig.tight_layout()
    fig.savefig(FIGURES / "coverage.png")
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    size_vs_error(), rounds_figure(), committee(), coverage()
    print("figures written to", FIGURES.resolve())
