"""Figures and summary tables of step 04, from the recorded results in ../results/.

Needs only numpy and matplotlib (no solver, no TensorFlow).

  ../figures/classifier_comparison.png      which model can predict "changes / no change"
  ../figures/gate_errors_by_step_size.png   where the gate is wrong, by size of the true step
  ../figures/benchmark_paths.png            stress error on the 18 benchmark paths
  ../figures/thesis_cycle_161.png           the cold cycle with a long plateau
  ../results/summary.md                     the tables quoted in the README

Run:  python make_figures.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.plotstyle import AQUA, BLUE, INK, MUTED, ORANGE, SURFACE, apply_style   # noqa: E402

from gate import THRESHOLD_EPS_TR, THRESHOLD_XI   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"

# One color per kind of run, the same in every figure. The reference is drawn in INK.
C_SOLVER, C_NETWORK, C_GATED_NETWORK = BLUE, ORANGE, AQUA

BENCHMARK = ["tension_T-050", "tension_T-035", "tension_T-020", "tension_T-005", "tension_T+010",
             "tension_T+025", "tension_T+040", "tension_T+055", "tension_T+070", "tension_T+085",
             "tension_T+100", "deepcycle_T+060", "compression_T+000", "minorloop_T+040",
             "thermomech_T+100", "random_0", "recovery_T-040", "Tsweep_eps0.04"]
TENSION = [p for p in BENCHMARK if p.startswith("tension")]
EIGHT = ["compression_T+000", "Tsweep_eps0.04", "recovery_T-040", "tension_T-050",
         "minorloop_T+040", "random_0", "thermomech_T+100", "deepcycle_T+060"]
MODEL_NAMES = {"majority": "always the majority class", "logreg+features": "logistic regression",
               "boosting": "gradient boosting", "boosting+features": "gradient boosting + features",
               "nn_64_64_32": "network 64-64-32", "nn_128x3": "network 128 × 3",
               "nn_128x3+features": "network 128 × 3 + features"}


def load_path_results():
    """{variant: {path: row}} with numeric fields converted."""
    table = defaultdict(dict)
    for row in csv.DictReader(open(RESULTS / "path_results.csv")):
        for key, value in row.items():
            if key not in ("path", "variant"):
                row[key] = float(value) if value else None
        table[row["variant"]][row["path"]] = row
    return table


def median(table, variant, paths, field="stress_mae"):
    return float(np.median([table[variant][p][field] for p in paths]))


# ------------------------------------------------------------------ tables

def write_summary(table):
    lines = ["## Benchmark paths (stress error in MPa)\n",
             "| Path | Steps | Gate + solver | Solver work | Network alone | Gate + network |",
             "|---|---:|---:|---:|---:|---:|"]
    for p in BENCHMARK:
        s, n, g = table["gate_solver"][p], table["c128_alone"][p], table["gate_c128"][p]
        lines.append(f"| `{p}` | {s['steps']:.0f} | {s['stress_mae']:.3f} | {s['solver_work']:.0%} | "
                     f"{n['stress_mae']:.2f} | {g['stress_mae']:.2f} |")
    for label, fn in (("Median", np.median), ("Mean", np.mean), ("Worst path", np.max)):
        cells = [fn([table[v][p][f] for p in BENCHMARK])
                 for v, f in (("gate_solver", "stress_mae"), ("gate_solver", "solver_work"),
                              ("c128_alone", "stress_mae"), ("gate_c128", "stress_mae"))]
        work = "" if label == "Worst path" else f"{cells[1]:.0%}"
        lines.append(f"| **{label}** | | {cells[0]:.3f} | {work} | {cells[2]:.2f} | {cells[3]:.2f} |")

    lines += ["\n## Medians by path set (stress error in MPa)\n",
              "| Run | 11 tension cycles | 8 mixed paths | all 18 |", "|---|---:|---:|---:|"]
    names = [("full_solver", "Full solver in the same loop (control)"),
             ("gate_solver_first_rule_tau1e-5", "Gate + full solver, one threshold 1e-5"),
             ("gate_solver_per_variable_tau1e-5", "Gate + per-variable solver, one threshold 1e-5"),
             ("gate_solver", "Gate + per-variable solver, per-variable thresholds"),
             ("c128_alone", "Network alone"),
             ("gate_c128_tau1e-5", "Gate + network, one threshold 1e-5"),
             ("gate_c128", "Gate + network, per-variable thresholds"),
             ("v17_alone", "Fine-tuned network of step 02 alone"),
             ("gate_v17", "Gate + fine-tuned network of step 02")]
    for variant, name in names:
        lines.append(f"| {name} | {median(table, variant, TENSION):.3f} | "
                     f"{median(table, variant, EIGHT):.3f} | {median(table, variant, BENCHMARK):.3f} |")

    lines += ["\n## Cold cycles with a long plateau\n",
              "| Run | Cycle 161: xi error | stress error | Cycle 213: xi error | stress error |",
              "|---|---:|---:|---:|---:|"]
    for variant, name in (("v17_alone", "Fine-tuned network of step 02"), ("gate_v17", "Gate + that network"),
                          ("c128_alone", "Network alone"), ("gate_c128", "Gate + network"),
                          ("gate_solver", "Gate + solver")):
        a, b = table[variant]["thesis_161"], table[variant]["thesis_213"]
        lines.append(f"| {name} | {a['xi_mae']:.4f} | {a['stress_mae']:.1f} | "
                     f"{b['xi_mae']:.4f} | {b['stress_mae']:.1f} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ------------------------------------------------------------------ figures

def classifier_comparison():
    study = json.load(open(RESULTS / "classifier_study.json"))
    grouped = study["grouped_cv"]
    models = list(MODEL_NAMES)[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    y = np.arange(len(models))
    values = [grouped[m]["both_right"] for m in models]
    ax.barh(y, values, height=0.62, color=BLUE)
    for yi, v in zip(y, values):
        ax.text(v + 0.008, yi, f"{v:.1%}", va="center", fontsize=9)
    ax.set_yticks(y, [MODEL_NAMES[m] for m in models])
    ax.set(xlim=(0.5, 1.06), xlabel="share of steps with both answers right, whole states held out")
    ax.set_title("Predicting “does ξ change, does ε$^{tr}$ change?”", loc="left")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    fig.savefig(FIGURES / "classifier_comparison.png")
    plt.close(fig)


def gate_errors_by_step_size():
    rows = list(csv.DictReader(open(RESULTS / "gate_errors_by_step_size.csv")))
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 5.4), sharex="col",
                             gridspec_kw={"height_ratios": [1, 1.25]})
    for col, (variable, symbol, tau) in enumerate((("xi", "Δξ", THRESHOLD_XI),
                                                   ("eps_tr", "Δε$^{tr}$", THRESHOLD_EPS_TR))):
        sel = [r for r in rows if r["variable"] == variable]
        zero, bins = sel[0], sel[1:]
        lo = np.array([float(r["step_lo"]) for r in bins])
        hi = np.array([float(r["step_hi"]) for r in bins])
        count = np.array([int(r["rows"]) for r in bins])
        errors = np.array([int(r["gate_errors"]) for r in bins])
        mid = np.sqrt(lo * hi)
        shown = (hi > 1e-9) & (count >= 30)

        top, bottom = axes[0, col], axes[1, col]
        top.bar(lo[shown], count[shown], width=(hi - lo)[shown] * 0.86, align="edge", color=MUTED, alpha=0.55)
        top.set(xscale="log", yscale="log", ylabel="steps in the data")
        top.set_title(f"|{symbol}|:  “changes” means more than {tau:.2e}".replace("e-0", "e-")
                      + f"\n({int(zero['rows']):,} further steps do not move at all)", loc="left")

        rate = errors[shown] / count[shown]
        harmful = mid[shown] > tau
        bottom.plot(mid[shown], rate, color=MUTED, lw=1.0, zorder=1)
        bottom.plot(mid[shown][~harmful], rate[~harmful], "o", color=MUTED, ms=7, mec=SURFACE, mew=1.5,
                    label="gate computes although nothing changes (costs work)")
        bottom.plot(mid[shown][harmful], rate[harmful], "o", color=ORANGE, ms=7, mec=SURFACE, mew=1.5,
                    label="gate holds although the variable changes (costs accuracy)")
        bottom.set(xscale="log", ylim=(0, 0.45), ylabel="share the gate gets wrong",
                   xlabel=f"size of the true step |{symbol}|")
        bottom.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        for ax in (top, bottom):
            ax.axvline(tau, color=INK, lw=1.0, ls="--")
            ax.set_xlim(1e-9, 1.0)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.savefig(FIGURES / "gate_errors_by_step_size.png")
    plt.close(fig)


def benchmark_paths(table):
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    y = np.arange(len(BENCHMARK))[::-1]
    alone = np.array([table["c128_alone"][p]["stress_mae"] for p in BENCHMARK])
    gated = np.array([table["gate_c128"][p]["stress_mae"] for p in BENCHMARK])
    solver = np.array([table["gate_solver"][p]["stress_mae"] for p in BENCHMARK])
    work = [table["gate_solver"][p]["solver_work"] for p in BENCHMARK]
    floor = 2e-3
    ax.hlines(y, np.minimum(alone, gated), np.maximum(alone, gated), color=MUTED, lw=1.0, zorder=1)
    ax.plot(np.maximum(solver, floor), y, "o", color=C_SOLVER, ms=9, mec=SURFACE, mew=2, label="gate + solver")
    ax.plot(alone, y, "o", color=C_NETWORK, ms=9, mec=SURFACE, mew=2, label="network alone")
    ax.plot(gated, y, "o", color=C_GATED_NETWORK, ms=9, mec=SURFACE, mew=2, label="gate + network")
    for yi, s in zip(y, solver):
        if s < floor:
            ax.text(floor * 1.25, yi, "below 0.001", va="center", fontsize=8.5, color=MUTED)
    for yi, w in zip(y, work):
        ax.text(1.02, yi, f"{w:.0%}", va="center", fontsize=9, transform=ax.get_yaxis_transform())
    ax.text(1.02, len(BENCHMARK) - 0.2, "solver\nwork", va="bottom", fontsize=9, color=MUTED,
            transform=ax.get_yaxis_transform())
    ax.set_yticks(y, BENCHMARK)
    ax.set(xscale="log", xlim=(floor * 0.8, 80), xlabel="free-run stress error [MPa], log scale")
    ax.set_title("18 benchmark paths: mean absolute stress error against the reference solver", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3)
    fig.savefig(FIGURES / "benchmark_paths.png")
    plt.close(fig)


def thesis_cycle(traces, table, cycle="thesis_161"):
    t = traces[cycle]
    steps = np.arange(len(t["eps"]))
    reference = np.array(t["reference"]["xi"])
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.0), sharex=True)
    panels = [("One-step network C128 (step 03), alone and gated", "c128_alone", "gate_c128", True),
              ("Fine-tuned network of step 02, alone and gated", "v17_alone", "gate_v17", False)]
    for ax, (title, alone, gated, with_solver) in zip(axes, panels):
        ax.plot(steps, reference, color=INK, lw=1.6, label="reference solver")
        ax.plot(steps, t[alone]["xi"], color=C_NETWORK, lw=1.4,
                label=f"network alone (ξ error {table[alone][cycle]['xi_mae']:.4f})")
        ax.plot(steps, t[gated]["xi"], color=C_GATED_NETWORK, lw=1.4,
                label=f"gate + network ({table[gated][cycle]['xi_mae']:.4f})")
        if with_solver:
            ax.plot(steps, t["gate_solver"]["xi"], color=C_SOLVER, lw=1.4, ls=(0, (4, 3)),
                    label=f"gate + solver ({table['gate_solver'][cycle]['xi_mae']:.4f})")
        ax.set(ylim=(0.77, 0.93), ylabel="martensite fraction ξ")
        ax.set_title(title, loc="left")
        ax.legend(loc="upper right", ncol=2)
    axes[1].set_xlabel("load step  (the plateau of a cold cycle at −35 °C; ξ axis zoomed to 0.77–0.93)")
    fig.tight_layout()
    fig.savefig(FIGURES / "thesis_cycle_161.png")
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    results = load_path_results()
    traces = json.load(open(RESULTS / "traces.json"))
    write_summary(results)
    classifier_comparison()
    gate_errors_by_step_size()
    benchmark_paths(results)
    thesis_cycle(traces, results)
    walk = traces["random_0"]
    missed = np.flatnonzero((np.array(walk["gate_solver"]["chg_xi"]) == 0)
                            & (np.array(walk["oracle_gate_solver"]["chg_xi"]) == 1))
    print("\nrandom_0: learned gate holds xi against the perfect gate at steps", missed.tolist())
