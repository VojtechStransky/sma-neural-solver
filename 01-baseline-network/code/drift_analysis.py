"""Where does the baseline network (v10) fail in the loop, and why?

For every benchmark cycle the script compares two levels of accuracy, for both
outputs (xi and etr):

  one step   the network gets the reference state at every step
  free run   the network gets its own output

On the cold cycles it also measures the one-step error separately on steps
where the reference does not move ("flat" steps).

  ../results/drift_summary.json
  ../figures/failing_cycles.png    the three worst cycles: xi, etr and stress, reference against free run
  ../figures/error_growth.png      error along the worst cycle: one step against free run

Run:  python drift_analysis.py
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.data import MODELS, load_cycles
from common.plotstyle import INK, MUTED, ORANGE, apply_style
from common.rollout import OnnxStep, free_run, teacher_forced
from common.stress import stress

STEP = pathlib.Path(__file__).resolve().parents[1]
FLAT = 1e-5              # reference |d xi| below this: the step is "flat"
COLD_LOADING_T = -28.0   # loaded at -30 deg C or colder
N_WORST = 3              # cycles shown in failing_cycles.png
TABLE_CYCLES = (101, 160, 161, 213)

REFERENCE = dict(color=INK, lw=1.5, label="reference solver")
NETWORK = dict(color=ORANGE, lw=1.9, ls=(0, (1, 1)), label="network, free run")


def analyse(model, cycles):
    """Errors of every cycle, the runs themselves, and the flat-step statistics of the cold cycles."""
    rows, runs = [], {}
    flat_signed, flat_abs, moving_abs, flat_share = [], [], [], []
    for c in cycles:
        one = teacher_forced(model, c) - c.Y
        run = free_run(model, c)
        runs[c.id] = (one, run)
        err = np.abs(run - c.Y)
        rows.append({"cycle": c.id, "loading_temperature": c.loading_temperature,
                     "one_step_xi": float(np.abs(one[:, 0]).mean()), "free_run_xi": float(err[:, 0].mean()),
                     "one_step_etr": float(np.abs(one[:, 1]).mean()), "free_run_etr": float(err[:, 1].mean())})
        if c.loading_temperature < COLD_LOADING_T:
            flat = np.abs(c.Y[:, 0] - c.X[:, 0]) < FLAT
            flat_signed.append(one[flat, 0]), flat_abs.append(np.abs(one[flat, 0]))
            moving_abs.append(np.abs(one[~flat, 0]))
            flat_share.append((err[flat, 0].sum(), err[:, 0].sum(), flat.sum(), len(flat)))
    share = np.array(flat_share).sum(0)
    cold = {
        "cycles": len(flat_share),
        "flat_steps_percent": 100 * share[2] / share[3],
        "one_step_error_on_flat_steps_mean_signed": float(np.concatenate(flat_signed).mean()),
        "one_step_error_on_flat_steps_mean_abs": float(np.concatenate(flat_abs).mean()),
        "one_step_error_on_moving_steps_mean_abs": float(np.concatenate(moving_abs).mean()),
        "free_run_error_located_on_flat_steps_percent": 100 * share[0] / share[1],
    }
    return rows, runs, cold


def plot_failing_cycles(cycles, runs, worst):
    """One row per cycle: xi and etr along the run, and the stress-strain curve."""
    fig, axes = plt.subplots(len(worst), 3, figsize=(13, 3.5 * len(worst)))
    for (a, b, s), cid in zip(axes, worst):
        c = next(x for x in cycles if x.id == cid)
        run = runs[cid][1]
        err = np.abs(run - c.Y).mean(0)
        a.plot(c.Y[:, 0], **REFERENCE), a.plot(run[:, 0], **NETWORK)
        a.set(xlabel="step", ylabel="martensite fraction ξ")
        a.set_title(f"Cycle {cid}, loaded at {c.loading_temperature:.0f} °C:  error of ξ {err[0]:.3f}", loc="left")
        b.plot(c.Y[:, 1], **REFERENCE), b.plot(run[:, 1], **NETWORK)
        b.set(xlabel="step", ylabel="transformation strain εᵗʳ")
        b.set_title(f"error of εᵗʳ {err[1]:.4f}", loc="left")
        sig_ref = stress(c.strain, c.Y[:, 0], c.Y[:, 1], c.temperature)
        sig_net = stress(c.strain, run[:, 0], run[:, 1], c.temperature)
        s.plot(c.strain, sig_ref, **REFERENCE), s.plot(c.strain, sig_net, **NETWORK)
        s.set(xlabel="strain ε", ylabel="stress σ [MPa]")
        s.set_title(f"error of the stress {np.abs(sig_net - sig_ref).mean():.0f} MPa", loc="left")
    axes[0, 0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(STEP / "figures" / "failing_cycles.png")
    plt.close(fig)


def plot_error_growth(cycles, runs, cid):
    """Error along one cycle: it stays small for one step and accumulates in the free run."""
    c = next(x for x in cycles if x.id == cid)
    one, run = runs[cid]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, k, name in zip(axes, (0, 1), ("martensite fraction ξ", "transformation strain εᵗʳ")):
        ax.plot(np.abs(run[:, k] - c.Y[:, k]), color=ORANGE, lw=1.8, label="free run (own output on the input)")
        ax.plot(np.abs(one[:, k]), color=MUTED, lw=1.2, label="one step (reference on the input)")
        ax.set(xlabel="step", ylabel=f"absolute error of {name.split()[-1]}", title=name)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2)
    fig.suptitle(f"Cycle {cid}: error at every step", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(STEP / "figures" / "error_growth.png")
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    cycles = load_cycles()
    model = OnnxStep(MODELS / "v10.onnx")
    rows, runs, cold = analyse(model, cycles)
    by_id = {r["cycle"]: r for r in rows}
    worst = [r["cycle"] for r in sorted(rows, key=lambda r: -r["free_run_xi"])[:N_WORST]]
    summary = {
        "cold_cycles": cold,
        "worst_cycles_by_free_run_error_of_xi": worst,
        "examples": {str(i): by_id[i] | {"ratio_xi": by_id[i]["free_run_xi"] / by_id[i]["one_step_xi"],
                                         "ratio_etr": by_id[i]["free_run_etr"] / by_id[i]["one_step_etr"]}
                     for i in sorted(set(TABLE_CYCLES) | set(worst))},
        "median_ratio_xi_all_cycles": float(np.median([r["free_run_xi"] / r["one_step_xi"] for r in rows])),
        "median_ratio_etr_all_cycles": float(np.median([r["free_run_etr"] / r["one_step_etr"] for r in rows])),
        "mean_over_cycles": {k: float(np.mean([r[k] for r in rows]))
                             for k in ("one_step_xi", "free_run_xi", "one_step_etr", "free_run_etr")},
    }
    plot_failing_cycles(cycles, runs, worst)
    plot_error_growth(cycles, runs, worst[0])
    (STEP / "results").mkdir(exist_ok=True)
    with open(STEP / "results" / "drift_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
