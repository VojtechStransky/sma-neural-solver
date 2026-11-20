"""Score a one-step model on the 250 benchmark cycles.

Two levels of accuracy are measured:
  one step   the model gets the reference state at every step
  free run   the model gets the reference state once, then its own output

Usage (from the repository root):
    python -m common.benchmark models/v17.onnx --out 02-rollout-fine-tuning/results/v17

Writes <out>_per_cycle.csv and <out>_summary.json.
"""
import argparse
import csv
import json

import numpy as np

from common.data import load_cycles, split_cycles
from common.rollout import OnnxStep, free_run, r_squared, teacher_forced

COLD_LOADING_T = -28.0     # "cold" cycle: loaded at -30 deg C (= Ms) or colder
LARGE_DEVIATION = 0.2      # a deviation of xi above this inside a cycle is reported separately


def score(model, cycles):
    """Per-cycle rows and a summary dict."""
    rows, ref, one, run = [], [], [], []
    for c in cycles:
        tf_pred = teacher_forced(model, c)
        fr_pred = free_run(model, c)
        err = np.abs(fr_pred - c.Y)
        rows.append({
            "cycle": c.id, "steps": len(c.X),
            "mean_temperature": round(c.mean_temperature, 2),
            "loading_temperature": round(c.loading_temperature, 2),
            "one_step_mae_xi": float(np.abs(tf_pred[:, 0] - c.Y[:, 0]).mean()),
            "free_run_mae_xi": float(err[:, 0].mean()),
            "free_run_mae_etr": float(err[:, 1].mean()),
            "free_run_max_xi": float(err[:, 0].max()),
            "finite": bool(np.isfinite(fr_pred).all()),
        })
        ref.append(c.Y), one.append(tf_pred), run.append(fr_pred)
    ref, one, run = np.concatenate(ref), np.concatenate(one), np.concatenate(run)

    col = lambda name: np.array([r[name] for r in rows])
    cold = col("loading_temperature") < COLD_LOADING_T
    held_out = np.zeros(len(rows), bool)
    held_out[split_cycles(len(rows))[2]] = True          # the 37 cycles never used in training
    worst = rows[int(np.argmax(col("free_run_mae_xi")))]
    span = ref.max(0) - ref.min(0)
    mae_xi = col("free_run_mae_xi")
    summary = {
        "cycles": len(rows), "steps": int(len(ref)),
        "one_step_r2_xi": r_squared(ref[:, 0], one[:, 0]),
        "one_step_r2_etr": r_squared(ref[:, 1], one[:, 1]),
        "one_step_mae_xi": float(np.abs(one[:, 0] - ref[:, 0]).mean()),
        "one_step_mae_etr": float(np.abs(one[:, 1] - ref[:, 1]).mean()),
        "one_step_mae_xi_percent_of_range": float(100 * np.abs(one[:, 0] - ref[:, 0]).mean() / span[0]),
        "one_step_mae_etr_percent_of_range": float(100 * np.abs(one[:, 1] - ref[:, 1]).mean() / span[1]),
        "free_run_mae_xi": float(mae_xi.mean()),
        "free_run_mae_xi_median": float(np.median(mae_xi)),
        "free_run_mae_etr": float(col("free_run_mae_etr").mean()),
        "free_run_r2_xi": r_squared(ref[:, 0], run[:, 0]),
        "free_run_r2_etr": r_squared(ref[:, 1], run[:, 1]),
        "cold_cycles": int(cold.sum()),
        "free_run_mae_xi_cold": float(mae_xi[cold].mean()),
        "free_run_mae_xi_not_cold": float(mae_xi[~cold].mean()),
        "held_out_cycles": int(held_out.sum()),
        "free_run_mae_xi_held_out": float(mae_xi[held_out].mean()),
        "diverged_cycles": int((~col("finite")).sum()),
        "cycles_with_large_deviation": int((col("free_run_max_xi") > LARGE_DEVIATION).sum()),
        "largest_deviation_xi": float(col("free_run_max_xi").max()),
        "worst_cycle": worst["cycle"], "worst_cycle_mae_xi": worst["free_run_mae_xi"],
    }
    return rows, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", help="path to an ONNX model")
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()

    rows, summary = score(OnnxStep(args.model), load_cycles())
    with open(args.out + "_per_cycle.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.out + "_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    for key, value in summary.items():
        print(f"{key:36s} {value:.6g}" if isinstance(value, float) else f"{key:36s} {value}")


if __name__ == "__main__":
    main()
