"""Verify the proximal-gradient solver on the public one-dimensional model.

Two tests:

A. Steps along loading paths (the realistic case). 60 loading programmes
   (strain cycles at fixed temperature, and temperature cycles at fixed
   strain) are simulated with the solver itself. At every step the solver's
   answer is compared with a brute-force minimum of the same energy.

B. Stress test: random old states with very large load jumps, far outside
   what a loading path produces.

Every step is solved with the four step sizes of prox_gradient_solver.py
(Newton step, one step per unknown in xi and etr, one step per unknown in xi
and q, one common step). The exact prox is the same in all four. The Newton
step may use 200 iterations, the others 2,000.

Brute force: brute_force_minimum of test_steps.py, a grid with repeated
zooming that also tries the kink lines.

Writes ../results/solver_verification.json and ../results/solver_verification.txt.

Run:  python verify_solver.py --workers 8          (about 10 minutes)
"""
import argparse
import json
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from prox_gradient_solver import directional_curvature, f_all, incremental_energy, solve_step, step_directions
from test_steps import (MAT, N_RANDOM, XI_START, brute_force_minimum, loading_programmes, random_states, stress,
                        update_memory)

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
MAX_ITER = {"newton": 200, "steps_xi_etr": 2000, "steps_xi_q": 2000, "one_step": 2000}
LABEL = {"newton": "Newton step", "steps_xi_etr": "steps for xi and etr",
         "steps_xi_q": "steps for xi and q", "one_step": "one common step"}


def compare(state):
    """All step sizes against brute force for one step."""
    xi_n, etr_n, M, eps, T = state
    F_ref, xi_ref, etr_ref = brute_force_minimum(*state, MAT)
    sig_ref = stress(xi_ref, etr_ref, eps, MAT)
    row = {"state": [float(v) for v in state], "xi_ref": float(xi_ref), "etr_ref": float(etr_ref)}
    for metric, max_iter in MAX_ITER.items():
        calls = 0

        def smooth(xi, q):
            nonlocal calls
            calls += 1
            return f_all(xi, q, xi_n, eps, T, MAT)
        xi, etr, iters, regime = solve_step(*state, MAT, metric=metric, max_iter=max_iter, smooth=smooth)
        F = float(incremental_energy(xi, etr, xi_n, etr_n, M, eps, T, MAT))
        row[metric] = {"iterations": iters, "evaluations": calls, "regime": regime,
                       "energy_gap": F - float(F_ref), "d_xi": float(xi - xi_ref),
                       "d_stress": float(stress(xi, etr, eps, MAT) - sig_ref)}
        if metric == "newton":                    # curvature of f at the solution
            hess = f_all(xi, xi * etr, xi_n, eps, T, MAT)[2]
            row["curvature"] = {"xi": float(hess[0, 0]), "q": float(hess[1, 1]),
                                "condition": float(np.linalg.cond(hess))}
            for pair in ("steps_xi_q", "steps_xi_etr"):
                row["curvature"]["rho_" + pair] = float(
                    directional_curvature(hess, step_directions(xi, etr, pair))[1])
    return row


def path_rows(programme):
    """Test A for one loading path. The path itself is simulated with the Newton step."""
    state, rows = (XI_START, 0.0, 0.0), []
    for eps, T in zip(*programme):
        rows.append(compare((*state, eps, T)))
        xi, etr, _, _ = solve_step(*state, eps, T, MAT)
        state = (xi, etr, update_memory(state[2], state[0], xi, etr))
    return rows


def report_curvature(rows, lines):
    """Curvature of f at the solutions, and how strongly the two step directions are coupled."""
    get = lambda key: np.array([r["curvature"][key] for r in rows])
    summary = {"xi_MPa": [float(get("xi").min()), float(get("xi").max())],
               "q_MPa": [float(get("q").min()), float(get("q").max())],
               "condition_max": float(get("condition").max())}
    lines.append(f"  curvature of f at the solutions [MPa]: along xi {summary['xi_MPa'][0]:.3g} to "
                 f"{summary['xi_MPa'][1]:.3g}, along q {summary['q_MPa'][0]:.3g} to {summary['q_MPa'][1]:.3g}; "
                 f"condition number of the Hessian up to {summary['condition_max']:.3g}")
    for pair in ("steps_xi_q", "steps_xi_etr"):
        rho = get("rho_" + pair)
        left = (1.0 + rho.max()) / (1.0 - rho.max())
        summary[pair] = {"rho_median": float(np.median(rho)), "rho_max": float(rho.max()),
                         "share_above_0.99": float((rho > 0.99).mean()), "condition_left_max": float(left)}
        lines.append(f"  coupling rho of the two step directions, {LABEL[pair]}: median {np.median(rho):.2f}, "
                     f"max {rho.max():.4f}, above 0.99 in {100 * (rho > 0.99).mean():.0f} % of the steps; "
                     f"condition number left after the scaling up to {left:.3g}")
    return summary


def report(rows, metric, lines):
    it = np.array([r[metric]["iterations"] for r in rows])
    calls = np.array([r[metric]["evaluations"] for r in rows])
    gap = np.array([r[metric]["energy_gap"] for r in rows])
    d_sig = np.abs([r[metric]["d_stress"] for r in rows])
    d_xi = np.abs([r[metric]["d_xi"] for r in rows])
    cap = MAX_ITER[metric]
    summary = {"steps": len(rows), "max_iter": cap,
               "iterations_median": float(np.median(it)), "iterations_p99": float(np.percentile(it, 99)),
               "iterations_max": int(it.max()), "not_converged": int((it >= cap).sum()),
               "evaluations_median": float(np.median(calls)), "evaluations_max": int(calls.max()),
               "energy_above_brute_force_max": float(gap.max()),
               "energy_below_brute_force_max": float(max(-gap.min(), 0.0)),
               "d_xi_max": float(d_xi.max()), "d_stress_p99": float(np.percentile(d_sig, 99)),
               "d_stress_max": float(d_sig.max()),
               "steps_off_by_more_than_0.01_MPa": int((d_sig > 0.01).sum()),
               "steps_off_by_more_than_0.1_MPa": int((d_sig > 0.1).sum()),
               "steps_with_higher_energy_than_brute_force": int((gap > 1e-9).sum()),
               "steps_with_lower_energy_than_brute_force": int((gap < -1e-9).sum())}
    lines += [f"  {LABEL[metric]} ({metric})",
              f"      iterations median {summary['iterations_median']:.0f}, p99 "
              f"{summary['iterations_p99']:.0f}, max {summary['iterations_max']} "
              f"(not converged in {cap}: {summary['not_converged']}); energy evaluations median "
              f"{summary['evaluations_median']:.0f}, max {summary['evaluations_max']}",
              f"      energy above brute force: max {gap.max():.1e} MPa "
              f"({summary['steps_with_higher_energy_than_brute_force']} steps above 1e-9); "
              f"below it: max {summary['energy_below_brute_force_max']:.1e} "
              f"({summary['steps_with_lower_energy_than_brute_force']} steps)",
              f"      |stress - brute force|: p99 {summary['d_stress_p99']:.1e}, max {d_sig.max():.1e} MPa "
              f"({summary['steps_off_by_more_than_0.1_MPa']} steps above 0.1 MPa); "
              f"|xi - brute force| max {d_xi.max():.1e}"]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=4)
    workers = parser.parse_args().workers
    rng = np.random.default_rng(0)
    programmes = loading_programmes(rng)
    states = random_states(rng)          # continues the same generator as the paths

    with ProcessPoolExecutor(workers) as pool:
        rows_a = [row for rows in pool.map(path_rows, programmes) for row in rows]
        rows_b = list(pool.map(compare, states, chunksize=10))

    regimes = {}
    for row in rows_a:
        regimes[row["newton"]["regime"]] = regimes.get(row["newton"]["regime"], 0) + 1
    out, lines = {}, [f"A. {len(rows_a)} steps along {len(programmes)} loading paths", f"   regimes: {regimes}"]
    out["paths"] = {m: report(rows_a, m, lines) for m in MAX_ITER}
    out["paths"]["regimes"] = regimes
    out["paths"]["curvature"] = report_curvature(rows_a, lines)
    lines += ["", f"B. {N_RANDOM} random states with large load jumps"]
    out["random"] = {m: report(rows_b, m, lines) for m in MAX_ITER}

    out["path_rows"], out["random_rows"] = rows_a, rows_b
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "solver_verification.json", "w") as fh:
        json.dump(out, fh)
    text = "\n".join(lines)
    (RESULTS / "solver_verification.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
