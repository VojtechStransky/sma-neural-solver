"""How fast the error falls, iteration by iteration: Nelder-Mead against the proximal solver.

Both solvers minimize the public one-dimensional energy, on the path steps of
benchmark_solvers.py in which the state must change. After every iteration
the current answer is read out and its stress is compared with the truth of
that step (the best known minimum, stored by benchmark_solvers.py).

  Nelder-Mead   nelder_mead.py, all five starting simplices, 3 phases x 100
                iterations; the answer after an iteration is the best vertex
                of the simplex.
  proximal      solve_step of prox_gradient_solver.py (Newton step); a step
                that has stopped keeps its final answer.

Needs ../results/solver_benchmark.json (python benchmark_solvers.py).
Writes ../results/benchmark_convergence.json and ../results/benchmark_convergence.txt.

Run:  python benchmark_convergence.py --workers 8      (about 15 minutes)
"""
import argparse
import json
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from nelder_mead import STARTS, nelder_mead
from prox_gradient_solver import solve_step
from test_steps import MAT, XI_START, must_change, stress, update_memory

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
NM_ITERATIONS, PROX_ITERATIONS = 300, 200
PERCENTILES = (10, 25, 50, 75, 90, 99)


def errors(args):
    """Stress error after every iteration: (5, 300) for Nelder-Mead, (300,) for the proximal solver."""
    state, sigma_true = args
    nm = []
    for start in STARTS:
        trace = []
        nelder_mead(state, start, trace=trace)
        nm.append([abs(stress(xi, etr, state[3], MAT) - sigma_true) for xi, etr in trace])
    trace = []
    _, _, stop, _ = solve_step(*state, MAT, max_iter=PROX_ITERATIONS, trace=trace)
    trace += [trace[-1]] * (NM_ITERATIONS - len(trace))
    prox = [abs(stress(xi, etr, state[3], MAT) - sigma_true) for xi, etr in trace]
    return np.array(nm), np.array(prox), stop


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    data = json.load(open(RESULTS / "solver_benchmark.json"))["figure_data"]

    jobs = []                                  # the path steps, rebuilt from the stored truth
    for (eps_seq, T_seq), truth in zip(data["programmes"], data["truth"]):
        state = (XI_START, 0.0, 0.0)
        for eps, T, (xi, etr, sigma) in zip(eps_seq, T_seq, truth):
            step = (*state, eps, T)
            if must_change(step):
                jobs.append((step, sigma))
            state = (xi, etr, float(update_memory(state[2], state[0], xi, etr)))

    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(errors, jobs, chunksize=10))
    nm = np.concatenate([r[0] for r in rows])            # (5 * steps, 300): every answer
    prox = np.array([r[1] for r in rows])                 # (steps, 300)
    stops = [int(r[2]) for r in rows]

    out = {"steps": len(jobs), "proximal_stop_iterations": stops}
    for name, e in (("nelder_mead", nm), ("proximal", prox)):
        out[name] = {"percentiles": {str(p): np.percentile(e, p, axis=0).tolist() for p in PERCENTILES},
                     "share_above_0.1_MPa": (100 * (e > 0.1).mean(axis=0)).tolist()}
    RESULTS.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS / "benchmark_convergence.json", "w"))

    lines = [f"{len(jobs)} path steps in which the state must change; stress error [MPa] after k iterations",
             f"  {'k':>4s} {'NM median':>11s} {'NM p99':>10s} {'NM >0.1 %':>10s} "
             f"{'prox median':>12s} {'prox p99':>10s} {'prox >0.1 %':>12s}"]
    for k in (1, 2, 3, 4, 6, 10, 20, 50, 100, 200, 300):
        i = k - 1
        lines.append(f"  {k:4d} {np.median(nm[:, i]):11.1e} {np.percentile(nm[:, i], 99):10.1e} "
                     f"{100 * (nm[:, i] > 0.1).mean():10.1f} {np.median(prox[:, i]):12.1e} "
                     f"{np.percentile(prox[:, i], 99):10.1e} {100 * (prox[:, i] > 0.1).mean():12.1f}")
    lines.append(f"  proximal solver stopped by iteration {max(stops)} in every step "
                 f"(median {int(np.median(stops))})")
    text = "\n".join(lines)
    (RESULTS / "benchmark_convergence.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
