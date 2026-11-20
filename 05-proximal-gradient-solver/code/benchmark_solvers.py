"""Benchmark: Nelder-Mead against the proximal-gradient solver, on the same energy.

Both solvers minimize the public one-dimensional energy of
00-problem-definition/code/energy_1d.py, so every difference is a difference
between the two methods and not between two models.

  Nelder-Mead   nelder_mead.py, with the schedule of the reference solver;
                all five starting simplices are run ("starts").
  proximal      solve_step of prox_gradient_solver.py, unchanged.

The truth for a step is the best known minimum: the lowest energy found by a
brute-force search (grid with zooming, test_steps.py) or by any solver
run on that step.

Parts
  A  single steps along 60 loading paths (old state = the true state)
  B  single steps from 1,000 random states with very large load jumps
  C  free run: each solver simulates the 60 paths alone, feeding back its
     own state; deviation from the true path
  D  accuracy against budget: both solvers with a reduced number of
     iterations, on the steps of part A
  E  wall time per step, one process, both solvers on the same steps

Cost is counted in energy evaluations (independent of the machine) and
measured in wall time (same machine, same plain NumPy code for both).

Writes ../results/solver_benchmark.json and ../results/solver_benchmark.txt.

Run:  python benchmark_solvers.py --workers 8      (about 25 minutes)
"""
import argparse
import json
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from nelder_mead import STARTS, nelder_mead, pi
from prox_gradient_solver import f_all, solve_step
from test_steps import (MAT, XI_START, brute_force_minimum, loading_programmes, must_change, random_states,
                        stress, update_memory)

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
NM_BUDGETS = (3, 10, 30, 100, 300)           # iterations per phase, part D
PROX_BUDGETS = (1, 2, 3, 4, 6, 8, 12, 20)    # iterations, part D
TIMING_REPEATS = 3


def proximal(state, max_iter=200):
    """Proximal-gradient step. Returns xi, etr, iterations, energy evaluations, regime."""
    xi_n, etr_n, _, eps, T = state
    calls = 0

    def smooth(xi, q):
        nonlocal calls
        calls += 1
        return f_all(xi, q, xi_n, eps, T, MAT)

    xi, etr, iterations, regime = solve_step(*state, MAT, max_iter=max_iter, smooth=smooth)
    return float(xi), float(etr), iterations, calls, regime


# ----------------------------------------------------------------- scoring

def single_step(state):
    """Both solvers and the brute force on one step. Returns the truth and one record per answer."""
    answers = {"proximal": proximal(state)}
    for i, start in enumerate(STARTS):
        answers[f"nm{i}"] = nelder_mead(state, start)
    candidates = [brute_force_minimum(*state, MAT)[1:]] + [a[:2] for a in answers.values()]
    energies = [pi(xi, etr, state) for xi, etr in candidates]
    xi_t, etr_t = candidates[int(np.argmin(energies))]
    F_t, sig_t = min(energies), stress(xi_t, etr_t, state[3], MAT)

    record = {"state": [float(v) for v in state], "truth": [float(xi_t), float(etr_t)],
              "must_change": bool(must_change(state)),
              "regime": answers["proximal"][4], "iterations": answers["proximal"][2]}
    for name, a in answers.items():
        record[name] = {"gap": pi(a[0], a[1], state) - F_t, "d_xi": a[0] - xi_t, "d_etr": a[1] - etr_t,
                        "d_stress": float(stress(a[0], a[1], state[3], MAT) - sig_t),
                        "evaluations": a[3] if name == "proximal" else a[2],
                        "held": bool(a[0] == state[0] and a[1] == state[1])}
    return (float(xi_t), float(etr_t)), record


def path_steps(programme):
    """Part A for one loading path: single steps along the true path."""
    eps_seq, T_seq = programme
    state, records = (XI_START, 0.0, 0.0), []
    for eps, T in zip(eps_seq, T_seq):
        (xi, etr), record = single_step((*state, eps, T))
        records.append(record)
        state = (xi, etr, float(update_memory(state[2], state[0], xi, etr)))
    return records


def free_run(args):
    """Part C: one solver simulates one loading path alone. Returns xi, etr, stress per step."""
    programme, solver = args
    state, out = (XI_START, 0.0, 0.0), []
    for eps, T in zip(*programme):
        step = (*state, eps, T)
        xi, etr = proximal(step)[:2] if solver == "proximal" else nelder_mead(step, STARTS[int(solver[2])])[:2]
        out.append([xi, etr, float(stress(xi, etr, eps, MAT))])
        state = (xi, etr, float(update_memory(state[2], state[0], xi, etr)))
    return out


def reduced_budget(args):
    """Part D: one step with reduced budgets. Returns {solver: {budget: (stress error, evaluations)}}."""
    state, truth = args
    sig_t = stress(*truth, state[3], MAT)
    out = {"proximal": {}, "nelder_mead": {}}
    for k in PROX_BUDGETS:
        xi, etr, _, evaluations, _ = proximal(state, max_iter=k)
        out["proximal"][k] = (abs(float(stress(xi, etr, state[3], MAT) - sig_t)), evaluations)
    for n in NM_BUDGETS:
        xi, etr, evaluations = nelder_mead(state, 1.0, iterations=n)
        out["nelder_mead"][n] = (abs(float(stress(xi, etr, state[3], MAT) - sig_t)), evaluations)
    return out


def wall_time(states):
    """Part E: seconds per step, the fastest of TIMING_REPEATS runs, one process."""
    times = {"proximal": [], "nelder_mead": []}
    for state in states:
        for name, solver in (("proximal", lambda: solve_step(*state, MAT)),
                             ("nelder_mead", lambda: nelder_mead(state, 1.0))):
            best = np.inf
            for _ in range(TIMING_REPEATS):
                t0 = time.perf_counter()
                solver()
                best = min(best, time.perf_counter() - t0)
            times[name].append(best)
    return times


# ----------------------------------------------------------------- summary

def percentiles(values):
    v = np.asarray(values, float)
    return {"median": float(np.median(v)), "p99": float(np.percentile(v, 99)), "max": float(v.max()),
            "mean": float(v.mean())}


def summarize_steps(records):
    """Accuracy and cost of both solvers over a list of single-step records."""
    nm = [f"nm{i}" for i in range(len(STARTS))]
    change = np.array([r["must_change"] for r in records])
    out = {"steps": len(records), "steps_where_the_state_must_not_change": int((~change).sum())}
    for label, names in (("proximal", ["proximal"]), ("nelder_mead", nm)):
        get = lambda key: np.array([[r[n][key] for n in names] for r in records], float)
        d_sig, gap = np.abs(get("d_stress")), get("gap")
        worst = d_sig.max(axis=1)                       # per step: the worst of the answers
        held = get("held").astype(bool)                 # the answer is exactly the old state
        stuck = held & change[:, None]                  # ... although the state must change
        out[label] = {
            "answers": int(d_sig.size),
            "stress_error_MPa": percentiles(d_sig.ravel()),
            "energy_gap_MPa": percentiles(gap.ravel()),
            "xi_error": percentiles(np.abs(get("d_xi")).ravel()),
            "etr_error": percentiles(np.abs(get("d_etr")).ravel()),
            "steps_off_by_more_than_MPa": {str(t): int((worst > t).sum()) for t in (0.01, 0.1, 1.0, 10.0)},
            "evaluations": percentiles(get("evaluations").ravel()),
            "exact_hold": f"{int(held[~change].all(axis=1).sum())} of {int((~change).sum())}",
            "no_change_although_the_state_must_change": {
                "steps": int(stuck.any(axis=1).sum()), "of_steps": int(change.sum()),
                "answers": int(stuck.sum()), "of_answers": int(change.sum() * len(names)),
                "largest_stress_error_MPa": float(d_sig[stuck].max()) if stuck.any() else 0.0,
                "share_of_the_answers_off_by_more_than_0.1_MPa": (
                    float((stuck & (d_sig > 0.1)).sum() / max((d_sig > 0.1).sum(), 1)))},
        }
        if len(names) > 1:                              # how much the five starts disagree
            spread = np.ptp(get("d_stress"), axis=1)
            out[label]["spread_between_starts_MPa"] = percentiles(spread)
            out[label]["steps_where_starts_differ_by_more_than_MPa"] = {
                str(t): int((spread > t).sum()) for t in (0.01, 0.1, 1.0)}
    return out


def by_regime(records):
    """99th percentile and maximum of the stress error, per regime of the true step."""
    out = {}
    for regime in sorted({r["regime"] for r in records}):
        rows = [r for r in records if r["regime"] == regime]
        nm = np.abs([[r[f"nm{i}"]["d_stress"] for i in range(len(STARTS))] for r in rows])
        px = np.abs([r["proximal"]["d_stress"] for r in rows])
        out[regime] = {"steps": len(rows),
                       "nelder_mead_p99": float(np.percentile(nm, 99)), "nelder_mead_max": float(nm.max()),
                       "proximal_p99": float(np.percentile(px, 99)), "proximal_max": float(px.max())}
    return out


def table(title, s):
    lines = [title, f"  {s['steps']} steps, the state must not change in "
                    f"{s['steps_where_the_state_must_not_change']} of them (optimality test of the old state)",
             f"  {'':34s}{'Nelder-Mead':>16s}{'proximal':>16s}"]

    def row(label, key, sub, fmt):
        lines.append(f"  {label:34s}{format(s['nelder_mead'][key][sub], fmt):>16s}"
                     f"{format(s['proximal'][key][sub], fmt):>16s}")

    for sub in ("median", "p99", "max"):
        row(f"stress error [MPa], {sub}", "stress_error_MPa", sub, ".2e")
    for sub in ("p99", "max"):
        row(f"energy above the minimum [MPa], {sub}", "energy_gap_MPa", sub, ".2e")
    row("|xi error|, max", "xi_error", "max", ".2e")
    row("|etr error|, max", "etr_error", "max", ".2e")
    for t in ("0.01", "0.1", "1.0", "10.0"):
        row(f"steps off by more than {t} MPa", "steps_off_by_more_than_MPa", t, "d")
    for sub in ("median", "max"):
        row(f"energy evaluations, {sub}", "evaluations", sub, ".0f")
    lines.append(f"  {'exact hold when nothing must change':34s}{s['nelder_mead']['exact_hold']:>16s}"
                 f"{s['proximal']['exact_hold']:>16s}")
    stuck = {k: s[k]["no_change_although_the_state_must_change"] for k in ("nelder_mead", "proximal")}
    lines.append(f"  {'no change although it must change':34s}"
                 + "".join(f"{str(stuck[k]['steps']) + ' of ' + str(stuck[k]['of_steps']):>16s}"
                           for k in ("nelder_mead", "proximal")))
    lines.append(f"  Nelder-Mead stuck at the old state: {stuck['nelder_mead']['answers']} of "
                 f"{stuck['nelder_mead']['of_answers']} answers, stress error up to "
                 f"{stuck['nelder_mead']['largest_stress_error_MPa']:.2f} MPa; these are "
                 f"{100 * stuck['nelder_mead']['share_of_the_answers_off_by_more_than_0.1_MPa']:.0f} % of its "
                 "answers that are more than 0.1 MPa off")
    sp = s["nelder_mead"]["spread_between_starts_MPa"]
    lines.append(f"  Nelder-Mead, spread of the five starts [MPa]: median {sp['median']:.1e}, "
                 f"p99 {sp['p99']:.1e}, max {sp['max']:.1e}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    programmes = loading_programmes(np.random.default_rng(0))
    solvers = ["proximal"] + [f"nm{i}" for i in range(len(STARTS))]
    out, text = {"starts": STARTS.tolist()}, []

    with ProcessPoolExecutor(args.workers) as pool:
        # A, B: single steps
        per_path = list(pool.map(path_steps, programmes))
        path_records = [r for records in per_path for r in records]
        random_records = [r for _, r in pool.map(single_step, random_states(np.random.default_rng(1)),
                                                 chunksize=20)]
        out["path_steps"], out["random_steps"] = summarize_steps(path_records), summarize_steps(random_records)
        out["path_steps_by_regime"] = by_regime(path_records)
        text += [table("A. Single steps along 60 loading paths", out["path_steps"]), "",
                 table("B. Single steps from random states with large load jumps", out["random_steps"]), ""]

        # C: free runs
        jobs = [(p, s) for p in programmes for s in solvers]
        runs = list(pool.map(free_run, jobs, chunksize=4))
        free = {s: [] for s in solvers}
        for (_, s), run in zip(jobs, runs):
            free[s].append(run)
        truth = [np.array([[*r["truth"], stress(*r["truth"], r["state"][3], MAT)] for r in records])
                 for records in per_path]
        out["free_run"] = {}
        for s in solvers:
            dev = [np.abs(np.array(run) - t) for run, t in zip(free[s], truth)]
            out["free_run"][s] = {
                "stress_mean_MPa": float(np.mean([d[:, 2].mean() for d in dev])),
                "stress_max_MPa": float(max(d[:, 2].max() for d in dev)),
                "xi_max": float(max(d[:, 0].max() for d in dev)),
                "etr_max": float(max(d[:, 1].max() for d in dev)),
                "end_of_path_stress_max_MPa": float(max(d[-1, 2] for d in dev)),
                "paths_off_by_more_than_1_MPa": int(sum(d[:, 2].max() > 1.0 for d in dev)),
                "per_path_stress_max_MPa": [float(d[:, 2].max()) for d in dev]}
        text.append("C. Free run along the 60 paths: deviation from the true path")
        text.append(f"  {'solver':12s}{'stress mean':>13s}{'stress max':>13s}{'xi max':>11s}{'etr max':>11s}"
                    f"{'paths > 1 MPa':>15s}")
        for s in solvers:
            f = out["free_run"][s]
            name = "proximal" if s == "proximal" else f"NM start {STARTS[int(s[2])]:.2f}"
            text.append(f"  {name:12s}{f['stress_mean_MPa']:13.2e}{f['stress_max_MPa']:13.2e}{f['xi_max']:11.1e}"
                        f"{f['etr_max']:11.1e}{f['paths_off_by_more_than_1_MPa']:15d}")
        text.append("")

        # D: accuracy against budget
        budget = list(pool.map(reduced_budget, [(r["state"], r["truth"]) for r in path_records], chunksize=20))
        out["budget"] = {}
        text.append("D. Accuracy against budget, single steps along the paths (Nelder-Mead: start 1.00)")
        for solver, budgets in (("nelder_mead", NM_BUDGETS), ("proximal", PROX_BUDGETS)):
            out["budget"][solver] = {}
            for b in budgets:
                err = np.array([row[solver][b][0] for row in budget])
                evaluations = np.array([row[solver][b][1] for row in budget])
                out["budget"][solver][str(b)] = {"evaluations_mean": float(evaluations.mean()),
                                                 "stress_error_MPa": percentiles(err)}
                unit = "iterations per phase" if solver == "nelder_mead" else "iterations"
                text.append(f"  {solver:12s}{b:4d} {unit:21s}{evaluations.mean():8.1f} evaluations   "
                            f"stress error p99 {np.percentile(err, 99):.1e}, max {err.max():.1e} MPa")
        text.append("")

    # E: wall time, with the pool closed
    times = wall_time([r["state"] for r in path_records])
    out["wall_time_ms"] = {k: percentiles(np.array(v) * 1e3) for k, v in times.items()}
    out["wall_time_ms"]["ratio_of_medians"] = (out["wall_time_ms"]["nelder_mead"]["median"]
                                               / out["wall_time_ms"]["proximal"]["median"])
    text.append("E. Wall time per step [ms], single steps along the paths, plain NumPy, one process")
    for k in ("nelder_mead", "proximal"):
        w = out["wall_time_ms"][k]
        text.append(f"  {k:12s} median {w['median']:.2f}, mean {w['mean']:.2f}, p99 {w['p99']:.2f}, max {w['max']:.2f}")

    # data for the figures
    out["figure_data"] = {
        "path_stress_error": {"proximal": [abs(r["proximal"]["d_stress"]) for r in path_records],
                              "nelder_mead": [[abs(r[f"nm{i}"]["d_stress"]) for i in range(len(STARTS))]
                                              for r in path_records]},
        "random_stress_error": {"proximal": [abs(r["proximal"]["d_stress"]) for r in random_records],
                                "nelder_mead": [[abs(r[f"nm{i}"]["d_stress"]) for i in range(len(STARTS))]
                                                for r in random_records]},
        "path_evaluations": {"proximal": [r["proximal"]["evaluations"] for r in path_records],
                             "nelder_mead": [r["nm4"]["evaluations"] for r in path_records]},
        "wall_time_ms": {k: (np.array(v) * 1e3).tolist() for k, v in times.items()},
        "programmes": [[np.asarray(e).tolist(), np.asarray(T).tolist()] for e, T in programmes],
        "truth": [t.tolist() for t in truth],
        "free_run": free,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "solver_benchmark.txt").write_text("\n".join(text) + "\n")
    with open(RESULTS / "solver_benchmark.json", "w") as fh:
        json.dump(out, fh)
    print("\n".join(text))


if __name__ == "__main__":
    main()
