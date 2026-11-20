"""Gated free run along one loading path: the gate decides, the solver or a network computes.

NEEDS THE REFERENCE SOLVER, which is not part of this repository. Point the
environment variable SMA_SOLVER_PATH at the directory that holds it:

    SMA_SOLVER_PATH=/path/to/solver python gated_simulation.py solver tension_T+100

Each load step:

  1. The gate (gate.py) predicts, for xi and for eps_tr, whether the variable changes.
  2. A variable predicted not to change keeps its exact old value.
  3. The other variables are computed by the chosen backend:
       solver   minimizes the energy only over the variables that change
                (one-variable search if one is held, the full solver if both change);
       network  one forward pass of an ONNX surrogate; the output of a held variable is discarded.
  4. The new state goes through the solver's state -> stress routine, the same for every backend.

The run carries its own state from step to step. It is never reset to the reference.

Controls:
    --always-compute   the gate always answers "changes" (reproduces the backend alone)
    --oracle           gate decisions are taken from the reference path (a perfect gate)

Backends and paths:
    backend:  solver | network (with --onnx model.onnx; add --total-strain for a network
              that was trained on total strain, such as the fine-tuned network of step 02)
    path:     any key of ../results/reference_paths.json

Output: a JSON file with the trajectory, the gate decisions and the errors against the reference.
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from gate import Gate, THRESHOLD_EPS_TR, THRESHOLD_XI

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
T_START = -20.0                    # the model is initialised at this temperature
THERMAL_EXPANSION, T_REFERENCE = 7e-6, -30.0

solver_dir = os.environ.get("SMA_SOLVER_PATH")
if not solver_dir:
    sys.exit("Set SMA_SOLVER_PATH to the directory of the reference solver (not part of this repository).")
sys.path.insert(0, solver_dir)
import par_fmin_search                                     # noqa: E402  (private solver)
import stress_strain_attut as reference                     # noqa: E402
from input_parameters import get_parameters                # noqa: E402
from main import _build_initial_internal_variables, _update_temperature_elasticity   # noqa: E402
from stress_strain_nn import stress_strain_nn_el           # noqa: E402


# ----------------------------------------------------------------- counting solver work

energy_evaluations = 0
_energy = reference.free_energy_all_el


def _counted_energy(*args, **kwargs):
    global energy_evaluations
    energy_evaluations += 1
    return _energy(*args, **kwargs)


reference.free_energy_all_el = _counted_energy    # every objective evaluation of the solver is counted


# ----------------------------------------------------------------- simulation loop

def set_temperature(par, T):
    """Temperature-dependent stiffness and reorientation stress."""
    _update_temperature_elasticity(T, par)
    par["sig_reo_o"] = par["sig_reo_o_To"] + np.tanh(T / 250.0 * 3.0) * 250.0 / 3.0 * par["d_T_sig_reo_o"]


def uniaxial_strain(eps, n):
    """Volume-preserving uniaxial strain in the six-component notation of the solver."""
    e = np.zeros((6, n))
    e[0], e[1], e[2] = eps, -eps / 2.0, -eps / 2.0
    return e


def state_of(iv):
    """(xi, eps_tr, M) of the material point."""
    return float(np.mean(iv["ksi"])), float(np.mean(iv["eps_tr"][0])), float(np.mean(iv["M_tr_cum"]))


class FixedPrediction:
    """Stands in for an ONNX session and returns one prescribed (xi, eps_tr).

    The solver's network routine turns that state into stress and updated
    internal variables, so every backend shares one state -> stress code path.
    """

    class _Input:
        name = "x"

    value = None

    def get_inputs(self):
        return [self._Input()]

    def run(self, _, feed):
        rows = np.asarray(next(iter(feed.values()))).shape[0]
        return [np.tile(self.value, (rows, 1))]


# ----------------------------------------------------------------- backends

def one_variable_search(e, T, iv, par, search_xi):
    """Minimize the energy over one variable; the other stays at its old value.

    Same energy and same Nelder-Mead routine as the full solver, with a
    two-point simplex: "no change" and one trial step.
    """
    par = par if isinstance(par, reference._ParamAdapter) else reference._ParamAdapter(par)
    n, K, delta = int(par["n"]), float(par["K"]), float(par["delta"])
    index, spread = np.arange(n), np.linspace(1e-2, 1.0, n)
    xi_old, etr_old = iv["ksi"].copy(), iv["eps_tr"].copy()
    simplex = np.zeros((2, 1, n))

    if search_xi:                              # variable: K * (xi - xi_n); eps_tr fixed
        trial = 0.05 * K * np.where(xi_old > 0.5, -1.0, 1.0)
        simplex[1, 0] = trial * spread
        fixed = etr_old[0].copy()

        def objective(x):
            return _counted_energy(e, e, T, np.vstack([x[0:1], fixed[None]]), iv, par, index)[0]
    else:                                      # variable: eps_tr; xi fixed
        D, asym = np.asarray(par["D"], float), float(par["asym"])
        norm = reference.norm_e2(etr_old, K, asym, D)
        etr_old[:, norm < 1e-4] += (np.array([1.0, -0.5, -0.5, 0, 0, 0]) * K / 10.0)[:, None]
        norm = reference.norm_e2(etr_old, K, asym, D)
        trial_norm = norm * K + 0.05 * np.where(norm > 0.5, -1.0, 1.0)
        trial = etr_old / np.where(norm > 0, norm * K, 1.0) * trial_norm
        simplex[0, 0], simplex[1, 0] = etr_old[0], trial[0] * spread

        def objective(x):
            return _counted_energy(e, e, T, np.vstack([np.zeros((1, x.shape[1])), x[0:1]]), iv, par, index)[0]

    best, _ = par_fmin_search.par_fmin_search3(objective, simplex, delta)
    if search_xi:
        return float(np.mean(xi_old + best[0] / K)), float(np.mean(iv["eps_tr"][0]))
    return float(np.mean(xi_old)), float(np.mean(best[0]))


def solver_backend(x, e, T, iv, par, change_xi, change_eps_tr):
    if change_xi and change_eps_tr:
        _, _, new = reference.stress_strain_attut_el(e, e, T, copy.deepcopy(iv), par)
        return float(np.mean(new["ksi"])), float(np.mean(new["eps_tr"][0]))
    return one_variable_search(e, T, iv, par, search_xi=change_xi)


def make_network_backend(onnx_file, total_strain):
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    def backend(x, *_):
        x = np.array(x, float)
        if total_strain:                       # add the thermal expansion back
            x[3] += THERMAL_EXPANSION * (x[4] - T_REFERENCE) * (1.0 - x[0])
        xi, eps_tr = session.run(None, {name: x[None].astype(np.float32)})[0][0]
        return float(xi), float(eps_tr)

    return backend


# ----------------------------------------------------------------- one run

def run(path, backend, gate, always_compute=False, oracle=False, cutoff=0.5):
    par = get_parameters()
    set_temperature(par, T_START)
    iv = _build_initial_internal_variables(T_START, par)
    carrier = FixedPrediction()
    n = int(par["n"])
    out = {key: [] for key in ("xi", "eps_tr", "sig", "change_xi", "change_eps_tr", "energy_evaluations")}
    global energy_evaluations

    for step, (eps, T) in enumerate(zip(path["eps"], path["T"])):
        set_temperature(par, T)
        e = uniaxial_strain(eps, n) - (T - par["To"]) * par["therm_exp"] * (1.0 - iv["ksi"])   # mechanical strain
        xi_n, etr_n, M_n = state_of(iv)
        x = np.array([xi_n, etr_n, M_n, e[0, 0], T])

        if always_compute:
            change_xi = change_eps_tr = True
        elif oracle:
            xi_ref_old = path["xi"][step - 1] if step else xi_n
            etr_ref_old = path["eps_tr"][step - 1] if step else etr_n
            change_xi = abs(path["xi"][step] - xi_ref_old) > 1e-5
            change_eps_tr = abs(path["eps_tr"][step] - etr_ref_old) > 1e-5
        else:
            change_xi, change_eps_tr = (bool(c) for c in gate.changes(x, cutoff)[0])

        before = energy_evaluations
        if change_xi or change_eps_tr:
            xi, eps_tr = backend(x, e, T, iv, par, change_xi, change_eps_tr)
        else:
            xi, eps_tr = xi_n, etr_n
        used = energy_evaluations - before

        carrier.value = np.array([xi if change_xi else xi_n, eps_tr if change_eps_tr else etr_n])
        sig, _, iv = stress_strain_nn_el(e, e, T, copy.deepcopy(iv), par, carrier)
        stress = float(sig[0, 0] - sig[1, 0])
        if not np.isfinite(stress) or abs(stress) > 1e5:
            raise FloatingPointError(f"stress diverged at step {step}")

        new_xi, new_etr, _ = state_of(iv)
        for key, value in zip(out, (new_xi, new_etr, stress, change_xi, change_eps_tr, used)):
            out[key].append(value)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("backend", choices=["solver", "network"])
    ap.add_argument("path")
    ap.add_argument("--onnx", help="ONNX surrogate for the network backend")
    ap.add_argument("--total-strain", action="store_true")
    ap.add_argument("--always-compute", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--cutoff", type=float, default=0.5, help="call 'changes' when P >= cutoff")
    ap.add_argument("--out", help="output JSON (default: ../results/run_<backend>_<path>.json)")
    args = ap.parse_args()

    path = json.load(open(RESULTS / "reference_paths.json"))[args.path]
    if args.backend == "network":
        if not args.onnx:
            ap.error("the network backend needs --onnx")
        backend = make_network_backend(args.onnx, args.total_strain)
    else:
        backend = solver_backend

    start = time.perf_counter()
    result = run(path, backend, Gate(), args.always_compute, args.oracle, args.cutoff)
    sig, xi = np.array(result["sig"]), np.array(result["xi"])
    summary = {
        "path": args.path, "backend": args.backend, "steps": len(sig),
        "always_compute": args.always_compute, "oracle": args.oracle, "cutoff": args.cutoff,
        "stress_mae": float(np.mean(np.abs(sig - path["sig"]))),
        "stress_max": float(np.max(np.abs(sig - path["sig"]))),
        "xi_mae": float(np.mean(np.abs(xi - path["xi"]))),
        "steps_computed": int(np.sum(np.array(result["change_xi"]) | np.array(result["change_eps_tr"]))),
        "energy_evaluations": int(np.sum(result["energy_evaluations"])),
        "seconds": time.perf_counter() - start,
    }
    out = args.out or RESULTS / f"run_{args.backend}_{args.path}.json"
    json.dump({**summary, **result}, open(out, "w"))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
