"""Would exact solver labels correct the drift? (the test behind the DAgger decision)

Run a network freely along the cold cycles and collect the states in which its
xi has drifted away from the reference by more than `--drift`. Ask the
reference solver for the next state from each drifted state and compare:

    network error = |network's next xi - reference next xi|
    solver error  = |solver's next xi  - reference next xi|

If the solver pulled the drifted state back, its error would be much smaller
than the network's, and training on such labels (DAgger) could remove the
drift. If the solver stays where it was put, there is nothing to learn from.

NEEDS THE REFERENCE SOLVER, which is not part of this repository. Set the
environment variable SMA_SOLVER_PATH to a directory that contains a module
`solver_label` with a function

    solver_step(X) -> Y     X: (N, 5) [xi_n, etr_n, M_n, eps, T],  Y: (N, 2) [xi, etr]

The recorded result of this test is in ../README.md.

Run from the repository root:
    SMA_SOLVER_PATH=/path/to/solver python 01-baseline-network/code/solver_probe.py
"""
import argparse
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.data import MODELS, load_cycles
from common.rollout import OnnxStep, update_memory

COLD_MEAN_T = -30.0
CONFIRMS = 0.005     # the solver "confirms" a state if it moves xi by less than this


def drifted_states(model, cycles, drift):
    """Inputs the free-running network saw while its xi was off by more than `drift`."""
    states, reference_next, network_next = [], [], []
    for c in cycles:
        if c.mean_temperature > COLD_MEAN_T:
            continue
        xi, etr, M = (float(v) for v in c.X[0, :3])
        for (xi_ref_in, _, _, eps, T), (xi_ref, _) in zip(c.X, c.Y):
            xi_new, etr_new = model([[xi, etr, M, eps, T]])[0]
            xi_new = float(np.clip(xi_new, 0.0, 1.0))
            if abs(xi - xi_ref_in) > drift:
                states.append([xi, etr, M, eps, T])
                reference_next.append(xi_ref), network_next.append(xi_new)
            M = update_memory(M, xi, xi_new, etr_new)
            xi, etr = xi_new, float(etr_new)
    return np.array(states), np.array(reference_next), np.array(network_next)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=str(MODELS / "v17.onnx"))
    ap.add_argument("--drift", type=float, default=0.02)
    ap.add_argument("--max-states", type=int, default=2000)
    args = ap.parse_args()

    solver_path = os.environ.get("SMA_SOLVER_PATH")
    if not solver_path:
        sys.exit("Set SMA_SOLVER_PATH to the directory with the reference solver (see the docstring).")
    sys.path.insert(0, solver_path)
    from solver_label import solver_step

    states, reference_next, network_next = drifted_states(OnnxStep(args.model), load_cycles(), args.drift)
    print(f"{len(states)} drifted states on cold cycles")
    if len(states) > args.max_states:
        keep = np.random.RandomState(0).choice(len(states), args.max_states, replace=False)
        states, reference_next, network_next = states[keep], reference_next[keep], network_next[keep]

    solver_next = solver_step(states)[:, 0]
    network_error = np.abs(network_next - reference_next).mean()
    solver_error = np.abs(solver_next - reference_next).mean()
    confirmed = (np.abs(solver_next - states[:, 0]) < CONFIRMS).mean()
    print(f"error of the next xi from drifted states: network {network_error:.4f}, solver {solver_error:.4f}")
    print(f"correction = 1 - solver/network = {1 - solver_error / network_error:+.2f}")
    print(f"states the solver confirms (moves xi by less than {CONFIRMS}): {100 * confirmed:.0f} %")


if __name__ == "__main__":
    main()
