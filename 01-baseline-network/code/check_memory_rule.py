"""Check the update rule of the memory variable M against the reference cycles.

The simulation loop computes M itself (the network does not predict it):

    M_next = max(M * min(xi_n / xi, 1), sqrt(1.5) * |etr|)

An early version of the loop had no `min(..., 1)`. This script applies both
versions to the reference transitions and compares with the M that the
reference solver carried into the next step.

Run:  python check_memory_rule.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.data import load_cycles
from common.rollout import SQRT15, XI_FLOOR

M_LIMIT = SQRT15 * 0.07      # sqrt(1.5) * largest transformation strain = 0.0857


def next_memory(M, xi_n, xi, etr, capped):
    ratio = xi_n / np.maximum(xi, XI_FLOOR)
    if capped:
        ratio = np.minimum(ratio, 1.0)
    return np.maximum(M * ratio, SQRT15 * np.abs(etr))


if __name__ == "__main__":
    M, xi_n, xi, etr, M_ref = [], [], [], [], []
    for c in load_cycles():
        M.append(c.X[:-1, 2]), xi_n.append(c.X[:-1, 0])
        xi.append(c.Y[:-1, 0]), etr.append(c.Y[:-1, 1])
        M_ref.append(c.X[1:, 2])              # what the solver carried into the next step
    M, xi_n, xi, etr, M_ref = (np.concatenate(a).astype(float) for a in (M, xi_n, xi, etr, M_ref))
    reverse = xi < xi_n
    print(f"{len(M):,} reference transitions, {reverse.sum():,} with reverse transformation")
    for capped in (True, False):
        error = np.abs(next_memory(M, xi_n, xi, etr, capped) - M_ref)
        print(f"{'with' if capped else 'without'} min(.., 1):  mean error {error.mean():.2e}, "
              f"largest {error.max():.2e}, on reverse steps {error[reverse].mean():.2e}")

    # Iterate both rules along each cycle, fed with the reference xi and etr.
    for capped in (True, False):
        worst = 0.0
        for c in load_cycles():
            m = float(c.X[0, 2])
            for x0, x1, e in zip(c.X[:, 0], c.Y[:, 0], c.Y[:, 1]):
                m = float(next_memory(m, x0, x1, e, capped))
                worst = max(worst, m)
        print(f"{'with' if capped else 'without'} min(.., 1):  largest M along the cycles "
              f"{worst:.4f}  (physical limit {M_LIMIT:.4f})")
