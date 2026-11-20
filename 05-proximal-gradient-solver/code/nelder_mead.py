"""Nelder-Mead step with the schedule of the reference solver, on the public energy.

The baseline of the benchmarks of step 05, written from the description of
the reference solver: variables (K * d_xi, etr), the old state as the first
vertex, three phases of 100 iterations, restarts of a collapsed simplex in
the first two phases, no convergence test. The reference solver runs five
differently scaled starting simplices (STARTS).
"""
import numpy as np

from prox_gradient_solver import XI_MIN, incremental_energy
from energy_1d import relative_strain          # found through the path set by prox_gradient_solver
from test_steps import MAT

STARTS = np.linspace(0.01, 1.0, 5)           # scales of the five starting simplices
TOL_COLLAPSE, RESTART = 5e-6, (5e-3, 1e-3, 0.0)


def pi(xi, etr, state):
    """The energy of a step, +inf outside 0 < xi < 1.

    The formula of the model is not meant for xi outside this range, and for
    xi < 0 it is unbounded below. The proximal solver has the same guard, and
    the reference implementation switches the term in question off there.
    """
    if not XI_MIN < xi < 1.0:
        return np.inf
    xi_n, etr_n, M, eps, T = state
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        value = float(incremental_energy(xi, etr, xi_n, etr_n, M, eps, T, MAT))
    return value if np.isfinite(value) else np.inf



def nelder_mead(state, start=1.0, iterations=100, trace=None):
    """Nelder-Mead step with the schedule of the reference solver.

    `start` scales the starting simplex, `iterations` is the budget of each of
    the three phases. Returns xi, etr and the number of energy evaluations.
    If `trace` is a list, the best vertex (xi, etr) is appended to it after
    every iteration.
    """
    xi_n, etr_n, M, eps, T = state
    K = MAT.etr_max
    calls = 0

    def energy(x):
        nonlocal calls
        calls += 1
        return pi(xi_n + x[0] / K, x[1], state)

    # Starting simplex: no change, a trial transformation, a trial reorientation.
    d_xi = -0.05 if xi_n > 0.5 else 0.05
    base = etr_n if relative_strain(etr_n, MAT) >= 1e-4 else etr_n + K / 10.0
    size = float(relative_strain(base, MAT)) * K
    trial = base * (1.0 + (-0.05 if size > 0.5 * K else 0.05) / size)
    X = np.array([[0.0, etr_n], [start * K * d_xi, etr_n], [0.0, start * trial]])
    F = np.array([energy(x) for x in X])

    for perturbation in RESTART:
        for _ in range(iterations):
            order = np.argsort(F, kind="stable")
            X, F = X[order], F[order]
            centre = 0.5 * (X[0] + X[1])
            reflected = 2.0 * centre - X[2]
            f_r = energy(reflected)
            if f_r < F[0]:                                   # expansion
                expanded = 2.0 * reflected - centre
                f_e = energy(expanded)
                X[2], F[2] = (expanded, f_e) if f_e < F[0] else (reflected, f_r)
            elif f_r < F[1]:                                 # reflection
                X[2], F[2] = reflected, f_r
            else:                                            # contraction, outside or inside
                toward = reflected if f_r < F[2] else X[2]
                contracted = 0.5 * (centre + toward)
                f_c = energy(contracted)
                if f_c < F[2]:
                    X[2], F[2] = contracted, f_c
                else:                                        # shrink towards the best vertex
                    for i in (1, 2):
                        X[i] = 0.5 * (X[0] + X[i])
                        F[i] = energy(X[i])
            if perturbation and X.std(axis=0, ddof=1).max() < TOL_COLLAPSE:
                best = int(np.argmin(F))
                others = [i for i in range(3) if i != best]
                X[others[0], 0] += perturbation
                X[others[1], 1] += perturbation
                F[others[0]], F[others[1]] = energy(X[others[0]]), energy(X[others[1]])
            if trace is not None:
                b = int(np.argmin(F))
                trace.append((xi_n + X[b, 0] / K, float(X[b, 1])))
    best = int(np.argmin(F))
    return xi_n + X[best, 0] / K, float(X[best, 1]), calls
