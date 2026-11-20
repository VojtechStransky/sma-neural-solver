"""The test problems of step 05 and their truth, shared by all scripts.

  loading_programmes   the 60 loading paths (strain cycles at a fixed
                       temperature, strain ramps followed by cooling and heating)
  random_states        old states with very large load jumps
  brute_force_minimum  the truth of a step: grid search with zooming
  must_change          the optimality test of the old state

Every step is minimized on the public one-dimensional energy of
00-problem-definition/code/energy_1d.py. A state is (xi_n, etr_n, M_n, eps, T).
"""
import numpy as np

from prox_gradient_solver import Material, corners, f_all, incremental_energy, project_polygon
from energy_1d import SQRT15, stress, update_memory  # noqa: F401  (found through prox_gradient_solver)

MAT = Material()
XI_START = 0.02                       # every path starts in almost pure austenite
N_PATHS, N_RANDOM = 60, 1000
XI_RANGE, COARSE, ZOOM, LEVELS, SHRINK = (0.004, 0.996), 60, 21, 26, 0.5     # brute force


def loading_programmes(rng):
    """(eps, T) sequences: strain cycles at fixed T, or a strain ramp followed by cooling and heating.

    Strain cycles are cut at 60 steps, so the longer ones end before the strain returns to zero.
    """
    programmes = []
    for i in range(N_PATHS):
        if i % 4 < 3:                                   # strain cycle, tension or compression
            T = rng.uniform(-70.0, 100.0)
            top = rng.uniform(0.02, 0.07) * (1.0 if rng.random() < 0.8 else -0.6)
            step = rng.choice([5e-4, 1e-3, 2.5e-3, 5e-3])
            n = max(int(round(abs(top) / step)), 2)
            eps = np.concatenate([np.linspace(0, top, n + 1)[1:], np.linspace(top, 0, n + 1)[1:]])
            programmes.append((eps[:60], np.full(len(eps[:60]), T)))
        else:                                           # strain ramp, then cool and heat
            level = rng.uniform(0.0, 0.04)
            n = 10
            T_hot, T_cold = rng.uniform(20.0, 80.0), rng.uniform(-80.0, -40.0)
            T = np.concatenate([np.full(n, T_hot), np.linspace(T_hot, T_cold, 21)[1:],
                                np.linspace(T_cold, T_hot, 21)[1:]])
            eps = np.concatenate([np.linspace(0, level, n + 1)[1:], np.full(40, level)])
            programmes.append((eps, T))
    return programmes


def random_states(rng, n=N_RANDOM):
    """Random old states with load jumps far larger than a step of a loading path."""
    states = []
    for _ in range(n):
        xi_n, etr_n = rng.uniform(0.03, 0.97), rng.uniform(-0.04, 0.065)
        M = SQRT15 * abs(etr_n) + rng.choice([0.0, rng.uniform(0.0, 0.02)])
        states.append((xi_n, etr_n, M, xi_n * etr_n + rng.uniform(-0.012, 0.016), rng.uniform(-80.0, 100.0)))
    return states


def brute_force_minimum(xi_n, etr_n, M, eps, T, mat=MAT):
    """Lowest point of Pi found by grid search with zooming. Returns Pi, xi, etr.

    A coarse grid over the admissible range, then repeated zooming around the
    best point in (xi, q). Every level also tries the kink lines (xi = xi_n,
    q = q_n, etr = etr_n), where the minimum often sits.
    """
    lo, hi = -mat.etr_max / mat.compression_factor, mat.etr_max

    def energy(xi, etr):
        with np.errstate(over="ignore"):
            return incremental_energy(xi, etr, xi_n, etr_n, M, eps, T, mat)

    xi, etr = np.meshgrid(np.linspace(*XI_RANGE, COARSE), np.linspace(lo, hi, COARSE))
    xi, etr = np.append(xi.ravel(), xi_n), np.append(etr.ravel(), etr_n)
    F = energy(xi, etr)
    k = int(np.argmin(F))
    best = (F[k], xi[k], etr[k])

    half_xi, half_q = 0.02, 0.002
    offsets = np.linspace(-1.0, 1.0, ZOOM)
    for _ in range(LEVELS):
        xi_c, q_c = best[1], best[1] * best[2]
        xi_z, q_z = np.meshgrid(np.clip(xi_c + half_xi * offsets, *XI_RANGE), q_c + half_q * offsets)
        xi_z, q_z = xi_z.ravel(), q_z.ravel()
        xi = np.concatenate([xi_z, np.full_like(xi_z, xi_n), xi_z, xi_z])
        etr = np.concatenate([q_z / xi_z, q_z / xi_n, xi_n * etr_n / xi_z, np.full_like(xi_z, etr_n)])
        F = energy(xi, etr)
        k = int(np.argmin(F))
        if F[k] < best[0]:
            best = (F[k], xi[k], etr[k])
        half_xi, half_q = half_xi * SHRINK, half_q * SHRINK
    return best


def must_change(state, mat=MAT):
    """Optimality test of the old state: does the state have to change in this step?

    The old state is the minimizer exactly when minus the gradient of the
    smooth part, taken at the old state, lies inside the elastic domain. This
    is a property of the step, computed without running a solver. (Comparing
    energies cannot decide it: near a threshold they agree to the last digit.)
    """
    xi_n, etr_n, M, eps, T = state
    grad = f_all(xi_n, xi_n * etr_n, xi_n, eps, T, mat)[1]
    return project_polygon(-grad, corners(xi_n, etr_n, M, mat))[1][0] != "inside"
