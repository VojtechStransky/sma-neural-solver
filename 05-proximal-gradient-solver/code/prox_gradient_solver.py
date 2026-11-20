"""Proximal-gradient solver for one SMA step, in the variables (xi, q).

Variables: u = (xi, q) with q = xi * etr (macroscopic transformation strain).
The old state u_n = (xi_n, q_n) and the load (eps, T) are known numbers.

    minimize   f(xi, q) + g(xi - xi_n, q - q_n)

    f(xi, q)   smooth and convex: free energy + the parts of the dissipation
               that have no kink
    g(d)       the kinks, as a function of the increment d = u - u_n:
               g(d) = max_i <c_i, d>, with the four corners c_i of the
               elastic domain C (a trapezoid, in MPa)

The file has two parts.

Part 1, the model: f with its gradient and Hessian, the thresholds, the
corners of the elastic domain, and the exact prox of g (the trial increment
minus its projection onto the trapezoid).

Part 2, the solver. One iteration:

    1. gradient and curvature of the smooth part f at the current point
    2. trial increment   w = d - H^-1 grad f          (d = current increment)
    3. exact prox        d+ = w - projection of w onto the elastic domain,
                         in the metric H               (prox_g)
    4. step halving until the energy does not rise     (at most `halvings` times)

The step size is the matrix H^-1. Four choices (`metric`), from the plain
ISTA step to the Newton step; the prox is exact in all of them:

    "one_step"      H = I / t with one step size t = 1 / (largest curvature)
    "steps_xi_q"    one step size per unknown, for xi and q
    "steps_xi_etr"  one step size per unknown, for xi and etr: the same
                    recipe in the directions "xi at fixed etr" and "etr at
                    fixed xi". This is the ISTA that converges in practice.
    "newton"        H = the Hessian of f (proximal Newton), the default

`solve_step` iterates until the step stops changing.

The energy is the public one-dimensional model of
00-problem-definition/code/energy_1d.py (Reuss stiffness, memory variable at
the start of the step); f + g equals its `incremental_energy`. The reference
implementation is not used.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "00-problem-definition" / "code"))
from energy_1d import Material, incremental_energy  # noqa: E402,F401

LN10 = np.log(10.0)


# ----------------------------------------------------------------- barriers
# The model keeps xi and etr in range with steep exponential barriers.

def _barrier_forward(xi, mat):
    """Upper barrier of xi (forward branch): value, first and second derivative."""
    b = 100.0 * 10.0 ** ((xi - (1.0 - mat.xi_lim)) / mat.xi_lim)
    k = LN10 / mat.xi_lim
    return b, k * b, k * k * b


def _barrier_reverse(xi, mat):
    """Lower barrier of xi (reverse branch): value, first and second derivative."""
    b = 100.0 * 10.0 ** ((mat.xi_lim - xi) / mat.xi_lim)
    k = -LN10 / mat.xi_lim
    return b, k * b, k * k * b


def _phi(etr, eps, mat):
    """Interaction energy per unit martensite and its barrier: value, phi', phi''."""
    c = mat.compression_factor if etr < 0 else 1.0
    K = mat.etr_max
    hard, incl = mat.E_int * c * c / K, 3.0 * mat.E_incl / K
    value = 0.5 * hard * etr ** 2 + 0.5 * incl * (etr - 2.0 * eps / 3.0) ** 2
    d1 = hard * etr + incl * (etr - 2.0 * eps / 3.0)
    d2 = hard + incl
    # barrier on |etr|
    r = c * abs(etr) / K
    k = LN10 / mat.xi_lim * c / K
    b = 100.0 * 10.0 ** ((r - (1.0 - mat.xi_lim)) / mat.xi_lim)
    return value + b, d1 + np.sign(etr) * k * b, d2 + k * k * b


# -------------------------------------------------------------- the kinks g

def thresholds(xi_n, etr_n, M_n, mat):
    """Y_F0, Y_R, tau0: the constants of g. They depend on the old state only."""
    Y_F0 = mat.dS * ((mat.T0 - mat.Ms) + (mat.Ms - mat.Mf) * xi_n) + _barrier_forward(xi_n, mat)[1]
    Y_R = (mat.dS * ((mat.Af - mat.T0) + (mat.As - mat.Af) * xi_n) + mat.sig_reo * M_n
           - _barrier_reverse(xi_n, mat)[1])
    return Y_F0, Y_R, mat.tau0


def corners(xi_n, etr_n, M_n, mat):
    """The four corners c_1..c_4 of the elastic domain C, counter-clockwise, shape (4, 2).

    Components are (force on xi, force on q) in MPa.
    """
    Y_F0, Y_R, tau0 = thresholds(xi_n, etr_n, M_n, mat)
    return np.array([[Y_F0, tau0],                       # c1: forward, q grows
                     [-Y_R - tau0 * etr_n, tau0],        # c3: reverse, etr grows
                     [-Y_R + tau0 * etr_n, -tau0],       # c4: reverse, etr shrinks
                     [Y_F0, -tau0]])                     # c2: forward, q shrinks


def g(d, C):
    """Kinked part of the energy for the increment d = (d_xi, d_q)."""
    return float(np.max(C @ np.asarray(d, float)))


# -------------------------------------------------------------- the smooth f

def f_all(xi, q, xi_n, eps, T, mat):
    """f, its gradient (2,) and its Hessian (2, 2) at (xi, q)."""
    etr = q / xi
    s = (1.0 - xi) / mat.E_A + xi / mat.E_M              # compliance, linear in xi
    ds = 1.0 / mat.E_M - 1.0 / mat.E_A
    sigma = (eps - q) / s
    phi, dphi, ddphi = _phi(etr, eps, mat)
    d_xi = xi - xi_n
    h = mat.dS * ((mat.Ms - mat.Mf) if d_xi >= 0 else (mat.Af - mat.As))
    barrier = _barrier_forward if d_xi >= 0 else _barrier_reverse
    b, db, ddb = barrier(xi, mat)
    b_n, db_n, _ = barrier(xi_n, mat)

    value = (0.5 * (eps - q) * sigma + (T - mat.T0) * mat.dS * xi + xi * phi
             + 0.5 * h * d_xi ** 2 + (b - b_n - db_n * d_xi))
    grad = np.array([-0.5 * sigma ** 2 * ds + (T - mat.T0) * mat.dS + phi - etr * dphi
                     + h * d_xi + (db - db_n),
                     -sigma + dphi])
    hess = np.array([[sigma ** 2 * ds ** 2 / s + etr ** 2 * ddphi / xi + h + ddb,
                      sigma * ds / s - etr * ddphi / xi],
                     [sigma * ds / s - etr * ddphi / xi,
                      1.0 / s + ddphi / xi]])
    return value, grad, hess


# -------------------------------------------------- projection and the prox

def project_polygon(p, V):
    """Closest point of the convex polygon with vertices V (counter-clockwise) to p.

    Returns the point and where it lies: ("inside", None), ("edge", i) for the
    edge from V[i] to V[i+1], or ("corner", i).
    """
    n = len(V)
    edges = np.roll(V, -1, axis=0) - V
    rel = p - V
    cross = edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]
    if np.all(cross >= 0.0):
        return p.copy(), ("inside", None)
    lam = np.einsum("ij,ij->i", rel, edges) / np.einsum("ij,ij->i", edges, edges)
    foot = V + np.clip(lam, 0.0, 1.0)[:, None] * edges
    i = int(np.argmin(np.einsum("ij,ij->i", p - foot, p - foot)))
    if 0.0 < lam[i] < 1.0:
        return foot[i], ("edge", i)
    return foot[i], ("corner", i if lam[i] <= 0.0 else (i + 1) % n)


def prox_g(w, C, metric):
    """argmin_d  g(d) + 1/2 (d - w)^T H (d - w),   g(d) = max_i <c_i, d>.

    `metric` is the positive definite matrix H (2, 2). With H = I/t this is
    the usual prox_{t g}. The solution is w minus the projection of w onto the
    elastic domain, taken in coordinates in which H is the identity:

        H = L L^T,   e = L^T d,   corners L^-1 c_i,   d = L^-T (omega - P(omega))

    Returns d and the location of the projection (the regime of the step).
    """
    L = np.linalg.cholesky(metric)
    V = np.linalg.solve(L, C.T).T                 # corners in the new coordinates
    omega = L.T @ w
    P, where = project_polygon(omega, V)
    return np.linalg.solve(L.T, omega - P), where


REGIME_OF = {("inside", None): "R0 hold",
             ("edge", 0): "R1 reorientation, xi held", ("edge", 2): "R1 reorientation, xi held",
             ("edge", 3): "forward transformation, q held",
             ("edge", 1): "R4 reverse transformation, etr held",
             ("corner", 0): "R2 forward with reorientation", ("corner", 3): "R2 forward with reorientation",
             ("corner", 1): "R3 reverse with reorientation", ("corner", 2): "R3 reverse with reorientation"}


# =================================================================== solver

XI_MIN = 1e-6


def _evaluate(d, u_n, C, smooth):
    """Energy f + g, gradient and Hessian of f for the increment d. The energy is inf outside the range."""
    xi, q = u_n + d
    if xi <= XI_MIN:
        return np.inf, None, None
    with np.errstate(over="ignore", invalid="ignore"):
        value, grad, hess = smooth(xi, q)
        value = value + g(d, C)
    return (value, grad, hess) if np.isfinite(value) else (np.inf, None, None)


def step_directions(xi, etr, metric):
    """The two directions (columns) that get one step size each, in the plane (xi, q).

    For xi and q they are the axes. For xi and etr they are (1, etr), "xi at
    fixed etr", and (0, xi), "etr at fixed xi", because q = xi * etr.
    """
    return np.eye(2) if metric == "steps_xi_q" else np.array([[1.0, 0.0], [etr, xi]])


def directional_curvature(hess, B):
    """Curvatures a_i of f along the columns b_i of B, and their coupling rho.

    rho = |b_1^T hess b_2| / sqrt(a_1 a_2) lies between 0 (independent
    directions) and 1 (the same direction as far as the curvature can tell).
    """
    curvature = B.T @ hess @ B
    a = np.diag(curvature)
    return a, abs(curvature[0, 1]) / np.sqrt(a[0] * a[1])


def step_metric(hess, xi, etr, metric):
    """The matrix H of one iteration: the step is H^-1 and the prox is taken in the metric H.

    All choices except "newton" bound the Hessian at the current point from
    above (the ISTA condition t <= 1/L, per direction). One step size per
    unknown means t_i = 1 / ((1 + rho) a_i) along the two step directions.
    """
    if metric == "newton":
        return hess
    if metric == "one_step":
        return np.linalg.eigvalsh(hess)[-1] * np.eye(2)
    B = step_directions(xi, etr, metric)
    a, rho = directional_curvature(hess, B)
    B_inv = np.linalg.inv(B)
    return B_inv.T @ np.diag((1.0 + rho) * a) @ B_inv


def _layer(d, here, u_n, C, smooth, metric, halvings):
    """One iteration from the increment d, where `here` = _evaluate(d).

    Returns the new increment, its evaluation, the regime and the size of the change.
    """
    energy, grad, hess = here
    xi, q = u_n + d
    hess = step_metric(hess, xi, q / xi, metric)
    w = d - np.linalg.solve(hess, grad)
    d_new, where = prox_g(w, C, hess)

    alpha = 1.0
    for _ in range(halvings + 1):
        trial = d + alpha * (d_new - d)
        if np.array_equal(trial, d):
            break
        there = _evaluate(trial, u_n, C, smooth)
        if there[0] <= energy:
            return trial, there, where, np.abs(trial - d)
        alpha *= 0.5
    return d, here, where, np.zeros(2)


def solve_step(xi_n, etr_n, M_n, eps, T, mat=Material(), metric="newton",
               max_iter=200, tol=(1e-11, 1e-12), halvings=30, smooth=None, trace=None):
    """Iterate to convergence. Returns xi, etr, the number of iterations and the regime.

    `smooth(xi, q)` returns f, its gradient and its Hessian. The default is
    f_all of the public model; pass another function to count the
    evaluations (verify_solver.py, benchmark_solvers.py).
    Every evaluation of the energy goes through it: one per iteration, plus
    one for every step halving.

    `metric` chooses the step size (see the top of the file). The Newton step
    converges within 20 iterations in every tested step; one step size per
    unknown needs up to a few hundred, so pass a larger `max_iter` with it.

    If `trace` is a list, the answer (xi, etr) after every iteration is
    appended to it (benchmark_convergence.py).
    """
    if smooth is None:
        def smooth(xi, q):
            return f_all(xi, q, xi_n, eps, T, mat)
    u_n = np.array([xi_n, xi_n * etr_n])
    C = corners(xi_n, etr_n, M_n, mat)
    d = np.zeros(2)
    here = _evaluate(d, u_n, C, smooth)
    for k in range(1, max_iter + 1):
        d, here, where, change = _layer(d, here, u_n, C, smooth, metric, halvings)
        if trace is not None:
            xi, q = u_n + d
            trace.append((xi_n, etr_n) if not d.any() else (xi, q / xi))
        if change[0] <= tol[0] and change[1] <= tol[1]:
            break
    if not d.any():                       # hold: the old state itself, not q / xi rounded
        return xi_n, etr_n, k, REGIME_OF[where]
    xi, q = u_n + d
    return xi, q / xi, k, REGIME_OF[where]
