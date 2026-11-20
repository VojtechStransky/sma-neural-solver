"""The change gate: does xi change in this step, and does eps_tr change?

A small network with two outputs. It is stored as plain numpy arrays
(../models/gate.npz), so using it needs neither TensorFlow nor the solver.

Input of one step (the same five numbers every surrogate in this repository gets):

    [xi_n, eps_tr_n, M_n, eps_mech, T]

    eps_mech is the mechanical strain: total strain minus the thermal expansion
    7e-6 * (T + 30) * (1 - xi_n).

Output: P(xi changes), P(eps_tr changes).

"Changes" means more than the solver itself can resolve:

    |xi - xi_n|         > 2 * delta / K = 1.43e-4
    |eps_tr - eps_tr_n| > delta         = 5e-6

with delta = 5e-6 (the solver's tolerance) and K = 0.07 (largest transformation strain).

Run this file for a small demonstration:  python gate.py
"""
from pathlib import Path

import numpy as np

SOLVER_TOLERANCE = 5e-6          # delta
MAX_TRANSFORMATION_STRAIN = 0.07  # K
THRESHOLD_XI = 2 * SOLVER_TOLERANCE / MAX_TRANSFORMATION_STRAIN
THRESHOLD_EPS_TR = SOLVER_TOLERANCE

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "gate.npz"


def swish(z):
    return z / (1.0 + np.exp(-z))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Gate:
    """Three dense layers of 128 units (swish) and two sigmoid outputs: 34,050 weights."""

    def __init__(self, weights_file=DEFAULT_WEIGHTS):
        stored = np.load(weights_file)
        self.mean, self.std = stored["mu"], stored["sd"]
        n_layers = sum(name.startswith("w") for name in stored.files) // 2
        self.layers = [(stored[f"w{2 * i}"], stored[f"w{2 * i + 1}"]) for i in range(n_layers)]

    def probabilities(self, x):
        """x: (N, 5) inputs -> (N, 2) probabilities [xi changes, eps_tr changes]."""
        h = (np.atleast_2d(np.asarray(x, float)) - self.mean) / self.std
        for weight, bias in self.layers[:-1]:
            h = swish(h @ weight + bias)
        weight, bias = self.layers[-1]
        return sigmoid(h @ weight + bias)

    def changes(self, x, cutoff=0.5):
        """Boolean decisions, same shape as `probabilities`."""
        return self.probabilities(x) >= cutoff


def change_labels(xi_n, eps_tr_n, xi, eps_tr):
    """The training labels: did each variable move by more than the solver resolves?"""
    return np.stack([np.abs(xi - xi_n) > THRESHOLD_XI,
                     np.abs(eps_tr - eps_tr_n) > THRESHOLD_EPS_TR], axis=-1)


if __name__ == "__main__":
    gate = Gate()
    # Austenite at 40 deg C, strained from rest. xi holds while the response is elastic and
    # switches on when the transformation starts. eps_tr (the orientation of the small
    # residual martensite fraction) follows the load from the first step.
    strains = np.array([0.004, 0.008, 0.010, 0.012, 0.014, 0.016])
    states = np.column_stack([np.full(6, 0.02), np.zeros(6), np.zeros(6), strains, np.full(6, 40.0)])
    print("xi_n = 0.02, eps_tr_n = 0, T = 40 deg C")
    print("  strain   P(xi changes)   P(eps_tr changes)")
    for eps, (p_xi, p_eps) in zip(strains, gate.probabilities(states)):
        print(f"  {eps:.3f}       {p_xi:6.3f}           {p_eps:6.3f}")
