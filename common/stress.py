"""Stress from the state, for the uniaxial loading used in this repository.

sigma = (C eps_el)_1 - (C eps_el)_2 with eps_el = (eps - xi * etr) * [1, -1/2, -1/2, 0, 0, 0].
The compliance of the mixture is the average of the phase compliances (Reuss).
The stiffness of martensite depends on the sign of etr (tension or compression)
and both phases depend weakly on temperature.

The same function is applied to the reference and to a model, so differences
in the stress come only from differences in (xi, etr).
"""
import numpy as np

GPA = 1000.0  # MPa


def _cubic(c11, c12, c44):
    c = np.zeros((6, 6))
    c[:3, :3] = c12
    c[np.arange(3), np.arange(3)] = c11
    c[np.arange(3, 6), np.arange(3, 6)] = c44
    return c * GPA


C_AUSTENITE = _cubic(185.0, 135.0, 25.0)
C_MARTENSITE_TENSION = np.array([
    [160.47, 129.77, 129.77, 0, 0, 0], [129.77, 180.47, 114.77, 0, 0, 0],
    [129.77, 114.77, 180.47, 0, 0, 0], [0, 0, 0, 32.85, 0, 0],
    [0, 0, 0, 0, 25.85, 0], [0, 0, 0, 0, 0, 25.85]]) * GPA
C_MARTENSITE_COMPRESSION = np.array([
    [199.5, 117.75, 117.75, 0, 0, 0], [117.75, 179.5, 118.75, 0, 0, 0],
    [117.75, 118.75, 179.5, 0, 0, 0], [0, 0, 0, 30.4, 0, 0],
    [0, 0, 0, 0, 40.4, 0], [0, 0, 0, 0, 0, 40.4]]) * GPA
# Change of stiffness with temperature: deviatoric part only.
DEVIATORIC = _cubic(4 / 3, -2 / 3, 1.0)
AXIS = np.array([1.0, -0.5, -0.5, 0.0, 0.0, 0.0])


def _saturating(T):
    """Temperature measure that saturates far from the reference."""
    return np.tanh(T / 250.0 * 3.0) * 250.0 / 3.0


def stress(eps, xi, etr, T):
    """Uniaxial stress [MPa] for arrays of strain, state and temperature."""
    eps, xi, etr, T = (np.atleast_1d(np.asarray(a, float)) for a in (eps, xi, etr, T))
    xi = np.clip(xi, 0.0, 1.0)[:, None, None]
    shift_a = (0.05 * _saturating(T))[:, None, None] * DEVIATORIC
    shift_m = (-0.05 * _saturating(T + 30.0))[:, None, None] * DEVIATORIC
    c_m = np.where((etr >= 0)[:, None, None], C_MARTENSITE_TENSION, C_MARTENSITE_COMPRESSION)
    compliance = xi * np.linalg.inv(c_m + shift_m) + (1 - xi) * np.linalg.inv(C_AUSTENITE + shift_a)
    stiffness = np.linalg.inv(compliance)
    eps_el = (eps - xi[:, 0, 0] * etr)[:, None] * AXIS
    sigma = np.einsum("nij,nj->ni", stiffness, eps_el)
    return sigma[:, 0] - sigma[:, 1]
