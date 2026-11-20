"""One-dimensional incremental energy of the NiTi model: Pi = Psi + D.

This is the uniaxial form written out in minimization-program.md. It follows
the thermomechanical model of Sedlak, Frost et al. (2012) with the calibration
used in this project. It is meant for understanding and for small checks. The
labels in the later steps come from the full reference implementation (six
strain components, stiffness that depends on temperature and orientation).

Two details of the reference implementation can be switched on, because they
matter for convexity:

  stiffness="linearized"  the modulus is linear in xi within the step,
                          E(xi_n) + (xi - xi_n) E'(xi_n), instead of the Reuss average
  memory="mean"           the reverse branch uses the mean of M over the step
                          instead of its value at the start

Units: stress and energy density in MPa, temperature in deg C, strain [-].

State at the start of a step:  xi_n, etr_n, M      (M = memory variable)
Load:                          eps (strain), T (temperature)
Unknowns:                      xi  (martensite fraction), etr (transformation strain)
"""
from dataclasses import dataclass

import numpy as np

SQRT15 = np.sqrt(1.5)


@dataclass(frozen=True)
class Material:
    etr_max: float = 0.07      # K, largest transformation strain in tension
    asym: float = 0.9          # tension-compression asymmetry
    dS: float = 0.40           # entropy difference austenite - martensite [MPa/K]
    T0: float = -30.0          # equilibrium temperature (= Ms in this calibration)
    Ms: float = -30.0          # martensite start
    Mf: float = -60.0          # martensite finish
    As: float = -45.0          # austenite start
    Af: float = -15.0          # austenite finish
    sig_reo: float = 100.0     # reorientation stress [MPa]
    E_A: float = 71_000.0      # Young's modulus of austenite [MPa]
    E_M: float = 46_000.0      # Young's modulus of martensite in tension [MPa]
    E_int: float = 30.0        # hardening during reorientation [MPa]
    E_incl: float = 200.0      # energy of martensite inclusions [MPa]
    xi_lim: float = 5e-3       # width of the barriers that keep xi and etr in range

    @property
    def tau0(self):
        """Reorientation threshold for the uniaxial strain component."""
        return self.sig_reo * SQRT15

    @property
    def compression_factor(self):
        """The limit strain in compression is etr_max / compression_factor."""
        return 1.0 / np.cos(np.arccos(1.0 - 2.0 * self.asym) / 3.0)


def modulus(xi, mat, xi_n=None):
    """Young's modulus of the mixture.

    Reuss average (compliances add). With `xi_n` given, its linearization
    around xi_n, which is what the reference implementation uses inside a step.
    """
    def reuss(x):
        return 1.0 / ((1.0 - x) / mat.E_A + x / mat.E_M)

    if xi_n is None:
        return reuss(xi)
    slope = -(1.0 / mat.E_M - 1.0 / mat.E_A) * reuss(xi_n) ** 2
    return reuss(xi_n) + (xi - xi_n) * slope


def relative_strain(etr, mat):
    """|etr| divided by its limit (1 = fully oriented martensite)."""
    c = np.where(np.asarray(etr) >= 0, 1.0, mat.compression_factor)
    return c * np.abs(etr) / mat.etr_max


def free_energy(xi, etr, eps, T, mat, barriers=True, xi_n=None):
    """Psi: elastic + chemical + interaction energy.

    Pass `xi_n` to use the stiffness linearized around the old state.
    """
    elastic = 0.5 * modulus(xi, mat, xi_n) * (eps - xi * etr) ** 2
    chemical = (T - mat.T0) * mat.dS * xi
    r = relative_strain(etr, mat)
    hardening = 0.5 * mat.E_int * mat.etr_max * r ** 2
    inclusions = 1.5 * mat.E_incl / mat.etr_max * (etr - 2.0 * eps / 3.0) ** 2
    psi = elastic + chemical + xi * (hardening + inclusions)
    if barriers:
        psi = psi + xi * 100.0 * 10.0 ** ((r - (1.0 - mat.xi_lim)) / mat.xi_lim)
    return psi


def dissipation(xi, etr, xi_n, etr_n, M, T, mat, barriers=True, memory="start"):
    """D: energy dissipated by going from (xi_n, etr_n) to (xi, etr).

    Forward (xi >= xi_n): new martensite forms already oriented, so the
    reorientation cost acts on the change of q = xi * etr.
    Reverse (xi < xi_n): the remaining martensite reorients, and the vanishing
    part pays sig_reo * M.
    """
    xi, etr = np.asarray(xi, float), np.asarray(etr, float)
    d_xi = xi - xi_n
    lim = mat.xi_lim
    if memory == "mean":
        M = M + 0.5 * np.maximum(SQRT15 * np.abs(etr) - M, 0.0)

    forward = (mat.tau0 * np.abs(xi * etr - xi_n * etr_n)
               + mat.dS * ((mat.T0 - mat.Ms) * d_xi
                           + 0.5 * (mat.Ms - mat.Mf) * (xi ** 2 - xi_n ** 2)))
    reverse = (mat.tau0 * xi * np.abs(etr - etr_n)
               + mat.sig_reo * M * (-d_xi)
               + mat.dS * ((mat.Af - mat.T0) * (-d_xi)
                           + 0.5 * (mat.As - mat.Af) * (xi_n ** 2 - xi ** 2)))
    if barriers:
        forward = forward + 100.0 * (10.0 ** ((xi - (1 - lim)) / lim)
                                     - 10.0 ** ((xi_n - (1 - lim)) / lim))
        reverse = reverse + 100.0 * (10.0 ** ((lim - xi) / lim)
                                     - 10.0 ** ((lim - xi_n) / lim))
    return np.where(d_xi >= 0, forward, reverse)


def incremental_energy(xi, etr, xi_n, etr_n, M, eps, T, mat=Material(), barriers=True,
                       stiffness="reuss", memory="start"):
    """Pi(xi, etr) = Psi + D. The constitutive step minimizes this."""
    linearize_at = xi_n if stiffness == "linearized" else None
    return (free_energy(xi, etr, eps, T, mat, barriers, linearize_at)
            + dissipation(xi, etr, xi_n, etr_n, M, T, mat, barriers, memory))


def stress(xi, etr, eps, mat=Material()):
    """sigma = dPsi/d eps for the elastic part."""
    return modulus(xi, mat) * (eps - xi * etr)


def update_memory(M, xi_n, xi, etr):
    """Memory variable after the step.

    It shrinks with the martensite that disappears (never grows on reverse)
    and is raised to the current strain norm. It stays below sqrt(1.5) * etr_max.
    """
    return np.maximum(M * np.minimum(xi_n / xi, 1.0), SQRT15 * np.abs(etr))
