"""Figures for the problem definition.

  ../figures/sma_behaviour.png      schematic stress-strain response in the two regimes
  ../figures/energy_landscape.png   Pi along etr: the kink and the pinned minimum

Run:  python plot_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from energy_1d import Material, SQRT15, dissipation, free_energy

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#104281"]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "legend.fontsize": 9, "legend.frameon": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "lines.linewidth": 2.0, "savefig.bbox": "tight", "savefig.dpi": 160,
})


def sma_behaviour():
    """Schematic loops (not computed): superelasticity and the shape-memory effect."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        ax.set_xlabel("strain  ε"), ax.set_ylabel("stress  σ")
        ax.set_xlim(0, 6.2), ax.set_ylim(0, 4.4)

    ax = axes[0]
    ax.set_title("Superelasticity  (T > A$_f$)")
    load = np.array([(0, 0), (0.55, 2.45), (4.15, 2.92), (4.85, 3.82)])
    unload = np.array([(4.85, 3.82), (4.3, 2.42), (0.7, 1.48), (0, 0)])
    ax.fill(*np.vstack([load, unload]).T, color=BLUE, alpha=0.10, lw=0)
    ax.plot(*load.T, color=BLUE), ax.plot(*unload.T, color=BLUE)
    ax.annotate("", (2.7, 2.73), (2.2, 2.665), arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.annotate("", (2.2, 1.87), (2.7, 2.0), arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.text(2.4, 3.15, "loading: austenite → martensite", ha="center")
    ax.text(2.9, 1.25, "unloading: back to austenite", ha="center")
    ax.text(3.4, 0.35, "strain fully recovered", ha="center", color=MUTED)

    ax = axes[1]
    ax.set_title("Shape-memory effect  (T < M$_f$)")
    load = np.array([(0, 0), (0.4, 1.5), (4.2, 1.78), (4.9, 3.7)])
    ax.plot(*load.T, color=ORANGE)
    ax.plot([4.9, 4.35], [3.7, 0], color=ORANGE, ls="--")
    ax.annotate("", (0.1, 0.12), (4.3, 0.12),
                arrowprops=dict(arrowstyle="->", color=AQUA, lw=2, connectionstyle="arc3,rad=0.12"))
    ax.text(2.3, 2.05, "martensite reorients", ha="center")
    ax.text(5.0, 1.7, "unloading", ha="left")
    ax.text(2.2, 0.75, "heating above A$_f$ restores the shape", ha="center")
    ax.text(4.5, 0.3, "residual\nstrain", ha="left", color=MUTED, fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sma_behaviour.png"))
    plt.close(fig)


def energy_landscape():
    """Cut of Pi along etr in fully martensitic material at -80 deg C."""
    mat = Material(E_M=55_260.0, sig_reo=105.0)     # modulus and threshold at -80 deg C
    xi, T, etr_n = 1.0, -80.0, 0.020
    M = SQRT15 * etr_n
    etr = np.linspace(-0.045, 0.069, 4000)

    def parts(eps):
        psi = free_energy(xi, etr, eps, T, mat)
        elastic = 0.5 * mat.E_M * (eps - etr) ** 2
        dis = dissipation(xi, etr, xi, etr_n, M, T, mat)
        return elastic, psi - elastic, dis

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    elastic, interaction, dis = parts(0.030)
    total = elastic + interaction + dis
    a.plot(etr, elastic - elastic.min(), color=MUTED, ls="--", lw=1.4, label="elastic")
    a.plot(etr, interaction - interaction.min(), color=AQUA, ls="-.", lw=1.6, label="interaction")
    a.plot(etr, dis, color=ORANGE, ls=":", lw=2.2, label="dissipation $\\mathcal{D}$ (non-smooth)")
    a.plot(etr, total - total.min(), color=BLUE, lw=2.4, label="Π = Ψ + $\\mathcal{D}$")
    a.axvline(etr_n, color=MUTED, lw=0.8, ls=":")
    a.annotate("kink at the\nprevious state ε$^{tr}_n$", xy=(etr_n, (total - total.min())[np.argmin(abs(etr - etr_n))]),
               xytext=(-0.006, 0.55), ha="center", arrowprops=dict(arrowstyle="->", color=MUTED))
    a.set(xlim=(-0.02, 0.06), ylim=(-0.2, 6), xlabel="transformation strain  ε$^{tr}$",
          ylabel="energy density [MPa]", title="(a) the terms at strain ε = 0.030")
    a.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2)

    for eps, color in zip([0.016, 0.020, 0.022, 0.027], BLUE_RAMP):
        total = sum(parts(eps))
        total = total - total.min()
        j = int(np.argmin(total))
        pinned = abs(etr[j] - etr_n) < 1e-4
        b.plot(etr, total, color=color, lw=1.8,
               label=f"ε = {eps:.3f}" + ("  (minimum pinned)" if pinned else ""))
        b.plot(etr[j], 0.0, "o", color=color, ms=8, mec=SURFACE, mew=2, zorder=3)
    b.axvline(etr_n, color=MUTED, lw=0.8, ls=":")
    b.set(xlim=(0.013, 0.031), ylim=(-0.03, 0.62), xlabel="transformation strain  ε$^{tr}$",
          ylabel="Π − min Π  [MPa]", title="(b) the minimum stays in the kink for ε ∈ [0.019, 0.023]")
    b.set_xticks(np.arange(0.014, 0.031, 0.004))
    b.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "energy_landscape.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    sma_behaviour()
    energy_landscape()
    print("figures written to", os.path.abspath(OUT))
