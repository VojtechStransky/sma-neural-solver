"""Numbers and figures that describe the datasets in `data/`.

  ../figures/input_coverage.png    distribution of the five inputs in the single-step pairs
  ../figures/benchmark_cycles.png  the 250 benchmark cycles: temperatures, lengths, one example

Run:  python describe_data.py
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.data import load_cycles, load_single_steps, split_cycles
from common.plotstyle import BLUE, INK, apply_style
from common.stress import stress

FIGURES = pathlib.Path(__file__).resolve().parent / "figures"
INPUTS = [("ξₙ", "martensite fraction"), ("εᵗʳₙ", "transformation strain"),
          ("Mₙ", "memory variable"), ("ε", "strain"), ("T [°C]", "temperature")]
NO_CHANGE = 1e-5     # |increment| below this counts as "no change"


def temperature_program(cycle):
    """'isothermal', 'ramp, then isothermal loading' or 'thermomechanical'."""
    d_eps = np.abs(np.diff(cycle.strain)) > 1e-6
    d_T = np.abs(np.diff(cycle.temperature)) > 1e-6
    if not d_T.any():
        return "isothermal"
    if not (d_T & d_eps).any() and d_eps.any():
        return "ramp, then isothermal loading"
    return "thermomechanical"


def print_numbers(X, dY, cycles):
    print(f"single-step pairs: {len(X):,}")
    print("ranges of the inputs:")
    for (symbol, meaning), lo, hi in zip(INPUTS, X.min(0), X.max(0)):
        print(f"  {symbol:8s} {lo:9.4g} .. {hi:9.4g}   {meaning}")
    still = (np.abs(dY) < NO_CHANGE).mean(0)
    print(f"pairs with no change: xi {100 * still[0]:.1f} %, etr {100 * still[1]:.1f} %")

    lengths = np.array([len(c.X) for c in cycles])
    print(f"\ncycles: {len(cycles)}, steps {lengths.sum():,} "
          f"(shortest {lengths.min()}, median {int(np.median(lengths))}, longest {lengths.max()})")
    T = np.array([c.loading_temperature for c in cycles])
    print(f"loading temperature: {T.min():.0f} .. {T.max():.0f} deg C")
    kinds = [temperature_program(c) for c in cycles]
    for kind in sorted(set(kinds)):
        print(f"  {kinds.count(kind):4d}  {kind}")
    print(f"  {sum(c.reversals == 1 for c in cycles):4d}  with one load reversal (load, then unload)")
    steps = np.concatenate([c.Y - c.X[:, :2] for c in cycles])
    still = (np.abs(steps) < NO_CHANGE).mean(0)
    print(f"steps of the cycles with no change: xi {100 * still[0]:.1f} %, etr {100 * still[1]:.1f} %")
    d_eps = np.concatenate([np.abs(np.diff(c.strain)) for c in cycles])
    print(f"strain step in the cycles: median of the moving steps {np.median(d_eps[d_eps > 1e-6]):.2g}")
    print("cycle split (fine-tuning / validation / test):", [len(s) for s in split_cycles(len(cycles))])


def input_coverage(X):
    fig, axes = plt.subplots(1, 5, figsize=(13, 2.7))
    for ax, column, (symbol, meaning) in zip(axes, X.T, INPUTS):
        ax.hist(column, bins=60, color=BLUE, linewidth=0)
        ax.set_xlabel(f"{symbol}   ({meaning})")
        ax.set_yticks([])
        ax.grid(False)
    axes[0].set_ylabel("number of pairs")
    fig.suptitle(f"Where the {len(X):,} single-step pairs lie", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "input_coverage.png")
    plt.close(fig)


def benchmark_cycles(cycles, example_id=17):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.6))
    a.hist([cy.loading_temperature for cy in cycles], bins=np.arange(-52.5, 105, 5), color=BLUE,
           edgecolor="#fcfcfb", linewidth=1)
    a.set(xlabel="temperature during loading [°C]", ylabel="number of cycles",
          title="250 cycles by temperature")
    b.hist([len(cy.X) for cy in cycles], bins=np.arange(0, 1400, 80), color=BLUE,
           edgecolor="#fcfcfb", linewidth=1)
    b.set(xlabel="steps in the cycle", ylabel="number of cycles", title="Cycle length")

    cy = next(x for x in cycles if x.id == example_id)
    sigma = stress(cy.strain, cy.Y[:, 0], cy.Y[:, 1], cy.temperature)
    c.plot(cy.strain, sigma, color=INK, lw=1.6)
    c.set(xlabel="strain ε", ylabel="stress σ [MPa]",
          title=f"Example: cycle {cy.id}, loaded at {cy.loading_temperature:.0f} °C")
    fig.tight_layout()
    fig.savefig(FIGURES / "benchmark_cycles.png")
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    X, dY = load_single_steps()
    cycles = load_cycles()
    print_numbers(X, dY, cycles)
    input_coverage(X)
    benchmark_cycles(cycles)
