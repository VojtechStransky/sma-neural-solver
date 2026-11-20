"""Figures of step 01 (baseline network v10).

  ../figures/architecture.png           the network and the loop around it
  ../figures/training_curves.png        the three error curves during training
  ../figures/one_step_correlation.png   predicted against reference state, one step
  ../figures/warm_cycle.png             a free run on a warm cycle

Run:  python make_figures.py
"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.data import MODELS, load_cycles
from common.plotstyle import AQUA, BLUE, INK, MUTED, ORANGE, SURFACE, apply_style
from common.rollout import OnnxStep, free_run, r_squared, teacher_forced
from common.stress import stress

STEP = pathlib.Path(__file__).resolve().parents[1]
FIGURES = STEP / "figures"
WARM_CYCLE = 19


def architecture():
    """Boxes and arrows: the network (blue) and the loop that feeds its output back (orange)."""
    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    ax.set(xlim=(0, 131), ylim=(0, 36))
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=SURFACE, edgecolor=color, linewidth=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(start, end, color=MUTED, style="-", bend=0.0):
        ax.annotate("", end, start, arrowprops=dict(arrowstyle="->", color=color, lw=1.4, ls=style,
                                                    connectionstyle=f"arc3,rad={bend}"))

    box(1, 17, 17, 10, "input\nξₙ, εᵗʳₙ, Mₙ, ε, T", MUTED)
    box(23, 25, 15, 7, "normalization", BLUE)
    box(23, 12, 15, 7, "4 products\nof the inputs", BLUE)
    box(43, 17, 13, 10, "dense 128\nswish", BLUE)
    box(60, 17, 14, 10, "3 residual\nblocks, 128", BLUE)
    box(78, 17, 12, 10, "dense 128", BLUE)
    box(95, 25, 11, 7, "head  Δξ", BLUE)
    box(95, 12, 11, 7, "head  Δεᵗʳ", BLUE)
    box(110, 25, 20, 7, "ξ = clip(ξₙ + Δξ, 0, 1)", MUTED)
    box(110, 12, 20, 7, "εᵗʳ = εᵗʳₙ + Δεᵗʳ", MUTED)
    box(32, 1, 62, 6, "simulation loop:  Mₙ₊₁ = max(Mₙ · min(ξₙ / ξ, 1),  √1.5 · |εᵗʳ|)", ORANGE)

    arrow((18, 24), (23, 28.5)), arrow((18, 20), (23, 15.5))
    arrow((38, 28.5), (43, 24)), arrow((38, 15.5), (43, 20))
    arrow((56, 22), (60, 22)), arrow((74, 22), (78, 22))
    arrow((90, 24), (95, 28.5)), arrow((90, 20), (95, 15.5))
    arrow((106, 28.5), (110, 28.5)), arrow((106, 15.5), (110, 15.5))
    arrow((120, 12), (94, 4), ORANGE, bend=-0.15)
    arrow((32, 4), (9.5, 17), ORANGE, "--", bend=-0.15)
    ax.text(13, 2.5, "state of the next step", color=INK, fontsize=8.5, ha="center")
    fig.savefig(FIGURES / "architecture.png")
    plt.close(fig)


def training_curves():
    h = np.load(STEP / "results" / "v10_training_history.npz")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.semilogy(h["epoch"], h["train"], color=BLUE, label="training loss (one step)")
    ax.semilogy(h["epoch"], h["one_step"], color=AQUA, ls="--", label="one-step error on single steps kept out of training")
    ax.semilogy(h["epoch"], h["cold_roll"], color=ORANGE, ls=":", lw=2.4,
                label="free-run error on the 37 validation cycles (cold cycles ×3)")
    ax.set(xlabel="epoch", ylabel="error (log scale, units differ between curves)",
           title="One-step error keeps falling, free-run error does not")
    ax.legend(loc="center right")
    fig.savefig(FIGURES / "training_curves.png")
    plt.close(fig)


def one_step_correlation(model, cycles):
    ref = np.concatenate([c.Y for c in cycles])
    pred = np.concatenate([teacher_forced(model, c) for c in cycles])
    shown = np.random.default_rng(0).choice(len(ref), 15_000, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax, k, name in zip(axes, (0, 1), ("martensite fraction ξ", "transformation strain εᵗʳ")):
        ax.plot(ref[shown, k], pred[shown, k], ".", color=BLUE, ms=2.5, alpha=0.35)
        lim = [ref[:, k].min(), ref[:, k].max()]
        ax.plot(lim, lim, color=MUTED, lw=0.8)
        ax.set(xlabel=f"reference {name}", ylabel="network, one step",
               title=f"{name}:  R² = {r_squared(ref[:, k], pred[:, k]):.5f}")
    fig.tight_layout()
    fig.savefig(FIGURES / "one_step_correlation.png")
    plt.close(fig)


def warm_cycle(model, cycles):
    c = next(x for x in cycles if x.id == WARM_CYCLE)
    run = free_run(model, c)
    ref_style = dict(color=INK, lw=1.5, label="reference solver")
    net_style = dict(color=ORANGE, lw=1.9, ls=(0, (1, 1)), label="network, free run")
    fig, (a, b, s) = plt.subplots(1, 3, figsize=(13, 3.7))
    a.plot(c.Y[:, 0], **ref_style), a.plot(run[:, 0], **net_style)
    a.set(xlabel="step", ylabel="martensite fraction ξ")
    b.plot(c.Y[:, 1], **ref_style), b.plot(run[:, 1], **net_style)
    b.set(xlabel="step", ylabel="transformation strain εᵗʳ")
    s.plot(c.strain, stress(c.strain, c.Y[:, 0], c.Y[:, 1], c.temperature), **ref_style)
    s.plot(c.strain, stress(c.strain, run[:, 0], run[:, 1], c.temperature), **net_style)
    s.set(xlabel="strain ε", ylabel="stress σ [MPa]")
    b.legend(loc="lower center")
    fig.suptitle(f"Cycle {c.id}, loaded at {c.loading_temperature:.0f} °C: "
                 f"free-run error of ξ {np.abs(run[:, 0] - c.Y[:, 0]).mean():.4f}", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "warm_cycle.png")
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    cycles = load_cycles()
    model = OnnxStep(MODELS / "v10.onnx")
    architecture()
    training_curves()
    one_step_correlation(model, cycles)
    warm_cycle(model, cycles)
    print("figures written to", FIGURES)
