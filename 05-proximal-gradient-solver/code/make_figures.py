"""Figures of step 05, from the formulas and from the files in ../results/.

  ../figures/unrolled_network.png      the planned chain of K layers (diagram for the plan pages)
  ../figures/elastic_domain.png        kink lines of g and the trapezoid C
  ../figures/prox_activation.png       the prox, one output at a time
  ../figures/nm_convergence.png        stress error after each iteration, Nelder-Mead and proximal (page 03)

Run:  python make_figures.py        (needs only the public energy)
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.plotstyle import AQUA, BLUE, INK, MUTED, ORANGE, SURFACE, apply_style  # noqa: E402
from prox_gradient_solver import Material, corners, prox_g, thresholds  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIGURES, RESULTS = HERE.parent / "figures", HERE.parent / "results"
EXAMPLE = (0.3, 0.05, 0.075)            # xi_n, etr_n, M_n of the example state
HOLD = "#eef1f4"


def unrolled_network():
    fig, ax = plt.subplots(figsize=(10.5, 3.3))
    ax.set(xlim=(0, 10.5), ylim=(0, 3.3))
    ax.axis("off")

    def box(x, y, w, h, text, color, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    fc=color, ec=MUTED, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, weight=weight)

    def arrow(x0, y0, x1, y1, **kw):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2, **kw))

    box(0.1, 1.15, 1.55, 1.0, "old state\n$\\xi_n,\\ \\varepsilon^{tr}_n,\\ M_n$\nload $\\varepsilon,\\ T$", SURFACE)
    x = 2.2
    for k in ("1", "2", "…", "K"):
        if k == "…":
            ax.text(x + 0.3, 1.65, "…", ha="center", va="center", fontsize=16, color=MUTED)
            arrow(x + 0.65, 1.65, x + 0.95, 1.65)
            x += 0.95
            continue
        box(x, 1.95, 1.45, 0.62, "gradient step\non smooth $f$", "#dbe9fb")
        box(x, 0.75, 1.45, 0.62, "prox of $g$:\nhold or move", "#fde3d6")
        arrow(x + 0.72, 1.95, x + 0.72, 1.38)
        ax.text(x + 0.72, 2.75, f"layer {k}", ha="center", weight="bold", fontsize=9.5)
        arrow(x + 1.48, 1.65, x + 1.88, 1.65)
        x += 1.9
    box(x, 1.15, 1.45, 1.0, "new state\n$\\xi,\\ \\varepsilon^{tr}$", "#d9f2e8")
    arrow(1.68, 1.65, 2.17, 1.65)
    ax.plot([0.9, 0.9, x - 1.0], [1.12, 0.35, 0.35], color=MUTED, lw=1, ls=":")
    for xk in (2.92, 4.82, x - 1.2):
        ax.annotate("", (xk, 0.73), (xk, 0.35), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1, ls=":"))
    ax.text(3.9, 0.12, "every layer reads the old state and the load: they fix the kinks and thresholds",
            ha="center", color=MUTED, fontsize=9)
    fig.savefig(FIGURES / "unrolled_network.png")
    plt.close(fig)


def elastic_domain():
    mat = Material()
    xi_n, etr_n, M = EXAMPLE
    Y_F0, Y_R, tau0 = thresholds(xi_n, etr_n, M, mat)
    C = corners(xi_n, etr_n, M, mat)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    # (a) the plane of increments: three kink lines, four linear pieces
    s = 1.0
    a.plot([0, 0], [-s, s], color=INK, lw=1.6)                       # xi = xi_n
    a.plot([0, s], [0, 0], color=ORANGE, lw=1.6)                     # q = q_n (forward side)
    a.plot([-s, 0], [-0.45 * s, 0], color=BLUE, lw=1.6)              # etr = etr_n (reverse side), schematic slope
    a.plot(0, 0, "o", color=INK, ms=7, mec=SURFACE, mew=1.5, zorder=3)
    for (x, y, text) in [(0.5, 0.55, "piece 1\nforward, $q$ grows"), (0.5, -0.55, "piece 2\nforward, $q$ shrinks"),
                         (-0.5, 0.35, "piece 3\nreverse, $\\varepsilon^{tr}$ grows"),
                         (-0.5, -0.72, "piece 4\nreverse, $\\varepsilon^{tr}$ shrinks")]:
        a.text(x, y, text, ha="center", va="center", fontsize=9.5)
    a.text(0.03, 0.93, "$\\xi = \\xi_n$", fontsize=9.5)
    a.text(0.97, 0.05, "$q = q_n$", ha="right", color=ORANGE, fontsize=9.5)
    a.text(-0.7, -0.2, "$q = \\varepsilon^{tr}_n\\,\\xi$", color=BLUE, fontsize=9.5, ha="center")
    a.annotate("old state", (0, 0), (0.22, -0.2), fontsize=9, color=MUTED,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    a.set(xlim=(-s, s), ylim=(-s, s), xticks=[], yticks=[],
          xlabel="change of martensite fraction  $\\Delta\\xi$", ylabel="change of  $q = \\xi\\varepsilon^{tr}$",
          title="(a) $g$ is linear in four pieces; kinks on three lines (schematic)")
    a.grid(False)

    # (b) the slopes of the four pieces are the corners of the elastic domain
    b.add_patch(Polygon(C, closed=True, fc=HOLD, ec=INK, lw=1.6, zorder=2))
    names = ["$c_1$", "$c_3$", "$c_4$", "$c_2$"]
    offsets = [(6, 6), (-6, 6), (-6, -12), (6, -12)]
    for c, name, off in zip(C, names, offsets):
        b.plot(*c, "o", color=INK, ms=6, zorder=3)
        b.annotate(name, c, xytext=off, textcoords="offset points", ha="left" if off[0] > 0 else "right")
    b.text(C[:, 0].mean(), 0, "R0\nhold", ha="center", va="center", fontsize=10, weight="bold", zorder=4)
    b.text(-5.5, tau0 + 22, "R1: reorientation,\n$\\xi$ held", ha="center", fontsize=9.5)
    b.text(-0.5, -tau0 - 82, "R1: reorientation,\n$\\xi$ held", ha="center", fontsize=9.5)
    b.text(Y_F0 + 2.0, 0, "forward\ntransformation,\n$q$ held", va="center", fontsize=9.5)
    b.text(-Y_R - 3.0, -20, "R4: reverse\ntransformation,\n$\\varepsilon^{tr}$ held", va="center", ha="right", fontsize=9.5)
    b.text(Y_F0 + 5.0, tau0 + 30, "R2: forward\n+ reorientation", fontsize=9.5)
    b.text(Y_F0 + 7.0, -tau0 - 45, "R2", fontsize=9.5)
    b.text(-Y_R - 11.0, tau0 + 30, "R3: reverse\n+ reorientation", fontsize=9.5, ha="right")
    b.text(-Y_R - 3.0, -tau0 - 45, "R3", fontsize=9.5, ha="right")
    b.set(xlim=(-46, 30), ylim=(-245, 215), xlabel="driving force on $\\xi$  [MPa]",
          ylabel="driving force on $q$  [MPa]",
          title="(b) elastic domain $C$ for $\\xi_n = 0.3$, $\\varepsilon^{tr}_n = 0.05$, $M_n = 0.075$")
    fig.tight_layout()
    fig.savefig(FIGURES / "elastic_domain.png")
    plt.close(fig)


def prox_activation():
    mat = Material()
    C = corners(*EXAMPLE, mat)
    H = np.eye(2)                                         # step size t = 1
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    w_xi = np.linspace(-40, 25, 521)
    for w_q, color, label in [(0.0, BLUE, "$w_q = 0$"), (200.0, ORANGE, "$w_q = +200$ (q pushed up)"),
                              (-200.0, AQUA, "$w_q = -200$ (q pushed down)")]:
        a.plot(w_xi, [prox_g(np.array([x, w_q]), C, H)[0][0] for x in w_xi], color=color, label=label)
    a.plot(w_xi, w_xi, color=MUTED, lw=0.9, ls=":", label="no kink")
    a.set(xlabel="trial change of $\\xi$:  $w_\\xi$", ylabel="actual change  $d_\\xi$",
          title="(a) $\\xi$: an asymmetric dead zone", ylim=(-30, 22))
    a.legend(loc="upper left")

    w_q = np.linspace(-320, 320, 641)
    for w_x, color, style, label in [(0.0, BLUE, "-", "$w_\\xi = 0$"),
                                     (-30.0, ORANGE, "--", "$w_\\xi = -30$ (reverse step)")]:
        b.plot(w_q, [prox_g(np.array([w_x, q]), C, H)[0][1] for q in w_q], color=color, ls=style, label=label)
    b.plot(w_q, w_q, color=MUTED, lw=0.9, ls=":", label="no kink")
    b.set(xlabel="trial change of $q$:  $w_q$", ylabel="actual change  $d_q$",
          title="(b) $q$: a soft threshold at $\\pm\\tau_0$", ylim=(-220, 220))
    b.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "prox_activation.png")
    plt.close(fig)


# ------------------------------------------------------------ against Nelder-Mead (page 03)

FLOOR = 1e-9            # stress errors below this are drawn at this value [MPa]


def nm_convergence():
    """Stress error after each iteration on the public energy: Nelder-Mead against the proximal solver."""
    d = json.load(open(RESULTS / "benchmark_convergence.json"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.0, 5.2))
    stops = np.sort(d["proximal_stop_iterations"])
    last = int(stops[-1])                      # the iteration by which every step has stopped
    for solver, color in (("nelder_mead", INK), ("proximal", BLUE)):
        pct = d[solver]["percentiles"]
        end = len(pct["50"]) if solver == "nelder_mead" else last
        it = np.arange(1, end + 1)
        band = lambda key: np.maximum(np.asarray(pct[key][:end]), FLOOR)
        a.fill_between(it, band("10"), band("90"), color=color, alpha=0.12, lw=0)
        a.fill_between(it, band("25"), band("75"), color=color, alpha=0.28, lw=0)
        a.plot(it, band("50"), color=color)
        b.plot(it, d[solver]["share_above_0.1_MPa"][:end], color=color)
        if solver == "proximal":
            a.plot(last, max(pct["50"][last - 1], FLOOR), "s", color=color, ms=6)
            b.plot(last, d[solver]["share_above_0.1_MPa"][last - 1], "s", color=color, ms=6)
    for ax in (a, b):
        for phase in (100, 200):
            ax.axvline(phase, color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(1, 300)
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 300], ["1", "2", "5", "10", "20", "50", "100", "200", "300"])
        ax.minorticks_off()
        ax.set_xlabel("iteration")
    a.text(100, 3e-9, " phase 2", color=MUTED, fontsize=8.5)
    a.text(200, 3e-9, " phase 3", color=MUTED, fontsize=8.5)
    a.set(yscale="log", ylim=(FLOOR, 1000), ylabel="stress error [MPa]")
    a.set_title("(a) Error after each iteration", fontsize=11, fontweight="bold", loc="left", pad=20)
    a.text(0.0, 1.02, f"{d['steps']} path steps in which the state must change", transform=a.transAxes,
           ha="left", va="bottom", fontsize=8.8, color=MUTED)
    b.set(ylim=(0, 102), ylabel="answers more than 0.1 MPa off [%]")
    b.set_title("(b) Answers still more than 0.1 MPa off", fontsize=11, fontweight="bold", loc="left", pad=20)
    b.text(0.0, 1.02, "Nelder–Mead: 5 answers per step (5 starting simplices); proximal: 1",
           transform=b.transAxes, ha="left", va="bottom", fontsize=8.8, color=MUTED)
    from matplotlib.patches import Patch
    L = plt.Line2D
    fig.legend(handles=[L([], [], color=INK, label="Nelder–Mead"),
                        L([], [], color=BLUE, label="proximal Newton"),
                        L([], [], color=MUTED, label="line: median"),
                        Patch(color=MUTED, alpha=0.45, label="dark band: 25th–75th percentile"),
                        Patch(color=MUTED, alpha=0.18, label="light band: 10th–90th percentile"),
                        L([], [], color=BLUE, marker="s", ls="none", ms=6,
                          label=f"proximal solver has stopped in every step (iteration {last})")],
               loc="lower left", bbox_to_anchor=(0.05, 0.0), ncol=3, fontsize=8.8, frameon=False)
    fig.tight_layout(w_pad=3.0, rect=(0, 0.1, 1, 1))
    fig.savefig(FIGURES / "nm_convergence.png")
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    FIGURES.mkdir(exist_ok=True)
    unrolled_network()
    elastic_domain()
    prox_activation()
    if (RESULTS / "benchmark_convergence.json").exists():
        nm_convergence()
    print("figures written to", FIGURES)
