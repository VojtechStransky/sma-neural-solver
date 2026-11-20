# 00 · Problem definition

## The problem in one paragraph

A constitutive model of the shape-memory alloy NiTi computes, at every load step, the new internal state of the
material by minimizing an energy. The objective is non-smooth, and the reference implementation solves it with
the Nelder–Mead method. That costs about 0.26 s per step, and the method has no optimality check and no
convergence theory for this problem. A finite-element simulation needs $10^5$ to $10^8$ such steps and cannot
tolerate a wrong one. The goal is a replacement that is **much faster and verifiably accurate**.

```math
(\xi,\ \varepsilon^{\mathrm{tr}})_{n+1} \;=\; \arg\min_{\xi,\ \varepsilon^{\mathrm{tr}}}\
\Big[\ \Psi(\varepsilon, T, \xi, \varepsilon^{\mathrm{tr}}) \;+\; \mathcal{D}\big(\xi - \xi_n,\ \varepsilon^{\mathrm{tr}} - \varepsilon^{\mathrm{tr}}_n\big)\ \Big]
```

$\Psi$ is the free energy, $\mathcal{D}$ the dissipation, $\xi$ the fraction of martensite and
$\varepsilon^{\mathrm{tr}}$ its transformation strain. The subscript $n$ marks the known state from the previous
step.

## Pages

| Page | Content |
|---|---|
| [physical-background.md](physical-background.md) | The material, the two effects, internal variables, the modelling framework |
| [minimization-program.md](minimization-program.md) | The full program: every term of $\Psi$ and $\mathcal{D}$, the kinks, convexity, calibration |

## Code

| File | What it does |
|---|---|
| [code/energy_1d.py](code/energy_1d.py) | The one-dimensional energy $\Pi = \Psi + \mathcal{D}$, the stress and the memory update |
| [code/plot_figures.py](code/plot_figures.py) | The two figures of this folder |

```
cd code
python plot_figures.py
```

Needs `numpy` and `matplotlib`.

## Four facts to take into the next steps

1. **The solution has a "no change" region.** For a whole range of loads the minimizer is exactly the previous
   state. A replacement must reproduce that exactly, or the state drifts.
2. **The model is applied to its own output.** Errors of single steps add up along a loading path, so one-step
   accuracy says little.
3. **The problem is convex in the right variables.** In $(\xi,\ q = \xi\varepsilon^{\mathrm{tr}})$ the model is
   strictly convex with one minimum; in the solver's own variables it is not convex.
4. **Nelder–Mead is accurate in a typical step, not in every step.** On the one-dimensional model its median
   stress error is $2\cdot10^{-6}$ MPa, but 30 of 2,602 path steps are more than 0.1 MPa off, and it stays at the
   old state in 48 of 2,324 steps where the state has to move
   ([benchmark](../05-proximal-gradient-solver/04-benchmark-against-nelder-mead.md)). It also needs about 650
   energy evaluations per step and has no convergence test.

## Where the model comes from

The constitutive model is the thermomechanical model of Sedlák, Frost, Benešová, Ben Zineb and Šittner,
*International Journal of Plasticity* 39 (2012), 132–151, [doi:10.1016/j.ijplas.2012.06.008](https://doi.org/10.1016/j.ijplas.2012.06.008),
in its one-dimensional form.

The first part of this project (steps 01 to 02) is described in: V. Stránský, P. Sedlák, M. Frost, P. Jaroš,
*Simulations of shape memory alloys within the generalized standard materials thermodynamical framework and by
machine learning-based models*, proceedings paper, 2026.
