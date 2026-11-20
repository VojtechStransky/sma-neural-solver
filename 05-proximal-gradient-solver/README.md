# 05 · Proximal-gradient solver

**Question.** Can the Nelder–Mead (downhill simplex) solver be replaced by a method that has a convergence
argument?

**Answer so far.** Yes, on the public one-dimensional model. In the variables $(\xi,\ q = \xi\varepsilon^{\mathrm{tr}})$ the problem
is convex and its kinked part has an exact proximal step: a projection onto a trapezoid whose sides are the
regimes of the material. An ISTA-type iteration built on it, with the step scaled by the curvature, matches a
brute-force minimum on all 3,602 tested steps of the public model, with a median of 4 iterations. In a
[benchmark against Nelder–Mead](04-benchmark-against-nelder-mead.md) it is more accurate and needs about a
hundred times fewer energy evaluations on the same energy. Turning the solver into a network with a fixed number
of layers (unrolling) is planned and not started.

## Context

November 2026, after the thesis. Steps 01 to 04 tried to *imitate* the reference solver with a network, or to
call it less often. This step replaces the optimization method itself.

## Motivation

The reference solver minimizes the energy with the Nelder–Mead (downhill simplex) method. It is slow (about 650
energy evaluations per step on the one-dimensional model), and it gives no reason to trust a single answer: it
runs a fixed budget, has no optimality test, and has no convergence theory for an objective with kinks. A
finite-element simulation needs both speed and a step that does not fail.

The networks of the earlier steps are fast but offer no guarantee either, and the hybrid of
[step 04](../04-gated-hybrid/) still calls Nelder–Mead.

The problem has more structure than either approach uses. In the variables $(\xi,\ q = \xi\varepsilon^{\mathrm{tr}})$
the model is strictly convex with one minimum. The reference code's linearized stiffness is outside that statement,
so the solver built here uses the exact mixture rule.

## The method in four lines

1. Split the energy: $\Pi = f + g$, with $f$ smooth and $g$ holding all the kinks of the dissipation.
2. Take a step on $f$ using its gradient and curvature. This gives a trial state.
3. Apply the prox of $g$: the trial increment minus its projection onto the elastic domain, a trapezoid. Inside
   it the state holds exactly; outside, the side that is hit decides which process is active.
4. Repeat until the step no longer changes.

This is the structure of ISTA (iterative shrinkage-thresholding). Two things differ from the textbook form:
the prox is a projection in two variables instead of a soft threshold, and one common step size does not
work. Two versions of the step do: one step size per unknown, taken for $\xi$ and $\varepsilon^{\mathrm{tr}}$
(plain ISTA), and the Hessian of $f$ (a proximal Newton step, the default), which needs about a third of the
iterations.

## Pages

| Page | Content | Status |
|---|---|---|
| [01 · From Nelder–Mead to a proximal-gradient step](01-from-nelder-mead-to-proximal-gradient.md) | the split $\Pi = f + g$, the thresholds, the iteration, why one threshold per variable is not enough | theory, checked |
| [02 · The exact prox for both unknowns](02-exact-joint-prox.md) | $g$ as a maximum of four lines; prox = projection onto a trapezoid; sides = regimes; Jacobian | theory, checked |
| [03 · Numerical checks and the Newton step](03-guarantees-and-open-gaps.md) | the solver against a brute-force minimum; why one step size fails and the step needs the curvature (Newton step, with its derivation) | solver run on the public model |
| [04 · Benchmark against Nelder–Mead](04-benchmark-against-nelder-mead.md) | accuracy, reliability and cost of both methods on the public energy | measured |
| [Plan: unrolling the solver into a network](plan-learned-unrolling.md) | a fixed number of layers; what could be learned | plan only |
| [Plan: training through the rollout](plan-rollout-training.md) | training along whole loading paths | plan only |

## Results so far

**The solver alone**, on the public one-dimensional model
([code/energy_1d.py](../00-problem-definition/code/energy_1d.py)):

| Result | Number | Where |
|---|---|---|
| The prox formula is optimal | 0 failures of 60,000 optimality tests | [page 02](02-exact-joint-prox.md) |
| Solver against a brute-force minimum, 2,602 steps along 60 loading paths | median 4 iterations, at most 14; never above the brute-force minimum | [page 03](03-guarantees-and-open-gaps.md) |
| Same, 1,000 random states with very large load jumps | median 8 iterations, at most 20; never above the brute-force minimum | page 03 |
| Plain ISTA step: one step size per unknown, for $\xi$ and $\varepsilon^{\mathrm{tr}}$ | converged in all 3,602 steps to the same minimum; median 13 iterations, at most 212 (path steps), 22 and 232 (random states) | page 03 |
| One common step size; one step size per unknown for $\xi$ and $q$ | not converged within 2,000 iterations in 59 % and 20 % of the path steps | page 03 |

**Against Nelder–Mead**, both minimizing the public energy ([page 04](04-benchmark-against-nelder-mead.md)).
The truth is the best known minimum of each step.

| 2,602 steps along 60 loading paths | Nelder–Mead | proximal |
|---|---|---|
| Stress error, 99th percentile [MPa] | 0.046 | $1\cdot10^{-5}$ |
| Stress error, largest [MPa] | 0.58 | $2\cdot10^{-5}$ |
| Steps more than 0.1 MPa off | 30 | 0 |
| Returns the old state although the state must change | 48 steps | 0 |
| Energy evaluations per step, median | 648 | 5 |
| Wall time per step, median (NumPy, busy machine) | 51 ms | 1.3 ms |

Nelder–Mead is a re-implementation with the schedule of the reference solver. Each Nelder–Mead step gives five
answers (five starting simplices), and a step counts as off when one of them is.

**Along loading paths.** Each solver simulated the 60 paths alone, feeding back its own state. The proximal
solver stays within $2\cdot10^{-5}$ MPa of the true path; Nelder–Mead deviates by up to 0.58 MPa and by more
than 0.1 MPa somewhere on 7 to 10 of the 60 paths.

**What these numbers do not show.**

- **Three-dimensional loading.** Everything here is uniaxial. The failures of Nelder–Mead reported from
  finite-element use are not tested.
- **A production timing.** Both solvers are plain NumPy. The ratio of energy evaluations (about 130) is the
  solid number; the wall times are not.

## Findings

1. The elastic domain of the material is a trapezoid in the plane of driving forces, and the exact proximal
   step is a projection onto it. The regimes of the material (hold, reorientation, forward, reverse) are its
   sides and corners.
2. With this step, the iteration finds the minimum in a handful of iterations on the public model and agrees
   with brute force in every tested step.
3. One step size per unknown works only for the right pair of unknowns. For $\xi$ and $q$ the two directions
   are coupled above 0.99 in 24 % of the steps and the iteration stalls. For $\xi$ and
   $\varepsilon^{\mathrm{tr}}$ the coupling stays below 0.88 and plain ISTA with the exact prox converges in
   every tested step. The Hessian step needs about a third of the iterations.
4. The fixed point of the iteration is the optimality condition, so every step can carry a residual that says
   whether it is solved. Nelder–Mead cannot provide that.
5. A step whose trial increment lies inside the trapezoid returns the old state exactly. The drift of
   [step 01](../01-baseline-network/) cannot occur.
6. Nelder–Mead loses accuracy where the solution lies on a kink line, and its typical failure is to stay at the
   old state when the state has to move (48 of 2,324 path steps). More iterations do not repair it.

## What did not work

- One common step size, and one step size per unknown for $\xi$ and $q$ (59 % and 20 % of the path steps
  unconverged after 2,000 iterations).
- Too few step halvings next to a barrier: an iteration stalled in one earlier test.
  The solver now allows 30 halvings.
- An earlier version that updated one unknown at a time needed an extra search direction to work
  ([page 01](01-from-nelder-mead-to-proximal-gradient.md)).

## Code

Three modules hold the method, the baseline and the test problems; the scripts only combine them.

| File | Kind | What it does |
|---|---|---|
| [code/prox_gradient_solver.py](code/prox_gradient_solver.py) | module | the method: the split into $f$ and $g$, the exact prox, the four step rules, `solve_step` |
| [code/nelder_mead.py](code/nelder_mead.py) | module | the baseline: Nelder–Mead with the schedule of the reference solver, on the same energy |
| [code/test_steps.py](code/test_steps.py) | module | the test problems and their truth: loading paths, random states, brute-force minimum, optimality test of the old state |
| [code/verify_solver.py](code/verify_solver.py) | script | the solver against a brute-force minimum, with the four step rules ([page 03](03-guarantees-and-open-gaps.md)) |
| [code/benchmark_solvers.py](code/benchmark_solvers.py) | script | Nelder–Mead against the proximal solver: single steps, free runs, accuracy for a given budget, wall time ([page 04](04-benchmark-against-nelder-mead.md)) |
| [code/benchmark_convergence.py](code/benchmark_convergence.py) | script | the error of both solvers after every iteration ([page 03](03-guarantees-and-open-gaps.md)); needs the output of `benchmark_solvers.py` |
| [code/make_figures.py](code/make_figures.py) | script | the figures of this step |

```
cd code
python verify_solver.py --workers 8          # about 10 minutes
python benchmark_solvers.py --workers 8      # about 25 minutes
python benchmark_convergence.py --workers 8  # about 15 minutes, after benchmark_solvers.py
python make_figures.py
```

Needs `numpy` and `matplotlib`; nothing in this step needs the reference implementation. The recorded outputs
are in [results/](results/).

## Next step

1. Add the residual check, so each step reports whether it is solved.
2. Treat the limits on $\xi$ and $\varepsilon^{\mathrm{tr}}$ as hard constraints inside the projection.
3. Derive and test the consistent tangent $\partial\sigma/\partial\varepsilon$.
4. Repeat the benchmark on three-dimensional loading, where the failures of Nelder–Mead were reported.
5. Only then: fix the number of iterations and learn parts of the step
   ([plan](plan-learned-unrolling.md), [plan](plan-rollout-training.md)).

## References

- K. Gregor, Y. LeCun (2010). Learning fast approximations of sparse coding. *ICML*.
- J. R. Hershey, J. Le Roux, F. Weninger (2014). Deep unfolding: model-based inspiration of novel deep
  architectures. [arXiv:1409.2574](https://arxiv.org/abs/1409.2574).
- Y. Yang, J. Sun, H. Li, Z. Xu (2016). Deep ADMM-Net for compressive sensing MRI. *Advances in Neural
  Information Processing Systems* 29. Extended as ADMM-CSNet, *IEEE Transactions on Pattern Analysis and
  Machine Intelligence*, 2018.
- D. Ito, S. Takabe, T. Wadayama (2019). Trainable ISTA for sparse signal recovery. *IEEE Transactions on
  Signal Processing* 67(12), 3113–3125.
- V. Monga, Y. Li, Y. C. Eldar (2021). Algorithm unrolling: interpretable, efficient deep learning for signal
  and image processing. *IEEE Signal Processing Magazine* 38(2), 18–44.
- I. Daubechies, M. Defrise, C. De Mol (2004). An iterative thresholding algorithm for linear inverse problems
  with a sparsity constraint. *Communications on Pure and Applied Mathematics* 57(11), 1413–1457.
- A. Beck, M. Teboulle (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems.
  *SIAM Journal on Imaging Sciences* 2(1), 183–202.
- J. J. Moreau (1965). Proximité et dualité dans un espace hilbertien. *Bulletin de la Société Mathématique de
  France* 93, 273–299.
- N. Parikh, S. Boyd (2014). Proximal algorithms. *Foundations and Trends in Optimization* 1(3), 127–239.
- J. D. Lee, Y. Sun, M. A. Saunders (2014). Proximal Newton-type methods for minimizing composite functions.
  *SIAM Journal on Optimization* 24(3), 1420–1443.
- J. C. Simo, T. J. R. Hughes (1998). *Computational Inelasticity*. Springer.
- P. Sedlák, M. Frost, B. Benešová, T. Ben Zineb, P. Šittner (2012). Thermomechanical model for NiTi-based shape
  memory alloys including R-phase and material anisotropy under multi-axial loadings. *International Journal of
  Plasticity* 39, 132–151.
