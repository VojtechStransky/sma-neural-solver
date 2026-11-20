# Benchmark: Nelder–Mead against the proximal solver

**Question.** When both methods minimize the same energy, how do they compare in accuracy, reliability and cost?

**Answer.** On the public one-dimensional model the proximal solver is at the minimum in every tested step
(stress error at most $2\cdot10^{-5}$ MPa) with a median of 5 energy evaluations. Nelder–Mead needs about 650
evaluations and is more than 0.1 MPa off in 30 of 2,602 path steps.

## Setup

| | |
|---|---|
| Energy | the public one-dimensional model ([energy_1d.py](../00-problem-definition/code/energy_1d.py)), the same for both solvers |
| Nelder–Mead | [nelder_mead.py](code/nelder_mead.py), written for this benchmark with the schedule of the reference solver: same variables, three phases of 100 iterations, restarts, five starting simplices |
| Proximal solver | [`solve_step`](code/prox_gradient_solver.py) with the exact gradient and Hessian |
| Script | [code/benchmark_solvers.py](code/benchmark_solvers.py) |
| Recorded output | [results/solver_benchmark.txt](results/solver_benchmark.txt) |

**Truth.** For every step, the *best known minimum*: the lowest energy found by a brute-force search (a grid
over the admissible range, then repeated zooming) or by any solver run on that step. All errors are measured
against it.

**Test sets.**

| Name | What it is |
|---|---|
| path steps | 2,602 single steps along 60 generated loading paths (45 strain cycles at a fixed temperature, in tension or compression, and 15 temperature cycles at a fixed strain). Each step starts from the true state of the path |
| random states | 1,000 random old states with load jumps far larger than a step of a loading path |
| free run | each solver simulates the 60 paths alone and feeds back its own state |

## Accuracy of a single step

Stress error against the best known minimum. Nelder–Mead gives five answers per step, one per starting simplex;
all of them are counted.

| | Path steps | | Random states | |
|---|---|---|---|---|
| | Nelder–Mead | proximal | Nelder–Mead | proximal |
| Steps | 2,602 | 2,602 | 1,000 | 1,000 |
| Stress error, median [MPa] | $2\cdot10^{-6}$ | $1\cdot10^{-6}$ | $2\cdot10^{-6}$ | $1\cdot10^{-6}$ |
| Stress error, 99th percentile [MPa] | 0.046 | $1\cdot10^{-5}$ | 0.066 | $1\cdot10^{-5}$ |
| Stress error, largest [MPa] | 0.58 | $2\cdot10^{-5}$ | 2.2 | $2\cdot10^{-5}$ |
| Steps more than 0.01 MPa off | 93 | 0 | 33 | 0 |
| Steps more than 0.1 MPa off | 30 | 0 | 16 | 0 |
| Steps more than 1 MPa off | 0 | 0 | 2 | 0 |
| Energy above the minimum, largest [MPa] | $2\cdot10^{-4}$ | $1\cdot10^{-14}$ | $6\cdot10^{-5}$ | $1\cdot10^{-14}$ |

A step counts as "off" when at least one of its answers is. The errors of about $10^{-5}$ MPa of the proximal
solver are the resolution of the truth, not of the solver: its energy is within $10^{-14}$ MPa of the minimum.

## Reliability

| Path steps | Nelder–Mead | proximal |
|---|---|---|
| Returns the old state although the state must change | 48 of 2,324 steps | 0 |
| Returns the old state when nothing must change | 248 of 278 steps | 278 of 278 |
| Steps not converged | no convergence test | 0 |

Whether the state must change is decided by the optimality condition of the old state (minus the gradient of
the smooth part inside or outside the elastic domain), not by comparing energies, which agree to the last digit
near a threshold. A Nelder–Mead step counts here when at least one of its five answers shows the behaviour.

Three observations:

1. **Nelder–Mead gets stuck at the old state.** The old state is a vertex of its starting simplex and sits in a
   kink of the energy. When the improvement lies in a narrow direction, the simplex collapses onto that vertex
   and the solver reports "no change". In one path step it did so with the driving force 38 MPa outside the
   elastic domain. Stuck answers are 27 % of all answers more than 0.1 MPa off.
2. **Identical problems get different answers.** The five starting simplices differ only in size. In 32 path
   steps their answers differ by more than 0.1 MPa. The proximal solver has no starting simplex; it starts from
   the old state and the problem has one minimum.
3. **The errors of Nelder–Mead sit in one regime.** Stress error against the best known minimum on the path
   steps, grouped by the regime of the true step:

| Regime of the step | Steps | Nelder–Mead stress error: 99th percentile / largest [MPa] | Proximal stress error: largest [MPa] |
|---|---|---|---|
| hold | 278 | $2\cdot10^{-6}$ / $4\cdot10^{-6}$ | $4\cdot10^{-6}$ |
| reorientation, $\xi$ held | 618 | 0.15 / 0.58 | $1\cdot10^{-5}$ |
| forward transformation with reorientation | 1,180 | $2\cdot10^{-5}$ / 0.024 | $2\cdot10^{-5}$ |
| reverse transformation with reorientation | 446 | $2\cdot10^{-5}$ / $3\cdot10^{-5}$ | $1\cdot10^{-5}$ |
| reverse transformation, $\varepsilon^{\mathrm{tr}}$ held | 42 | 0.011 / 0.017 | $3\cdot10^{-7}$ |
| forward transformation, $q$ held | 38 | 0.055 / 0.055 | $4\cdot10^{-7}$ |

Where both unknowns move, Nelder–Mead is accurate. It loses accuracy where the solution lies on a kink line
(one unknown held), which is where a method that only compares function values has the least to go on.

## Along a loading path

Each solver simulates the 60 paths alone. Deviation from the true path:

| Solver | Stress, mean [MPa] | Stress, largest [MPa] | Paths with more than 0.1 MPa somewhere |
|---|---|---|---|
| proximal | $2\cdot10^{-6}$ | $2\cdot10^{-5}$ | 0 of 60 |
| Nelder–Mead, starting simplex scaled by 0.01 | 0.0025 | 0.56 | 10 of 60 |
| Nelder–Mead, scale 0.26 | 0.0026 | 0.58 | 10 of 60 |
| Nelder–Mead, scale 0.51 | 0.0020 | 0.46 | 7 of 60 |
| Nelder–Mead, scale 0.75 | 0.0023 | 0.37 | 9 of 60 |
| Nelder–Mead, scale 1.00 | 0.0022 | 0.46 | 10 of 60 |

The errors of Nelder–Mead do not build up along these paths: the largest deviation in a free run (0.58 MPa)
equals the largest single-step error.

## Cost

| 2,602 path steps | Nelder–Mead | proximal |
|---|---|---|
| Energy evaluations per step, median | 648 | 5 |
| Energy evaluations per step, largest | 737 | 42 |
| Wall time per step, median | 51 ms | 1.3 ms |
| Wall time per step, 99th percentile | 126 ms | 3.9 ms |
| Wall time per step, largest | 159 ms | 7.4 ms |

How to read these numbers:

- **Evaluations are the solid measure.** They do not depend on the machine. An evaluation of the proximal solver
  returns the energy with its gradient and Hessian.
- **Wall time is plain NumPy on a workstation that was busy with other jobs.** Both solvers were timed on the
  same steps, in turn, taking the fastest of three runs. Repeated runs gave a ratio of 40 to 50. The absolute
  values are upper bounds.
- **The cost of Nelder–Mead is fixed**, because it has no convergence test. The cost of the proximal solver
  follows the difficulty of the step: 1 evaluation when nothing changes, 5 in a typical step, up to 65 for a
  random state with a very large load jump.

More budget does not repair Nelder–Mead. Its largest error stays at 0.46 MPa from 30 to 300 iterations per
phase, because a simplex that is stuck at the old state stays there. The proximal solver is at the minimum in
every path step after 12 iterations; stopped after 6 it is still 1.2 MPa off in one step, so it has to be run
to convergence.

## Limits of this benchmark

- One-dimensional loading only. The failures of Nelder–Mead reported from three-dimensional simulations are not
  tested here.
- The Nelder–Mead is a re-implementation with the schedule of the reference solver, not the original code.
  Outside $0 < \xi < 1$ the public energy formula is unbounded below, and without a guard Nelder–Mead left the
  range on 4 of the 60 paths. Both solvers therefore see an infinite energy there.
- The public energy keeps $\xi$ and $\varepsilon^{\mathrm{tr}}$ in range with steep barriers. They set the
  largest curvatures of the problem ([page 03](03-guarantees-and-open-gaps.md)).
- The tangent $\partial\sigma/\partial\varepsilon$ is not part of the comparison; the proximal solver does not
  provide one yet.
- Timings are for unoptimized NumPy code on a shared machine.

## Reproduce

```
cd code
python benchmark_solvers.py --workers 8      # about 25 minutes
```
