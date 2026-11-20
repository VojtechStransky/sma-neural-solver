# A faster and more reliable solver for a shape-memory alloy material model

A material model for the shape-memory alloy NiTi solves a small non-smooth optimization problem at every load
step. The reference solver (Nelder–Mead) needs about 0.26 s per step, has no optimality check and no convergence
theory for this problem. A finite-element simulation calls it $10^5$ to $10^8$ times.

This repository documents the search for a replacement that is **fast and verifiably accurate**:
first neural networks that imitate the solver, then a proximal-gradient solver that uses the structure of the
problem. It records what was tried, what was measured, and what turned out to be wrong.

```math
(\xi,\ \varepsilon^{\mathrm{tr}})_{n+1} \;=\; \arg\min_{\xi,\ \varepsilon^{\mathrm{tr}}}\
\Big[\ \Psi(\varepsilon, T, \xi, \varepsilon^{\mathrm{tr}}) \;+\; \mathcal{D}\big(\xi - \xi_n,\ \varepsilon^{\mathrm{tr}} - \varepsilon^{\mathrm{tr}}_n\big)\ \Big]
```

The step is applied to its own output thousands of times in a row, so small errors add up. That one fact shapes
most of the results below.

## Steps

**Problem**

| Step | Question | Result |
|---|---|---|
| [00 · Problem definition](00-problem-definition/) | What is being solved, and what is wrong with the current solver? | The physics, the program, convexity in the right variables, the one-dimensional energy |

**Part I: a network in place of the solver** (the thesis)

| Step | Question | Result |
|---|---|---|
| [01 · Baseline network](01-baseline-network/) | What does a stable one-step network need, and where does it still fail? | Predict increments, keep the fed-back state in range by construction, plain value loss: $R^2$ 0.99995 per step, 0 of 250 runs diverge. It still drifts on steps where nothing should change (free-run error 26–77× the one-step error on cold cycles), and the solver itself accepts 99 % of the drifted states |
| [02 · Rollout fine-tuning](02-rollout-fine-tuning/) | What removes the drift? | 15 epochs of training along the network's own trajectory: free-run error 0.0061 → 0.0032 (held-out cycles 0.0074 → 0.0050), one-step accuracy unchanged, 146× faster than the solver |

**Part II: why the networks stopped improving** (after the thesis)

| Step | Question | Result |
|---|---|---|
| [03 · Architectures and active learning](03-architectures-and-active-learning/) | Does a better architecture or solver-in-the-loop data collection help? | Network shape beats size (39 % lower error at 1.4× the weights); extra rounds of data and committees add little; one seed varies ±50–60 % |

**Part III: replace the optimizer, not only imitate it**

| Step | Question | Result |
|---|---|---|
| [04 · Gated hybrid](04-gated-hybrid/) | Can a classifier decide when anything needs computing? | A small gate is right on 98 % of unseen states. Gate + solver: 0.15 MPa median over 18 paths at 50 % of the solver's work, but a few wrong "no change" calls leave 10.6 MPa on a random path |
| [05 · Proximal-gradient solver](05-proximal-gradient-solver/) | Can Nelder–Mead be replaced by a method with a convergence argument? | Yes, on the public model. The exact proximal step is a projection onto a trapezoid whose sides are the regimes, and an ISTA-type iteration with it matches a brute-force minimum on all 3,602 tested steps of the public model (median 4 iterations). [Benchmarked against Nelder–Mead](05-proximal-gradient-solver/04-benchmark-against-nelder-mead.md) on the same energy it is more accurate (0 against 30 of 2,602 steps more than 0.1 MPa off) with about a hundred times fewer energy evaluations. Unrolling it into a network is planned, not started |

Every step folder has a `README.md` (context, motivation, method, results, findings, what did not work, next
step), a `code/` folder and a `figures/` folder.

## What carried over from every step

1. **One-step accuracy does not predict accuracy in the loop.** A network with $R^2 = 0.99995$ on single steps
   drifted by 26 times its one-step error over a loading cycle. Only free-run results decide.
2. **Check the data before the model.** An early dataset did not describe the call the network receives at run
   time (total strain where the loop passes the mechanical strain). A perfect imitator of those labels was
   still 12 MPa off; with corrected labels 0.16 MPa. No architecture could have passed that.
3. **One training run proves little.** Changing only the seed moves the error of the same recipe by up to
   ±50–60 % (step 03). For the networks of steps 01–04, usually 3 to 5 trainings were run, depending on how hard
   the training was, and the best one was kept for testing. A reported number is therefore the best of a few
   runs, not a typical one, and differences smaller than the seed spread are indicative, not established. The
   proximal solver of step 05 is deterministic and has no seed to vary.
4. **Negative results are results.** Larger networks, more data in the same region, committees and several
   stabilizing loss terms did not help. They are reported with their numbers.
5. **Structure beats capacity.** An exact "no change" rule did more than any increase in network size, and the
   exact proximal step removes the need to learn it at all.
6. **Test your own premise.** The project assumed Nelder–Mead sometimes misses the minimum. On the
   one-dimensional model it is accurate in a typical step, and its misses sit in the tail: 30 of 2,602 path steps
   more than 0.1 MPa off, and 48 of 2,324 steps where it stays at the old state although the state has to move
   ([benchmark](05-proximal-gradient-solver/04-benchmark-against-nelder-mead.md)).

## Layout

```
00-problem-definition/         physics, the minimization program, the one-dimensional energy
01-baseline-network/           Part I    a stable one-step network, and where it fails
02-rollout-fine-tuning/                  what removes the drift
03-architectures-and-active-learning/    Part II   shape, size, committees, data collection
04-gated-hybrid/               Part III  a classifier decides when to compute
05-proximal-gradient-solver/             a solver with an exact proximal step, benchmarked against Nelder–Mead; unrolling is planned
common/                        shared code: data loading, the free-run loop, the benchmark, plot style
data/                          single-step training set and the 250 benchmark cycles, with their description
models/                        the two networks of Part I (ONNX and weights)
references/                    literature
```

Inside every step:

```
README.md     context, motivation, method, metric, results, findings, what did not work, next step
code/         the scripts of the step
figures/      the figures the README shows
results/      recorded numbers the figures and tables are made from
```

## What is in the repository, and what is not

- **Data and models.** `data/` and `models/` hold outputs of the reference solver and the networks trained on
  them, so the results of Part I can be re-scored. Step 04 keeps its own small model and data.
- **Two kinds of scripts.** Most scripts run from the files in this repository. Scripts that create new labels or
  score the stress on arbitrary paths need the reference implementation of the constitutive model. They read its
  location from the environment variable `SMA_SOLVER_PATH` and say so in their first lines.
- **Not included:** the reference implementation itself.

## Running the code

```
pip install -r requirements.txt
python -m common.benchmark models/v17.onnx --out v17_check     # re-score the fine-tuned network on the 250 cycles
cd 00-problem-definition/code && python plot_figures.py
```

## Credits

The constitutive model is by P. Sedlák, M. Frost and co-workers (Institute of Thermomechanics of the Czech
Academy of Sciences). References: [references/REFERENCES.md](references/REFERENCES.md).
