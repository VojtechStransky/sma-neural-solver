# Plan: training through the rollout

**Status: plan only.** Nothing on this page has been implemented or run.

**Status: plan.** Nothing on this page has been run.

## Two different unrollings

The word "unrolling" covers two loops in this project. They are easy to confuse.

| | Loop | Length | What is repeated |
|---|---|---|---|
| inside one time step | the iterations of the solver ([plan](plan-learned-unrolling.md)) | $K$ layers, 3 to 8 | gradient step + prox, from the same old state |
| along the loading path | the time steps of the simulation | $N$ steps, hundreds to thousands | the whole constitutive step, each from the output of the previous one |

Page 05 trains the first loop on single rows: the old state is always the true one. In a simulation the old
state is the network's own previous output. [Step 01](../01-baseline-network/) showed what that difference costs a
black-box network: a one-step error of 0.0006 in $\xi$ became 0.0214 in free run on a cold cycle, because a
small one-sided error was added at every step of a hold and nothing pulled the state back.
[Step 02](../02-rollout-fine-tuning/) fixed most of it by training along the network's own trajectory.

## Why the unrolled network starts from a better place

- **Holds are exact.** When the trial increment lies inside the elastic domain, the layer returns exactly the
  old state. The drift of step 01 came from outputs that should have been zero and were not. Here a zero is
  produced by the activation, not learned.
- **The memory variable stays outside.** $M$ is updated by its closed formula after the step, as in every
  earlier model.

What can still go wrong in the blind version (L3): the learned driving force can be slightly wrong near a
threshold. Then a step that should hold moves, or the reverse. That error is no longer a small drift at every
step; it is a wrong decision at single steps, and the state after it is a different valid state. As step 01
showed, the solver itself would not correct such a state. This is the failure mode the rollout training has to
address.

## Plan

**Stage 1: single steps.** Train on rows as in the [plan for learned unrolling](plan-learned-unrolling.md), with deep supervision over the $K$ layers and a
smoothed activation. Choose the model by free-run error.

**Stage 2: through time.** Fine-tune on windows of $N$ consecutive steps of recorded loading paths. The network
runs freely inside the window: its output, with the memory update, is the next old state. The loss is the
single-step loss plus the error of the window, in the form that worked in step 02:

```math
\mathcal{L} = \mathcal{L}_{\text{step}} + \frac{1}{N}\sum_{j=1}^{N}\left[\ 2\,\frac{\big(\xi_j - \xi_j^{\text{ref}}\big)^2}{s_\xi^2}
 + \frac{\big(\varepsilon^{\mathrm{tr}}_j - \varepsilon^{\mathrm{tr,ref}}_j\big)^2}{s_\varepsilon^2}\ \right].
```

$\xi_j$, $\varepsilon^{\mathrm{tr}}_j$ are the network's outputs at step $j$ of its own run;
the reference values come from the recorded path, so the solver is not called during training. Gradients flow
through $N \times K$ applications of the layer.

Settings carried over from step 02, where they were found to matter:

- keep the single-step term at full weight, so the one-step accuracy is not traded away;
- scheduled sampling: feed the reference state with a probability that falls from 0.8 to 0.4, not to zero;
- a short window first ($N = 16$ was the memory limit there; the network here is 100 times smaller, so longer
  windows are possible);
- select the checkpoint by free-run error on held-out paths, not by the last epoch.

New for this network:

- a **stress term** in the loss, since $\sigma = E(\xi)(\varepsilon - q)$ is differentiable and stress is what is
  scored;
- **dead gradients** multiply along the window: a hold passes no gradient to the layers before it, in either
  loop. The smoothed activation is used during training and the exact one at deployment;
- training paths should include the situations where the earlier networks failed: temperature sweeps under
  load, constrained recovery, minor loops.

**Stage 3: a check at run time.** After the last layer, evaluate the residual of
[page 01](01-from-nelder-mead-to-proximal-gradient.md) (the distance of the driving forces from the
subdifferential of $g$). For the levels that use the energy this is cheap and gives every step a certificate.
For the blind network it needs one evaluation of the energy gradient per step; whether that is acceptable is an
open design question.

## How it will be judged

Same rules as the rest of the repository: free-run stress error on the 40
paths, median of at least three seeds, number of diverged paths and worst path reported, criteria written
before the fit.

| Bar | Value | From |
|---|---|---|
| best trained network of the project | 5.2 MPa | small regime-specific networks; not part of this repository |
| the converged solver of this step on the same paths | to be measured | [README](README.md), next step 1 |

The blind network has to beat 5.2 MPa to be worth having. The levels that use the energy have to match the
converged solver with a small fixed number of layers.

## References

- S. Bengio, O. Vinyals, N. Jaitly, N. Shazeer (2015). Scheduled sampling for sequence prediction with recurrent
  neural networks. *Advances in Neural Information Processing Systems* 28.
- S. Ross, G. J. Gordon, J. A. Bagnell (2011). A reduction of imitation learning and structured prediction to
  no-regret online learning. *AISTATS*, 627–635.
