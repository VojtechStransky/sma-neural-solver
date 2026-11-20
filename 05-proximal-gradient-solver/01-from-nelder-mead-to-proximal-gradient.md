# From Nelder–Mead to a proximal-gradient step

The reference solver treats the energy as a black box and searches it with the Nelder–Mead (downhill simplex)
method. This page takes the energy apart instead: a smooth part, which gets a gradient step, and a kinked part,
which gets an exact "hold or move" step. That is the proximal-gradient method, known in signal processing as ISTA.

Code for everything on this page: the first part of
[code/prox_gradient_solver.py](code/prox_gradient_solver.py).

## Known and unknown quantities

| Role | Symbol | Meaning |
|---|---|---|
| **unknowns** | $\xi,\ \varepsilon^{\mathrm{tr}}$ | martensite fraction and transformation strain at the end of the step |
| unknown used by the method | $q = \xi\,\varepsilon^{\mathrm{tr}}$ | macroscopic transformation strain; the method works in $(\xi, q)$ and returns $\varepsilon^{\mathrm{tr}} = q/\xi$ |
| known: load | $\varepsilon,\ T$ | strain and temperature at the end of the step |
| known: old state | $\xi_n,\ \varepsilon^{\mathrm{tr}}_n,\ M_n$ | converged values of the previous step; $q_n = \xi_n\varepsilon^{\mathrm{tr}}_n$ |
| increments | $\Delta\xi = \xi - \xi_n$, $\ \Delta q = q - q_n$ | always new minus old |
| constants | $\Delta S,\ T_0,\ M_s,\ M_f,\ A_s,\ A_f,\ \sigma_{\mathrm{reo}},\ E_A,\ E_M$ | calibration, see [the program](../00-problem-definition/minimization-program.md) |
| constant | $\tau_0 = \sigma_{\mathrm{reo}}\sqrt{3/2}$ | reorientation threshold, 122.5 MPa |
| iteration | $k$, $t$ | iteration counter and step size inside one time step |

$q$ is not a new modelling quantity. It is the strain that is subtracted in the elastic term
$\tfrac12 E(\xi)(\varepsilon - \xi\varepsilon^{\mathrm{tr}})^2$ and the quantity whose change the forward
dissipation measures. Only the name is new.

## The problem

```math
\big(\xi,\ \varepsilon^{\mathrm{tr}}\big) = \arg\min_{\xi,\ \varepsilon^{\mathrm{tr}}}\ \Pi,
\qquad \Pi = \Psi(\varepsilon, T, \xi, \varepsilon^{\mathrm{tr}}) + \mathcal{D}(\xi, \varepsilon^{\mathrm{tr}};\ \xi_n, \varepsilon^{\mathrm{tr}}_n, M_n)
```

```math
\mathcal{D} =
\begin{cases}
\tau_0\,\big|\xi\varepsilon^{\mathrm{tr}} - \xi_n\varepsilon^{\mathrm{tr}}_n\big|
 + \Delta S\big[(T_0 - M_s)(\xi - \xi_n) + \tfrac12 (M_s - M_f)(\xi^2 - \xi_n^2)\big], & \xi \ge \xi_n \\[6pt]
\tau_0\,\xi\,\big|\varepsilon^{\mathrm{tr}} - \varepsilon^{\mathrm{tr}}_n\big| + \sigma_{\mathrm{reo}} M_n\,(\xi_n - \xi)
 + \Delta S\big[(A_f - T_0)(\xi_n - \xi) + \tfrac12 (A_s - A_f)(\xi_n^2 - \xi^2)\big], & \xi < \xi_n
\end{cases}
```

$\Psi$ is smooth. $\mathcal{D}$ has kinks at the old state. In $(\xi, q)$ the whole problem is strictly convex
and has one minimum. The reference code
linearizes the stiffness within a step, which is outside that statement, so the solver built here uses the exact
mixture rule $1/E(\xi) = (1-\xi)/E_A + \xi/E_M$.

## Step 1: separate the kinks from the smooth rest

Every term of $\mathcal{D}$ grows linearly with the distance from the old state, except the two terms with
$\xi^2 - \xi_n^2$. Each of those is a linear part plus a parabola around $\xi_n$:

```math
\tfrac12 (M_s - M_f)(\xi^2 - \xi_n^2) = (M_s - M_f)\,\xi_n\,(\xi - \xi_n) + \tfrac12 (M_s - M_f)(\xi - \xi_n)^2 ,
```

```math
\tfrac12 (A_s - A_f)(\xi_n^2 - \xi^2) = (A_s - A_f)\,\xi_n\,(\xi_n - \xi) + \tfrac12 (A_f - A_s)(\xi - \xi_n)^2 .
```

The linear parts stay with the kinks. The parabolas move to the smooth side. The result is the same function,
regrouped:

```math
\Pi \;=\; \underbrace{\Psi + \tfrac12\,h_\pm\,(\xi - \xi_n)^2}_{f\ \text{(smooth)}} \;+\; \underbrace{g}_{\text{kinked}},
\qquad h_+ = \Delta S\,(M_s - M_f),\quad h_- = \Delta S\,(A_f - A_s)
```

($h_+$ applies for $\xi \ge \xi_n$, $h_-$ for $\xi < \xi_n$; both are 12 MPa in this calibration.) Written in
$(\xi, q)$:

```math
f(\xi, q) = \frac{(\varepsilon - q)^2}{2\,s(\xi)} + (T - T_0)\,\Delta S\,\xi + \xi\,\varphi\!\Big(\frac{q}{\xi},\ \varepsilon\Big) + \tfrac12\,h_\pm\,(\xi - \xi_n)^2,
\qquad s(\xi) = \frac{1-\xi}{E_A} + \frac{\xi}{E_M}
```

```math
g(\Delta\xi, \Delta q) =
\begin{cases}
Y_F^0\,\Delta\xi + \tau_0\,|\Delta q|, & \Delta\xi \ge 0 \\[4pt]
Y_R\,|\Delta\xi| + \tau_0\,\big|\Delta q - \varepsilon^{\mathrm{tr}}_n\,\Delta\xi\big|, & \Delta\xi < 0
\end{cases}
```

with the two thresholds

```math
Y_F^0 = \Delta S\big[(T_0 - M_s) + (M_s - M_f)\,\xi_n\big],
\qquad
Y_R = \Delta S\big[(A_f - T_0) + (A_s - A_f)\,\xi_n\big] + \sigma_{\mathrm{reo}}\,M_n .
```

$Y_F^0$, $Y_R$ and $\tau_0$ contain only values of the old state. They are fixed numbers during the step.
$g$ is zero at the old state and grows linearly away from it, so it holds all the kinks. $\varphi$ is the
interaction energy of [the program](../00-problem-definition/minimization-program.md).

The model also has steep barriers that keep $\xi$ and $\varepsilon^{\mathrm{tr}}$ in range. They are smooth and
go into $f$; the slope of the $\xi$-barrier at $\xi_n$ is added to $Y_F^0$ and $Y_R$, so that the remainder has
no kink.

**Checked** (2,000 random points): $f + g$ equals $\Pi$ to $2\cdot10^{-15}$ relative; the thresholds equal the
one-sided slopes of $\Pi - f$ at the old state to $10^{-7}$.

## Step 2: the gradient of the smooth part

With the stress $\sigma = (\varepsilon - q)/s(\xi)$ and $\varepsilon^{\mathrm{tr}} = q/\xi$:

```math
\frac{\partial f}{\partial \xi} = -\tfrac12\,\sigma^2\Big(\frac{1}{E_M} - \frac{1}{E_A}\Big) + \Delta S\,(T - T_0)
 + \varphi - \varepsilon^{\mathrm{tr}}\,\varphi' + h_\pm\,(\xi - \xi_n),
\qquad
\frac{\partial f}{\partial q} = -\sigma + \varphi' .
```

($\varphi'$ is the derivative of $\varphi$ with respect to its first argument.) The negative gradient is the pair
of thermodynamic driving forces of the model, converted to the variables $(\xi, q)$ and corrected by the
hardening parabola. Everything on the right-hand side is a current iterate or a known number.

## Step 3: the iteration

```math
u^{k+1} = \operatorname{prox}_{t g}\!\big(u^{k} - t\,\nabla f(u^{k})\big),
\qquad
\operatorname{prox}_{t g}(v) = \arg\min_{u}\Big\{\, g(u - u_n) + \frac{1}{2t}\,\lVert u - v\rVert^2 \Big\},
\qquad u = (\xi, q).
```

In words: take a gradient step on the smooth part, which gives a trial state $v$. Then find the state that stays
as close as possible to $v$ while paying the dissipation $g$ for moving away from the old state. $g$ pulls the
result back to the old state (hold); the quadratic term pulls it to the trial state (move).

Start from $u^0 = u_n$. In the first iteration the trial increment is $t$ times the driving force at the old
state, like the elastic predictor of a return-mapping algorithm.

For $g(x) = \lambda|x|$ in one variable the prox is the soft threshold: zero inside $[-t\lambda, t\lambda]$,
shifted by $t\lambda$ outside. The next page derives the same thing for the two-variable $g$ above.

## Why one soft threshold per variable is not enough

It is tempting to treat $\xi$ and $\varepsilon^{\mathrm{tr}}$ separately: a dead zone for $\xi$ and a soft
threshold for $\varepsilon^{\mathrm{tr}}$. The kinks do not allow it.

- **The forward reorientation kink is not aligned with either variable.** It lies where
  $\xi\,\varepsilon^{\mathrm{tr}} = \xi_n\varepsilon^{\mathrm{tr}}_n$, a curve in the
  $(\xi, \varepsilon^{\mathrm{tr}})$ plane. A method that changes one variable at a time meets this kink at an
  angle and can zig-zag along it or stop on it.
- **The reverse threshold depends on what the other variable does.** For the example state
  $\xi_n = 0.3$, $\varepsilon^{\mathrm{tr}}_n = 0.05$, $M_n = 0.075$, reverse transformation starts at a driving
  force of $-16.0$ MPa if the transformation strain is being pushed up, and at $-3.8$ MPa if it is being pushed
  down.
- **The first experiment showed it in practice.** An earlier version of the iteration updated one unknown at a
  time. It needed an extra search along the line $\xi\varepsilon^{\mathrm{tr}} = \text{const}$ to work.

In $(\xi, q)$ the forward kink becomes the straight line $q = q_n$, and the prox of $g$ has an exact closed form
for both unknowns at once ([page 02](02-exact-joint-prox.md)).

## The same structure as ISTA for sparse coding

| Part of the step | ISTA / LISTA (sparse coding) | This problem |
|---|---|---|
| unknowns | code $x$ | $(\xi,\ q)$ |
| smooth part $f$ | $\tfrac12\lVert Ax - y\rVert^2$ | $\Psi + \tfrac12 h_\pm(\xi - \xi_n)^2$ |
| kinked part $g$ | $\lambda\lVert x\rVert_1$ | $\mathcal{D}$ without the parabolas |
| gradient | $A^{\mathsf T}(Ax - y)$, linear in $x$ | step 2 above, not linear |
| step size | $1/L$ with $L$ the largest eigenvalue of $A^{\mathsf T}A$ | must be scaled per direction ([page 03](03-guarantees-and-open-gaps.md)) |
| prox (the activation) | soft threshold: interval $[-\lambda, \lambda]$ | projection onto a trapezoid ([page 02](02-exact-joint-prox.md)) |

## Where the iteration stops

A point $u^\ast$ is a fixed point of the iteration, for any step size $t > 0$, exactly when

```math
0 \in \nabla f(u^\ast) + \partial g(u^\ast - u_n) .
```

This is the optimality condition of the minimization, and it is the incremental form of the normality rule of
the material model: the driving forces lie in the subdifferential of the dissipation. So the iteration cannot
stop anywhere except at the solution, and the distance of $-\nabla f$ from $\partial g$ is a residual that can
be checked after any number of iterations. Nelder–Mead offers no such test.
