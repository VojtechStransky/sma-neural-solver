"""Active learning along a loading path: where the network and the solver disagree, keep the solver's answer.

At every load step the network advances its own state. The solver is called
from the SAME state. If the two results differ by more than a threshold, the
solver's result is saved as a new training row, and the solver's trajectory is
followed for a few more steps (those rows are saved as well).

The network keeps running on its own state, so its errors are allowed to
compound: the new rows come from the states the network actually visits.

The two steppers are passed in as functions, so this file needs no solver:

    step(state, eps, T) -> new state            state = (xi, etr, M)

In the study `solver_step` was the reference solver (not part of this
repository) and `network_step` the network inside the simulation loop.
"""
import numpy as np

ETR_MAX = 0.07


def loading_path(T_target, eps_max, eps_step=0.005, cycles=1, T_start=-20.0):
    """Temperature ramp at zero strain in 1 K steps, then `cycles` of loading to eps_max and back."""
    T, eps = [], []
    direction = 1.0 if T_target >= T_start else -1.0
    for i in range(1, int(round(abs(T_target - T_start))) + 1):
        T.append(T_start + direction * i)
        eps.append(0.0)
    n_half = int(round(abs(eps_max) / eps_step))
    ramp = np.sign(eps_max) * eps_step * np.arange(1, n_half + 1)
    for _ in range(cycles):
        for e in list(ramp) + list(ramp[::-1] - np.sign(eps_max) * eps_step):
            T.append(T_target)
            eps.append(float(e))
    return np.array(eps), np.array(T)


def disagreement_percent(network, solver):
    """Relative difference of the two results, in percent, per variable.

    The floors in the denominators are needed because both variables pass
    through zero. They also make the rule much stricter where the state is
    small: with xi = 0.02 it fires at |d xi| > 0.002, during loading only at
    about 0.027. In the study 62 % of the harvested rows sat at zero strain
    for this reason.
    """
    d_xi = 100.0 * abs(network[0] - solver[0]) / max(abs(solver[0]), 0.01)
    d_etr = 100.0 * abs(network[1] - solver[1]) / max(abs(solver[1]), 0.01 * ETR_MAX)
    return d_xi, d_etr


def harvest_path(network_step, solver_step, state, eps_path, T_path, threshold=10.0, follow_steps=10):
    """Run one loading path. Returns the new rows and a small log.

    A row is [xi_n, etr_n, M_n, eps, T, xi, etr]: a state, a load and the solver's answer.
    """
    rows, log = [], {"steps": 0, "triggers": 0, "solver_calls": 0, "abs_error_xi": []}
    follow_state, remaining = None, 0
    for eps, T in zip(eps_path, T_path):
        network_new = network_step(state, eps, T)
        solver_new = solver_step(state, eps, T)                     # reference from the network's own state
        log["solver_calls"] += 1
        log["abs_error_xi"].append(abs(network_new[0] - solver_new[0]))
        disagree = max(disagreement_percent(network_new, solver_new)) > threshold
        log["triggers"] += disagree

        if remaining > 0:                                           # following the solver after a trigger
            followed = solver_step(follow_state, eps, T)
            log["solver_calls"] += 1
            rows.append([*follow_state, eps, T, *followed[:2]])
            follow_state, remaining = followed, remaining - 1
        elif disagree:                                              # new trigger: save it, then follow
            rows.append([*state, eps, T, *solver_new[:2]])
            follow_state, remaining = solver_new, follow_steps

        state = network_new                                         # the network keeps its own history
        log["steps"] += 1
    log["mean_abs_error_xi"] = float(np.mean(log.pop("abs_error_xi")))
    return np.array(rows).reshape(-1, 7), log


if __name__ == "__main__":
    # Self-test with toy steppers: the "solver" relaxes xi towards a target, the "network" is slightly off.
    def solver(state, eps, T):
        xi = state[0] + 0.5 * (min(max(eps * 12.0, 0.02), 0.98) - state[0])
        return (xi, 0.05 * xi, max(state[2], 0.06 * xi))

    def network(state, eps, T):
        xi, etr, M = solver(state, eps, T)
        return (xi * (1.15 if eps > 0.05 else 1.02), etr, M)     # wrong only at large strain

    eps, T = loading_path(T_target=20.0, eps_max=0.08)
    rows, log = harvest_path(network, solver, (0.02, 0.0, 0.0), eps, T)
    print(f"{log['steps']} steps, {log['triggers']} triggers, {len(rows)} rows harvested, "
          f"{log['solver_calls']} solver calls, mean |error xi| {log['mean_abs_error_xi']:.4f}")
