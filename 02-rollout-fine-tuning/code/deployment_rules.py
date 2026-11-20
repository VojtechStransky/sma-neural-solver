"""Rules applied in the simulation loop, without retraining the network.

dead band   On a confirmed flat section, hold xi and etr instead of adding the
            network's tiny increments. A section is confirmed after `confirm`
            consecutive predicted changes of xi below `enter`. It ends when a
            predicted change reaches `leave` (a switch with hysteresis).

re-anchor   Every `every` steps, replace the state by the reference state.
            This is an ORACLE: it shows what an occasional exact correction
            could give. A real solver call from a drifted state would not
            return the reference state on a flat section (see step 01).

Scores both models with each rule on the 250 cycles.

Run from the repository root:
    python 02-rollout-fine-tuning/code/deployment_rules.py
"""
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.data import MODELS, load_cycles
from common.rollout import OnnxStep, free_run

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
COLD_LOADING_T = -28.0


class DeadBand:
    def __init__(self, enter=0.0015, leave=0.004, confirm=3, etr_band=0.001):
        self.enter, self.leave, self.confirm, self.etr_band = enter, leave, confirm, etr_band
        self.holding, self.count = False, 0

    def __call__(self, state, prediction, eps, T):
        xi, etr, _ = state
        xi_new, etr_new = prediction
        change = abs(float(np.clip(xi_new, 0.0, 1.0)) - xi)
        if self.holding:
            if change >= self.leave:            # a real transformation: release
                self.holding, self.count = False, 0
        else:
            self.count = self.count + 1 if change < self.enter else 0
            self.holding = self.count >= self.confirm
        if self.holding:
            xi_new = xi
            if abs(etr_new - etr) < self.etr_band:
                etr_new = etr
        return xi_new, etr_new


class ReanchorToReference:
    def __init__(self, cycle, every=20):
        self.reference, self.every, self.t = cycle.Y, every, 0

    def __call__(self, state, prediction, eps, T):
        self.t += 1
        return tuple(self.reference[self.t - 1]) if self.t % self.every == 0 else prediction


RULES = {
    "network only": lambda cycle: None,
    "dead band": lambda cycle: DeadBand(),
    "re-anchor every 20 steps (oracle)": lambda cycle: ReanchorToReference(cycle),
}


def main():
    cycles = load_cycles()
    cold = np.array([c.loading_temperature < COLD_LOADING_T for c in cycles])
    table = {}
    for name in ("v10", "v17"):
        model = OnnxStep(MODELS / f"{name}.onnx")
        for rule, make in RULES.items():
            start = time.perf_counter()
            mae = np.array([np.abs(free_run(model, c, make(c))[:, 0] - c.Y[:, 0]).mean() for c in cycles])
            seconds = time.perf_counter() - start
            table[f"{name}, {rule}"] = {
                "free_run_mae_xi": float(mae.mean()), "median": float(np.median(mae)),
                "cold": float(mae[cold].mean()), "worst_cycle": float(mae.max()),
                "ms_per_step": 1000 * seconds / sum(len(c.X) for c in cycles)}
            print(f"{name:4s} {rule:36s} mean {mae.mean():.4f}  median {np.median(mae):.4f}  "
                  f"cold {mae[cold].mean():.4f}  worst {mae.max():.4f}")
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "deployment_rules.json", "w") as fh:
        json.dump(table, fh, indent=1)


if __name__ == "__main__":
    main()
