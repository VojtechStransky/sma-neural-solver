"""Loading of the datasets in `data/` (see data/README.md).

Column order everywhere:
    inputs   X = [xi_n, etr_n, M_n, eps, T]
    outputs  Y = [xi, etr]
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
MODELS = REPO / "models"


@dataclass
class Cycle:
    """One loading path computed by the reference solver."""
    id: int
    X: np.ndarray        # (steps, 5) solver inputs; columns 3, 4 are the load program
    Y: np.ndarray        # (steps, 2) solver outputs

    @property
    def strain(self):
        return self.X[:, 3]

    @property
    def temperature(self):
        return self.X[:, 4]

    @property
    def mean_temperature(self):
        return float(self.temperature.mean())

    @property
    def loading_temperature(self):
        """Mean temperature over the steps in which the strain changes.

        For most cycles this is the temperature of the mechanical test.
        Purely thermal cycles fall back to the mean over all steps.
        """
        moving = np.abs(np.diff(self.strain)) > 1e-6
        if not moving.any():
            return self.mean_temperature
        return float(self.temperature[1:][moving].mean())

    @property
    def reversals(self):
        """Number of changes of the loading direction (1 = load, then unload)."""
        d = np.diff(self.strain)
        sign = np.sign(d[np.abs(d) > 1e-5])
        return int((np.diff(sign) != 0).sum()) if len(sign) else 0


def load_cycles(path=DATA / "cycles.npz"):
    """The 250 benchmark cycles, in the order of the file."""
    d = np.load(path)
    return [Cycle(int(cid), d["X"][o:o + n], d["Y"][o:o + n])
            for cid, o, n in zip(d["cycle_ids"], d["offsets"], d["lengths"])]


def load_single_steps(path=DATA / "singlestep.npz"):
    """Single-step pairs: inputs X and target increments [d_xi, d_etr]."""
    d = np.load(path)
    return d["X"], d["Y_delta"]


def split_cycles(n_cycles, seed=123, test_fraction=0.15, val_fraction=0.15):
    """Cycle-level split used for fine-tuning: (train, validation, test) index lists.

    With 250 cycles this gives 176 / 37 / 37.
    """
    order = np.random.RandomState(seed).permutation(n_cycles)
    n_test = max(1, int(test_fraction * n_cycles))
    n_val = max(1, int(val_fraction * n_cycles))
    return (sorted(order[n_test + n_val:].tolist()),
            sorted(order[n_test:n_test + n_val].tolist()),
            sorted(order[:n_test].tolist()))
