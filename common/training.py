"""Pieces shared by the baseline training (step 01) and the rollout fine-tuning (step 02)."""
import numpy as np
import tensorflow as tf

from common.data import load_cycles, load_single_steps, split_cycles
from common.rollout import SQRT15, XI_FLOOR

XI_WEIGHT = 2.0          # xi is the main state variable; its error counts twice
BARRIER_WEIGHT = 5.0     # penalty on xi outside [0, 1] before clipping
COLD_BELOW = -5.0        # cycles with a mean temperature below this are weighted ...
COLD_WEIGHT = 3.0        # ... three times when the checkpoint is selected
VALIDATION_WINDOWS, VALIDATION_STEPS = 256, 80
GRADIENT_CLIP = 5.0


class TrainingData:
    """Training pairs, held-out pairs and free-run validation windows.

    Value training uses the single-step pairs plus the consecutive-step pairs of
    the training cycles, shuffled as individual pairs. The validation and test
    cycles are never used for training.
    """

    def __init__(self):
        self.cycles = load_cycles()
        self.train_ids, self.val_ids, self.test_ids = split_cycles(len(self.cycles))
        X, dY = load_single_steps()
        fine_X = np.concatenate([self.cycles[i].X for i in self.train_ids])
        fine_dY = np.concatenate([self.cycles[i].Y - self.cycles[i].X[:, :2] for i in self.train_ids])
        self.X = np.concatenate([X, fine_X]).astype(np.float32)
        self.dY = np.concatenate([dY, fine_dY]).astype(np.float32)
        self.heldout_X = np.concatenate([self.cycles[i].X for i in self.val_ids]).astype(np.float32)
        self.heldout_dY = np.concatenate(
            [self.cycles[i].Y - self.cycles[i].X[:, :2] for i in self.val_ids]).astype(np.float32)

        # Scales that make the two outputs comparable.
        self.var_d_xi, self.var_d_etr = (float(v) for v in self.dY.var(axis=0))
        Y_all = np.concatenate([c.Y for c in self.cycles])
        self.std_xi, self.std_etr = (float(v) for v in Y_all.std(axis=0))

    def pairs(self, batch):
        return (tf.data.Dataset.from_tensor_slices((self.X, self.dY))
                .shuffle(200_000).batch(batch, drop_remainder=True).prefetch(2))

    def windows(self, cycle_ids, steps, max_windows, seed):
        """Windows of consecutive steps cut from the given cycles.

        Returns a dict of float32 arrays (windows, steps[, ...]):
        start state, load program, reference outputs, reference M at each input,
        and the cold weight of the window's cycle.
        """
        owner, start = [], []
        for i in cycle_ids:
            n = len(self.cycles[i].X) - steps
            owner += [i] * max(n, 0)
            start += list(range(max(n, 0)))
        pick = np.random.RandomState(seed).choice(len(owner), min(max_windows, len(owner)), replace=False)
        X = np.stack([self.cycles[owner[k]].X[start[k]:start[k] + steps] for k in pick])
        Y = np.stack([self.cycles[owner[k]].Y[start[k]:start[k] + steps] for k in pick])
        cold = np.array([self.cycles[owner[k]].mean_temperature < COLD_BELOW for k in pick])
        f32 = lambda a: a.astype(np.float32)
        return {"state": f32(X[:, 0, :3]), "eps": f32(X[:, :, 3]), "T": f32(X[:, :, 4]),
                "xi": f32(Y[:, :, 0]), "etr": f32(Y[:, :, 1]), "M": f32(X[:, :, 2]),
                "weight": f32(np.where(cold, COLD_WEIGHT, 1.0))}


def value_loss(model, x, d_ref, data):
    """One-step loss: scaled squared errors of the increments + barrier on xi."""
    d = model(x, training=True)
    error = (XI_WEIGHT * tf.reduce_mean((d[:, 0] - d_ref[:, 0]) ** 2) / data.var_d_xi
             + tf.reduce_mean((d[:, 1] - d_ref[:, 1]) ** 2) / data.var_d_etr)
    xi = x[:, 0] + d[:, 0]
    barrier = tf.reduce_mean(tf.nn.relu(xi - 1.0) ** 2 + tf.nn.relu(-xi) ** 2)
    return error + BARRIER_WEIGHT * barrier


def advance(model, state, eps, T, training=False):
    """One step of the simulation loop in TensorFlow. state = [xi, etr, M] per row.

    The same rule as common.rollout.update_memory, written for batches.
    """
    d = model(tf.concat([state, eps[:, None], T[:, None]], axis=1), training=training)
    xi = tf.clip_by_value(state[:, 0] + d[:, 0], 0.0, 1.0)
    etr = state[:, 1] + d[:, 1]
    ratio = tf.minimum(state[:, 0] / tf.maximum(xi, XI_FLOOR), 1.0)
    M = tf.maximum(state[:, 2] * ratio, tf.abs(etr) * SQRT15)
    return tf.stack([xi, etr, M], axis=1)


def free_run_error(model, win, data):
    """Cold-weighted mean error of free runs over the validation windows."""
    state = tf.constant(win["state"])
    error = tf.zeros(len(win["state"]))
    steps = win["eps"].shape[1]
    for t in range(steps):
        state = advance(model, state, win["eps"][:, t], win["T"][:, t])
        error += (XI_WEIGHT * tf.abs(state[:, 0] - win["xi"][:, t]) / data.std_xi
                  + tf.abs(state[:, 1] - win["etr"][:, t]) / data.std_etr)
    w = win["weight"]
    return float(tf.reduce_sum(w * error / steps) / tf.reduce_sum(w))


def apply_gradients(optimizer, tape, loss, model):
    grads = [tf.clip_by_norm(g, GRADIENT_CLIP) for g in tape.gradient(loss, model.trainable_variables)]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def cosine(epoch, epochs, high, low, warmup=0):
    if epoch < warmup:
        return high * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(epochs - warmup, 1)
    return low + 0.5 * (high - low) * (1 + np.cos(np.pi * progress))
