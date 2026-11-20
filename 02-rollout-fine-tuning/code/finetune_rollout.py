"""Fine-tune the baseline along its own trajectory (gives v17 in the thesis).

For every window of K consecutive steps of a training cycle, the network is
unrolled K times under the simulation loop. The loss is the scaled squared
distance of that trajectory from the reference, added to the unchanged
one-step loss.

Scheduled sampling: at each step of the window the input state is the
reference with probability p, otherwise the network's own previous output.
p falls from 0.8 to 0.4 during training.

The targets come from the stored reference cycles. The solver is not called.

Run from the repository root (CPU is enough; about 2 GB of memory at K = 16):
    python 02-rollout-fine-tuning/code/finetune_rollout.py --out runs/v17
    python 02-rollout-fine-tuning/code/finetune_rollout.py --out runs/smoke --epochs 1 --max-batches 5
"""
import argparse
import pathlib
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.data import MODELS
from common.network import build_network, export_onnx
from common.training import (VALIDATION_STEPS, VALIDATION_WINDOWS, XI_WEIGHT, TrainingData,
                             advance, apply_gradients, cosine, free_run_error, value_loss)

SEED = 441
PAIR_BATCH = 1024
LR_HIGH, LR_LOW = 2e-4, 1e-5
P_REFERENCE_START, P_REFERENCE_END = 0.8, 0.4


def rollout_loss(model, win, p_reference, data):
    """Mean scaled squared error of the unrolled trajectory, cold windows weighted."""
    state = win["state"]
    steps = win["eps"].shape[1]
    total = 0.0
    for t in range(steps):
        if t > 0:   # scheduled sampling: sometimes put the reference state back
            reference = tf.stack([win["xi"][:, t - 1], win["etr"][:, t - 1], win["M"][:, t]], axis=1)
            use_reference = tf.cast(tf.random.uniform((tf.shape(state)[0], 1)) < p_reference, tf.float32)
            state = use_reference * reference + (1.0 - use_reference) * state
        state = advance(model, state, win["eps"][:, t], win["T"][:, t], training=True)
        error = (XI_WEIGHT * (state[:, 0] - win["xi"][:, t]) ** 2 / data.var_d_xi
                 + (state[:, 1] - win["etr"][:, t]) ** 2 / data.var_d_etr)
        total += tf.reduce_sum(win["weight"] * error) / tf.reduce_sum(win["weight"])
    return total / steps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default=str(MODELS / "v10.weights.h5"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--K", type=int, default=16, help="unrolled steps per window")
    ap.add_argument("--window-batch", type=int, default=128)
    ap.add_argument("--max-windows", type=int, default=40_000)
    ap.add_argument("--rollout-weight", type=float, default=1.0)
    ap.add_argument("--max-batches", type=int, default=None, help="for smoke tests")
    ap.add_argument("--no-export", action="store_true", help="skip the ONNX export")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    data = TrainingData()
    model = build_network(data.X)
    model.load_weights(args.base)
    val = data.windows(data.val_ids, VALIDATION_STEPS, VALIDATION_WINDOWS, seed=7)
    train = data.windows(data.train_ids, args.K, args.max_windows, seed=0)
    windows = (tf.data.Dataset.from_tensor_slices(train).shuffle(len(train["state"]))
               .batch(args.window_batch, drop_remainder=True).prefetch(2))
    pairs = iter(data.pairs(PAIR_BATCH).repeat())
    optimizer = tf.keras.optimizers.Adam(LR_HIGH)
    print(f"{len(train['state']):,} windows of {args.K} steps; "
          f"free-run error of the base model {free_run_error(model, val, data):.4f}")

    @tf.function
    def train_step(x, d_ref, win, p_reference):
        with tf.GradientTape() as tape:
            one_step = value_loss(model, x, d_ref, data)
            unrolled = rollout_loss(model, win, p_reference, data)
            loss = one_step + args.rollout_weight * unrolled
        apply_gradients(optimizer, tape, loss, model)
        return one_step, unrolled

    best_error, best_weights, history = np.inf, None, []
    for epoch in range(args.epochs):
        optimizer.learning_rate = cosine(epoch, args.epochs, LR_HIGH, LR_LOW)
        p_reference = P_REFERENCE_START + (P_REFERENCE_END - P_REFERENCE_START) * epoch / max(args.epochs - 1, 1)
        log = []
        for k, win in enumerate(windows):
            if args.max_batches and k >= args.max_batches:
                break
            x, d_ref = next(pairs)
            log.append([float(v) for v in train_step(x, d_ref, win, tf.constant(p_reference, tf.float32))])
        # Late epochs run mostly on the network's own output and can get worse:
        # keep the weights with the best held-out free run, not the last ones.
        free_run = free_run_error(model, val, data)
        if free_run < best_error:
            best_error, best_weights = free_run, model.get_weights()
        history.append((epoch, *np.mean(log, axis=0), free_run))
        print("epoch %2d  one-step %.5f  unrolled %.5f  free run (held out, cold-weighted) %.4f" % history[-1],
              f" p_reference {p_reference:.2f}")

    model.set_weights(best_weights)
    model.save_weights(out / "model.weights.h5")
    np.savez(out / "loss_history.npz", **dict(zip(
        ("epoch", "one_step", "unrolled", "free_run"), np.array(history).T)))
    if not args.no_export:
        export_onnx(model, out / "model.onnx")
    print(f"saved to {out}; best free-run error {best_error:.4f}")


if __name__ == "__main__":
    main()
