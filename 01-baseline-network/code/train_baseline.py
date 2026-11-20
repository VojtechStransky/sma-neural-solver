"""Train the baseline network (called v10 in the thesis).

Loss: scaled squared error of the increments, with xi counted twice, plus a
barrier that keeps xi inside [0, 1]. Nothing else.

The checkpoint is not chosen by this loss. After every epoch in the second half
of training, the averaged weights (EMA) are run freely over validation windows,
and the weights with the smallest cold-weighted free-run error are kept.

Run from the repository root (a GPU is recommended; 400 epochs):
    python 01-baseline-network/code/train_baseline.py --out runs/v10
    python 01-baseline-network/code/train_baseline.py --out runs/smoke --epochs 2 --max-batches 20
"""
import argparse
import pathlib
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.network import build_network, export_onnx
from common.training import (VALIDATION_STEPS, VALIDATION_WINDOWS, TrainingData,
                             apply_gradients, cosine, free_run_error, value_loss)

SEED, BATCH = 441, 1024
LR_HIGH, LR_LOW, WARMUP_FRACTION = 1e-3, 1e-5, 0.08
EMA_DECAY = 0.95
SELECT_FROM_FRACTION = 0.55      # checkpoints are compared only in the converged part


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--max-batches", type=int, default=None, help="for smoke tests")
    ap.add_argument("--no-export", action="store_true", help="skip the ONNX export")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    data = TrainingData()
    model = build_network(data.X)
    val = data.windows(data.val_ids, VALIDATION_STEPS, VALIDATION_WINDOWS, seed=7)
    optimizer = tf.keras.optimizers.Adam(LR_HIGH)
    print(f"{len(data.X):,} training pairs, {model.count_params():,} parameters")

    @tf.function
    def train_step(x, d_ref):
        with tf.GradientTape() as tape:
            loss = value_loss(model, x, d_ref, data)
        apply_gradients(optimizer, tape, loss, model)
        return loss

    def held_out_one_step():
        d = model(data.heldout_X, training=False).numpy()
        return float(np.mean((d[:, 0] - data.heldout_dY[:, 0]) ** 2) / data.var_d_xi
                     + np.mean((d[:, 1] - data.heldout_dY[:, 1]) ** 2) / data.var_d_etr)

    warmup = int(WARMUP_FRACTION * args.epochs)
    select_from = int(SELECT_FROM_FRACTION * args.epochs)
    ema = model.get_weights()
    best_error, best_weights, history = np.inf, None, []
    for epoch in range(args.epochs):
        optimizer.learning_rate = cosine(epoch, args.epochs, LR_HIGH, LR_LOW, warmup)
        losses = []
        for k, (x, d_ref) in enumerate(data.pairs(BATCH)):
            if args.max_batches and k >= args.max_batches:
                break
            losses.append(float(train_step(x, d_ref)))
        weights = model.get_weights()
        ema = [EMA_DECAY * a + (1 - EMA_DECAY) * w for a, w in zip(ema, weights)]

        free_run = np.nan
        if epoch >= select_from:
            model.set_weights(ema)
            free_run = free_run_error(model, val, data)
            if free_run < best_error:
                best_error, best_weights = free_run, [w.copy() for w in ema]
            model.set_weights(weights)
        history.append((epoch, np.mean(losses), held_out_one_step(), free_run))
        print("epoch %3d  train %.5f  one-step (held out) %.5f  free run (cold-weighted) %.4f" % history[-1])

    model.set_weights(best_weights if best_weights is not None else ema)
    model.save_weights(out / "model.weights.h5")
    np.savez(out / "loss_history.npz", **dict(zip(
        ("epoch", "train", "one_step", "free_run"), np.array(history).T)))
    if not args.no_export:
        export_onnx(model, out / "model.onnx")
    print("saved to", out)


if __name__ == "__main__":
    main()
