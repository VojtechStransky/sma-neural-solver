"""Train the change gate (two outputs: xi changes, eps_tr changes).

Data: solver-labelled single steps in the run-time convention, as a .mat file
with one array of rows

    [xi_n, eps_tr_n, M_n, eps_mech, T, xi, eps_tr, ...]

(solver-labelled single steps in the run-time convention; not stored in this repository). The labels are
computed from each row: a variable "changes" if it moves by more than the
solver resolves (see gate.py).

Rows that share a previous state (xi_n, eps_tr_n, M_n) always stay on the same
side of a split. The data hold many rows per previous state, so a random row
split would put near-duplicates on both sides.

Usage:
    python train_gate.py --data all_pathconv.mat
        80 % of the states train, 10 % stop the training, 10 % test.
        Writes ../models/gate.npz and ../results/gate_report.json.

    python train_gate.py --data all_pathconv.mat --out-of-fold
        Five gates, each trained without one fifth of the states, so every row
        gets a prediction from a gate that never saw its state.
        Writes ../results/gate_out_of_fold.json and gate_errors_by_step_size.csv.
"""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

from gate import THRESHOLD_EPS_TR, THRESHOLD_XI, change_labels

ROOT = Path(__file__).resolve().parent.parent
MAX_TRANSFORMATION_STRAIN = 0.07


def load_steps(mat_file):
    """Inputs X (N, 5), labels Y (N, 2), state id per row, and the step sizes."""
    arrays = [v for k, v in loadmat(mat_file).items() if not k.startswith("__")]
    rows = arrays[0].astype(float)
    valid = (np.isfinite(rows).all(axis=1) & (rows[:, 5] >= 0) & (rows[:, 5] <= 1)
             & (np.abs(rows[:, 6]) <= MAX_TRANSFORMATION_STRAIN))
    rows = rows[valid]
    X = rows[:, :5]
    Y = change_labels(rows[:, 0], rows[:, 1], rows[:, 5], rows[:, 6]).astype("float32")
    _, state = np.unique(np.round(rows[:, :3], 12), axis=0, return_inverse=True)
    steps = np.abs(rows[:, 5] - rows[:, 0]), np.abs(rows[:, 6] - rows[:, 1])
    return X, Y, state.ravel(), steps


def fit(X, Y, train, validation, seed):
    """Fit one gate; returns (keras model, input mean, input std)."""
    mean, std = X[train].mean(axis=0), X[train].std(axis=0) + 1e-12
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [tf.keras.Input((5,))]
        + [tf.keras.layers.Dense(128, activation="swish") for _ in range(3)]
        + [tf.keras.layers.Dense(2, activation="sigmoid")])
    model.compile(tf.keras.optimizers.Adam(2e-3), "binary_crossentropy")
    model.fit((X[train] - mean) / std, Y[train],
              validation_data=((X[validation] - mean) / std, Y[validation]),
              epochs=300, batch_size=256, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-5)])
    return model, mean, std


def predict(model, mean, std, X):
    return model.predict((X - mean) / std, verbose=0, batch_size=4096)


def shuffled_states(state, seed):
    ids = np.unique(state)
    np.random.default_rng(seed).shuffle(ids)
    return ids


def train_single(X, Y, state):
    ids = shuffled_states(state, seed=0)
    test = np.isin(state, ids[:len(ids) // 10])
    validation = np.isin(state, ids[len(ids) // 10:len(ids) // 5])
    train = ~(test | validation)

    model, mean, std = fit(X, Y, train, validation, seed=0)
    P = predict(model, mean, std, X[test])
    truth = Y[test].astype(bool)
    report = {"rows": int(len(X)), "test_rows": int(test.sum()),
              "tau_xi": THRESHOLD_XI, "tau_eps_tr": THRESHOLD_EPS_TR}
    for i, name in enumerate(("xi", "eps_tr")):
        report[name] = {"acc": float(((P[:, i] >= 0.5) == truth[:, i]).mean()),
                        "auc": float(roc_auc_score(truth[:, i], P[:, i])),
                        "p_change": float(truth[:, i].mean())}
    report["both_right"] = float(((P >= 0.5) == truth).all(axis=1).mean())
    weights = {f"w{i}": w for i, w in enumerate(model.get_weights())}
    return weights | {"mu": mean, "sd": std}, report


def out_of_fold(X, Y, state, steps, folds=5):
    ids = shuffled_states(state, seed=0)
    fold_of_state = {s: i % folds for i, s in enumerate(ids)}
    fold = np.array([fold_of_state[s] for s in state])
    P = np.zeros_like(Y)
    for k in range(folds):
        test = fold == k
        rest_ids = shuffled_states(state[~test], seed=k)
        validation = np.isin(state, rest_ids[:len(rest_ids) // 9]) & ~test
        model, mean, std = fit(X, Y, ~test & ~validation, validation, seed=k)
        P[test] = predict(model, mean, std, X[test])
        print(f"fold {k}: accuracy {((P[test] >= 0.5) == Y[test].astype(bool)).mean(axis=0)}", flush=True)

    truth, called = Y.astype(bool), P >= 0.5
    summary = {"thresholds": {"xi": THRESHOLD_XI, "eps_tr": THRESHOLD_EPS_TR},
               "both_right": float((called == truth).all(axis=1).mean())}
    table = []
    edges = np.logspace(-12, 0, 25)                      # half-decade bins of the true step size
    for i, name in enumerate(("xi", "eps_tr")):
        y, c, wrong = truth[:, i], called[:, i], called[:, i] != truth[:, i]
        summary[name] = {"correct_hold": int((~y & ~c).sum()), "correct_compute": int((y & c).sum()),
                         "needless_compute": int((~y & c).sum()), "missed_change": int((y & ~c).sum()),
                         "accuracy": float((~wrong).mean()), "rows": int(len(y))}
        zero = steps[i] < edges[0]
        table.append([name, 0.0, edges[0], int(zero.sum()), int(wrong[zero].sum())])
        for lo, hi in zip(edges[:-1], edges[1:]):
            inside = (steps[i] >= lo) & (steps[i] < hi)
            table.append([name, lo, hi, int(inside.sum()), int(wrong[inside].sum())])
    return summary, table


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help=".mat file of solver-labelled steps")
    ap.add_argument("--out-of-fold", action="store_true")
    ap.add_argument("--model-out", default=ROOT / "models" / "gate.npz")
    ap.add_argument("--results-dir", default=ROOT / "results", type=Path)
    args = ap.parse_args()

    X, Y, state, steps = load_steps(args.data)
    print(f"{len(X)} rows, {state.max() + 1} previous states; "
          f"xi changes in {Y[:, 0].mean():.1%}, eps_tr in {Y[:, 1].mean():.1%} of the rows", flush=True)

    if args.out_of_fold:
        summary, table = out_of_fold(X, Y, state, steps)
        json.dump(summary, open(args.results_dir / "gate_out_of_fold.json", "w"), indent=1)
        with open(args.results_dir / "gate_errors_by_step_size.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["variable", "step_lo", "step_hi", "rows", "gate_errors"])
            writer.writerows(table)
        print(json.dumps(summary, indent=1))
    else:
        weights, report = train_single(X, Y, state)
        np.savez(args.model_out, **weights)
        json.dump(report, open(args.results_dir / "gate_report.json", "w"), indent=1)
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
