"""First study: can "does the state change in this step?" be predicted at all?

Compares simple baselines, gradient boosting and small networks on
solver-labelled single steps, under two evaluation schemes:

    group    5-fold cross-validation in which all rows of one previous state
             (xi_n, eps_tr_n, M_n) stay in one fold. This is the honest score.
    random   one random 80/20 split of the rows. The data hold about 40 rows per
             previous state, so the same state lands on both sides and the
             score comes out too high.

Data: a .mat file with one array of rows [xi_n, eps_tr_n, M_n, eps, T, xi, eps_tr, ...]
(the original 61,845-row dataset of step 03; it is not stored in this folder).

Usage:
    python classifier_study.py --data red_dataset_1.mat --split group
    python classifier_study.py --data red_dataset_1.mat --split random --models nn_128x3
    python classifier_study.py --data red_dataset_1.mat --tau 1e-4      # threshold sensitivity

Prints one line per model and writes ../results/classifier_run_<split>_tau<tau>.json.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from scipy.io import loadmat
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_steps(mat_file, tau):
    """Inputs X (N, 5), labels Y (N, 2) = [xi changes, eps_tr changes], state id per row."""
    rows = [v for k, v in loadmat(mat_file).items() if not k.startswith("__")][0]
    X = rows[:, :5].copy()
    Y = np.stack([np.abs(rows[:, 5] - rows[:, 0]) > tau,
                  np.abs(rows[:, 6] - rows[:, 1]) > tau], axis=1).astype(np.float32)
    _, state = np.unique(np.round(rows[:, :3], 12), axis=0, return_inverse=True)
    return X, Y, state.ravel()


def with_features(X):
    """Two hand-made inputs: an elastic-strain proxy and the same scaled by temperature."""
    elastic = X[:, 3] - X[:, 1]
    return np.c_[X, elastic, elastic * (X[:, 4] + 273.15) / 300.0]


def group_folds(state, k=5, seed=0):
    ids = np.unique(state)
    np.random.default_rng(seed).shuffle(ids)
    fold_of_state = {s: i % k for i, s in enumerate(ids)}
    fold = np.array([fold_of_state[s] for s in state])
    return [(np.flatnonzero(fold != i), np.flatnonzero(fold == i)) for i in range(k)]


def random_split(n, seed=0, test_fraction=0.2):
    order = np.random.default_rng(seed).permutation(n)
    cut = int(n * test_fraction)
    return [(order[cut:], order[:cut])]


# ------------------------------------------------------------------ models
# Each model is a function (X_train, Y_train, state_train, X_test) -> probabilities (N_test, 2).

def majority(X, Y, state, X_test):
    return np.tile(Y.mean(axis=0), (len(X_test), 1))


def sklearn_model(make, features):
    def predict(X, Y, state, X_test):
        if features:
            X, X_test = with_features(X), with_features(X_test)
        scaler = StandardScaler().fit(X)
        return np.stack([make().fit(scaler.transform(X), Y[:, head]).predict_proba(scaler.transform(X_test))[:, 1]
                         for head in (0, 1)], axis=1)
    return predict


def network(units, features, seed=0):
    def predict(X, Y, state, X_test):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        if features:
            X, X_test = with_features(X), with_features(X_test)
        # a tenth of the training states is held out to stop the training
        ids = np.unique(state)
        np.random.default_rng(seed).shuffle(ids)
        stop = np.isin(state, ids[:len(ids) // 10])
        scaler = StandardScaler().fit(X[~stop])
        model = tf.keras.Sequential(
            [tf.keras.Input((X.shape[1],))]
            + [tf.keras.layers.Dense(u, activation="swish") for u in units]
            + [tf.keras.layers.Dense(2, activation="sigmoid")])
        model.compile(tf.keras.optimizers.Adam(2e-3), "binary_crossentropy")
        model.fit(scaler.transform(X[~stop]), Y[~stop],
                  validation_data=(scaler.transform(X[stop]), Y[stop]),
                  epochs=300, batch_size=256, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
                             tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-5)])
        return model.predict(scaler.transform(X_test), verbose=0, batch_size=4096)
    return predict


def boosting():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08)


MODELS = {
    "majority": majority,
    "logreg": sklearn_model(lambda: LogisticRegression(max_iter=2000), features=False),
    "logreg+features": sklearn_model(lambda: LogisticRegression(max_iter=2000), features=True),
    "boosting": sklearn_model(boosting, features=False),
    "boosting+features": sklearn_model(boosting, features=True),
    "nn_64_64_32": network((64, 64, 32), features=False),
    "nn_128x3": network((128, 128, 128), features=False),
    "nn_128x3+features": network((128, 128, 128), features=True),
}


def scores(truth, P):
    out = {}
    for head, name in enumerate(("xi", "eps_tr")):
        y, p = truth[:, head], P[:, head]
        out[name] = {"acc": float(((p >= 0.5) == y).mean()),
                     "balanced_acc": float(balanced_accuracy_score(y, p >= 0.5)),
                     "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")}
    out["both_right"] = float(((P >= 0.5) == truth).all(axis=1).mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", choices=["group", "random"], default="group")
    ap.add_argument("--tau", type=float, default=1e-5, help="'changes' threshold, same for both variables")
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()

    X, Y, state = load_steps(args.data, args.tau)
    print(f"{len(X)} rows, {state.max() + 1} previous states; "
          f"xi changes in {Y[:, 0].mean():.3f}, eps_tr in {Y[:, 1].mean():.3f} of the rows", flush=True)
    folds = group_folds(state) if args.split == "group" else random_split(len(X))

    results = {}
    for name in args.models.split(","):
        start = time.time()
        P, tested = np.zeros_like(Y), np.zeros(len(Y), bool)
        for train, test in folds:
            P[test] = MODELS[name](X[train], Y[train], state[train], X[test])
            tested[test] = True
        results[name] = scores(Y[tested].astype(bool), P[tested])
        r = results[name]
        print(f"{name:18s} xi {r['xi']['acc']:.4f}   eps_tr {r['eps_tr']['acc']:.4f}   "
              f"both right {r['both_right']:.4f}   [{time.time() - start:.0f} s]", flush=True)
    json.dump(results, open(RESULTS / f"classifier_run_{args.split}_tau{args.tau:g}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
