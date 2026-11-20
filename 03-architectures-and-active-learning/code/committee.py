"""Committees of independently trained networks: aggregation, the limit of averaging, disagreement as a warning.

    aggregate(predictions)            median over the members
    saturation(k, error)              fit error(k) = floor + slope / k
    auc(score, flag)                  how well a score separates flagged rows from the rest

Run this file to refit the saturation curves from ../results/committee.json.
"""
import json
import pathlib

import numpy as np


def aggregate(predictions, how="median"):
    """predictions: (members, rows, outputs). The median is robust to one bad member."""
    return np.median(predictions, axis=0) if how == "median" else np.mean(predictions, axis=0)


def saturation(k, error):
    """Least-squares fit of error(k) = floor + slope / k.

    Averaging k members removes the part of the error that differs between
    members (it shrinks like 1/k). The floor is the part they all share, and
    no committee size gets below it.
    """
    k, error = np.asarray(k, float), np.asarray(error, float)
    floor, slope = np.linalg.lstsq(np.column_stack([np.ones_like(k), 1.0 / k]), error, rcond=None)[0]
    return float(floor), float(slope)


def auc(score, flag):
    """Probability that a flagged row has a higher score than an unflagged one (Mann-Whitney)."""
    positive, negative = score[flag], score[~flag]
    ranks = np.argsort(np.argsort(np.concatenate([positive, negative]))) + 1
    return float((ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2)
                 / (len(positive) * len(negative)))


def disagreement_auc(predictions, truth):
    """Does the spread between members point at the rows where the committee is wrong?

    A row is flagged when the committee's xi error is in the worst 10 %.
    """
    error = np.abs(aggregate(predictions) - truth)[:, 0]
    return auc(predictions.std(axis=0)[:, 0], error > np.percentile(error, 90))


if __name__ == "__main__":
    results = json.load(open(pathlib.Path(__file__).resolve().parent / ".." / "results" / "committee.json"))
    for name, r in results.items():
        k = [p["k"] for p in r["curve"]]
        floor, slope = saturation(k, [p["holdout_median"] for p in r["curve"]])
        single = np.array(r["single_spread"]["mae_per_member"])
        spread = 100 * (single.max() - single.min()) / 2 / single.mean()
        print(f"{name:5s} single members: mean {single.mean():.5f}, spread +/-{spread:.0f} %;  "
              f"committee: error(k) = {floor:.6f} + {slope:.6f}/k;  "
              f"AUC of the member spread as a failure flag: {r['trust']['auc_std_ksi']:.3f}")
