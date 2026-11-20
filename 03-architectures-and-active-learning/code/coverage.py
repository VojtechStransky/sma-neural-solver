"""How much of the input space does a dataset cover?

    occupied_cells(x, bins)                 cells of a regular grid that contain at least one row
    nearest_neighbour_distance(x, train)    distance of each row to the closest training row

Both work on standardized inputs. In the study the cell count showed that the
reachable states form a thin set (1,101 of 32,768 cells of an 8^5 grid), and
the nearest-neighbour distance replaced a min/max box test that never fired.
"""
import numpy as np


def occupied_cells(x, bins=8, lower=None, upper=None):
    """Set of occupied cells of a bins^d grid over [lower, upper] (default: the range of x)."""
    lower = x.min(0) if lower is None else lower
    upper = x.max(0) if upper is None else upper
    index = np.clip(((x - lower) / (upper - lower + 1e-12) * bins).astype(int), 0, bins - 1)
    return set(map(tuple, index))


def nearest_neighbour_distance(x, train, chunk=2000):
    """Euclidean distance to the nearest training row, in units of the training standard deviations."""
    mean, std = train.mean(0), train.std(0) + 1e-12
    a, b = (x - mean) / std, (train - mean) / std
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        d2 = ((a[i:i + chunk, None, :] - b[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = np.sqrt(d2.min(1))
    return out


if __name__ == "__main__":
    # Synthetic illustration: points near a curve inside a 5-D box occupy few cells.
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 1, (20000, 2))
    thin = np.column_stack([t[:, 0], t[:, 1], t[:, 0] * t[:, 1], t[:, 0] ** 2, np.sin(3 * t[:, 1])])
    thin += 0.01 * rng.standard_normal(thin.shape)
    box = rng.uniform(thin.min(0), thin.max(0), thin.shape)
    for name, x in (("2-D sheet in 5-D", thin), ("uniform in the box", box)):
        print(f"{name:20s} occupies {len(occupied_cells(x)):6d} of {8 ** 5} cells")
    inside = nearest_neighbour_distance(thin[:500], thin[500:5000])
    outside = nearest_neighbour_distance(box[:500], thin[500:5000])
    print(f"median distance to the nearest training row: on the sheet {np.median(inside):.3f}, off it {np.median(outside):.3f}")
