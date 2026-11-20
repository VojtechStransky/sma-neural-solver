"""The simulation loop around a one-step model.

The model predicts the new (xi, etr). The loop updates the memory variable M
by the rule of the constitutive model and feeds the result back as the next
input. This file is the single place where that rule is written.
"""
import numpy as np

SQRT15 = float(np.sqrt(1.5))
XI_FLOOR = 0.02          # guards the division when xi is close to zero


def update_memory(M, xi_n, xi, etr):
    """Memory variable after a step.

    M shrinks with the martensite that disappears and is raised to the norm of
    the current transformation strain. The factor is capped at 1: without the
    cap M grows without bound during reverse transformation.
    """
    ratio = min(xi_n / max(xi, XI_FLOOR), 1.0)
    return max(M * ratio, abs(etr) * SQRT15)


class OnnxStep:
    """A one-step model stored as ONNX: [xi_n, etr_n, M_n, eps, T] -> [xi, etr]."""

    def __init__(self, path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, np.float32))
        return self.session.run(None, {self.input_name: X})[0]


def teacher_forced(model, cycle):
    """One-step predictions with the reference state on the input. Returns (steps, 2)."""
    return model(cycle.X)


def free_run(model, cycle, on_step=None):
    """Autoregressive run along one cycle. Returns predictions (steps, 2).

    Only the first state comes from the reference. After that the model
    receives its own output. `on_step(state, prediction, eps, T)` may replace
    the prediction (used for the deployment rules of step 02).
    """
    xi, etr, M = (float(v) for v in cycle.X[0, :3])
    out = np.zeros((len(cycle.X), 2))
    for t, (eps, T) in enumerate(cycle.X[:, 3:5]):
        xi_new, etr_new = model([[xi, etr, M, eps, T]])[0]
        if on_step is not None:
            xi_new, etr_new = on_step((xi, etr, M), (xi_new, etr_new), eps, T)
        xi_new = float(np.clip(xi_new, 0.0, 1.0))
        M = update_memory(M, xi, xi_new, etr_new)
        xi, etr = xi_new, float(etr_new)
        out[t] = xi, etr
    return out


def r_squared(true, pred):
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    return float(1.0 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2))
