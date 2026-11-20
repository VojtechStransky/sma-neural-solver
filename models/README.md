# Trained networks

| File | What | Described in |
|---|---|---|
| `v10.onnx`, `v10.weights.h5` | baseline network, trained on single steps | [step 01](../01-baseline-network/) |
| `v17.onnx`, `v17.weights.h5` | `v10` fine-tuned along its own trajectory | [step 02](../02-rollout-fine-tuning/) |

The ONNX files map `[xi_n, etr_n, M_n, eps, T]` (float32, any batch size) to `[xi, etr]`, with $\xi$ already
clipped to $[0, 1]$. The memory variable $M$ is updated by the simulation loop, see
[common/rollout.py](../common/rollout.py). The weight files load into `common.network.build_network`.

The names `v10` and `v17` are the run numbers used in the thesis.

Re-score either network on the 250 benchmark cycles (from the repository root):

```
python -m common.benchmark models/v17.onnx --out v17_check
```
