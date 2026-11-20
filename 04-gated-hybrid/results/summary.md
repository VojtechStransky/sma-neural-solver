## Benchmark paths (stress error in MPa)

| Path | Steps | Gate + solver | Solver work | Network alone | Gate + network |
|---|---:|---:|---:|---:|---:|
| `tension_T-050` | 62 | 0.474 | 44% | 19.75 | 3.17 |
| `tension_T-035` | 47 | 0.704 | 51% | 2.91 | 1.83 |
| `tension_T-020` | 32 | 1.395 | 86% | 3.74 | 3.32 |
| `tension_T-005` | 47 | 0.019 | 64% | 3.32 | 2.69 |
| `tension_T+010` | 62 | 0.008 | 50% | 3.27 | 1.76 |
| `tension_T+025` | 77 | 0.005 | 40% | 4.39 | 1.36 |
| `tension_T+040` | 92 | 0.014 | 33% | 5.55 | 1.21 |
| `tension_T+055` | 107 | 0.007 | 28% | 6.70 | 1.14 |
| `tension_T+070` | 122 | 0.464 | 25% | 7.44 | 1.59 |
| `tension_T+085` | 137 | 0.206 | 22% | 7.97 | 1.27 |
| `tension_T+100` | 152 | 0.112 | 20% | 8.35 | 1.21 |
| `deepcycle_T+060` | 240 | 0.268 | 58% | 6.55 | 4.55 |
| `compression_T+000` | 44 | 0.000 | 52% | 1.91 | 1.24 |
| `minorloop_T+040` | 80 | 0.013 | 59% | 4.13 | 2.36 |
| `thermomech_T+100` | 80 | 0.112 | 88% | 4.64 | 4.49 |
| `random_0` | 150 | 10.642 | 46% | 3.95 | 5.69 |
| `recovery_T-040` | 89 | 0.670 | 68% | 8.95 | 7.08 |
| `Tsweep_eps0.04` | 138 | 0.196 | 70% | 50.71 | 41.43 |
| **Median** | | 0.154 | 50% | 5.10 | 2.10 |
| **Mean** | | 0.850 | 50% | 8.57 | 4.85 |
| **Worst path** | | 10.642 |  | 50.71 | 41.43 |

## Medians by path set (stress error in MPa)

| Run | 11 tension cycles | 8 mixed paths | all 18 |
|---|---:|---:|---:|
| Full solver in the same loop (control) | 0.000 | 0.005 | 0.001 |
| Gate + full solver, one threshold 1e-5 | 0.032 | 0.337 | 0.190 |
| Gate + per-variable solver, one threshold 1e-5 | 0.027 | 0.231 | 0.105 |
| Gate + per-variable solver, per-variable thresholds | 0.112 | 0.232 | 0.154 |
| Network alone | 5.553 | 5.596 | 5.097 |
| Gate + network, one threshold 1e-5 | 1.421 | 4.555 | 2.467 |
| Gate + network, per-variable thresholds | 1.589 | 4.516 | 2.098 |
| Fine-tuned network of step 02 alone | 11.775 | 21.495 | 15.917 |
| Gate + fine-tuned network of step 02 | 9.772 | 20.145 | 14.923 |

## Cold cycles with a long plateau

| Run | Cycle 161: xi error | stress error | Cycle 213: xi error | stress error |
|---|---:|---:|---:|---:|
| Fine-tuned network of step 02 | 0.0073 | 10.9 | 0.0041 | 14.8 |
| Gate + that network | 0.0048 | 10.5 | 0.0014 | 10.1 |
| Network alone | 0.0331 | 14.1 | 0.0179 | 8.9 |
| Gate + network | 0.0351 | 12.5 | 0.0198 | 9.8 |
| Gate + solver | 0.0001 | 0.3 | 0.0002 | 0.3 |
