# Initial sanity-check results

These numbers were produced on **2026-08-21** using the committed NumPy reference implementation. They verify the pipeline; they are **not** a final thesis benchmark or a claim of state-of-the-art performance.

## Synthetic smoke run

Command: `python scripts/run_smoke.py`.

Configuration: abrupt regime changes, seed 7, 240 observations, five variables, 72-observation training prefix, and 168 one-step streaming predictions.

| Method | MSE | MAE |
| --- | ---: | ---: |
| Persistence | 0.02242 | 0.11955 |
| Static VAR | 0.01838 | 0.10814 |
| Online VAR | 0.01505 | 0.09764 |
| Graph-conditioned NumPy SSM | 0.01488 | 0.09733 |

The final synthetic directed graph has F1 ≈ 0.667 and SHD = 3 under the smoke configuration's threshold convention.

## Real ETTh1 sanity run

Command: `python scripts/run_ett_baselines.py --dataset ETTh1 --max-rows 4000`.

Configuration: first 4,000 official observations, seven variables, a 2,400-observation chronological training prefix, training-prefix-only standardization, and 1,600 one-step predictions.

| Method | Standardized MSE | Standardized MAE |
| --- | ---: | ---: |
| Persistence | 0.21311 | 0.30021 |
| Static VAR | 0.24234 | 0.35541 |
| Online VAR | 0.19598 | 0.30457 |
| Graph-conditioned NumPy SSM | 0.21204 | 0.32518 |

## Interpretation

The graph-conditioned scaffold improves slightly over online VAR in one small synthetic smoke run but does **not** beat online VAR on the tested ETTh1 slice. The two-seed quick benchmark also shows scenarios where the graph-conditioned prototype is effectively tied with, or worse than, online VAR.

This negative result is scientifically useful: the intended neural architecture and ablations must demonstrate an actual benefit beyond online regression. Do not present these sanity checks as evidence that the proposed thesis method already succeeds.
