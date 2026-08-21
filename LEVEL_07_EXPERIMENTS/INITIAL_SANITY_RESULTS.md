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

## Full five-seed synthetic baseline

Command: `python scripts/run_benchmark.py --config configs/benchmark.yaml`.

Configuration: five scenarios, five paired seeds, 1,600 observations per run, six variables, and four model families. The table reports mean MSE over five seeds.

| Scenario | Persistence | Static VAR | Online VAR | Graph-conditioned NumPy SSM |
| --- | ---: | ---: | ---: | ---: |
| Stationary | 0.05236 | **0.03263** | 0.03418 | 0.03440 |
| Abrupt | 0.05311 | 0.03978 | **0.03434** | 0.03451 |
| Gradual | 0.05264 | 0.03944 | **0.03420** | 0.03435 |
| Recurring | 0.05252 | 0.03578 | **0.03427** | 0.03444 |
| Confounded | 0.05864 | 0.05314 | **0.04211** | 0.04256 |

Static VAR wins the stationary negative control, while online VAR is the strongest mean-MSE baseline under all four changing/confounded scenarios. This is an appropriate starting point for a thesis: the proposed richer architecture still has to earn its complexity.

## Interpretation

The graph-conditioned scaffold improves slightly over online VAR in one small synthetic smoke run but does **not** beat online VAR on the tested ETTh1 slice or in the full five-seed scenario means. The smoke-run difference therefore should not be interpreted as a robust research result.

This negative result is scientifically useful: the intended neural architecture and ablations must demonstrate an actual benefit beyond online regression. Do not present these sanity checks as evidence that the proposed thesis method already succeeds.
