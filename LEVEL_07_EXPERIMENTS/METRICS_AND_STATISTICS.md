# Metrics and statistical reporting

## Forecasting

MSE and MAE are mandatory. Report both total-stream results and an explicit post-change window, for example the first 25 or 100 labeled observations after each synthetic change.

For probabilistic models, report Gaussian CRPS, negative log-likelihood, 80%/95% coverage, and mean interval width. Point-only models do not have a meaningful CRPS unless a justified predictive distribution is added.

## Graph recovery

Report directed structural Hamming distance, edge precision, recall, and F1 using a declared threshold and the `[target, parent]` convention. This repository counts a reversed edge as two edits. Exclude self-lags from structural metrics by default.

Use only datasets with justified graph labels.

## Adaptation

Detection delay; tolerance-window alarm precision/recall; false alarms per 1,000 steps; post-change recovery time; and cumulative regret relative to a clearly named comparator.

Define recovery in advance. A robust variant requires the smoothed loss to stay within a chosen tolerance of its pre-change reference for multiple consecutive observations.

## Efficiency

Average latency, p50/p95/p99 latency, graph-update-only latency, peak resident/GPU memory, parameter count, and total wall-clock runtime. Include graph discovery cost rather than timing only the forecast head.

## Statistics

Run at least five paired seeds. Report per-seed values, mean ± standard deviation, paired bootstrap confidence intervals or another justified paired comparison, and the direction/magnitude of the effect. Correct for multiple comparisons if presenting many confirmatory tests.

Time-series samples are autocorrelated. Do not treat every forecast timestamp as an independent observation for naive significance tests; prefer seed-level pairing or a justified blocked bootstrap.

## Proposed causal-forecast consistency

Define this exploratory quantity before reporting it, for example the correlation across seeds/scenarios between improvement in graph edge F1 and reduction in post-change forecasting loss. Label it descriptive; correlation is not proof that graph quality caused forecast improvement.
