# Level 01 — Foundations

Master the vocabulary before changing the model. Use [`MATHEMATICAL_PRIMER.md`](MATHEMATICAL_PRIMER.md) for equations and [`IDENTIFIABILITY_CHECKLIST.md`](IDENTIFIABILITY_CHECKLIST.md) before using the word “causal.”

## Study sequence

1. Multivariate autoregression, stationarity, trend, seasonality, and conditional forecasting.
2. Structural causal models, interventions, directed graphs, confounding, and identifiability.
3. Granger predictability versus causal interpretation.
4. Time-unrolled lagged graphs versus contemporaneous DAGs.
5. Classical state-space models, Kalman filtering, structured state spaces, S4, and selective SSMs.
6. Online learning, forgetting factors, prequential evaluation, concept drift, and dynamic regret.
7. Proper scoring rules, Gaussian CRPS, interval coverage, and calibration under drift.

## Minimum working definitions

- `D`: number of observed variables.
- `t`: current observation index.
- `H`: forecast horizon.
- `A_t[target, parent]`: estimated lag-one influence of a parent at time `t-1` on a target at time `t`.
- `h_t`: forecasting hidden state; this is not the same object as the graph.
- `λ`: RLS forgetting factor; smaller values adapt faster but add estimation variance.
- “Online”: a model may update only with observations whose labels have actually been revealed.
- “Causal”: requires explicit identification assumptions; prediction alone is insufficient.
