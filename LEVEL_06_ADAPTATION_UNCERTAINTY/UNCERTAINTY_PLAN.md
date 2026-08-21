# Uncertainty implementation plan

## Level 1: deterministic forecasting

Current baseline: point predictions, MSE, and MAE. This level establishes whether graph conditioning improves the mean forecast at all.

## Level 2: Gaussian predictive head

Add a learned `μ_t` and `σ_t = softplus(s_t) + ε`. Train with Gaussian negative log-likelihood; evaluate Gaussian CRPS, empirical interval coverage, and interval width.

This captures predictive uncertainty under a chosen conditional family, but it is **not** automatically graph-posterior uncertainty.

## Level 3: graph ensemble

Fit several graph estimators with different seeds, bootstrap windows, forgetting factors, or posterior approximations. Propagate each graph through the same forecaster and decompose variability across graph-conditioned predictions.

Call this an **ensemble approximation** unless a valid Bayesian posterior is actually derived and implemented.

## Level 4: adaptive conformal calibration

Maintain a trailing buffer of past absolute residuals or normalized residuals. Update interval width only after the relevant target has arrived. Evaluate coverage before changes, shortly after changes, and during recovered regimes.

Reference: Gibbs and Candès, [Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170).

## Metrics

Gaussian CRPS; negative log-likelihood; 80% and 95% empirical interval coverage; mean interval width; calibration error by regime; and worst post-change coverage.

Do not claim distribution-free conditional coverage or Bayesian graph calibration unless the required theoretical conditions are demonstrated.
