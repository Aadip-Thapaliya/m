# Level 06 — Regime adaptation and uncertainty

Read [`REGIME_CHANGE.md`](REGIME_CHANGE.md) for detection and recurring-regime handling, then [`UNCERTAINTY_PLAN.md`](UNCERTAINTY_PLAN.md) for achievable probabilistic extensions.

Current code includes Page-Hinkley residual monitoring and a bounded graph-prototype memory. It does **not** yet implement a Bayesian graph posterior, calibrated predictive intervals, or conformal online coverage; those are clearly marked research extensions.

## Adaptation timescales

- Fast: update graph coefficients after each revealed observation.
- Medium: detect persistent residual changes and reset or inflate the graph covariance.
- Slow: recognize recurring regimes and reuse graph prototypes where justified.
- Optional uncertainty layer: estimate forecast variance and recalibrate intervals online.
