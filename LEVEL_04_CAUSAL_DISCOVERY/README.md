# Level 04 — Online causal discovery

The first implementation estimates a lag-one directed predictive graph with shared-covariance recursive least squares. Read [`ONLINE_GRAPH_ALGORITHM.md`](ONLINE_GRAPH_ALGORITHM.md), [`GRAPH_ASSUMPTIONS.md`](GRAPH_ASSUMPTIONS.md), and [`BASELINE_TOOLKIT.md`](BASELINE_TOOLKIT.md).

Implementation: [`src/ocg_ssm/discovery/online_var.py`](../src/ocg_ssm/discovery/online_var.py).

## Important distinction

The implementation returns a **lagged predictive graph**. Interpreting it as a causal graph requires justified assumptions. The lagged graph is not forced to be a DAG because feedback at different time steps is valid.

## Suggested progression

1. Learn a stationary lag-one graph from a synthetic process.
2. Verify edge direction and coefficient signs against known ground truth.
3. Introduce abrupt graph changes and vary the forgetting factor.
4. Add sparse shrinkage and graph-threshold sweeps.
5. Compare with windowed PCMCI and DYNOTEARS where dependencies can be installed.
6. Evaluate gradual changes, recurring regimes, latent confounding, and noise.
7. Add multi-lag, nonlinear, or contemporaneous discovery only after the first protocol is reproducible.
