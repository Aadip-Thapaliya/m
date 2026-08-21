# Streaming graph-update algorithm

## Data representation

Let `A_t[i, j]` denote the estimated effect of variable `j` at `t-1` on variable `i` at `t`. For the initial version use one lag, continuous variables, a linear structural approximation, and a shared covariance matrix.

## Per-observation update

1. Start with previous observation `x_(t-1)` and coefficient matrix `W_(t-1)`.
2. Produce `x̂_t = x_(t-1)^T W_(t-1)` before accessing the target.
3. Reveal `x_t` and compute `e_t = x_t - x̂_t`.
4. Pass the mean squared residual to a drift detector.
5. Compute the shared RLS gain and update all target coefficients simultaneously.
6. Apply a modest proximal sparsity update if requested.
7. Return `A_t = W_t^T` in the repository's `[target, parent]` convention.
8. If a regime alarm fires, optionally inflate/reset the covariance and store the old graph prototype.

## Complexity

For one lag and `D` variables:

- Shared covariance: `D × D`, giving `O(D²)` memory.
- Gain computation and covariance update: `O(D²)`.
- Coefficient outer-product update: `O(D²)`.
- Dense graph-conditioned message passing: `O(D²)`.

For `K` lags the feature dimension becomes `DK`; a dense shared covariance requires `O(D²K²)` memory and update work. A sparse or diagonal approximation is needed before claiming the same scaling for arbitrary `K`.

## Recommended sweeps

`forgetting_factor ∈ {0.90, 0.95, 0.97, 0.985, 0.995, 1.0}`.

`sparsity_threshold ∈ {0.0, 0.02, 0.05, 0.10, 0.20}`.

Evaluate edge F1 and post-change MSE together; a sparse graph can look structurally cleaner while worsening forecasting.
