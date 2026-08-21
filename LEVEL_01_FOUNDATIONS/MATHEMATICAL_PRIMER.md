# Mathematical primer

## 1. Forecasting task

For observations `x_t ∈ R^D`, estimate `x_(t+1:t+H)` given only `x_(1:t)`. For a one-step linear process:

`x_t = B_t x_(t-1) + ε_t`.

An entry `B_t[i, j]` maps parent variable `j` at the previous time to target variable `i` now.

## 2. Regime changes

Abrupt: `B_t = B^(1)` before `τ`, `B^(2)` after `τ`.

Gradual: `B_t = (1 - α_t) B^(1) + α_t B^(2)` with a smoothly changing `α_t`.

Recurring: `B^(1) → B^(2) → B^(1)`.

Confounded: a latent `u_t` affects multiple observed variables, violating causal sufficiency.

## 3. Shared-covariance RLS

For previous observation `z_t`, covariance `P_(t-1)`, coefficient matrix `W_(t-1)`, and forgetting factor `λ`:

`k_t = P_(t-1) z_t / (λ + z_t^T P_(t-1) z_t)`

`e_t = x_t - z_t^T W_(t-1)`

`W_t = W_(t-1) + k_t e_t^T`

`P_t = (P_(t-1) - k_t z_t^T P_(t-1)) / λ`

The adjacency used throughout the code is `A_t = W_t^T`. With one lag, both the covariance update and matrix outer product are `O(D²)`; memory is also `O(D²)`.

## 4. State-space update

A conventional discrete system has:

`h_(t+1) = F h_t + B x_t`

`ŷ_(t+1) = C h_(t+1) + D_skip x_t`.

A graph-conditioned selective version replaces fixed parameters with functions of the current graph and observation:

`m_t = A_t x_t`

`Δ_t = softplus(f_Δ(x_t, encode(A_t)))`

`h_(t+1) = exp(Δ_t F) ⊙ h_t + Δ_t B_t ⊙ project(x_t, m_t)`

`ŷ_(t+1) = C_t h_(t+1) + residual(m_t)`.

Parameterize diagonal state decay as `F = -exp(ρ)` to keep the transition stable.

## 5. Complexity caveat

For context length `L`, variables `D`, and state width `N`, the dense reference implementation costs `O(L(D² + DN))`. It is linear in `L`, not linear in `D`. Sparse graph multiplication can reduce the `D²` component to `O(|E|)` if sparsity is exploited by the implementation.

## 6. Proper streaming order

At index `t`, use history through `t-1` to create a prediction. Reveal `x_t`; record its loss; update the graph, detector, normalization, and forecasting state. Never update on `x_t` before scoring the prediction for `x_t`.
