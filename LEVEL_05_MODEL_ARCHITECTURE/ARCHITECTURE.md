# Proposed architecture

## Inputs

Observation context `X ∈ R^(B×L×D)` and estimated graph `A ∈ R^(B×D×D)`, using target rows and parent columns.

## Module 1: online graph estimator

Update the lagged graph from revealed observations only. Start with shared-covariance RLS; optionally compare a windowed PCMCI or DYNOTEARS estimate.

## Module 2: graph encoder

Encode each target's incoming adjacency row with a small multilayer perceptron. This creates node-specific graph context without requiring a heavy graph neural network in the first thesis version.

## Module 3: graph message

Compute `m_t = A_t x_t`. This lets each target aggregate currently estimated parent information. Dense multiplication costs `O(D²)`.

## Module 4: stable selective state transition

Parameterize the base diagonal transition with `F = -exp(ρ)`. Generate graph-aware `Δ_t`, `B_t`, and `C_t` from the current observation and graph embedding. Evolve the hidden state with a stable exponential decay and graph-aware input gate.

## Module 5: prediction head

Read out a per-variable state vector and optionally add a small graph-message residual. For probabilistic forecasting, add a positive scale head with `softplus` and train with Gaussian negative log-likelihood.

## Module 6: adaptation controller

Feed the **pre-update** residual score to Page-Hinkley or ADWIN. On an alarm, increase graph plasticity and optionally restore a similar stored regime graph.

## What to prove experimentally

1. The graph input affects forecasts even when parameter count is controlled.
2. The graph update occurs after, not before, the scored forecast target is revealed.
3. An incorrect graph can degrade performance; document the failure honestly.
4. Sequence runtime scales roughly linearly with context length under fixed graph size.
