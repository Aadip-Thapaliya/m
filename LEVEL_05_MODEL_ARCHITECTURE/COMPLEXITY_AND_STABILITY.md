# Complexity and stability accounting

Let `L` be context length, `D` the number of variables, `N` the per-variable state width, and `E` the number of active graph edges.

| Component | Dense cost | Sparse alternative | Notes |
| --- | --- | --- | --- |
| Shared lag-one RLS graph update | `O(D²)` per observation | Depends on approximation | Shared covariance is essential for the quoted scaling |
| Graph message `A x` | `O(D²)` | `O(E)` | Sparse gain requires an actual sparse implementation |
| Diagonal state transition | `O(DN)` | Same | Avoid dense `DN × DN` transition matrices |
| Graph-aware gate generation | `O(DN)` for fixed graph-embedding width | Same | Width assumptions must be stated |
| Full sequential reference | `O(L(D² + DN))` | `O(L(E + DN))` | Linear in `L`, not in `D` |
| Multi-lag dense covariance | `O(D²K²)` | Depends on approximation | `K` is the number of lags |
| NOTEARS-style matrix exponential | Generally super-quadratic | Approximation-dependent | Do not describe unspecified exact DAG projection as `O(D²)` |

## Stability checklist

- Parameterize base decay as a negative exponential.
- Use positive time steps via `softplus`.
- Clip or normalize graph weights if their spectral radius destabilizes propagation.
- Track gradient norms, state norms, graph density, and NaN occurrences.
- Use chronological validation; choose hyperparameters without inspecting future test targets.
- Benchmark an identity or diagonal graph to separate self-memory from cross-variable conditioning.
