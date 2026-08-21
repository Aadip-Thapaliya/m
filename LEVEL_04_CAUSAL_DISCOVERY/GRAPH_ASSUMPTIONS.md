# Graph semantics and assumptions

## Lagged graph

An edge `j(t-1) → i(t)` respects time ordering. The unrolled temporal graph is acyclic even if the collapsed variable graph contains a feedback cycle.

Example: `x(t-1) → y(t)` and `y(t-1) → x(t)` are jointly valid lagged relationships. Forcing the collapsed graph to be a DAG would delete one real mechanism.

## Contemporaneous graph

An edge `j(t) → i(t)` occurs within the same sampling interval. Acyclicity can be meaningful here, but it requires additional assumptions or intervention/domain evidence to orient the relationship.

The helper `contemporaneous_order_projection()` is safe only when a scientifically defensible ordering is supplied. It is not a universal structure-learning method.

## Hidden confounding

If latent `u(t)` affects both observed variables, a pairwise or multivariate lagged regression may introduce spurious edges. The included `confounded` synthetic scenario is specifically designed to expose this failure.

## Graph evaluation

Report directed structural Hamming distance consistently: in this repository a reversed edge costs **two** edits because one edge is removed and the opposite edge is added. Exclude self-lags from edge F1 unless explicitly analyzing them.

## Language to use

Prefer “estimated lagged predictive-dependency graph” until causal sufficiency, lag adequacy, absence or treatment of contemporaneous confounding, and model assumptions have been stated.
