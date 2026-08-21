# Proposal audit: issues to fix before implementation

## 1. The proposed “first” claim is incorrect

Huang, Zhang, Gong, and Glymour published [Causal Discovery and Forecasting in Nonstationary Environments with State-Space Models](https://proceedings.mlr.press/v97/huang19g.html) at ICML 2019. Their work already combines nonstationarity, causal discovery, forecasting, and state-space models. A closely named August 2026 publisher result also describes a causal-graph-convolution-enhanced Mamba forecasting framework; its full text was not accessible during verification.

**Replace:** “the first state-space model combining online causal discovery with causal conditioning.”

**With:** “an empirical investigation of online lagged graph tracking as an explicit conditioning mechanism for selective neural state-space forecasting under controlled regime shifts.”

Any narrower “first” claim requires a fresh, documented literature search immediately before submission.

## 2. CDT is not simply a completely static causal method

The [CDT paper](https://arxiv.org/abs/2505.16308) explicitly describes a **dynamic causal adapter** that refines a discovered causal structure during model training. Its distinction from the proposed work is that this is not automatically the same as target-by-target online streaming updates after deployment. Compare the actual training and inference protocols; do not claim CDT never adapts its graph.

## 3. Lagged graphs and DAGs are not the same object

For a lag-one process, edges connect `X_j(t-1)` to `X_i(t)`. The time-unrolled graph is acyclic by construction. The collapsed variable graph may legitimately contain `i → j` and `j → i` because they occur at different time points.

- Apply NOTEARS-style acyclicity only to an explicitly modeled contemporaneous graph.
- Record lag, edge direction, instantaneous assumptions, and any hidden confounding.
- Do not remove valid lagged feedback loops merely to produce a DAG visualization.

## 4. Exact DAG projection does not automatically cost `O(D²)`

The standard smooth NOTEARS constraint uses `trace(exp(A ⊙ A)) - D`; a generic dense matrix exponential is not an `O(D²)` operation. A first-order approximation does not guarantee exact acyclicity. An `O(D²)` claim is defensible for a shared-covariance lag-one RLS update or a fixed known variable ordering, not for an unspecified exact projection.

## 5. Observational predictability is not automatically causation

Granger-style predictive edges become causal claims only under stated assumptions such as causal sufficiency, adequate lag specification, no relevant measurement aggregation, stable noise assumptions, and an appropriate model family. Hidden confounders and contemporaneous effects can invalidate edge interpretation.

Use the wording **estimated lagged predictive graph** unless the identification assumptions are explicit and credible.

## 6. Real ETT data do not contain a known causal graph

Use SHD, edge F1, precision, recall, and recovery time against true graph changes on synthetic data or a benchmark with justified graph labels. On ETT and ordinary electricity, weather, or traffic datasets, report forecasting, calibration, stability, sparsity, latency, and sensitivity to graph perturbation; do not invent structural ground truth.

## 7. Uncertainty has two distinct sources

- **Graph or epistemic uncertainty:** uncertain structure, coefficients, and mechanism changes.
- **Forecast or aleatoric uncertainty:** irreducible noise conditional on a graph and observed history.

A deterministic adjacency and Gaussian output head do not constitute a Bayesian graph posterior. If a posterior is not implemented, describe the result as an ensemble, bootstrap approximation, or conformal interval instead.

## 8. The timeline in the PDF needs explicit year labels

The proposal is dated July 2026 and lists August through January without years. The natural interpretation is August–December 2026 followed by January 2027; confirm this with the supervisor and adapt to the university's actual deadline.

## 9. Writing and terminology corrections

Use **causal**, not “casual”; **non-stationary**, not “non-stationery”; and **acyclic**, not “without cycles,” when referring specifically to a DAG. Write `S-Mamba`, `C-Mamba` or `CMamba` according to the paper, and distinguish `Mamba`, `Mamba-2`, and `Mamba-3` rather than treating them as identical architectures.

## 10. Scope control

Priority A: correct streaming protocol, synthetic ground truth, static and online baselines, graph conditioning, forecast/graph evaluation.

Priority B: recurring-regime memory, short-horizon forecasts, ETT, calibration.

Priority C: formal tracking guarantees, full Bayesian graph posterior, fused kernels, large traffic graphs, and 2026 foundation-model baselines. These are stretch goals, not prerequisites for a solid Bachelor thesis.
