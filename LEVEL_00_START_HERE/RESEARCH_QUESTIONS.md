# Refined research questions and testable hypotheses

## Primary question

Under controlled changes in a lagged multivariate data-generating mechanism, does explicitly conditioning a stable state-space forecaster on an online-estimated directed dependency graph improve forecasting and recovery relative to static and adaptive graph-agnostic comparators?

## RQ1: tracking

How accurately and quickly does shared-covariance recursive least squares track sparse, changing lag-one structures under abrupt, gradual, and recurring changes?

**Measure:** SHD, directed edge F1, precision, recall, edge-weight error, and post-change recovery time on synthetic data.

## RQ2: forecasting

Does graph-conditioned state propagation improve MSE and MAE after mechanism shifts compared with persistence, static VAR, online VAR, and a graph-agnostic state-space ablation?

**Measure:** overall prequential error, post-change-window error, cumulative regret, and per-seed confidence intervals.

## RQ3: mechanism

Are improvements attributable to the estimated graph rather than merely online parameter updates or extra model capacity?

**Measure:** true-graph oracle, learned graph, frozen graph, random graph, shuffled graph, empty graph, and graph-agnostic ablations.

## RQ4: robustness

How do missing edges, reversed edges, graph density, hidden confounding, noise, change magnitude, and delayed target feedback alter performance?

**Measure:** graph perturbation sweeps, interaction plots, calibration, and qualitative failure analysis.

## RQ5: practicality

What are the empirical per-step runtime, peak memory, and adaptation overhead as variable count and lag increase?

**Measure:** CPU and GPU where available, warm-up-excluded timing, `D ∈ {5, 10, 25, 50, 100}`, and explicit dense/sparse complexity assumptions.

## Falsifiable hypotheses

H1: Online graph updates reduce short-window post-change forecasting error relative to a frozen graph.

H2: The benefit of graph conditioning increases when the true process is sparse and mechanisms change materially.

H3: Incorrect or confounded graphs can erase or reverse any forecasting benefit.

H4: A shared-covariance lag-one graph update scales approximately quadratically in variable count under the implemented dense linear algebra.

H5: If online VAR matches the graph-conditioned model, the additional state-space mechanism is not justified for that scenario.
