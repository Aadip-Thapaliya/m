# Dataset selection rationale

## Tier 1: mandatory causal evidence

Use all five included synthetic scenarios first. Only synthetic observations provide exact pointwise graph labels and controlled change points, which makes SHD, edge F1, drift detection accuracy, and recovery time interpretable.

The stationary scenario is a negative control. If an online model appears superior only because its evaluation accidentally sees the target before prediction, the stationary case and a strict prequential test should expose the flaw.

## Tier 2: realistic forecasting evidence

Use ETTh1 and ETTh2 for manageable CPU experiments. Add ETTm1 and ETTm2 after the protocol, normalization, and horizon implementation are correct. Report forecasting error, predictive intervals, runtime, and graph stability; do not report an invented true ETT graph.

## Tier 3: broader domains

Download electricity, solar, traffic, exchange-rate, and Jena climate files locally. High-dimensional traffic and electricity may require sparse graph estimation, candidate parent screening, or a smaller subset.

## Tier 4: real-world causal benchmarks

CausalRivers is more suitable when a thesis claim truly requires domain-supported graph evaluation. Carefully distinguish river flow topology, physical causality, and the benchmark's scoring assumptions.

## Dataset-specific risks

- ETT transformer loads may be correlated through unobserved demand, weather, and operational controls.
- A road-network adjacency is a spatial prior, not automatically a time-directed causal graph.
- Exchange rates have policy and macroeconomic latent confounders.
- Climate variables can have contemporaneous interactions and measurement aggregation.
- Synthetic success may depend on the generator matching the assumptions of a linear RLS estimator.
