# Baseline selection and fairness

| Priority | Baseline | Why it matters | Present status |
| --- | --- | --- | --- |
| P0 | Persistence | Detects whether a complex model beats trivial last-value prediction | Implemented |
| P0 | Static ridge VAR | Tests whether offline correlations are enough | Implemented |
| P0 | Online RLS VAR | Separates online adaptation from any state-space contribution | Implemented |
| P0 | Graph-conditioned NumPy SSM | End-to-end causal-conditioning scaffold | Implemented |
| P1 | Graph-agnostic SSM with matched parameters | Tests whether the graph itself helps | Planned neural ablation |
| P1 | Oracle-graph SSM | Upper bound under the generator's true graph | Planned |
| P1 | Frozen-graph SSM | Measures the value of graph adaptation | Planned |
| P1 | S-Mamba | Official Mamba forecasting comparison | External code available |
| P1 | CMamba | Channel-correlation state-space comparison | External code available |
| P1 | DTAF | Strong nonstationary forecasting comparison | Official code available |
| P2 | PCMCI + static/online forecaster | Separates graph estimator choice from the forecaster | Optional dependency |
| P2 | DYNOTEARS + forecaster | Temporal causal graph reference | Optional dependency and compute constraints |
| P2 | RevIN + MLP/Transformer | Distribution-shift normalization comparison | Planned |
| P2 | AdaRNN, DLinear, PatchTST, iTransformer | Broader forecasting context | Prioritize by available compute |
| P3 | CGT or TECamba adaptations | Nearby causal-anomaly methods | Different task; avoid unfair direct claims |

## Fairness rules

- Same observations, splits, context, horizon, scaling, and target-reveal schedule.
- Similar hyperparameter budget and number of random seeds.
- Separate offline and online method categories.
- Report model parameter count and graph-estimation overhead.
- Distinguish published benchmark numbers from reproduced results under your protocol.
- Record installation failures and missing official code rather than pretending a baseline was executed.
