# Closest-work and novelty matrix

| Work | Forecasting | Explicit causal graph | Online deployment graph updates | State-space backbone | Main distinction or caution |
| --- | --- | --- | --- | --- | --- |
| Huang et al., ICML 2019 | Yes | Yes, under stated assumptions | Adaptive Bayesian forecasting; inspect exact discovery procedure | Yes | Directly invalidates a broad “first causal state-space forecaster” claim |
| CDT, 2025 | Yes | Yes | A dynamic adapter is described; true post-deployment streaming must be checked | Transformer | Do not incorrectly label its learned structure completely static |
| DYNOTEARS, 2020 | Discovery | Yes | Batch method in its standard form | No | Handles temporal structure but is not itself the target forecasting architecture |
| PCMCI/Tigramite | Discovery | Yes, under assumptions | Standard estimation is typically window/batch based | No | Strong causal-discovery comparator; account for hidden confounding and lags |
| S-Mamba, 2024 | Yes | No explicit identified causal graph | No explicit graph updates | Selective SSM | Useful graph-agnostic SSM baseline |
| CMamba, 2024 | Yes | Channel correlations, not necessarily identified causality | Not an online causal graph method | Selective SSM | Channel dependence and causal structure should not be conflated |
| TSMamba, 2024 | Yes | No verified causal graph | Not the proposed streaming graph mechanism | Mamba-style | Foundation-scale comparisons may exceed the thesis compute budget |
| DTAF, AAAI 2026 | Yes | No explicit causal graph in the stated method | Temporal/frequency adaptation, not graph tracking | Separate architecture | Official code exists; valuable nonstationary forecasting baseline |
| CGT, 2026 | Forecasting for anomaly detection | Yes, time-lagged graph prior | Streaming thresholds; graph update details must be checked | Transformer | Different principal task and metrics |
| TECamba, 2026 | Forecasting signals for anomaly detection | Transfer-entropy-based causal modeling | Dynamic patterns are described | Verify original model | Publisher full-text restrictions may prevent faithful reimplementation |
| August 2026 causal-graph-enhanced Mamba publisher result | Multistep forecasting according to title | According to title | Not verified | Mamba according to title | Inspect full text before making any first-of-its-kind claim |
| Proposed Bachelor thesis | Yes | Estimated lagged graph with declared assumptions | Explicit predict → reveal → graph update loop | Graph-conditioned selective SSM | Evaluate a narrowly specified *combination*; do not assume novelty without comparison |

## Defensible contribution statement

> This thesis investigates whether a computationally explicit online lagged-graph estimator can improve a graph-conditioned selective state-space forecaster under controlled mechanism shifts, and quantifies when graph-estimation errors, latent confounding, or adaptation costs remove the benefit.

## Claims that require additional proof

- “First” anything: requires a documented current literature search and precise architecture definition.
- “Causal graph recovery” on real ETT data: requires credible external graph labels or intervention evidence.
- “Linear complexity”: must specify that it is linear in sequence length, and separately account for `D²` graph operations.
- “Calibrated uncertainty”: requires actual intervals, appropriate coverage metrics, and shift-specific evaluation.
- “No retraining”: online updates are still parameter or graph adaptation; explain exactly which parameters remain frozen.
