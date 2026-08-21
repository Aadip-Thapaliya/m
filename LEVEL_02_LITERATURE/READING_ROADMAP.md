# Reading roadmap

## Stage A: closest direct prior work

1. **Huang et al. 2019:** establishes that nonstationary causal discovery, forecasting, and state-space models have already been combined. Record its assumptions, inference machinery, and whether its model uses modern selective neural state spaces.
2. **CDT 2025:** studies causal-aligned forecasting with a discovered graph and dynamic causal adapter. Compare training-time refinement versus genuinely online deployment-time updating.
3. **CausalRivers 2025:** explains why real-world causal graph benchmarking is difficult and offers a more defensible domain than ordinary ETT graph labels.
4. **CausalTime 2023:** provides semi-realistic ground-truth causal benchmark ideas.

## Stage B: graph discovery

5. **NOTEARS:** differentiable acyclicity for a contemporaneous DAG; understand why the matrix exponential is not automatically quadratic.
6. **DYNOTEARS:** adapts continuous structure learning to temporal lagged and contemporaneous dependencies.
7. **PCMCI/Tigramite:** conditional-independence-based time-series causal discovery.
8. **CD-NOD:** uses heterogeneity and nonstationarity in discovery instead of treating all changes as nuisance.
9. **Semi-stationary causal discovery:** inspect how graph assumptions evolve across regimes.

## Stage C: state-space forecasting

10. **S4:** structured state-space foundations.
11. **Mamba:** input-conditioned selective state spaces and linear sequence-length scaling.
12. **Mamba-2:** state-space duality and architecture refinements.
13. **S-Mamba, CMamba, MambaTS, TSMamba, TimeMachine:** examine variable mixing, channel dependence, temporal direction, and implementation availability.
14. **Mamba-3:** treat as a contemporary comparison or discussion item; do not assume implementation parity or fair compute access.

## Stage D: nonstationary forecasting and credible baselines

15. **DTAF:** official repository available; strong nonstationary comparator.
16. **AdaRNN and RevIN:** adaptation and normalization baselines.
17. **DLinear, PatchTST, iTransformer, TimesNet, Autoformer, FEDformer:** contextualize performance and avoid comparing only against weak methods.

## Stage E: uncertainty and anomaly-oriented neighbors

18. **Adaptive conformal inference:** useful for distribution-shift-aware intervals.
19. **CGT:** causal graph restrictions and Gaussian uncertainty, but aimed at anomaly detection.
20. **TECamba:** causality-guided anomaly detection under nonstationarity; verify task and access before proposing it as a forecasting baseline.

## Notes to capture for every paper

Citation; official paper URL; official code URL; data; graph semantics; lag handling; contemporaneous assumptions; whether the graph changes during training, evaluation, or deployment; forecast horizon; update latency; computational cost; uncertainty mechanism; reproducibility; and the exact distinction from this thesis.
