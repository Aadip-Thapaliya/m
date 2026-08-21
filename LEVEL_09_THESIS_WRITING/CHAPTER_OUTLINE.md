# Suggested thesis chapter outline

## 1. Introduction

Describe multivariate forecasting, nonstationary mechanisms, why a fixed dependency graph can become stale, the practical cost of frequent rediscovery, and the narrowed contribution.

## 2. Background and assumptions

Introduce multivariate autoregression, structural causal models, lagged versus contemporaneous graphs, identifiability caveats, state-space recurrence, selective SSMs, and streaming/prequential evaluation.

## 3. Related work

Start with Huang et al. 2019 because it is the closest conceptual predecessor. Then cover NOTEARS/DYNOTEARS/PCMCI, CDT, S-Mamba/CMamba/Mamba variants, DTAF/AdaRNN/RevIN, CausalRivers/CausalTime, and anomaly-focused CGT/TECamba.

Conclude with a precise comparison matrix rather than an unsupported “first” claim.

## 4. Method

Define graph convention and assumptions; derive shared-covariance RLS; present graph shrinkage and drift detection; define graph-conditioned input, selective transition, and readout; explain stable diagonal parameterization; and account for dense/sparse complexity.

## 5. Experimental design

Document generators, exact graph labels, ETT licensing, splits, preprocessing, model families, horizons, seeds, update schedules, metrics, ablations, hardware, and statistical comparisons.

## 6. Results

Present stationary controls, abrupt/gradual/recurring adaptation, graph F1 and SHD over time, post-change forecasting loss, oracle/random/frozen graph comparisons, ETT forecasting, sensitivity to hidden confounding, and runtime scaling.

## 7. Discussion and limitations

Discuss observational identification, generator mismatch, hidden variables, contemporaneous effects, dense graph cost, detector false alarms, real-world graph-label absence, compute limits, uncertainty calibration, and nearby 2019–2026 literature.

## 8. Conclusion

Answer each refined research question directly. Separate supported findings from future work such as nonlinear discovery, intervention data, multi-lag sparse estimators, and full graph-posterior approximations.

## Appendices

Hyperparameters, graph conventions, dataset cards, license attribution, supplementary ablations, reproducibility commands, and selected code diagrams.
