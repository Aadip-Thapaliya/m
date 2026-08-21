# Claim-safety checklist

- [ ] Huang et al. ICML 2019 is discussed before making any novelty statement.
- [ ] Recent causal-graph-enhanced Mamba work has been checked immediately before submission.
- [ ] CDT's dynamic adapter is described accurately.
- [ ] “Causal” claims list the assumptions required beyond directed predictability.
- [ ] Lagged feedback cycles are not incorrectly forbidden by a contemporaneous DAG constraint.
- [ ] Complexity specifies whether scaling is in sequence length, variables, edges, state width, and lag count.
- [ ] No exact `O(D²)` DAG-projection guarantee is asserted without a real proof.
- [ ] Every forecast is generated before its target is revealed.
- [ ] Scaling and normalization do not use future test data.
- [ ] SHD/F1 are reported only where graph ground truth is defensible.
- [ ] Hidden confounding and wrong-lag failure modes are reported.
- [ ] A random-graph, frozen-graph, online-VAR, and graph-free control is considered.
- [ ] Graph uncertainty is not confused with predictive-output uncertainty.
- [ ] Any coverage or calibration claim has matching empirical evidence.
- [ ] The NumPy reference is not presented as a fused CUDA Mamba kernel.
- [ ] Published external scores are not compared unfairly to a different streaming protocol.
- [ ] All redistributed ETT files remain unchanged and retain their original attribution/license.
- [ ] Findings, speculative explanations, and future work are clearly separated.
