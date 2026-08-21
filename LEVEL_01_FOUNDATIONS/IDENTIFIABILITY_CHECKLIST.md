# Causal-identifiability checklist

Before labeling an estimated directed edge causal, state whether all of the following are justified:

1. The relevant causal parents are observed, or the method explicitly accommodates latent confounders.
2. Sampling frequency is high enough that meaningful mechanisms are not hidden inside one observation interval.
3. The selected lag range contains the relevant causal delays.
4. Contemporaneous effects are absent, externally ordered, intervention-identified, or modeled with an appropriate method.
5. The conditional model family is rich enough for the data-generating mechanism.
6. The disturbances satisfy the independence or distributional assumptions required by the discovery method.
7. Mechanism changes are not confused with pure marginal-distribution or measurement shifts.
8. Selection bias, missingness, sensor saturation, and aggregation have been assessed.
9. Faithfulness, Markov assumptions, or invariance assumptions are declared if the method needs them.
10. The graph direction convention and any temporal alignment offset are written down.

If the assumptions are not credible, report a **directed predictive-dependency graph** and discuss causal interpretation as a limitation.
