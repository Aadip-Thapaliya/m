# Ablation matrix

| Group | Conditions | Scientific question |
| --- | --- | --- |
| Graph information | Empty, identity, learned, frozen, random, shuffled, oracle | Is structural information useful at all? |
| Graph update | Never, every step, every 4 steps, every 24 steps, alarm-triggered | Is the benefit due to update frequency or graph quality? |
| Conditioning site | Input only, transition only, readout only, all sites | Where does graph information improve the state-space computation? |
| Estimator | Shared RLS, sliding-window ridge, windowed PCMCI, optional DYNOTEARS | Does the result depend on one discovery algorithm? |
| Sparsity | Threshold `0, 0.02, 0.05, 0.10, 0.20` | Is sparsity helping or destroying weak true parents? |
| Adaptation | No detector, Page-Hinkley, ADWIN, prototype memory | Which adaptation mechanism justifies its overhead? |
| Mechanism shift | Stationary, abrupt, gradual, recurring | Under which drift pattern does conditioning help? |
| Failure stress | Hidden confounder, missed edge, reversed edge, wrong lag | How fragile is the claimed causal benefit? |
| Scale | `D=5, 10, 25, 50, 100`; fixed and increasing state width | Does empirical scaling match the complexity argument? |
| Uncertainty | Point-only, Gaussian, ensemble, adaptive conformal | Are intervals useful and calibrated under change? |

## Minimum mandatory ablation

Learned online graph versus frozen graph versus random graph versus online VAR, repeated across at least five seeds on stationary and abrupt-change scenarios.

If the random graph performs as well as the learned graph, do not claim a meaningful causal mechanism benefit.
