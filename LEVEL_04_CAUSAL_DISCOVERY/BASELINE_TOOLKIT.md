# Causal-discovery baseline toolkit

| Tool | Official source | Good use | Main caution |
| --- | --- | --- | --- |
| Tigramite / PCMCI | https://github.com/jakobrunge/tigramite | Conditional-independence time-series graphs; climate examples | Configure lags, significance, independence tests, and assumptions explicitly |
| DYNOTEARS / CausalNex | https://github.com/mckinsey/causalnex | Temporal lagged and contemporaneous continuous optimization | Batch cost; Python-version compatibility; exact acyclicity is not quadratic |
| causal-learn | https://github.com/py-why/causal-learn | PC, FCI, and broader discovery comparisons | Tabular methods need careful temporal feature construction |
| LiNGAM | https://github.com/cdt15/lingam | Linear non-Gaussian and VAR-LiNGAM alternatives | Requires assumptions such as suitable non-Gaussian disturbances |
| Salesforce CausalAI | https://github.com/salesforce/causalai | Time-series causal discovery and inference workflows | Check supported Python/dependency versions and benchmark parity |
| River | https://github.com/online-ml/river | ADWIN and online-stream processing baselines | Drift alarms are not causal graph discovery |

Use windowed reruns of batch methods as approximate streaming comparators, and include their recomputation time. Report any baseline that could not be installed or reproduced rather than silently excluding it.
