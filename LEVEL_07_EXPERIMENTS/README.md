# Level 07 — Experimental design

Use [`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md), [`BASELINES.md`](BASELINES.md), [`ABLATION_MATRIX.md`](ABLATION_MATRIX.md), and [`METRICS_AND_STATISTICS.md`](METRICS_AND_STATISTICS.md) as the evaluation contract before tuning the model. [`INITIAL_SANITY_RESULTS.md`](INITIAL_SANITY_RESULTS.md) records verified initial runs, including a real-data case where online VAR outperforms the graph-conditioned prototype.

Run the portable benchmark with:

```bash
python scripts/run_benchmark.py --quick
python scripts/run_benchmark.py --config configs/benchmark.yaml
```

The supplied runner is a correct initial synthetic benchmark. Full ETT neural-model training, long-horizon forecasting, and external baselines remain research implementation tasks.
