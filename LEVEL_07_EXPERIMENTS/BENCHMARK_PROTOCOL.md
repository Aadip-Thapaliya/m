# Benchmark protocol

## Synthetic evaluation

Scenarios: stationary, abrupt, gradual, recurring, and confounded.

Minimum seeds: `7, 21, 42, 87, 123`. Begin with six variables and one lag; expand to 10, 25, 50, and 100 variables only after the base protocol is stable.

Choose a chronological training prefix before any evaluation change point. At every streaming index:

1. Construct the forecast using history available at that instant.
2. Reveal the true observation.
3. Record forecasting error and, when appropriate, pre-update graph status.
4. Update the graph, drift detector, state, and normalization.
5. Record adaptation cost and any alarm.

Keep offline baselines frozen after training. Permit online updates to methods designed for online evaluation, but give all adaptive methods the same target-reveal timing.

## ETT evaluation

Start with ETTh1 and ETTh2, then ETTm1 and ETTm2. Define the exact chronological splits, context length, horizons, scaler, and target variables. Preserve standard benchmark settings when comparing published numbers; otherwise call the result a custom streaming benchmark.

Use forecasting, calibration, graph sparsity/stability, and runtime metrics. Do not report SHD or edge F1 against a fabricated ETT graph.

## Forecast horizons

First implement and validate `H = 1`. Then add short horizons such as `H = 12` and `H = 24`, with delayed label revelation handled correctly. Standard long-horizon settings such as 96/192/336/720 are optional and require matching baseline implementations.

## Resource reporting

Record CPU/GPU model, package versions, batch size, thread count, graph-update frequency, context length, number of variables, warm-up handling, mean and percentile latency, peak memory, and total adaptation time.

## Failure analyses

Inspect hidden confounding, wrong lag, graph threshold mismatch, dense true graphs, low signal-to-noise ratio, delayed labels, sudden noise-only changes, normalization leakage, and detector false positives.
