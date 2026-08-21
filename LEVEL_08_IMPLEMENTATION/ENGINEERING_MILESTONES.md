# Engineering milestones

## Milestone 1: reliable baseline

Run tests, regenerate the five synthetic datasets, execute the smoke experiment, inspect graph conventions, and reproduce persistence/static VAR/online VAR results over five seeds.

Exit criterion: the prequential-order test fails if prediction and update order are swapped; RLS recovers a simple known graph.

## Milestone 2: discovery evaluation

Record graph F1 and SHD over time rather than only at the final step. Add graph snapshots around every known regime change. Sweep forgetting factors and sparsity thresholds.

Exit criterion: abrupt graph changes produce measurable recovery curves and meaningful stationary negative controls.

## Milestone 3: neural graph-conditioned SSM

Install PyTorch; create a proper training dataset; train the selective-SSM reference on a chronological prefix; compare a matched graph-free variant; save deterministic configurations and checkpoints locally.

Exit criterion: model forward shape, stability, loss decrease, and update timing are independently tested.

## Milestone 4: real ETT forecasting

Implement a reader for all four unchanged ETT files; create past-only scalers and windows; establish ETTh1/ETTh2 one-step and short-horizon protocols; add ETTm1/ETTm2 when memory allows.

Exit criterion: every model uses identical splits, scaling, context, horizon, and target-reveal order.

## Milestone 5: adaptation and uncertainty

Compare Page-Hinkley to ADWIN; implement a recurring-graph memory ablation; optionally add Gaussian output and adaptive conformal intervals.

Exit criterion: change alarms, graph updates, forecasting recovery, and empirical coverage are plotted on the same time axis.

## Milestone 6: thesis-grade reporting

Run paired seeds, produce summary tables and failure cases, document limitations, cite upstream datasets and papers, and inspect novelty against current literature one final time.

Exit criterion: every central claim maps to a figure, table, reproducible command, and clearly named dataset.
