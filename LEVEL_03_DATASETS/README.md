# Level 03 — Datasets and ground truth

This level contains real-world data, directly runnable synthetic causal datasets, provenance, licensing, and download instructions. Read [`DATASET_SELECTION.md`](DATASET_SELECTION.md), [`PREPROCESSING_PROTOCOL.md`](PREPROCESSING_PROTOCOL.md), and the machine-readable [`DATASET_CATALOG.csv`](DATASET_CATALOG.csv).

## Included real-world datasets

The directory [`real/ETT/`](real/ETT/) contains the **complete and unmodified** official ETT-small files:

| File | Frequency | Rows | Observed variables | Source Git blob SHA |
| --- | --- | ---: | ---: | --- |
| `ETTh1.csv` | Hourly | 17,420 | 7 | `a52c4925778c07c1ef1a2cf6fd01594919717d9e` |
| `ETTh2.csv` | Hourly | 17,420 | 7 | `6e3760778598fdee5208213d3fce122637307eec` |
| `ETTm1.csv` | 15 minutes | 69,680 | 7 | `62c9f979e2dc5b05d9cc5b38891e60e6baf77417` |
| `ETTm2.csv` | 15 minutes | 69,680 | 7 | `f9d46027c987d442d2d8f81308d1d8c4d1d35db8` |

Columns are `date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT`. No verified causal graph or true regime-change label accompanies these observations.

## Included synthetic datasets

[`synthetic/`](synthetic/) contains five 1,200-row, six-variable datasets. Each scenario includes:

- `observations.csv`: indexed multivariate observations and regime identifiers.
- `ground_truth_graphs.npz`: exact lag-one adjacency at every time index.
- `metadata.json`: change points, graph convention, seed, and representative regime graphs.

The scenarios are `stationary`, `abrupt`, `gradual`, `recurring`, and `confounded`.

```bash
python scripts/generate_synthetic.py --all --steps 4000 --variables 12 --seed 123
```

## Additional local-only downloads

```bash
python scripts/download_datasets.py --list
python scripts/download_datasets.py electricity traffic solar exchange jena_climate
```

Additional electricity, traffic, solar, and exchange-rate archives are linked to their source rather than redistributed because the upstream repository does not expose an explicit license. Jena climate redistribution also requires checking its original terms.

## Better real-world graph benchmarks

- CausalRivers: https://github.com/causalrivers/causalrivers
- CausalTime paper: https://arxiv.org/abs/2310.01753
- Tigramite examples: https://github.com/jakobrunge/tigramite
- Caltrans PeMS provenance: https://dot.ca.gov/programs/traffic-operations/mpr/pems-source

Use these only after confirming access terms, graph-label meaning, and compatibility with the thesis forecasting protocol.
