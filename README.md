# OCG-SSM Thesis Research Hub

**Online Causal-Graph-Conditioned State Space Models for Non-Stationary Multivariate Time Series Forecasting**

Author: **Aadip Thapaliya** · Bachelor thesis · University of Europe for Applied Sciences

This repository organizes the complete thesis workflow into ten progressive levels: prerequisites, verified literature, real and synthetic datasets, causal discovery, model design, adaptation, evaluation, implementation, and thesis writing. It also contains a CPU-runnable research scaffold, four unmodified official ETT datasets, five generated causal-regime datasets, and the original thesis proposal.

> **Critical novelty correction:** Huang et al. already published *Causal Discovery and Forecasting in Nonstationary Environments with State-Space Models* at ICML 2019. Recent work also combines causal graph learning and Mamba-style architectures. Do **not** claim to be the first causal state-space forecaster. The defensible contribution is a carefully specified combination of **online lagged graph tracking, explicit graph-conditioned selective state transitions, regime adaptation, uncertainty evaluation, and leakage-safe streaming benchmarks**.

## Research levels

| Level | Directory | What it gives you |
| --- | --- | --- |
| 00 | [`LEVEL_00_START_HERE`](LEVEL_00_START_HERE/) | Original proposal, proposal audit, corrected research questions, first-week checklist |
| 01 | [`LEVEL_01_FOUNDATIONS`](LEVEL_01_FOUNDATIONS/) | Causal inference, time-series graphs, state-space models, online learning, notation |
| 02 | [`LEVEL_02_LITERATURE`](LEVEL_02_LITERATURE/) | Verified paper catalog, annotated reading roadmap, novelty matrix, BibTeX bibliography |
| 03 | [`LEVEL_03_DATASETS`](LEVEL_03_DATASETS/) | Four real ETT datasets, five synthetic datasets, dataset cards, download tools, licensing |
| 04 | [`LEVEL_04_CAUSAL_DISCOVERY`](LEVEL_04_CAUSAL_DISCOVERY/) | Streaming RLS, identifiability assumptions, lagged versus contemporaneous graphs |
| 05 | [`LEVEL_05_MODEL_ARCHITECTURE`](LEVEL_05_MODEL_ARCHITECTURE/) | Graph-conditioned selective SSM design, complexity accounting, PyTorch reference |
| 06 | [`LEVEL_06_ADAPTATION_UNCERTAINTY`](LEVEL_06_ADAPTATION_UNCERTAINTY/) | Change detection, recurring-regime memory, probabilistic and conformal extensions |
| 07 | [`LEVEL_07_EXPERIMENTS`](LEVEL_07_EXPERIMENTS/) | Benchmark protocols, baselines, metrics, ablations, statistics, reproducibility |
| 08 | [`LEVEL_08_IMPLEMENTATION`](LEVEL_08_IMPLEMENTATION/) | Code map, installation, commands, milestone-by-milestone engineering plan |
| 09 | [`LEVEL_09_THESIS_WRITING`](LEVEL_09_THESIS_WRITING/) | Chapter outline, writing templates, claim-safety checklist, timeline, defense preparation |

## Start in five minutes

```bash
git clone https://github.com/Aadip-Thapaliya/m.git
cd m
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_smoke.py
```

Generate or refresh all synthetic datasets:

```bash
python scripts/generate_synthetic.py --all
python scripts/run_benchmark.py --quick
python scripts/run_ett_baselines.py --dataset ETTh1 --max-rows 4000
python scripts/download_datasets.py --list
python scripts/download_papers.py --level essential --limit 8
```

For Tigramite, causal-learn, River, and the PyTorch selective-SSM prototype:

```bash
python -m pip install -e '.[research]'
```

## Data already included

| Dataset | Domain | Observations | Variables | Ground-truth causal graph |
| --- | --- | ---: | ---: | --- |
| ETTh1 | Electricity transformer, hourly | 17,420 | 7 | No |
| ETTh2 | Electricity transformer, hourly | 17,420 | 7 | No |
| ETTm1 | Electricity transformer, 15-minute | 69,680 | 7 | No |
| ETTm2 | Electricity transformer, 15-minute | 69,680 | 7 | No |
| Synthetic stationary | Controlled SVAR | 1,200 | 6 | Yes, each time step |
| Synthetic abrupt | Abrupt causal shifts | 1,200 | 6 | Yes, each time step |
| Synthetic gradual | Smooth causal drift | 1,200 | 6 | Yes, each time step |
| Synthetic recurring | Returning causal regimes | 1,200 | 6 | Yes, each time step |
| Synthetic confounded | Hidden-common-cause stress test | 1,200 | 6 | Observed lagged graph only |

ETT files are included **unchanged** under their original **CC BY-ND 4.0** license. See [`LEVEL_03_DATASETS/real/ETT/ATTRIBUTION.md`](LEVEL_03_DATASETS/real/ETT/ATTRIBUTION.md). Other electricity, traffic, solar, exchange-rate, and Jena climate datasets can be downloaded locally using `scripts/download_datasets.py`; they are not redistributed without clear permission.

## What is implemented now

- Reproducible regime-switching structural VAR generator with pointwise graph ground truth.
- Shared-covariance recursive least squares for a lag-one directed predictive graph.
- `O(D²)` online graph updates for fixed lag and a stable graph-conditioned NumPy SSM prototype.
- Page-Hinkley drift detection and bounded recurring-regime memory.
- Optional PyTorch graph-conditioned diagonal selective SSM reference model.
- Persistence, static VAR, online VAR, and graph-conditioned SSM baselines.
- Real ETT readers, training-prefix-only standardization, and a one-step ETT streaming baseline runner.
- Strict predict → reveal → update streaming evaluation; MSE, MAE, Gaussian CRPS, SHD, precision, recall, F1, regret, and recovery-time helpers.
- Unit tests, reproducible YAML configurations, a smoke run, benchmark runner, and GitHub Actions.

The portable implementation is a **working research baseline**, not evidence that the proposed thesis method has already been experimentally validated. The PyTorch model is a readable sequential reference, not a fused CUDA Mamba kernel.

## Three rules for scientifically sound results

1. A lagged causal graph can contain feedback cycles in its collapsed representation. Apply a DAG constraint only to a separately identified **contemporaneous** graph.
2. Forecast before revealing the target, update only after it becomes available, and fit normalization parameters on past data alone.
3. Report graph-recovery metrics only when justified ground truth exists. ETT, electricity, weather, and traffic observations alone do not provide a verified causal graph.

## Key sources

- Huang et al., ICML 2019: https://proceedings.mlr.press/v97/huang19g.html
- Gu and Dao, Mamba: https://arxiv.org/abs/2312.00752
- CDT: https://arxiv.org/abs/2505.16308
- DTAF and official code: https://arxiv.org/abs/2511.08229 · https://github.com/decisionintelligence/DTAF
- Official ETT data: https://github.com/zhouhaoyi/ETDataset
- CausalRivers: https://arxiv.org/abs/2503.17452 · https://github.com/causalrivers/causalrivers

See [`LEVEL_02_LITERATURE/references.bib`](LEVEL_02_LITERATURE/references.bib) and [`LEVEL_02_LITERATURE/paper_catalog.csv`](LEVEL_02_LITERATURE/paper_catalog.csv) for the full verified reading list.

## Licensing

Original code and documentation: MIT. Bundled third-party data retain their own licenses. Research papers remain at their official source; the paper downloader retrieves openly available copies for local research and does not grant redistribution rights.
