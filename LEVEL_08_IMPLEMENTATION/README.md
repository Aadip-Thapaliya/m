# Level 08 — Implementation roadmap

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/run_smoke.py
```

Install the heavier optional stack only when needed:

```bash
python -m pip install -e '.[research]'
```

## Code map

| Module | Responsibility |
| --- | --- |
| `src/ocg_ssm/data/synthetic.py` | Controlled stationary, abrupt, gradual, recurring, and confounded regimes |
| `src/ocg_ssm/data/windows.py` | Chronological splits and leakage-safe windows |
| `src/ocg_ssm/data/ett.py` | Official ETT reader and training-prefix-only standardization |
| `src/ocg_ssm/discovery/online_var.py` | Shared-covariance lag-one recursive least squares |
| `src/ocg_ssm/discovery/graph.py` | Graph thresholding, density, and explicit contemporaneous ordering |
| `src/ocg_ssm/adaptation/drift.py` | Page-Hinkley change detection |
| `src/ocg_ssm/adaptation/memory.py` | Bounded graph-prototype memory |
| `src/ocg_ssm/models/baselines.py` | Persistence, static VAR, and online VAR |
| `src/ocg_ssm/models/portable.py` | Executable NumPy graph-conditioned state-space prototype |
| `src/ocg_ssm/models/torch_model.py` | Optional graph-conditioned PyTorch selective SSM reference |
| `src/ocg_ssm/evaluation/metrics.py` | Forecast, graph, uncertainty, regret, and recovery metrics |
| `src/ocg_ssm/evaluation/prequential.py` | Strict predict/reveal/update loop |
| `src/ocg_ssm/experiment.py` | End-to-end synthetic experiment orchestration |

Read [`ENGINEERING_MILESTONES.md`](ENGINEERING_MILESTONES.md) before expanding the code.
