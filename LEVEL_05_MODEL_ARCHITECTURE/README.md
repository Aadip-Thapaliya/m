# Level 05 — Graph-conditioned state-space architecture

Read [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`COMPLEXITY_AND_STABILITY.md`](COMPLEXITY_AND_STABILITY.md), then inspect both reference implementations:

- CPU-runnable NumPy baseline: [`src/ocg_ssm/models/portable.py`](../src/ocg_ssm/models/portable.py).
- Optional PyTorch selective-SSM reference: [`src/ocg_ssm/models/torch_model.py`](../src/ocg_ssm/models/torch_model.py).

The PyTorch reference is deliberately readable and sequential. It is **not** the official fused Mamba selective-scan kernel and must not be described as such in the thesis.

## Suggested ablations

Graph-conditioned input only; graph-conditioned transition only; graph-conditioned readout only; all three; frozen graph; shuffled graph; empty graph; oracle graph; and graph-agnostic state-space model with matched parameter count.
