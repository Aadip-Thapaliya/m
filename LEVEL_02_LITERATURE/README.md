# Level 02 — Verified literature

Read [`READING_ROADMAP.md`](READING_ROADMAP.md) in order, compare direct neighbors in [`NOVELTY_MATRIX.md`](NOVELTY_MATRIX.md), import [`references.bib`](references.bib) into Zotero or your LaTeX project, and filter [`paper_catalog.csv`](paper_catalog.csv) when you need a machine-readable bibliography.

## Reading priority

**Essential:** Huang et al. 2019; Mamba; NOTEARS; DYNOTEARS; PCMCI; CDT; S-Mamba; CMamba; DTAF; CausalRivers; CausalTime.

**Important:** Mamba-2; Mamba-3; TSMamba; MambaTS; AdaRNN; RevIN; iTransformer; PatchTST; DLinear; nonstationary causal discovery; adaptive conformal inference.

**Advanced:** anomaly-oriented CGT and TECamba; semi-stationary discovery; change-point-specific conformal methods; foundation-scale or 2026 graph-Mamba variants.

## Obtain open papers locally

```bash
python scripts/download_papers.py --level essential --limit 12
python scripts/download_papers.py --level important --limit 20
```

Downloaded papers are intentionally ignored by Git. Publisher access and redistribution rights vary; use the official landing page when no openly accessible PDF is recorded.

The catalog was checked on **2026-08-21**. Recheck the latest work immediately before final submission, particularly because an additional causal-graph-enhanced Mamba publisher result appeared in August 2026.
