# Level 00 — Start here

## Objective

Transform the attached thesis proposal into a defensible, reproducible Bachelor thesis with a realistic implementation scope.

## Read in this order

1. [`thesis_proposal.pdf`](thesis_proposal.pdf): your original proposal.
2. [`PROPOSAL_AUDIT.md`](PROPOSAL_AUDIT.md): corrections to novelty, terminology, methodology, complexity, and baselines.
3. [`RESEARCH_QUESTIONS.md`](RESEARCH_QUESTIONS.md): research questions that can actually be answered with the available data.
4. [`FIRST_WEEK_CHECKLIST.md`](FIRST_WEEK_CHECKLIST.md): concrete setup and reading tasks.

## Recommended minimum viable thesis

**Question:** Does online tracking of a sparse lagged dependency graph improve multivariate one-step and short-horizon forecasting after controlled causal regime changes relative to static VAR, online VAR, and graph-agnostic state-space baselines?

**Minimum deliverables:** one synthetic generator with exact regime graphs; two real-world ETT datasets; three honest baselines; one graph-conditioned state-space implementation; prequential forecasting metrics; synthetic graph-recovery metrics; a graph-noise ablation; and a clear limitations discussion.

Do not promise formal identification, uncertainty calibration, GPU-scale throughput, and state-of-the-art results simultaneously unless each claim is independently demonstrated.
