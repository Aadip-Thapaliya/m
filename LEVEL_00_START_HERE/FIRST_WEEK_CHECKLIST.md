# First-week execution checklist

- [ ] Confirm supervisor expectations, thesis submission date, required citation style, and acceptable compute budget.
- [ ] Read Huang et al. (ICML 2019), Mamba, DYNOTEARS, CDT, and DTAF before drafting the novelty claim.
- [ ] Run `python -m unittest discover -s tests -v` and `python scripts/run_smoke.py`.
- [ ] Inspect all five committed synthetic datasets and their ground-truth graph metadata.
- [ ] Inspect the four bundled ETT files and read their CC BY-ND 4.0 license.
- [ ] Choose one graph convention: `adjacency[target, parent, lag]`.
- [ ] Decide whether the first thesis version models lagged edges only or also contemporaneous effects.
- [ ] Freeze an initial prequential protocol: training prefix, forecast horizon, update order, and seeds.
- [ ] Reproduce persistence, static VAR, and online VAR before tuning a neural architecture.
- [ ] Define the exact question that would make the graph-conditioned model worth its added complexity.
- [ ] Schedule a supervisor discussion specifically about the ICML 2019 overlap.
