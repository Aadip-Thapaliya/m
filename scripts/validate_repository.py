#!/usr/bin/env python
"""Check that research levels, citations, licenses, and sample data are usable."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEVELS = (
    "LEVEL_00_START_HERE",
    "LEVEL_01_FOUNDATIONS",
    "LEVEL_02_LITERATURE",
    "LEVEL_03_DATASETS",
    "LEVEL_04_CAUSAL_DISCOVERY",
    "LEVEL_05_MODEL_ARCHITECTURE",
    "LEVEL_06_ADAPTATION_UNCERTAINTY",
    "LEVEL_07_EXPERIMENTS",
    "LEVEL_08_IMPLEMENTATION",
    "LEVEL_09_THESIS_WRITING",
)


def main() -> None:
    errors = []
    for level in LEVELS:
        if not (ROOT / level / "README.md").is_file():
            errors.append(f"Missing level guide: {level}/README.md")
    required = (
        "README.md",
        "LEVEL_00_START_HERE/thesis_proposal.pdf",
        "LEVEL_02_LITERATURE/references.bib",
        "LEVEL_02_LITERATURE/paper_catalog.csv",
        "LEVEL_03_DATASETS/DATASET_CATALOG.csv",
        "LEVEL_03_DATASETS/real/ETT/ATTRIBUTION.md",
        "LEVEL_03_DATASETS/real/ETT/LICENSE",
        "configs/smoke.yaml",
    )
    for path in required:
        if not (ROOT / path).is_file():
            errors.append(f"Missing required file: {path}")

    paper_catalog = ROOT / "LEVEL_02_LITERATURE" / "paper_catalog.csv"
    if paper_catalog.exists():
        with paper_catalog.open(newline="", encoding="utf-8") as stream:
            papers = list(csv.DictReader(stream))
        if len(papers) < 20:
            errors.append(f"Expected at least 20 papers, found {len(papers)}")
    else:
        papers = []

    datasets = ROOT / "LEVEL_03_DATASETS" / "synthetic"
    scenarios = [path.name for path in datasets.iterdir() if path.is_dir()] if datasets.exists() else []
    if len(scenarios) < 5:
        errors.append(f"Expected five committed synthetic scenarios, found {len(scenarios)}")

    if errors:
        raise SystemExit("Repository validation failed:\n- " + "\n- ".join(errors))
    print(json.dumps({"levels": len(LEVELS), "papers": len(papers), "synthetic_scenarios": scenarios}, indent=2))


if __name__ == "__main__":
    main()
