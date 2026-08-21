#!/usr/bin/env python
"""Download official benchmark files with atomic writes and optional Git-blob verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "ETTh1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "filename": "ETTh1.csv",
        "git_blob_sha": "a52c4925778c07c1ef1a2cf6fd01594919717d9e",
        "license": "CC-BY-ND-4.0; distribute only unchanged files with attribution",
    },
    "ETTh2": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
        "filename": "ETTh2.csv",
        "git_blob_sha": "6e3760778598fdee5208213d3fce122637307eec",
        "license": "CC-BY-ND-4.0; distribute only unchanged files with attribution",
    },
    "ETTm1": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
        "filename": "ETTm1.csv",
        "git_blob_sha": "62c9f979e2dc5b05d9cc5b38891e60e6baf77417",
        "license": "CC-BY-ND-4.0; distribute only unchanged files with attribution",
    },
    "ETTm2": {
        "url": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
        "filename": "ETTm2.csv",
        "git_blob_sha": "f9d46027c987d442d2d8f81308d1d8c4d1d35db8",
        "license": "CC-BY-ND-4.0; distribute only unchanged files with attribution",
    },
    "electricity": {
        "url": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz",
        "filename": "electricity.txt.gz",
        "license": "No explicit upstream repository license; download for local research only",
    },
    "traffic": {
        "url": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz",
        "filename": "traffic.txt.gz",
        "license": "No explicit upstream repository license; download for local research only",
    },
    "solar": {
        "url": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz",
        "filename": "solar_AL.txt.gz",
        "license": "No explicit upstream repository license; download for local research only",
    },
    "exchange": {
        "url": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz",
        "filename": "exchange_rate.txt.gz",
        "license": "No explicit upstream repository license; download for local research only",
    },
    "jena_climate": {
        "url": "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
        "filename": "jena_climate_2009_2016.csv.zip",
        "license": "Consult original Max Planck Institute and archive terms before redistribution",
    },
}


def git_blob_sha(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def download(name: str, destination: Path, force: bool = False) -> Path:
    entry = DATASETS[name]
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / entry["filename"]
    if target.exists() and not force:
        print(f"Already present: {target}")
        return target
    request = Request(entry["url"], headers={"User-Agent": "ocg-ssm-thesis/0.1"})
    with urlopen(request, timeout=120) as response:
        payload = response.read()
    expected = entry.get("git_blob_sha")
    if expected and git_blob_sha(payload) != expected:
        raise ValueError(f"Upstream content for {name} does not match its recorded Git blob SHA")
    with tempfile.NamedTemporaryFile(dir=destination, delete=False) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    temporary.replace(target)
    print(f"Downloaded {name}: {len(payload):,} bytes -> {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", choices=tuple(DATASETS) + ("all",))
    parser.add_argument("--output", type=Path, default=ROOT / "LEVEL_03_DATASETS" / "downloads")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list or not args.datasets:
        print(json.dumps(DATASETS, indent=2))
        return
    names = list(DATASETS) if "all" in args.datasets else args.datasets
    for name in names:
        download(name, args.output, args.force)


if __name__ == "__main__":
    main()
