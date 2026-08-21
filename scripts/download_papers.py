#!/usr/bin/env python
"""Download openly available PDFs individually; never redistribute restricted papers."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("essential", "important", "advanced", "all"), default="essential")
    parser.add_argument("--output", type=Path, default=ROOT / "LEVEL_02_LITERATURE" / "downloaded_papers")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (ROOT / "LEVEL_02_LITERATURE" / "paper_catalog.csv").open(newline="", encoding="utf-8") as stream:
        papers = list(csv.DictReader(stream))

    count = 0
    for paper in papers:
        if count >= args.limit:
            break
        if args.level != "all" and paper["priority"] != args.level:
            continue
        pdf_url = paper.get("open_pdf_url", "")
        if not pdf_url:
            print(f"No verified open PDF; see publisher page: {paper['title']}")
            continue
        output = args.output / f"{paper['id']}.pdf"
        if output.exists():
            print(f"Already present: {output.name}")
            continue
        try:
            request = Request(pdf_url, headers={"User-Agent": "ocg-ssm-thesis/0.1 academic research"})
            with urlopen(request, timeout=90) as response:
                payload = response.read()
            if not payload.startswith(b"%PDF"):
                raise ValueError("The response did not contain a PDF document")
            output.write_bytes(payload)
            print(f"Downloaded {output.name}: {len(payload):,} bytes")
            count += 1
            time.sleep(1.0)
        except (HTTPError, URLError, ValueError) as exc:
            print(f"Could not download {paper['id']}: {exc}")


if __name__ == "__main__":
    main()
