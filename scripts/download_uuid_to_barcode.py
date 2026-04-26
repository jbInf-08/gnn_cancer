"""
Build a TCGA case UUID <-> submitter_id (barcode) mapping via the GDC REST API.
Writes `data/metadata/uuid_to_barcode.csv` (gitignored) by default.

Usage:
  python scripts/download_uuid_to_barcode.py --project TCGA-BRCA
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

GDC_API = "https://api.gdc.cancer.gov/cases"

DEFAULT_OUT = Path("data/metadata/uuid_to_barcode.csv")


def fetch_cases(project: str, page_size: int = 1000) -> list[tuple[str, str]]:
    """Return (case_id uuid, submitter_id) for all cases in a project."""
    out: list[tuple[str, str]] = []
    offset = 0
    fields = "case_id,submitter_id"
    flt = {
        "op": "in",
        "content": {
            "field": "projects.project_id",
            "value": [project],
        },
    }
    while True:
        params = {
            "filters": json.dumps(flt),
            "fields": fields,
            "format": "json",
            "size": str(page_size),
            "from": str(offset),
        }
        url = f"{GDC_API}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=120) as r:
            payload = json.loads(r.read().decode())
        hits = payload.get("data", {}).get("hits", [])
        if not hits:
            break
        for h in hits:
            cid = h.get("case_id")
            sid = h.get("submitter_id")
            if cid and sid:
                out.append((str(cid).lower(), str(sid)))
        if len(hits) < page_size:
            break
        offset += page_size
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default="TCGA-BRCA", help="GDC project_id e.g. TCGA-BRCA")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT, help="Output CSV path")
    args = p.parse_args()
    try:
        pairs = fetch_cases(args.project)
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print("error: GDC request failed:", e, file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uuid", "barcode"])
        w.writerows(pairs)
    print(f"Wrote {len(pairs)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
