"""Download the Gate 2 / Gate 3 corpora from the PMC Open Access S3 bucket.

The papers are third-party works, so the repo only keeps the manifests.
Usage: uv run python scripts/fetch_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parents[1]
S3 = "https://pmc-oa-opendata.s3.amazonaws.com"


def fetch(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "trialgeist"}), timeout=60).read()
    except Exception as e:  # missing format for this article is normal (XML-only deposits)
        print(f"  skip {url}: {e}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    time.sleep(0.4)
    return True


def main() -> int:
    bad = 0
    neg = ROOT / "tests" / "corpus" / "negative"
    for m in json.loads((neg / "manifest.json").read_text()):
        pid, d = m["pmcid"], neg / m["pmcid"]
        fetch(f"{S3}/{pid}.1/{pid}.1.pdf", d / "paper.pdf")
        fetch(f"{S3}/{pid}.1/{pid}.1.xml", d / "paper.nxml")
        pdf = d / "paper.pdf"
        if m.get("pdf_sha256") and pdf.exists() and hashlib.sha256(pdf.read_bytes()).hexdigest() != m["pdf_sha256"]:
            print(f"  {pid}: sha256 mismatch (PMC may have re-issued the PDF)")
            bad += 1
    ras = ROOT / "tests" / "corpus" / "raster"  # bitmap KM figures (scripts/build_raster_corpus.py)
    if (ras / "manifest.json").exists():
        for m in json.loads((ras / "manifest.json").read_text()):
            pid, d = m["pmcid"], ras / m["pmcid"]
            fetch(m["pdf_url"], d / "paper.pdf")
            fetch(m["xml_url"], d / "paper.nxml")
            pdf = d / "paper.pdf"
            if m.get("pdf_sha256") and pdf.exists() and hashlib.sha256(pdf.read_bytes()).hexdigest() != m["pdf_sha256"]:
                print(f"  {pid}: sha256 mismatch (PMC may have re-issued the PDF)")
                bad += 1
    pos = ROOT / "tests" / "corpus" / "positive"
    controls = json.loads((ROOT / "research" / "positive_controls.json").read_text())["controls"]
    for pid in sorted({c["paper"]["pmcid"] for c in controls if c["paper"].get("pmcid")}):
        fetch(f"{S3}/{pid}.1/{pid}.1.xml", pos / pid / "paper.nxml")
    print("done" + (f", {bad} hash mismatches" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
