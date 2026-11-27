"""Gate RC: real open-access RCTs whose Kaplan-Meier figures are bitmaps.

Every paper in tests/corpus/raster/manifest.json (built by scripts/build_raster_corpus.py,
retraction-checked) is scanned with the raster digitiser.  No PROVEN_IMPOSSIBLE is
allowed, and a raster finding built on OCR-read numbers at risk must never carry a
certificate.  Reading coverage is reported, not asserted: refusing a figure is sound.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from trialgeist.scan import scan_pdf
from trialgeist.verdict import Tier

CORPUS = Path(__file__).parent / "corpus" / "raster"
MANIFEST = json.loads((CORPUS / "manifest.json").read_text()) if (CORPUS / "manifest.json").exists() else []
pytestmark = pytest.mark.skipif(not shutil.which("tesseract"), reason="raster digitiser needs tesseract")


@pytest.mark.parametrize("entry", MANIFEST, ids=[m["pmcid"] for m in MANIFEST])
def test_raster_corpus_no_false_proofs(entry):
    pdf = CORPUS / entry["pmcid"] / "paper.pdf"
    if not pdf.exists():
        pytest.skip("PDF not downloaded: uv run python scripts/fetch_corpus.py")
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == entry["pdf_sha256"]
    report = scan_pdf(pdf, raster=True)
    proofs = [f for f in report.findings if f.tier is Tier.PROVEN_IMPOSSIBLE]
    assert not proofs, [(f.title, f.detail[:200]) for f in proofs]
    for f in report.findings:
        if f.title.startswith("Lead (OCR-read"):
            assert f.certificate is None
    # coverage, reported (not asserted) in the CI job summary
    raster = [f for f in report.findings if any(p.source == "raster" for p in f.provenance)]
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            if entry is MANIFEST[0]:
                fh.write("| paper | raster KM arms read | tiers |\n|---|---|---|\n")
            fh.write(f"| {entry['pmcid']} | {len(raster)} | {', '.join(sorted({f.tier.name for f in raster})) or '-'} |\n")
