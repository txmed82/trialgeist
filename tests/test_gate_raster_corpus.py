"""Gate RC: real open-access RCTs whose Kaplan-Meier figures are bitmaps.

Every paper in tests/corpus/raster/manifest.json (built by scripts/build_raster_corpus.py,
retraction-checked) is scanned with the raster digitiser.  No PROVEN_IMPOSSIBLE is
allowed except the reporting errors a person has checked against the figure and recorded
in the manifest (``verified_errors``): those must still be found, and their certificates
must verify.  A raster finding built on OCR-read numbers at risk never carries a
certificate.  Reading coverage is reported, not asserted: refusing a figure is sound.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from trialgeist.certificates import run_certificate
from trialgeist.scan import scan_pdf
from trialgeist.verdict import Tier

CORPUS = Path(__file__).parent / "corpus" / "raster"
MANIFEST = json.loads((CORPUS / "manifest.json").read_text()) if (CORPUS / "manifest.json").exists() else []
pytestmark = pytest.mark.skipif(not shutil.which("tesseract"), reason="raster digitiser needs tesseract")


def test_raster_corpus_manifest_present():
    if not MANIFEST:
        pytest.skip("no raster manifest: uv run python scripts/build_raster_corpus.py")


@pytest.mark.parametrize("entry", MANIFEST, ids=[m["pmcid"] for m in MANIFEST])
def test_raster_corpus_no_false_proofs(entry, capsys, tmp_path):
    pdf = CORPUS / entry["pmcid"] / "paper.pdf"
    if not pdf.exists():
        pytest.skip("PDF not downloaded: uv run python scripts/fetch_corpus.py")
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == entry["pdf_sha256"]
    report = scan_pdf(pdf, raster=True)
    proofs = [f for f in report.findings if f.tier is Tier.PROVEN_IMPOSSIBLE]
    known = entry.get("verified_errors", [])

    def matches(f, k):
        # one panel feeds several KM engines (staircase KM-SAT, checkpoint, joint checkpoint):
        # any of them may prove the recorded error; every such proof must still verify below
        family = k.get("checks", [k["check"]])
        return f.check in family and any(p.page == k["page"] - 1 for p in f.provenance)

    unexpected = [f for f in proofs if not any(matches(f, k) for k in known)]
    assert not unexpected, [(f.title, f.detail[:200]) for f in unexpected]
    for k in known:
        hits = [f for f in proofs if matches(f, k)]
        assert hits, f"verified error no longer found: {k['figure']}"
        for f in hits:
            assert f.certificate is not None
            ok, _, out = run_certificate(f.certificate, tmp_path, f.id)
            assert ok, out
    for f in report.findings:
        if f.title.startswith("Lead (OCR-read"):
            assert f.certificate is None
    # coverage, reported (not asserted) in the CI job summary
    raster = [f for f in report.findings if any(p.source == "raster" for p in f.provenance)]
    row = f"| {entry['pmcid']} | {len(raster)} | {', '.join(sorted({f.tier.name for f in raster})) or '-'} |\n"
    with capsys.disabled():  # also in the job log, which (unlike the summary) is readable via the API
        print(f"\nraster coverage {row.strip()}")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            if entry is MANIFEST[0]:
                fh.write("| paper | raster KM arms read | tiers |\n|---|---|---|\n")
            fh.write(row)
