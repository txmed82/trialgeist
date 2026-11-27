"""Gate 3: clean high-impact negative controls, end to end.

Every paper in tests/corpus/negative/manifest.json (open-access RCTs from PMC)
is scanned from its PDF and, when present, its JATS XML.  Zero
PROVEN_IMPOSSIBLE findings are allowed.  STATISTICALLY_EXTREME and
REGISTRY_DISCORDANT are reported but are not accusations.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from trialgeist.scan import scan_pdf, scan_xml
from trialgeist.verdict import Tier

CORPUS = Path(__file__).parent / "corpus" / "negative"
MANIFEST = json.loads((CORPUS / "manifest.json").read_text()) if (CORPUS / "manifest.json").exists() else []


def test_gate3_corpus_size():
    present = [m for m in MANIFEST if (CORPUS / m["pmcid"] / "paper.pdf").exists()]
    if not present:
        pytest.skip("corpus not downloaded: uv run python scripts/fetch_corpus.py")
    assert len(present) >= 20, f"only {len(present)} negative-control PDFs present"


@pytest.mark.parametrize("entry", MANIFEST, ids=[m["pmcid"] for m in MANIFEST])
def test_gate3_no_false_proofs(entry):
    d = CORPUS / entry["pmcid"]
    pdf = d / "paper.pdf"
    if not pdf.exists():
        pytest.skip("PDF not downloaded")
    if entry.get("pdf_sha256"):
        assert hashlib.sha256(pdf.read_bytes()).hexdigest() == entry["pdf_sha256"]
    findings = list(scan_pdf(pdf).findings)
    if (d / "paper.nxml").exists():
        findings += scan_xml(d / "paper.nxml").findings
    tiers = Counter(f.tier.value for f in findings)
    print(entry["pmcid"], dict(tiers))
    assert tiers[Tier.CONSISTENT.value] > 0, "extraction found nothing to check"
    # A proof on a clean paper is acceptable only if it is a hand-verified true error in the published
    # table (recorded with its justification in the manifest); everything else is a false alarm.
    known = [k["printed"] for k in entry.get("known_true_impossibilities", [])]
    bad = [(f.title, f.provenance[0].text if f.provenance else None) for f in findings if f.tier is Tier.PROVEN_IMPOSSIBLE
           and not any(k in (p.text or "") for k in known for p in f.provenance)]
    assert not bad, bad
