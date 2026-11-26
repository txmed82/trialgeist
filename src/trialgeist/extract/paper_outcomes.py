"""Deterministic extraction of the paper's stated primary outcome(s) and randomized N."""

from __future__ import annotations

import re

import pymupdf

from ..registry.diff import PaperClaims
from ..verdict import Provenance

PRIMARY_RE = re.compile(
    r"(?:the\s+)?primary\s+(?:efficacy\s+)?(?:end[\s-]?point|outcome(?:\s+measure)?)\s+(?:was|were|is|included|consisted of|:)\s+([^.;]{8,300})",
    re.I,
)
RAND_RE = re.compile(
    r"(?:a total of\s+)?(\d{1,3}(?:,\d{3})*|\d+)\s+(?:patients|participants|subjects|adults|children|women|men|infants)?\s*"
    r"(?:were|had been|underwent)\s+(?:randomly assigned|randomi[sz]ed|randomization|randomisation)",
    re.I,
)


def paper_from_text(text: str, page_of=None) -> PaperClaims:
    flat = re.sub(r"-\n(?=[a-z])", "", text)
    flat = re.sub(r"\s+", " ", flat)
    pc = PaperClaims()
    seen = set()
    for m in PRIMARY_RE.finditer(flat):
        s = m.group(1).strip()
        key = s.lower()[:60]
        if key in seen:
            continue
        seen.add(key)
        pc.primary_outcomes.append(s)
        pc.provenance.append(Provenance(source="pdf", text=m.group(0)[:300], note="primary outcome statement"))
        if len(pc.primary_outcomes) >= 3:
            break
    ns = [int(m.group(1).replace(",", "")) for m in RAND_RE.finditer(flat)]
    if ns:
        pc.randomized_n = max(ns)  # the overall count is the largest randomization statement
    return pc


def paper_from_pdf(path: str) -> PaperClaims:
    doc = pymupdf.open(path)
    return paper_from_text("\n".join(p.get_text() for p in doc))
