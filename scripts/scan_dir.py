"""Scan every PDF in a directory and print a compact tier summary (dev helper)."""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

from trialgeist.scan import scan_pdf

root = Path(sys.argv[1])
pdfs = sorted(root.rglob("*.pdf"))
for pdf in pdfs:
    t = time.time()
    try:
        r = scan_pdf(pdf)
    except Exception as e:  # noqa: BLE001
        print(pdf.parent.name, "ERROR", type(e).__name__, e)
        continue
    print(pdf.parent.name, f"{time.time() - t:.1f}s", r.n_claims, dict(Counter(f.tier.value for f in r.findings)), flush=True)
    for f in r.findings:
        if f.tier.value in ("PROVEN_IMPOSSIBLE", "STATISTICALLY_EXTREME", "REGISTRY_DISCORDANT"):
            p = f.provenance[0] if f.provenance else None
            print("   ", f.tier.value, f.check, f.title[:110], "|", p and (p.page, (p.text or "")[:50], (p.note or "")[:90]))
