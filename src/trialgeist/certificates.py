"""Standalone certificate scripts.

Each certificate is a self-contained Python file (<80 lines) that re-derives a
contradiction from the *printed strings* with exact rational arithmetic (and
Z3 for bounded-domain multiset problems).  It never imports trialgeist, so a
sceptical reader only has to trust ~30 lines they can read.

Exit status 0 + a line starting ``VERIFIED`` means the contradiction holds.
Exit status 1 means the claim did not reproduce.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

# Closed hull of {half-up (both directions), half-even, half-down, truncation}
# widened by 1e-9 relative slack.  Must stay identical to numbers.proof_interval.
HULL_SRC = '''\
def hull(s, nonneg=False):
    """Closed interval of true values that print as s under half-up,
    half-even, half-down, truncation or cascaded rounding, plus float slack."""
    s = s.strip().replace("\\u00b7", ".").replace("\\u2212", "-").rstrip("%").strip()
    neg = s.startswith("-"); s = s.lstrip("+-")
    d = len(s.split(".")[1]) if "." in s else 0
    v = -F(s) if neg else F(s); h = F(1, 2 * 10**d)
    lo = v - 2 * h if (neg or v == 0) else v - h
    hi = v + h if neg else v + 2 * h
    pad = F(1, 10**9) * (max(abs(lo), abs(hi)) + 2 * h)  # float slack
    lo, hi = lo - h / 9 - pad, hi + h / 9 + pad  # h/9: cascaded rounding (x.x46 -> x.x5 -> x.x)
    return (max(lo, F(0)) if nonneg else lo), hi
'''

HEADER = '''\
#!/usr/bin/env python3
"""TRIALGEIST certificate {cid}  [{check}]
Claim: {claim}
Inputs as printed: {inputs}
Assumptions: {assumptions}
Run `python3 {fname}`: exit 0 and VERIFIED = contradiction reproduced."""
'''


def render(cid: str, check: str, claim: str, inputs: dict, assumptions: list[str], body: str) -> str:
    head = HEADER.format(
        cid=cid,
        check=check,
        claim=claim.replace('"""', "'''"),
        inputs=json.dumps(inputs, ensure_ascii=False, default=str),
        assumptions="; ".join(assumptions) if assumptions else "none beyond the printed values",
        fname=f"certificate_{cid}.py",
    )
    return head + body


def run_certificate(source: str, workdir: Path, cid: str = "x", timeout: float = 10.0) -> tuple[bool, float, str]:
    """Execute a certificate in a fresh interpreter. Returns (verified, seconds, stdout)."""
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"certificate_{cid}.py"
    path.write_text(source)
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=timeout)
    dt = time.perf_counter() - t0
    out = proc.stdout + proc.stderr
    return proc.returncode == 0 and "VERIFIED" in proc.stdout, dt, out


def line_count(source: str) -> int:
    return len(source.rstrip("\n").splitlines())
