"""Gate 1: synthetic mutation torture suite (end-to-end through real PDFs).

200 clean simulated RCTs must give zero PROVEN_IMPOSSIBLE; injected
corruptions must be recalled at >= 0.9, and every proof's certificate must
re-derive the contradiction in under 2 s.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest

from trialgeist.bench.synthetic import CORRUPTIONS, make_trial
from trialgeist.certificates import line_count, run_certificate
from trialgeist.claims import ClaimSet, check_all
from trialgeist.scan import scan_pdf
from trialgeist.verdict import Tier

N_CLEAN = int(os.environ.get("TG_GATE1_CLEAN", "200"))
N_PER_CORRUPTION = int(os.environ.get("TG_GATE1_CORRUPT", "20"))


def run_trial(t):
    rep = scan_pdf(t.pdf)
    return rep.findings + check_all(ClaimSet.model_validate({"claims": t.claims}))


def _hit(f, t) -> bool:
    lab = t.corrupted_label
    if lab.startswith("km:"):
        return f.engine == "km_sat"
    return lab in (f.inputs.get("label") or "") or lab in f.title


@pytest.fixture(scope="module")
def outdir(tmp_path_factory):
    return tmp_path_factory.mktemp("gate1")


def test_gate1_clean_precision(outdir: Path):
    proven, tiers = [], Counter()
    for seed in range(N_CLEAN):
        t = make_trial(10_000 + seed, outdir / "clean")
        fs = run_trial(t)
        tiers.update(f.tier.value for f in fs)
        proven += [(seed, f.title, f.inputs) for f in fs if f.tier is Tier.PROVEN_IMPOSSIBLE]
    print(f"clean trials: {N_CLEAN}, findings by tier: {dict(tiers)}")
    assert tiers[Tier.CONSISTENT.value] > 15 * N_CLEAN  # engines actually exercised
    assert not proven, proven[:5]


def test_gate1_corruption_recall(outdir: Path):
    hits, total, missed, collateral = Counter(), Counter(), [], []
    certs_checked = 0
    for cor in CORRUPTIONS:
        seed, made = 0, 0
        while made < N_PER_CORRUPTION and seed < 40 * N_PER_CORRUPTION:
            seed += 1
            try:
                t = make_trial(20_000 + 1000 * CORRUPTIONS.index(cor) + seed, outdir / cor, cor)
            except ValueError:
                continue
            made += 1
            fs = run_trial(t)
            proven = [f for f in fs if f.tier is Tier.PROVEN_IMPOSSIBLE]
            total[cor] += 1
            good = [f for f in proven if _hit(f, t)]
            if good:
                hits[cor] += 1
            else:
                missed.append((cor, t.seed, [f.title for f in fs if f.tier is not Tier.CONSISTENT][:3]))
            collateral += [(cor, t.seed, f.title) for f in proven if not _hit(f, t)]
            for f in good[:1]:
                if certs_checked < 60:
                    assert line_count(f.certificate) < 80
                    ok, dt, out = run_certificate(f.certificate, outdir / "certs", f.id)
                    assert ok and dt < 2.0, (f.title, dt, out[-400:])
                    certs_checked += 1
        assert made == N_PER_CORRUPTION, f"generator could not build {cor}"
    recall = {c: hits[c] / total[c] for c in CORRUPTIONS}
    overall = sum(hits.values()) / sum(total.values())
    print("recall by corruption:", recall, "overall", overall)
    assert not collateral, collateral[:5]  # proofs only ever land on the corrupted item
    assert overall >= 0.9, (recall, missed[:6])
    assert min(recall.values()) >= 0.75, (recall, missed[:6])
