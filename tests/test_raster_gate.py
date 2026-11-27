"""Gate R (raster): rasterised synthetic KM figures never yield a false proof; edits are caught.

Size via TG_RASTER_N (default 40 figures per side; the full gate uses 200)."""

from __future__ import annotations

import os
import random
import shutil
from collections import Counter

import pymupdf
import pytest

from trialgeist.bench.synthetic import km, render_km, simulate_survival
from trialgeist.claims import check_claim
from trialgeist.extract.raster import find_raster_km, raster_claims
from trialgeist.verdict import Tier

pytestmark = pytest.mark.skipif(not shutil.which("tesseract"), reason="raster KM needs tesseract for axis labels")
N = int(os.environ.get("TG_RASTER_N", "40"))
PALETTES = [
    [(0.8, 0.1, 0.1), (0.1, 0.2, 0.8)],
    [(0, 0, 0), (0.55, 0.55, 0.55)],
    [(0.85, 0.45, 0.0), (0.0, 0.45, 0.7), (0.0, 0.6, 0.3)],
    [(0.1, 0.1, 0.1), (0.8, 0.1, 0.1)],
]


def figure(rnd: random.Random, tmp_path, idx: int, corrupt: bool):
    narms = rnd.choice([2, 2, 3])
    pal = rnd.choice([p for p in PALETTES if len(p) >= narms])
    arms, lam0 = [], rnd.uniform(0.015, 0.05)
    for ai in range(narms):
        n = rnd.randint(20, 60) if corrupt else rnd.randint(25, 300)  # small arms: resolvable drops
        obs = simulate_survival(rnd, n, lam0 * rnd.uniform(0.7, 1.4))
        risk = [(tau, sum(1 for u, _ in obs if u >= tau)) for tau in range(0, 37, 6)]
        cens = sorted({t for t, e in obs if not e and t < 36})[:25] if rnd.random() < 0.7 else []
        arms.append([f"Arm {ai + 1}", km(obs), risk, cens])
    edited = False
    if corrupt:  # remove deaths from the table: fewer leave the risk set than the curve's drops need
        ai = rnd.randrange(narms)
        steps, risk = arms[ai][1], arms[ai][2]
        for m in range(len(risk) - 1):
            t0, t1 = risk[m][0], risk[m + 1][0]
            # as Gate 1's km_risk_table: one fewer patient leaves than the drops drawn need,
            # with drops a raster can resolve: > 2.5% of the axis, about 5 x the digitising tolerance
            inside = [s for s in steps if t0 + 0.5 < s[0] < t1 - 0.5 and s[1] - s[2] > 0.025]
            if len(inside) >= 2 and not any(abs(s[0] - t0) <= 0.5 or abs(s[0] - t1) <= 0.5 for s in steps):
                new = risk[m][1] - len(inside) + 1
                risk[m + 1] = (t1, new)
                for j in range(m + 2, len(risk)):
                    risk[j] = (risk[j][0], min(risk[j][1], new))
                edited = True
                break
    dpi, q, rising = rnd.choice([110, 150, 200, 300]), rnd.choice([60, 75, 85, 95]), rnd.random() < 0.3
    width = rnd.choice([0.6, 1.0, 1.4])
    with pymupdf.open() as src:
        render_km(src.new_page(width=612, height=792), arms, rising=rising, colors=pal, width=width)
        clip = pymupdf.Rect(40, 105, 520, 410)
        jpg = src[0].get_pixmap(dpi=dpi, clip=clip).tobytes("jpeg", jpg_quality=q)
        with pymupdf.open() as out:
            page = out.new_page(width=612, height=792)
            page.insert_text((50, 90), "Figure 2. Kaplan-Meier estimates" + (" of cumulative mortality" if rising else ""), fontsize=10)
            page.insert_image(clip, stream=jpg)
            for x0, y0, x1, y1, w, *_ in src[0].get_text("words"):
                if y0 > clip.y1:
                    page.insert_text((x0, y1 - 2), w, fontsize=9)
            path = tmp_path / f"g{idx}.pdf"
            out.save(path)
    pymupdf.TOOLS.store_shrink(100)
    return path, edited, (dpi, q, rising, width, len(pal), narms)


def run(path):
    plots, notes, _ = find_raster_km(str(path))
    return [f for p in plots for c in raster_claims(p) for f in check_claim(c)], plots, notes


def test_gate_raster_clean_precision(tmp_path):
    rnd = random.Random(2026)
    read, total, bad = 0, 0, []
    why = Counter()
    for i in range(N):
        path, _, cfg = figure(rnd, tmp_path, i, corrupt=False)
        fs, plots, notes = run(path)
        total += 1
        read += bool(plots)
        if not plots:
            why.update(n.split(": ", 1)[-1][:60] for n in notes)
        bad += [(cfg, f.detail[:200]) for f in fs if f.tier is Tier.PROVEN_IMPOSSIBLE]
    print(f"raster clean: {read}/{total} figures digitised; unread because: {dict(why)}")
    assert not bad, bad
    assert read >= 0.6 * total


def test_gate_raster_recall(tmp_path):
    rnd = random.Random(7)
    caught = planted = 0
    for i in range(N):
        path, edited, cfg = figure(rnd, tmp_path, i, corrupt=True)
        if not edited:
            continue
        fs, plots, _ = run(path)
        if not plots:
            continue
        planted += 1
        caught += any(f.tier is Tier.PROVEN_IMPOSSIBLE for f in fs)
    print(f"raster recall: {caught}/{planted} edited at-risk tables caught (among digitised figures)")
    assert planted >= 5 and caught >= 0.5 * planted
