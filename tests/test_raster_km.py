"""Raster KM digitiser: synthetic figures rasterised at several resolutions and JPEG qualities."""

from __future__ import annotations

import shutil

import pymupdf
import pytest

from trialgeist.bench.synthetic import COLORS, render_km
from trialgeist.claims import check_claim
from trialgeist.extract.raster import find_raster_km, raster_claims
from trialgeist.verdict import Tier

from test_km_rising import km_from_times, risk

pytestmark = pytest.mark.skipif(not shutil.which("tesseract"), reason="raster KM needs tesseract for axis labels")

ARMS = [("Drug", 40, [2, 3, 5, 8, 8, 13, 17, 22, 29]), ("Placebo", 38, [1, 2, 4, 4, 7, 9, 11, 15, 20, 25, 31])]
PLOT = pymupdf.Rect(40, 105, 520, 410)  # axes, tick labels and axis title of render_km


def _band_edges(page, arms, colors):
    """A pale stepped confidence band per arm (lower and upper edge, with error-bar ticks),
    in a tint of the arm colour, as GraphPad / survminer draw them."""
    L, R, T, B = 90.0, 500.0, 120.0, 380.0
    X = lambda t: L + (R - L) * float(t) / 36
    Y = lambda s: B - (B - T) * min(1.0, max(0.0, float(s)))
    for (name, steps, _, _), col in zip(arms, colors):
        tint = tuple(1 - 0.35 * (1 - v) for v in col)
        for sign in (-1, 1):
            pts, lvl = [(X(0), Y(1))], 1.0
            for k, (t, b, a) in enumerate(steps):
                w = 0.04 + 0.01 * k  # the band widens with time
                pts += [(X(t), Y(float(b) + sign * w)), (X(t), Y(float(a) + sign * (w + 0.01)))]
                page.draw_line((X(t) + 4, Y(float(a) + sign * (w + 0.01)) - 3), (X(t) + 4, Y(float(a) + sign * (w + 0.01)) + 3), color=tint, width=0.8)
            page.draw_polyline(pts, color=tint, width=1.2)


def rasterised(tmp_path, dpi=200, quality=85, rising=False, edit=None, table_in_image=False, bands=False):
    taus = list(range(0, 37, 6))
    arms = []
    for name, n, deaths in ARMS:
        rk = risk(n, deaths, taus)
        if edit and name == "Drug":
            rk = [(t, k + edit if t == 12 else k) for t, k in rk]
        arms.append((name, km_from_times(n, deaths), rk, [36]))
    with pymupdf.open() as src:
        pg = src.new_page(width=612, height=792)
        if bands:
            _band_edges(pg, arms, COLORS)
        render_km(pg, arms, rising=rising)
        clip = pymupdf.Rect(PLOT.x0, 80, PLOT.x1, 520) if table_in_image else PLOT
        pix = src[0].get_pixmap(dpi=dpi, clip=clip)
        jpg = pix.tobytes("jpeg", jpg_quality=quality)
        with pymupdf.open() as out:
            page = out.new_page(width=612, height=792)
            page.insert_text((50, 90), "Figure 2. Kaplan-Meier estimates" + (" of cumulative mortality" if rising else " of overall survival"), fontsize=10)
            page.insert_image(clip, stream=jpg)
            if not table_in_image:  # keep the numbers-at-risk table as text, as most journals print it
                text_page = src[0]
                for x0, y0, x1, y1, w, *_ in text_page.get_text("words"):
                    if y0 > PLOT.y1:
                        page.insert_text((x0, y1 - 2), w, fontsize=9)
            path = tmp_path / f"r_{dpi}_{quality}_{rising}_{edit}_{table_in_image}_{bands}.pdf"
            out.save(path)
    return path


def tiers(path):
    plots, notes, _ = find_raster_km(str(path))
    return [f for p in plots for c in raster_claims(p) for f in check_claim(c)], plots, notes


@pytest.mark.parametrize("dpi,quality", [(150, 75), (200, 85), (300, 95)])
@pytest.mark.parametrize("rising", [False, True])
def test_clean_raster_never_proven(tmp_path, dpi, quality, rising):
    fs, plots, notes = tiers(rasterised(tmp_path, dpi, quality, rising))
    assert plots, notes
    assert sum(len(p.traces) for p in plots) == 2, [len(p.traces) for p in plots]
    assert fs and all(f.tier is Tier.CONSISTENT for f in fs), [f.detail for f in fs if f.tier is not Tier.CONSISTENT]


@pytest.mark.parametrize("dpi", [150, 300])
def test_raster_detects_risk_table_edit(tmp_path, dpi):
    fs, plots, notes = tiers(rasterised(tmp_path, dpi, 85, edit=4))
    assert plots, notes
    proven = [f for f in fs if f.tier is Tier.PROVEN_IMPOSSIBLE]
    assert proven and proven[0].certificate


def test_ocr_table_is_only_a_lead(tmp_path):
    fs, plots, notes = tiers(rasterised(tmp_path, 300, 95, edit=4, table_in_image=True))
    assert plots and all(p.ocr_numbers for p in plots), notes
    assert not [f for f in fs if f.tier is Tier.PROVEN_IMPOSSIBLE]


def test_axis_break_before_last_label_is_not_bridged():
    from trialgeist.extract.raster import _linear

    # 0..240 every 30 px-units, then "//" and 360 placed where 270 would be
    pairs = [(100 + 40 * k, 30.0 * k) for k in range(9)] + [(100 + 40 * 9, 360.0)]
    assert _linear(pairs, want_negative=False) is None
    # an interior OCR misread is tolerated
    bad = [(100 + 40 * k, 30.0 * k) for k in range(10)]
    bad[4] = (bad[4][0], 128.0)
    ax = _linear(bad, want_negative=False)
    assert ax is not None and abs(ax.value(100 + 40 * 4) - 120.0) < 1e-6


@pytest.mark.parametrize("dpi,quality", [(150, 75), (300, 90)])
def test_confidence_band_edges_are_never_read_as_curves(tmp_path, dpi, quality):
    """Stepped band edges in a tint of the arm colour must not be digitised: with the at-risk
    table in the text layer, a band edge read as the curve could otherwise be 'proven'."""
    path = rasterised(tmp_path, dpi, quality, bands=True)
    found, plots, notes = tiers(path)
    assert not [f for f in found if f.tier is Tier.PROVEN_IMPOSSIBLE]
    for p in plots:
        for t in p.traces:
            assert min(t.color) < 120, (t.color, notes)  # a line colour, never a pale tint
