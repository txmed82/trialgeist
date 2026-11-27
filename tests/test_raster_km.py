"""Raster KM digitiser: synthetic figures rasterised at several resolutions and JPEG qualities."""

from __future__ import annotations

import random
import shutil

import numpy as np
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


# ---------------------------------------------------------------- edge reader (thick strokes)

def _stroke(levels, xs_drop, w, H=600, W=700, ticks=()):
    """A staircase stroked with a square pen of width w: plateau rows [y - w//2, y + w//2],
    drops drawn as vertical strokes centred on the drop column."""
    from trialgeist.extract.raster import Panel
    m = np.zeros((H, W), bool)
    edges = [40] + list(xs_drop) + [W - 40]
    for (x0, x1), y in zip(zip(edges, edges[1:]), levels):
        m[y - w // 2: y - w // 2 + w, x0: x1 + w // 2] = True
    for x, (y0, y1) in zip(xs_drop, zip(levels, levels[1:])):
        m[y0 - w // 2: y1 - w // 2 + w, x - w // 2: x - w // 2 + w] = True
    for x, y in ticks:  # censor marks: | ticks 3w tall on a plateau
        m[y - 2 * w: y + 2 * w, x - w // 2: x - w // 2 + w] = True
    return m, Panel(x0=30, y1=H - 20, x1=W - 30, y0=20, lw=1)


def _level_at(steps, x, start):
    lvl = start
    for sx, b, a in steps:
        if sx <= x:
            lvl = a
    return lvl


@pytest.mark.parametrize("w", [4, 6, 9])
def test_edge_reader_resolves_drops_smaller_than_the_line(w):
    from trialgeist.extract.raster import trace, trace_edges
    rnd = random.Random(w)
    xs, levels, x, y = [], [60], 60, 60
    while x < 600:
        x += rnd.randint(2, 14) + (w > 6) * 6  # a 9 px pen needs some plateaus longer than itself
        y += rnd.randint(2, 4)  # drops of 2-4 px under a 4-9 px pen
        xs.append(x); levels.append(y)
    m, p = _stroke(levels, xs, w, ticks=[(xs[5] + 1, levels[6]), (xs[20] + 1, levels[21])])
    other = np.zeros_like(m)
    assert trace(m, p, other) is None  # the plateau reader cannot see them
    t = trace_edges(m, p, other)
    assert t is not None and len(t.steps) >= 0.5 * len(xs)
    # every read level is within half a line width + 1 px of the true curve
    for xx in range(int(t.x_start) + w, int(t.x_end) - w):
        true = _level_at(list(zip(xs, levels, levels[1:])), xx, 60)
        got = _level_at(t.steps, xx, 60)
        assert abs(got - true) <= 1.5 or any(abs(xx - d) <= w for d in xs), (xx, true, got)


def test_edge_reader_refuses_when_the_edges_disagree():
    from trialgeist.extract.raster import trace_edges
    rnd = random.Random(3)
    xs, levels, x, y = [], [60], 60, 60
    while x < 600:
        x += rnd.randint(4, 14)
        y += rnd.randint(2, 4)
        xs.append(x); levels.append(y)
    m, p = _stroke(levels, xs, 6)
    # a same-colour band glued under the line from x = 200 on (a CI band edge, a second arm):
    # the bottom edge no longer follows the top edge
    for xx in range(200, 640):
        y = _level_at(list(zip(xs, levels, levels[1:])), xx, 60)
        m[y: y + 14 + (xx // 7) % 9, xx] = True
    why: list = []
    t = trace_edges(m, p, np.zeros_like(m), why)
    # the glued band breaks double entry over most of the curve: the reader must refuse, and
    # say why (a reading that silently stopped at the band would be a different, weaker check)
    assert t is None, (t.x_end, len(t.steps))
    assert any("disagree" in str(w) for w in why), why



def test_sloped_polyline_is_not_a_staircase():
    """Points joined by sloping lines read as runs of sub-event drops; one event of a 25-patient
    arm is 4% of the axis, so 1 px 'drops' every 3 px are a slope and the curve is refused."""
    from trialgeist.extract.raster import Trace, _sloped

    slope = Trace((0, 0, 0), [(100 + 3 * k, 50 + k, 51 + k) for k in range(12)], 90, 200, 0.0, 2)
    assert _sloped(slope, event_px=13.0)
    stairs = Trace((0, 0, 0), [(100 + 40 * k, 50 + 13 * k, 63 + 13 * k) for k in range(6)], 90, 400, 0.0, 2)
    assert not _sloped(stairs, event_px=13.0)
    # lone 1 px jitter on long plateaus is not a slope
    jitter = Trace((0, 0, 0), [(100 + 90 * k, 50 + k, 51 + k) for k in range(5)], 90, 600, 0.0, 2)
    assert not _sloped(jitter, event_px=3.3)
    # when one event is below 3 px the test cannot tell and does not refuse
    assert not _sloped(slope, event_px=2.0)



def test_tick_label_crop_stops_before_the_axis_title():
    from trialgeist.extract.raster import _first_band

    crop = np.full((60, 80, 3), 255, np.uint8)
    crop[4:14, 30:50] = 0     # the tick label
    crop[30:40, 5:75] = 0     # the axis title further down
    assert _first_band(crop, 0).shape[0] <= 16
    left = np.full((40, 120, 3), 255, np.uint8)
    left[15:25, 5:15] = 0     # rotated y-axis title, far left
    left[15:25, 90:110] = 0   # the tick label next to the axis
    out = _first_band(left, 1)
    assert out.shape[1] <= 26 and out[:, :, 0].min() == 0



def test_axis_scale_refuses_ambiguous_rising_axes():
    from trialgeist.extract.km_vector import axis_scale

    assert axis_scale(1.0, rising=False) == 1.0
    assert axis_scale(100, rising=False) == 0.01
    assert axis_scale(40, rising=True) == 0.01
    assert axis_scale(0.5, rising=True) is None   # 0-0.5 probability or 0-0.5 %: cannot tell
    assert axis_scale(1.25, rising=False) is None  # neither a probability nor a percent top


def test_linear_drops_one_interior_misread_among_four_labels():
    from trialgeist.extract.raster import _linear

    ax = _linear([(100, 0), (200, 12), (300, 25), (400, 36)], want_negative=False)  # "24" read as "25"
    assert ax is not None and abs(ax.value(300) - 24) < 1e-6


def test_ocr_risk_rows_stay_with_their_own_panel():
    from trialgeist.extract.km_vector import Axis
    from trialgeist.extract.raster import Panel, ocr_risk_rows

    p = Panel(x0=100, y1=300, x1=500, y0=100, lw=2)
    ax = Axis(a=0.0, b=0.1, residual=0.0, labels=[(100, 0.0), (200, 10.0), (300, 20.0), (400, 30.0)])
    w = lambda x, y, t: (x - 8, y, x + 8, y + 12, t, 90)  # noqa: E731
    words = [w(60, 330, "risk"), w(100, 330, "50"), w(200, 330, "40"), w(300, 330, "30"),
             # the next panel down (starting at y = 400) has its own table
             w(60, 730, "risk"), w(100, 730, "60"), w(200, 730, "55"), w(300, 730, "50")]
    rows = ocr_risk_rows(words, p, ax, bottom=400)
    assert [[n for _, n in r] for _, r in rows] == [[50, 40, 30]]
    # rows read from the glyph height, not a 6 px grid: one row whose words jitter by 5 px
    words2 = [w(60, 330, "risk"), w(100, 330, "50"), w(200, 335, "40"), w(300, 331, "30")]
    assert [[n for _, n in r] for _, r in ocr_risk_rows(words2, p, ax)] == [[50, 40, 30]]
    # a two-checkpoint table is still a table
    words3 = [w(60, 330, "risk"), w(100, 330, "50"), w(200, 330, "40")]
    assert [[n for _, n in r] for _, r in ocr_risk_rows(words3, p, ax)] == [[50, 40]]



def _render_axes(xlabels, ylabels, grey=(0.45, 0.45, 0.45), size=7, title=True):
    """A survminer-style frame with no tick marks and small grey labels, rasterised."""
    from trialgeist.extract.raster import Panel

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=220)
    x0, x1, y0, y1 = 60, 280, 20, 170
    page.draw_line((x0, y1), (x1, y1), color=(0, 0, 0), width=1)
    page.draw_line((x0, y0), (x0, y1), color=(0, 0, 0), width=1)
    for i, s in enumerate(xlabels):
        x = x0 + 10 + i * (x1 - x0 - 20) / (len(xlabels) - 1)
        page.insert_text((x - 1.9 * len(s), y1 + 12), s, fontsize=size, color=grey)
    for i, s in enumerate(ylabels):
        y = y0 + 5 + i * (y1 - y0 - 10) / (len(ylabels) - 1)
        page.insert_text((x0 - 6 - 3.9 * len(s), y + 2.5), s, fontsize=size, color=grey)
    if title:
        page.insert_text((150, y1 + 30), "Time (months)", fontsize=8, color=(0, 0, 0))
        page.insert_text((22, 130), "Survival probability", fontsize=8, color=(0, 0, 0), rotate=90)
    pix = page.get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
    k = 200 / 72
    return img, Panel(x0=int(x0 * k) + 1, y1=int(y1 * k), x1=int(x1 * k), y0=int(y0 * k), lw=2), k


def test_blob_axis_reads_small_grey_labels_without_tick_marks():
    from trialgeist.extract.raster import blob_axis

    img, p, k = _render_axes(["0", "10", "20", "30"], ["1.00", "0.75", "0.50", "0.25", "0.00"])
    ax = blob_axis(img, p, "x")
    assert ax is not None and [v for _, v in ax.labels] == [0, 10, 20, 30]
    # the scale is exact: label i sits at x0 + 10 + i * 200/3 points
    assert abs(ax.value((60 + 10 + 200 / 3) * k) - 10) < 0.15
    ay = blob_axis(img, p, "y")
    assert ay is not None and sorted(v for _, v in ay.labels) == [0, 0.25, 0.5, 0.75, 1.0]
    assert abs(ay.value((20 + 5 + 2 * 140 / 4) * k) - 0.5) < 0.01


def test_blob_axis_refuses_labels_that_are_not_on_one_linear_scale():
    from trialgeist.extract.raster import blob_axis

    # evenly spaced labels whose values are not (a categorical or log axis): no scale
    img, p, _ = _render_axes(["0", "4", "8", "12", "24", "48"], ["6", "4", "2", "0"], title=False)
    assert blob_axis(img, p, "x") is None
    # control: the same layout with linear values is read, so the refusal above is the rule
    img, p, _ = _render_axes(["0", "4", "8", "12", "16", "20"], ["6", "4", "2", "0"], title=False)
    assert blob_axis(img, p, "x") is not None
