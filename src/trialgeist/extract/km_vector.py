"""Vector-first Kaplan-Meier extraction from PDF drawing operators.

Pipeline per page:

1. collect axis-aligned line segments from ``page.get_drawings()`` grouped by
   stroke colour/width;
2. build a monotone DAG (x and PDF-y non-decreasing, i.e. survival
   non-increasing) and take the longest left-to-right path per colour as the
   candidate survival staircase (censor ticks become dead-end branches);
3. calibrate axes from numeric tick labels (snapped to vector tick marks when
   present) with a linear least-squares fit, refusing poor fits / log axes;
4. read the "No. at risk" table: rows of integers aligned with the x tick
   labels underneath the plot;
5. emit :class:`KMArmClaim` objects (one per curve) with all candidate risk
   rows, plus bbox provenance for every vertex.

Rising curves (cumulative mortality / events plotted as 1 - S(t)) are read
too, but only when the page calls them Kaplan-Meier; cumulative incidence
with competing risks (Aalen-Johansen, Fine-Gray) is not 1 - KM and is refused.

Refusals (no claim emitted): raster figures, rising curves without Kaplan-Meier
wording, pages mentioning adjusted / weighted / competing-risk curves, poor
calibration, no risk table.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction

import pymupdf

from ..verdict import Provenance
from .pdf_text import Row, rows_from_page

SEG_TOL = 0.15  # pt, endpoint matching
NUM_WORD = re.compile(r"^[\-\u2212]?\d+(?:[.\u00b7]\d+)?%?$")
ADJUSTED_RE = re.compile(r"adjusted (survival|curves?)|weighted|IPTW|inverse.probability|cumulative incidence|"
                         r"competing.(risks?|events?)|Aalen|Fine.(and.)?Gray|subdistribution", re.I)
# A rising curve is read as 1 - KM only when nothing on the page suggests it might be a
# cumulative incidence function instead; these captions are ambiguous, so refuse them.
RISING_REFUSE_RE = re.compile(r"cumulative (risk|probability|proportion)|competing|cause.specific|subdistribution|"
                              r"Aalen|Fine.(and.)?Gray|incidence function", re.I)
KAPLAN_RE = re.compile(r"Kaplan[\s\-\u2010-\u2014]*Meier", re.I)
RISK_LABEL_RE = re.compile(r"(no\.?|number|n)\s*(of patients\s*)?at\s*risk|at risk", re.I)


@dataclass
class Seg:
    x0: float
    y0: float
    x1: float
    y1: float
    color: tuple
    width: float

    @property
    def horizontal(self) -> bool:
        return abs(self.y0 - self.y1) < 1e-3 and abs(self.x1 - self.x0) > 1e-3

    @property
    def vertical(self) -> bool:
        return abs(self.x0 - self.x1) < 1e-3 and abs(self.y1 - self.y0) > 1e-3


@dataclass
class Staircase:
    color: tuple
    width: float
    points: list[tuple[float, float]]  # PDF coords along the path
    rising: bool = False  # True: the plotted value goes up (1 - S), PDF y decreases

    @property
    def bbox(self):
        xs, ys = [p[0] for p in self.points], [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class Axis:
    a: float  # value = a + b * coord
    b: float
    residual: float
    labels: list[tuple[float, float]]  # (coord, value)

    def value(self, coord: float) -> float:
        return self.a + self.b * coord


@dataclass
class KMPlot:
    page: int
    curves: list[Staircase]
    xaxis: Axis
    yaxis: Axis
    y_scale: float  # 1 for probability axis, 0.01 for percent axis
    risk_rows: list[tuple[str, list[tuple[float, int]]]]  # (label, [(time, count)])
    notes: list[str] = field(default_factory=list)


def _segments(page: pymupdf.Page) -> list[Seg]:
    segs = []
    for d in page.get_drawings():
        color = tuple(round(c, 3) for c in (d.get("color") or ())) or None
        if color is None:
            continue
        w = d.get("width") or 0.0
        for it in d["items"]:
            if it[0] == "l":
                p, q = it[1], it[2]
                x0, y0, x1, y1 = p.x, p.y, q.x, q.y
                if (x1, y1) < (x0, y0):
                    x0, y0, x1, y1 = x1, y1, x0, y0
                segs.append(Seg(x0, y0, x1, y1, color, w))
    return segs


def _key(x: float, y: float) -> tuple[int, int]:
    return (round(x / SEG_TOL), round(y / SEG_TOL))


MAX_STAIRCASES_PER_STYLE = 32


def staircases(segs: list[Seg], min_steps: int = 3, rising: bool = False, notes: list[str] | None = None) -> list[Staircase]:
    by_style: dict[tuple, list[Seg]] = defaultdict(list)
    for s in segs:
        if s.horizontal or s.vertical:
            by_style[(s.color, round(s.width, 2))].append(s)
    out = []
    for (color, width), ss in by_style.items():
        # edges oriented so x and y never decrease (survival never increases)
        adj: dict[tuple, list[tuple[tuple, tuple[float, float]]]] = defaultdict(list)
        coords: dict[tuple, tuple[float, float]] = {}
        for s in ss:
            if s.horizontal:
                a, b = (s.x0, s.y0), (s.x1, s.y1)
            else:  # falling: PDF y increases along the path; rising: it decreases
                down = ((s.x0, s.y0), (s.x1, s.y1)) if s.y0 <= s.y1 else ((s.x1, s.y1), (s.x0, s.y0))
                a, b = (down[1], down[0]) if rising else down
            ka, kb = _key(*a), _key(*b)
            coords.setdefault(ka, a)
            coords.setdefault(kb, b)
            adj[ka].append((kb, b))
        # longest path in the DAG by x-extent then edge count; repeat on what is
        # left so several panels drawn in the same style each yield a curve
        order = sorted(coords, key=lambda k: (coords[k][0], -coords[k][1] if rising else coords[k][1]))
        # Used EDGES are retired, not vertices: two arms drawn in one style share their
        # (t=0, S=1) start point, and retiring it would truncate the second curve.
        dead: set[tuple[tuple, tuple]] = set()
        for it in range(MAX_STAIRCASES_PER_STYLE + 1):
            if it == MAX_STAIRCASES_PER_STYLE:
                if notes is not None:
                    notes.append(f"more than {MAX_STAIRCASES_PER_STYLE} staircases in stroke style {color}/{width}; the rest were not read")
                break
            best: dict[tuple, tuple[float, int, tuple | None]] = {}
            for k in reversed(order):
                cand = (0.0, 0, None)
                for nk, _ in adj.get(k, []):
                    if nk == k or (k, nk) in dead:
                        continue
                    ext = coords[nk][0] - coords[k][0] + best.get(nk, (0.0, 0, None))[0]
                    cnt = 1 + best.get(nk, (0.0, 0, None))[1]
                    if (ext, cnt) > cand[:2]:
                        cand = (ext, cnt, nk)
                best[k] = cand
            start = max(best, key=lambda k: best[k][:2], default=None)
            if start is None or best[start][2] is None:
                break  # no edges left
            path, keys, k = [coords[start]], [start], start
            while best[k][2] is not None:
                k = best[k][2]
                path.append(coords[k])
                keys.append(k)
            dead.update(zip(keys, keys[1:]))
            drops = sum(1 for p, q in zip(path, path[1:]) if (p[1] - q[1] if rising else q[1] - p[1]) > 1e-3)
            if drops >= min_steps:  # short fragments are retired too, and the search goes on
                out.append(Staircase(color, width, path, rising))
    return out


def _fit(pairs: list[tuple[float, float]]) -> Axis | None:
    if len(pairs) < 3:
        return None
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    if sxx == 0:
        return None
    b = sum((p[0] - mx) * (p[1] - my) for p in pairs) / sxx
    a = my - b * mx
    res = max(abs(a + b * c - v) / abs(b) for c, v in pairs) if b else 1e9  # residual in pt
    return Axis(a, b, res, pairs)


def _num(t: str) -> float | None:
    t = t.replace("\u2212", "-").replace("\u00b7", ".").rstrip("%")
    try:
        return float(t)
    except ValueError:
        return None


def _snap(coord: float, ticks: list[float], tol: float = 4.0) -> float:
    near = [t for t in ticks if abs(t - coord) <= tol]
    return min(near, key=lambda t: abs(t - coord)) if near else coord


def plot_frame(sc: Staircase, segs: list[Seg]) -> tuple[float, float, float, float]:
    """(left, top, right, bottom) of the plot area from long axis lines, else the curve bbox."""
    x0, y0, x1, y1 = sc.bbox
    w = max(x1 - x0, 1.0)
    hs = [s for s in segs if s.horizontal and s.y0 >= y1 - 1 and s.x1 - s.x0 >= 0.6 * w and s.x0 <= x0 + 5 and s.y0 - y1 < 400]
    # the y-axis line must span this curve (stacked panels share x, not y)
    vs = [s for s in segs if s.vertical and s.x0 <= x0 + 1 and x0 - s.x0 < 60 and s.y1 - s.y0 >= 0.3 * max(y1 - y0, 1.0)
          and s.y0 <= y0 + 1 and s.y1 >= y1 - 1]
    bottom = min((s.y0 for s in hs), default=y1)
    left = max((s.x0 for s in vs), default=x0)
    top = max((s.y0 for s in vs if s.x0 == left), default=y0)  # innermost spanning line
    return left, top, x1, bottom


def calibrate(page: pymupdf.Page, sc: Staircase, segs: list[Seg], group: list[Staircase] | None = None):
    """Axes for ``sc``; the y-axis must also be valid for every curve in ``group``."""
    x0, y0, _, y1 = plot_frame(sc, segs)
    curves_top = min(c.bbox[1] for c in (group or [sc]))
    words = [w for w in page.get_text("words") if NUM_WORD.match(w[4])]
    # y labels: right edge left of the plot, vertically within the plot band
    ycand = [w for w in words if w[2] <= x0 + 2 and w[2] >= x0 - 60 and y0 - 30 <= (w[1] + w[3]) / 2 <= y1 + 30]
    groups: dict[int, list] = defaultdict(list)
    for w in ycand:
        groups[round(w[2] / 3)].append(w)
    hticks = [s.y0 for s in segs if s.horizontal and abs(s.x1 - s.x0) < 8 and s.x1 <= x0 + 8]
    yaxis = None
    for g in sorted(groups.values(), key=len, reverse=True):
        pairs = [(_snap((w[1] + w[3]) / 2, hticks), _num(w[4])) for w in g]
        pairs = [(c, v) for c, v in pairs if v is not None]
        ax = _fit(pairs)
        if ax and ax.b < 0 and ax.residual < 1.0:
            yaxis = ax
            break
        # broken axis ("//"): the label above the break is off the line.  Drop the
        # topmost label only, and only if the curve stays below the remaining ones,
        # so values are never read across the break.
        if len(pairs) >= 4:
            top = min(pairs)
            rest = [p for p in pairs if p is not top]
            ax = _fit(rest)
            if ax and ax.b < 0 and ax.residual < 1.0 and curves_top >= min(c for c, _ in rest) - 2:  # every curve below the break
                yaxis = ax
                break
    # x labels: a row below the plot with increasing values left to right
    x_right = sc.bbox[2]
    xcand = [w for w in words if y1 < (w[1] + w[3]) / 2 < y1 + 40 and x0 - 20 <= (w[0] + w[2]) / 2 <= x_right + 40]
    rows: dict[int, list] = defaultdict(list)
    for w in xcand:
        rows[round((w[1] + w[3]) / 2 / 3)].append(w)
    vticks = [s.x0 for s in segs if s.vertical and abs(s.y1 - s.y0) < 8 and s.y0 >= y1 - 8]
    xaxis = None
    for key in sorted(rows):
        g = rows[key]
        pairs = [(_snap((w[0] + w[2]) / 2, vticks), _num(w[4])) for w in g]
        pairs = sorted((c, v) for c, v in pairs if v is not None)
        ax = _fit(pairs)
        if ax and ax.b > 0 and ax.residual < 1.0:
            xaxis = ax
            break
    return xaxis, yaxis


def risk_rows(page: pymupdf.Page, xaxis: Axis, below_y: float, x_lo: float, x_hi: float) -> list[tuple[str, list[tuple[float, int]]]]:
    """Rows of integers under the plot whose x positions fall on a regular time grid.

    Counts in parentheses (events/censored) are ignored.  Only the horizontal
    band of this panel [x_lo, x_hi] is read, so side-by-side panels stay apart.
    """
    rows = rows_from_page(page)
    label_rows = [r for r in rows if r.y > below_y and r.y < below_y + 80 and RISK_LABEL_RE.search(r.text)]
    unlabelled = not label_rows
    if unlabelled:
        # Many journals label the rows only with the arm names and put "number at risk" in the
        # caption or legend.  Then only rows starting right under the plot are read, only if the
        # page speaks of numbers at risk, and each must be non-increasing (counts at risk).
        if not RISK_LABEL_RE.search(page.get_text()):
            return []
        first = next((r for r in rows if below_y - 3 < r.y < below_y + 25 and
                      sum(1 for w in r.words if re.fullmatch(r"\d[\d,]*", w.text) and x_lo - 15 <= (w.x0 + w.x1) / 2 <= x_hi + 15) >= 3), None)
        if first is None:
            return []
        start_y = first.y
    else:
        start_y = label_rows[0].y
    tick_vals = sorted(v for _, v in xaxis.labels)
    tick_step = min((b - a for a, b in zip(tick_vals, tick_vals[1:]) if b > a), default=None)
    out = []
    for r in rows:
        if r.y < start_y - 1 or r.y > start_y + 90:
            continue
        ints = [w for w in r.words if re.fullmatch(r"\d[\d,]*", w.text) and x_lo - 15 <= (w.x0 + w.x1) / 2 <= x_hi + 15]
        if len(ints) < 3 or not tick_step:
            continue
        ts = [xaxis.value((w.x0 + w.x1) / 2) for w in ints]
        diffs = sorted(b - a for a, b in zip(ts, ts[1:]))
        step = diffs[len(diffs) // 2] if diffs else tick_step
        # the grid step must be a simple fraction of the tick-label step
        ratio = tick_step / step if step > 0 else 0
        if step <= 0 or abs(ratio - round(ratio)) > 0.1 or round(ratio) < 1:
            continue
        step = tick_step / round(ratio)
        aligned = []
        for w, t in zip(ints, ts):
            snapped = round(t / step) * step
            if abs(t - snapped) > 0.25 * step:
                aligned = []
                break
            aligned.append((snapped, int(w.text.replace(",", ""))))
        times = [t for t, _ in aligned]
        if unlabelled and any(b > a for (_, a), (_, b) in zip(aligned, aligned[1:])):
            continue
        if len(aligned) >= 3 and len(set(times)) == len(times) and times == sorted(times):
            label = " ".join(w.text for w in r.words if not re.fullmatch(r"[\d,()]+", w.text) and (w.x1 < x_lo + 5))
            out.append((label.strip(), aligned))
    return out


def find_km_plots(path: str) -> tuple[list[KMPlot], list[str]]:
    with pymupdf.open(path) as doc:
        return _find_km_plots(doc)


def _find_km_plots(doc) -> tuple[list[KMPlot], list[str]]:
    plots, notes = [], []
    for pi, page in enumerate(doc):
        segs = _segments(page)
        text = page.get_text()
        scs = staircases(segs, notes=notes)
        if KAPLAN_RE.search(text):
            if RISING_REFUSE_RE.search(text):
                if staircases(segs, rising=True):
                    notes.append(f"page {pi}: rising curves skipped (caption may describe cumulative incidence, not 1 - KM)")
            else:
                scs += staircases(segs, rising=True, notes=notes)
        if not scs:
            continue
        if ADJUSTED_RE.search(text):
            notes.append(f"page {pi}: staircases skipped (adjusted/weighted/cumulative-incidence wording)")
            continue
        # cluster staircases that share a plot area
        used = set()
        for i, sc in enumerate(scs):
            if i in used:
                continue
            group = [sc]
            for j in range(i + 1, len(scs)):
                b1, b2 = sc.bbox, scs[j].bbox
                if scs[j].rising == sc.rising and abs(b1[0] - b2[0]) < 15 and not (b2[1] > b1[3] + 10 or b1[1] > b2[3] + 10):
                    group.append(scs[j])
                    used.add(j)
            xaxis, yaxis = calibrate(page, sc, segs, group)
            if not xaxis or not yaxis:
                notes.append(f"page {pi}: staircase found but axes could not be calibrated")
                continue
            y_scale = axis_scale(max(v for _, v in yaxis.labels), sc.rising)
            if y_scale is None:
                notes.append(f"page {pi}: y axis could be probability or percent (rising curve, labels <= 1.5): refused")
                continue
            below = plot_frame(sc, segs)[3]
            rr = risk_rows(page, xaxis, below, min(c.bbox[0] for c in group), max(c.bbox[2] for c in group))
            if not rr:
                notes.append(f"page {pi}: KM curves without a parsable numbers-at-risk table")
                continue
            plots.append(KMPlot(pi, group, xaxis, yaxis, y_scale, rr))
    return plots, notes


def axis_scale(vmax: float, rising: bool) -> float | None:
    """Probability (1) or percent (0.01) y-axis from its largest label, or None if ambiguous.
    A falling KM curve starts at S = 1, so its axis tops out near 1 or near 100. A rising
    1 - S(t) axis may be zoomed: labels up to 1 can be probabilities or percentages
    (0-1%), and reading a percent axis as probabilities would invent huge drops.
    Labels above 100 are neither (counts, events per 1,000 persons, mean cumulative events)."""
    if vmax > 105:
        return None
    if vmax > 1.5:
        return 0.01
    if vmax <= 1.05 and not rising:
        return 1.0
    return None


def plot_to_claims(plot: KMPlot, eps_pt: float = 1.0):
    """One KMArmClaim per curve; every printed risk row is a candidate pairing."""
    from ..claims import KMArmClaim, KMRiskRow, KMStepIn

    ys = plot.y_scale
    eps = abs(plot.yaxis.b) * ys * max(eps_pt, 3 * plot.yaxis.residual)
    eps_t = abs(plot.xaxis.b) * max(eps_pt, 3 * plot.xaxis.residual)
    rows = [KMRiskRow(label=lab, risk_times=[f"{t:.6f}" for t, _ in row], at_risk=[n for _, n in row])
            for lab, row in plot.risk_rows]
    out = []
    for ci, sc in enumerate(plot.curves):
        steps = []
        for p, q in zip(sc.points, sc.points[1:]):
            if (p[1] - q[1] if sc.rising else q[1] - p[1]) > 1e-3 and abs(q[0] - p[0]) < 1e-3:
                t = plot.xaxis.value(p[0])
                b, a = plot.yaxis.value(p[1]) * ys, plot.yaxis.value(q[1]) * ys
                if sc.rising:  # plotted 1 - S(t)
                    b, a = 1 - b, 1 - a
                steps.append(KMStepIn(t=f"{t:.6f}", before=f"{min(max(b, 0), 1):.6f}", after=f"{min(max(a, 0), 1):.6f}"))
        prov = [Provenance(source="pdf", page=plot.page, bbox=sc.bbox, text=f"KM curve {ci} (stroke {sc.color}){' plotted as 1 - S(t)' if sc.rising else ''}",
                           note="vertices: " + ";".join(f"{x:.2f},{y:.2f}" for x, y in sc.points[:400]))]
        out.append(KMArmClaim(label=f"page {plot.page + 1} curve {ci + 1}", alternatives=rows, steps=steps,
                              eps=f"{eps:.6f}", eps_t=f"{eps_t:.6f}", provenance=prov))
    return out
