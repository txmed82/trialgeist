"""Raster figures: inventory every embedded image, digitise raster Kaplan-Meier plots.

Soundness rules (a false PROVEN_IMPOSSIBLE is the one unacceptable outcome):

* Axes: the plot frame is an L of long dark pixel runs.  Tick labels are read
  by OCR (tesseract, optional) and accepted only if values and positions are
  exactly linear (residual < 1.5 px): a misread digit breaks the progression.
* Curve: per colour, a left-to-right trace that follows a plateau and drops to
  a lower plateau only through a drawn riser (a vertical run of curve pixels
  joining both levels).  Legend swatches, text and stray marks are never
  joined by a riser.  Censor ticks straddle the level and are ignored.  Gaps
  (one curve hidden under another) keep the level; a lower level after a gap
  becomes a drop whose time uncertainty covers the whole gap.
* Tolerances: level +/- (half line width + 2 px), time +/- 3 px, converted to
  axis units.  The KM engine already treats each drop as a cluster of one or
  more event times with possible hidden events in between.
* Numbers at risk: from the PDF text layer they support proofs.  Read by OCR
  from inside the image they only ever yield a lead (``ocr_numbers=True``
  caps the verdict at INCONCLUSIVE in ``claims.check_claim``).
"""

from __future__ import annotations

import colorsys
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pymupdf

from ..verdict import Provenance
from .km_vector import ADJUSTED_RE, axis_scale, KAPLAN_RE, RISING_REFUSE_RE, RISK_LABEL_RE, Axis, _fit, risk_rows

KM_WORDS_RE = re.compile(r"kaplan|survival|at risk|time to (first )?event|event[- ]free|cumulative (mortality|probability)", re.I)
NUM_TOKEN = re.compile(r"^[\-\u2212]?\d+(?:[.,\u00b7]\d+)?%?$")
MIN_PX = 120  # smaller images are too coarse to digitise


@dataclass
class FigureImage:
    page: int
    rect: tuple[float, float, float, float]  # PDF points
    xref: int
    width: int
    height: int
    page_text: str
    km_candidate: bool
    png: bytes = b""


@dataclass
class Panel:
    x0: int  # y-axis column
    y1: int  # x-axis row
    x1: int  # right end of the x-axis
    y0: int  # top of the y-axis
    lw: int  # axis line width


@dataclass
class Trace:
    color: tuple[int, int, int]
    steps: list[tuple[float, float, float]]  # (x px, level before px, level after px)
    x_start: float
    x_end: float
    gap_px: float  # widest occlusion gap that hid a drop
    lw: float
    label: str = ""
    row: int | None = None  # risk row paired through the legend
    method: str = "plateau"  # "plateau" (line centre) or "edges" (thick stroke, double-entry edges)
    rising: bool = False  # read in the flipped orientation (plotted as 1 - S(t))
    gaps: list[float] | None = None  # per drop: px of occlusion before it was seen (None: gap_px for all)


@dataclass
class RasterKM:
    figure: FigureImage
    panel: Panel
    xaxis: Axis  # pixel -> time
    yaxis: Axis  # pixel -> plotted value
    y_scale: float
    rising: bool
    traces: list[Trace]
    risk: list[tuple[str, list[tuple[float, int]]]]
    ocr_numbers: bool
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ images


def _rgb(doc, xref: int) -> np.ndarray | None:
    try:
        pix = pymupdf.Pixmap(doc, xref)
        if pix.colorspace is None or pix.colorspace.n != 3:
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        if pix.alpha:
            pix = pymupdf.Pixmap(pix, 0)
        arr = np.frombuffer(pix.samples_mv, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
        del pix
        pymupdf.TOOLS.store_shrink(100)  # drop MuPDF's cached decode (up to 256 MB)
        return arr
    except Exception:
        return None


def figure_images(doc) -> list[FigureImage]:
    out = []
    for pno, page in enumerate(doc):
        text = page.get_text()
        if not KM_WORDS_RE.search(text):
            # inventory only: placing images runs a device that decodes them, so skip it
            for info in page.get_images(full=True):
                if info[2] >= MIN_PX and info[3] >= MIN_PX:
                    out.append(FigureImage(pno, (0.0, 0.0, 0.0, 0.0), info[0], info[2], info[3], "", False))
            continue
        for info in page.get_image_info(xrefs=True):
            xref, w, h = info.get("xref", 0), info["width"], info["height"]
            if not xref or w < MIN_PX or h < MIN_PX:
                continue
            a, b, c, d, _, _ = info["transform"]
            if abs(b) > 1e-6 or abs(c) > 1e-6 or a <= 0 or d <= 0:
                continue  # rotated / flipped placement: refuse rather than guess
            rect = pymupdf.Rect(info["bbox"])
            if rect.width < 100 or rect.height < 60:
                continue
            out.append(FigureImage(pno, tuple(rect), xref, w, h, text, bool(KM_WORDS_RE.search(text))))
        pymupdf.TOOLS.store_shrink(100)
    return out


# ------------------------------------------------------------------ OCR


def ocr_words(img: np.ndarray, scale: float | None = None) -> list[tuple[float, float, float, float, str, float]]:
    """(x0, y0, x1, y1, text, conf) in original pixels; [] when tesseract is unavailable.

    Figures are exported anywhere from 500 to 5000 px wide; tesseract reads best with
    glyphs ~20-40 px tall, so small images are upscaled 2x and large ones downscaled 2x."""
    exe = shutil.which("tesseract")
    if not exe:
        return []
    if scale is None:
        m = max(img.shape[:2])
        scale = 2 if m < 1600 else 0.5 if m > 3200 else 1
    if scale == 2:
        big = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
    elif scale == 0.5:
        h2, w2 = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
        big = (img[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2, 3).sum(axis=(1, 3), dtype=np.uint16) // 4).astype(np.uint8)
    else:
        big = img
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "f.ppm"  # plain binary PPM: no PyMuPDF pixmaps (scaled pixmaps leak native memory)
        with open(p, "wb") as fh:
            fh.write(b"P6 %d %d 255\n" % (big.shape[1], big.shape[0]))
            fh.write(np.ascontiguousarray(big, dtype=np.uint8).tobytes())
        del big
        try:
            r = subprocess.run([exe, str(p), "-", "--psm", "11", "tsv"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return []
    out = []
    for line in r.stdout.splitlines()[1:]:
        f = line.split("\t")
        if len(f) < 12 or not f[11].strip():
            continue
        x, y, ww, hh, conf = (float(v) for v in f[6:11])
        out.append((x / scale, y / scale, (x + ww) / scale, (y + hh) / scale, f[11].strip(), conf))
    return out


def _val(t: str) -> float | None:
    t = t.replace("\u2212", "-").replace("\u00b7", ".").replace(",", ".").rstrip("%")
    try:
        return float(t)
    except ValueError:
        return None


# ------------------------------------------------------------------ axes


def _runs(v: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive [start, end] runs of True."""
    if not v.any():
        return []
    d = np.diff(np.concatenate(([0], v.astype(np.int8), [0])))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    return list(zip(s.tolist(), e.tolist()))


def _lines(dark: np.ndarray, min_len: int) -> list[tuple[int, int, int, int]]:
    """Long runs along axis 1 merged across adjacent rows: (row0, row1, start, end)."""
    out: list[list[int]] = []
    open_: list[list[int]] = []
    for r in range(dark.shape[0]):
        runs = [(a, b) for a, b in _runs(dark[r]) if b - a + 1 >= min_len]
        nxt = []
        for a, b in runs:
            m = next((L for L in open_ if L[1] == r - 1 and abs(L[2] - a) <= 3 and abs(L[3] - b) <= 3), None)
            if m:
                m[1], m[2], m[3] = r, min(m[2], a), max(m[3], b)
                nxt.append(m)
            else:
                L = [r, r, a, b]
                out.append(L)
                nxt.append(L)
        open_ = nxt
    return [tuple(L) for L in out]  # type: ignore[misc]


def find_panels(img: np.ndarray) -> list[Panel]:
    """Plot frames: a long horizontal dark line whose left end meets a long vertical one."""
    # grey axes (ggplot/survminer exports, downsampled lines) are mid-grey, not black
    dark = img.astype(np.int16).mean(axis=2) < 160
    h, w = dark.shape
    hl = _lines(dark, max(60, int(0.15 * w)))  # (y0, y1, x0, x1)
    vl = _lines(dark.T, max(50, int(0.08 * h)))  # (x0, x1, y0, y1); 4-row grids have short y-axes
    panels: list[Panel] = []
    tick = max(25, int(0.015 * max(h, w)))  # tick length scales with the export resolution
    for ya, yb, a, b in hl:
        for xa, xb, ta, tb in vl:
            # the x-axis starts at (or overhangs left of) the y-axis by at most a tick, and the
            # y-axis ends at (or overhangs below) the x-axis by at most a tick
            if a - 3 <= xa <= a + tick and ya - 3 <= tb <= yb + tick and ta < ya - min(40, 0.15 * h) and b > xb + min(60, 0.15 * w):
                p = Panel(x0=xb, y1=ya, x1=b, y0=ta, lw=max(1, min(yb - ya + 1, xb - xa + 1)))
                if not any(abs(q.x0 - p.x0) <= 8 and abs(q.y1 - p.y1) <= 8 for q in panels):
                    panels.append(p)
    # a large KM riser meeting the start of the next plateau is also an "L": such a frame lies
    # inside a real panel (grid panels sit side by side, never one inside another)
    def inside(p: Panel, q: Panel) -> bool:
        return q is not p and q.x0 + 4 < p.x0 < q.x1 and q.y0 - 6 <= p.y0 and p.y1 < q.y1 - 4
    return [p for p in panels if not any(inside(p, q) for q in panels)]


def clean_for_ocr(img: np.ndarray, panels: list[Panel]) -> np.ndarray:
    """Whiten axis lines and the tick marks attached to them: OCR otherwise reads a
    tick next to "0.8" as a digit ("0.85").  Labels are separated from ticks by a gap.
    Anti-aliased tick edges count as tick (threshold 200), plus a 1 px margin."""
    out = img.copy()
    dark = img.astype(np.int16).mean(axis=2) < 200
    h, w = dark.shape
    for p in panels:
        tick = int(0.03 * max(w, h)) + p.lw + 6
        for y in range(max(0, p.y0 - 3), min(h, p.y1 + p.lw + 3)):  # y-axis and its ticks
            for a, b in _runs(dark[y, : p.x0 + p.lw + 2]):
                if b >= p.x0 - p.lw - 1 and (b - a) <= tick + 2 * p.lw:
                    out[max(0, y - 1): y + 2, max(0, a - 1): b + 1] = 255
        out[max(0, p.y0 - 2): p.y1 + p.lw + 2, max(0, p.x0 - p.lw - 1): p.x0 + p.lw + 2] = 255
        for x in range(max(0, p.x0 - 3), min(w, p.x1 + 3)):  # x-axis and its ticks
            for a, b in _runs(dark[p.y1 - p.lw - 1:, x]):
                a, b = a + p.y1 - p.lw - 1, b + p.y1 - p.lw - 1
                if a <= p.y1 + p.lw + 1 and (b - a) <= tick + 2 * p.lw:
                    out[a: b + 2, max(0, x - 1): x + 2] = 255
        out[max(0, p.y1 - p.lw - 1): p.y1 + p.lw + 2, max(0, p.x0 - 2): p.x1 + 2] = 255
    return out


def _cluster(items, key, tol: float) -> list[list]:
    """Group items whose key values chain within tol (tick labels share a baseline / right edge)."""
    out: list[list] = []
    for it in sorted(items, key=key):
        if out and key(it) - key(out[-1][-1]) <= tol:
            out[-1].append(it)
        else:
            out.append([it])
    return out


def _linear(pairs: list[tuple[float, float]], want_negative: bool) -> Axis | None:
    """An exact linear scale through the labels.  One INTERIOR label may be dropped as
    an OCR misread if at least four remain on the line.  An end label is never dropped:
    an axis broken before its last (or first) label ("//") would otherwise be read
    straight across the break."""
    def ok(ax):
        return ax is not None and (ax.b < 0) == want_negative and ax.b != 0 and ax.residual < 1.5

    pairs = sorted(pairs)
    ax = _fit(pairs)
    if ok(ax):
        return ax
    if len(pairs) >= 4:
        fits = [_fit(pairs[:i] + pairs[i + 1:]) for i in range(1, len(pairs) - 1)]
        good = [f for f in fits if ok(f)]
        if len(good) == 1:  # exactly one interior label is the odd one out
            return good[0]
    return None


def x_ticks(img: np.ndarray, p: Panel) -> list[float]:
    """Centres of tick marks hanging below (or rising above) the x-axis line."""
    dark = img.astype(np.int16).mean(axis=2) < 140
    h = dark.shape[0]
    n = max(4, p.lw)
    below = dark[min(h - 1, p.y1 + p.lw + 1): min(h, p.y1 + p.lw + 1 + n), p.x0 - p.lw - 2: p.x1 + 3].all(axis=0)
    above = dark[max(0, p.y1 - n - 1): p.y1 - 1, p.x0 - p.lw - 2: p.x1 + 3].all(axis=0)
    out = []
    for col in (below, above):
        for a, b in _runs(col):
            if b - a + 1 <= 3 * p.lw + 3:
                out.append(p.x0 - p.lw - 2 + (a + b) / 2)
    return sorted(set(out))


def y_ticks(img: np.ndarray, p: Panel) -> list[float]:
    """Centres of tick marks sticking out left of the y-axis line."""
    dark = img.astype(np.int16).mean(axis=2) < 140
    n = max(4, p.lw)
    x1 = p.x0 - p.lw - 1
    if x1 - n < 0:
        return []
    left = dark[max(0, p.y0 - 2): p.y1 + 3, x1 - n: x1].all(axis=1)
    return [max(0, p.y0 - 2) + (a + b) / 2 for a, b in _runs(left) if b - a + 1 <= 3 * p.lw + 3]


def _ocr_label(crop: np.ndarray, scale: int) -> tuple[str, float] | None:
    """One tick label: tesseract single-line mode with a numeric whitelist.  None unless
    exactly one confident numeric token is read."""
    exe = shutil.which("tesseract")
    if not exe or crop.size == 0 or crop.min() > 200:
        return None
    c = np.repeat(np.repeat(crop, scale, axis=0), scale, axis=1) if scale > 1 else crop
    c = np.pad(c, ((15, 15), (15, 15), (0, 0)), constant_values=255)
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "t.ppm"
        with open(f, "wb") as fh:
            fh.write(b"P6 %d %d 255\n" % (c.shape[1], c.shape[0]))
            fh.write(np.ascontiguousarray(c, dtype=np.uint8).tobytes())
        try:
            r = subprocess.run([exe, str(f), "-", "--psm", "7", "-c", "tessedit_char_whitelist=0123456789.-%",
                                "tsv"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
    toks = []
    for line in r.stdout.splitlines()[1:]:
        q = line.split("\t")
        if len(q) >= 12 and q[11].strip():
            toks.append((q[11].strip(), float(q[10])))
    if len(toks) != 1 or toks[0][1] < 60 or not NUM_TOKEN.match(toks[0][0]):
        return None
    return toks[0]


def _first_band(crop: np.ndarray, axis: int) -> np.ndarray:
    """Keep only the label nearest the plot: the first band of inked rows below an x-axis
    (axis 0), or the last band of inked columns left of a y-axis (axis 1). The axis title
    beyond a blank gap would otherwise join the crop and make OCR refuse the label."""
    if crop.size == 0:
        return crop
    ink = (crop.astype(np.int16).mean(axis=2) < 160).any(axis=1 - axis)
    idx = np.flatnonzero(ink)
    if not len(idx):
        return crop
    gap = 3
    if axis == 0:
        a = b = idx[0]
        for i in idx[1:]:
            if i - b > gap:
                break
            b = i
        return crop[max(0, a - 2): b + 3]
    a = b = idx[-1]
    for i in idx[::-1][1:]:
        if a - i > gap + 3:  # digits of one label are closer than this
            break
        a = i
    return crop[:, max(0, a - 2): b + 3]


def tick_axis(img: np.ndarray, p: Panel, ticks: list[float], axis: str) -> Axis | None:
    """Fallback calibration: OCR the label next to each detected tick mark on its own.
    Full-figure OCR drops isolated one-digit labels ("0", "6"); a crop centred on the tick
    reads them, and the label position is the tick itself.  The same exact-linearity rule
    applies, so a misread or a spurious tick refuses the axis rather than bending it."""
    # detached axes (Stata, base R): the other axis line overhangs the frame corner and
    # looks like a tick right beside the real origin tick; it carries no label of its own
    edge = p.x0 if axis == "x" else p.y1
    ticks = [t for t in ticks if not (abs(t - edge) <= p.lw + 2 and any(0 < abs(u - t) < max(20, 2 * p.lw) for u in ticks))]
    if len(ticks) < 3:
        return None
    h, w = img.shape[:2]
    height, width = p.y1 - p.y0, p.x1 - p.x0
    scale = 3 if height < 600 else 2
    gap = max(4, p.lw) + 2
    pairs = []
    for i, t in enumerate(ticks):
        near = min([abs(t - u) for j, u in enumerate(ticks) if j != i] or [60.0])
        half = 0.48 * near
        if half < 6:
            continue  # two marks this close are not both ticks; read neither
        if axis == "x":
            y0 = p.y1 + p.lw + gap
            crop = _first_band(img[y0: int(min(h, y0 + max(18, 0.09 * height))), int(max(0, t - half)): int(min(w, t + half))], 0)
        else:
            x1 = max(1, p.x0 - p.lw - gap)
            crop = _first_band(img[int(max(0, t - half)): int(min(h, t + half)), int(max(0, x1 - 0.2 * width)): max(0, x1)], 1)
        got = _ocr_label(crop, scale)
        v = _val(got[0]) if got else None
        if v is not None:
            pairs.append((t, v))
    if len(pairs) < 3:
        return None
    return _linear(sorted(pairs), want_negative=(axis == "y"))


def calibrate_panel(p: Panel, words, ticks: list[float] | None = None) -> tuple[Axis | None, Axis | None, list[str]]:
    notes = []
    num = [(x0, y0, x1, y1, _val(t), c) for x0, y0, x1, y1, t, c in words if NUM_TOKEN.match(t) and c >= 50]
    num = [n for n in num if n[4] is not None]
    width, height = p.x1 - p.x0, p.y1 - p.y0
    # y labels: right edge just left of the axis, centre within the panel band
    yc = [n for n in num if p.x0 - 0.3 * width <= n[2] <= p.x0 + 2 and p.y0 - 12 <= (n[1] + n[3]) / 2 <= p.y1 + 12]
    yaxis = None
    for g in sorted(_cluster(yc, lambda n: n[2], 8), key=len, reverse=True):
        yaxis = _linear([((n[1] + n[3]) / 2, n[4]) for n in g], want_negative=True)
        if yaxis:
            break
    if yaxis is None:
        notes.append("y tick labels not read as an exact linear scale")
    xc = [n for n in num if p.y1 < (n[1] + n[3]) / 2 <= p.y1 + 0.2 * height + 25 and p.x0 - 20 <= (n[0] + n[2]) / 2 <= p.x1 + 20]
    xaxis = None
    for g in sorted(_cluster(xc, lambda n: n[3], 12), key=lambda g: min(n[1] for n in g)):
        xaxis = _linear(sorted(((n[0] + n[2]) / 2, n[4]) for n in g), want_negative=False)
        if xaxis:
            break
    if xaxis is None and ticks:
        # survminer layout: plot, risk table, then a repeated x-axis carrying the labels.
        # Accept a lower row only if >= 3 label centres sit on the plot's own tick marks.
        tol = max(3.0, p.lw + 1.0)
        low = [n for n in num if p.y1 + 0.2 * height + 25 < (n[1] + n[3]) / 2 <= p.y1 + 0.9 * height
               and p.x0 - 20 <= (n[0] + n[2]) / 2 <= p.x1 + 20]
        for g in sorted(_cluster(low, lambda n: n[3], 12), key=lambda g: min(n[1] for n in g)):
            pairs = sorted(((n[0] + n[2]) / 2, n[4]) for n in g)
            on_tick = sum(1 for c, _ in pairs if any(abs(c - t) <= tol for t in ticks))
            ax = _linear(pairs, want_negative=False)
            if ax and on_tick >= 3 and on_tick >= len(pairs) - 1:
                xaxis = ax
                break
    if xaxis is None:
        notes.append("x tick labels not read as an exact linear scale")
    return xaxis, yaxis, notes


# ------------------------------------------------------------------ curves


def curve_colors(img: np.ndarray, p: Panel, text_mask: np.ndarray) -> list[tuple[int, int, int]]:
    sub = img[p.y0: p.y1 - p.lw, p.x0 + p.lw + 1: p.x1 + 1].astype(np.int16)
    tm = text_mask[p.y0: p.y1 - p.lw, p.x0 + p.lw + 1: p.x1 + 1]
    mx, mn = sub.max(axis=2), sub.min(axis=2)
    # ink: dark (black / dark grey curves) or strongly chromatic.  Light tints (CI bands,
    # anti-aliasing halos, which mix curve colour with white) are not curve colours.
    ink = ((mx < 150) | ((mx - mn > 80) & (mn < 140))) & ~tm
    if not ink.any():
        return []
    q = (sub[ink] // 32).astype(np.int32)
    keys = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    counts = np.bincount(keys, minlength=512)
    width = sub.shape[1]
    cands: list[tuple[int, int, int]] = []
    for k in np.argsort(counts)[::-1][:16]:
        if counts[k] < 0.25 * width:
            break
        c = (int(k // 64) * 32 + 16, int(k // 8 % 8) * 32 + 16, int(k % 8) * 32 + 16)
        if not any(_same_ink(c, d) for d in cands):  # most frequent first: anti-aliased shades merge into it
            cands.append(c)
    return cands[:4]


def _hsv(c: tuple[int, int, int]) -> tuple[float, float, float]:
    return colorsys.rgb_to_hsv(*(v / 255 for v in c))


def _same_ink(c: tuple[int, int, int], d: tuple[int, int, int]) -> bool:
    """Same curve colour up to anti-aliasing / JPEG (lighter or darker shade of one hue)."""
    if sum((a - b) ** 2 for a, b in zip(c, d)) <= 70 ** 2:
        return True
    (h1, s1, v1), (h2, s2, v2) = _hsv(c), _hsv(d)
    if s1 > 0.3 and s2 > 0.3:
        dh = min(abs(h1 - h2), 1 - abs(h1 - h2))
        return dh < 0.05
    return False


def _hue_sat(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pixel hue (0-1), saturation and min channel, float32."""
    f = img.astype(np.float32)
    mx, mn = f.max(axis=2), f.min(axis=2)
    d = mx - mn
    sat = np.where(mx > 0, d / np.maximum(mx, 1), 0).astype(np.float32)
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    dd = np.maximum(d, 1e-6)
    h = np.where(mx == r, ((g - b) / dd) % 6, np.where(mx == g, (b - r) / dd + 2, (r - g) / dd + 4)) / 6
    return h.astype(np.float32), sat, mn


def ink_mask(img: np.ndarray, c: tuple[int, int, int], hs=None) -> np.ndarray:
    """Pixels of a curve drawn in colour c.  For chromatic colours this includes the line
    where a translucent confidence band of another colour is painted over it (red under a
    green band looks brown): same hue family, still saturated, not a pale tint."""
    m = color_mask(img, c)
    h0, s0, _ = _hsv(c)
    if s0 > 0.3:
        h, sat, mn = hs if hs is not None else _hue_sat(img)
        dh = np.abs(h - h0)
        dh = np.minimum(dh, 1 - dh)
        # never lighter than the line itself: a band's tint of the SAME hue is not the line
        m |= (dh < 0.06) & (sat > 0.3) & (mn < min(170, min(c) + 100))
    return m


def color_mask(img: np.ndarray, c: tuple[int, int, int], tol: int = 60) -> np.ndarray:
    acc = np.zeros(img.shape[:2], dtype=np.int32)
    for k in range(3):  # one channel at a time: no H x W x 3 int32 temporary
        d = img[:, :, k].astype(np.int16) - c[k]
        acc += d.astype(np.int32) ** 2
    return acc <= tol * tol


def _closed_runs(col: np.ndarray, hole: int, off: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in _runs(col):
        if out and a - out[-1][1] - 1 <= hole:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return [(a + off, b + off) for a, b in out]


def trace(mask: np.ndarray, p: Panel, other_ink: np.ndarray, why: list | None = None) -> Trace | None:
    """Follow one monotone (image-y non-decreasing) staircase through ``mask``.

    ``other_ink`` (the other curves' pixels, never this curve's own) is what may
    hide a stretch of this curve."""
    x_lo, x_hi = p.x0 + p.lw + 1, p.x1
    y_lo, y_hi = p.y0, p.y1 - p.lw  # exclusive bottom: never read the axis line
    cols = {x: [(a + y_lo, b + y_lo) for a, b in _runs(mask[y_lo:y_hi, x])] for x in range(x_lo, x_hi + 1)}
    # line width: the most common vertical run length (plateaus dominate a staircase),
    # capped at 2.5% of the panel height so a filled band is never taken for a line
    cap = max(8, int(0.025 * (y_hi - y_lo)))
    lens = np.bincount([b - a + 1 for x in cols for a, b in cols[x] if b - a + 1 <= cap], minlength=cap + 1)
    if lens.sum() < 0.3 * (x_hi - x_lo):
        if why is not None:
            why.append(("too few thin runs", int(lens.sum())))
        return None
    lw = float(np.searchsorted(np.cumsum(lens), lens.sum() / 2))  # median run length
    if lens[max(1, int(lw) - 2): int(lw) + 3].sum() < 0.3 * (x_hi - x_lo):
        if why is not None:
            why.append(("no dominant line width", lw))
        return None
    tmax = lw * 1.25 + 2.5  # thickest run still read as a plateau
    # start: leftmost column with a thin run, preferring the highest level
    x = next((xx for xx in range(x_lo, x_hi + 1) if any(b - a + 1 <= tmax for a, b in cols[xx])), None)
    if x is None:
        return None
    level = min((a + b) / 2 for a, b in cols[x] if b - a + 1 <= tmax)
    x_start, steps, gap_max, last_seen, last_cover = x, [], 0.0, x, x
    gaps: list[float] = []
    max_gap = max(12, int(0.12 * (x_hi - x_lo)))
    while x < x_hi:
        x += 1
        runs = cols[x]
        on = [r for r in runs if r[0] - 1.5 <= level <= r[1] + 1.5]
        if on:
            r = on[0]
            # A riser may straddle two columns and JPEG speckle breaks thin risers into
            # fragments: read it from columns x and x+1 together with 1-2 px holes closed.
            rr = _closed_runs(mask[y_lo:y_hi, x: min(x + 2, x_hi + 1)].any(axis=1), 2, y_lo)
            r2 = next((q for q in rr if q[0] - 1.5 <= level <= q[1] + 1.5), r)
            if r2[1] - level > tmax:  # a run leaving the level downwards: riser
                bottom = r2[1]
                for a2, b2 in runs:  # the riser continues past another curve crossing it
                    # gap no wider than a crossing line (+ anti-aliasing), with the other curve's ink in it
                    if a2 > bottom and a2 - bottom - 1 <= tmax + 2 and other_ink[bottom + 1: a2, max(0, x - 1): x + 2].any():
                        bottom = b2
                # next plateau: a thin run near the riser bottom within a few columns
                nxt = None
                for xx in range(x, min(x + 4, x_hi) + 1):
                    for a, b in cols[xx]:
                        # +2: the same JPEG hole the riser closing allows between riser and plateau
                        if b - a + 1 <= tmax and abs((a + b) / 2 - (bottom - lw / 2)) <= lw / 2 + 3.5 and (a + b) / 2 > level + 1.5:
                            nxt = (xx, (a + b) / 2)
                            break
                    if nxt:
                        break
                if nxt is None:
                    continue  # censor tick or a riser without a plateau: keep the level
                steps.append((float(x), level, nxt[1]))
                gaps.append(0.0)
                level, x, last_seen = nxt[1], nxt[0], nxt[0]
            else:
                level = level if abs((r[0] + r[1]) / 2 - level) > 1.0 or r[1] - r[0] + 1 > tmax else (r[0] + r[1]) / 2
                last_seen = x
            continue
        # nothing of this colour at the level: occluded, or the curve ended / dropped unseen
        if any(a - 1.5 <= level <= b + 1.5 for a, b in _runs(other_ink[y_lo:y_hi, x]) for a, b in [(a + y_lo, b + y_lo)]):
            last_cover = x
            continue  # hidden under another curve
        if x - max(last_seen, last_cover) > max_gap:
            break
        # resume below after a gap?  only through occlusion: in every gap column another
        # curve's ink covers the old level or the new one (the drop happened under it)
        below = [((a + b) / 2) for a, b in runs if b - a + 1 <= tmax and (a + b) / 2 > level + 1.5]
        if below:
            new = min(below)
            gap_cols = range(last_seen + 1, x)
            band = int(lw / 2 + 1.5)

            def covered(gx, yy):
                return other_ink[max(0, int(yy) - band): int(yy) + band + 1, gx].any()

            def continues(yy, n=6):  # the new plateau is this curve's own line for n columns
                return x + n <= x_hi and all(any(b2 - a2 + 1 <= tmax and abs((a2 + b2) / 2 - yy) <= 1.5 for a2, b2 in cols[x + k])
                                             for k in range(n))

            hidden = (bool(gap_cols) and all(covered(gx, level) or covered(gx, new) for gx in gap_cols)
                      and continues(new))
            if hidden:
                steps.append(((last_seen + x) / 2, level, new))
                gaps.append((x - last_seen) / 2)
                gap_max = max(gap_max, (x - last_seen) / 2)
                level, last_seen = new, x
    if len(steps) < 3 or last_seen - x_start < 0.3 * (x_hi - x_lo):
        if why is not None:
            why.append(("short", x_start, last_seen, x, len(steps), level, lw))
        return None
    return Trace((0, 0, 0), steps, float(x_start), float(last_seen), gap_max, lw, gaps=gaps)


def trace_edges(mask: np.ndarray, p: Panel, other_ink: np.ndarray, why: list | None = None) -> Trace | None:
    """Edge reading of a thick stroked staircase whose drops are smaller than its line width.

    A pen of width w draws a drop at x_r as a vertical stroke over [x_r - w/2, x_r + w/2], so the
    stroke's TOP edge is the curve shifted right by w/2 (minus w/2) and its BOTTOM edge is the
    curve shifted left by w/2 (plus w/2).  Both edges are read independently and must describe
    the same staircase (bottom = top + thickness, k ~ w columns earlier): a column counts only
    if the edges agree within 1.5 px. At least 25% of the width must be confirmed, at most 10% of
    the columns with a candidate may disagree, and no unconfirmed run may exceed the occlusion
    limit; otherwise the curve is refused. Unconfirmed columns are unknown, and a drop seen after
    them gets their width as extra time tolerance.  Drops are taken from
    the top edge, placed back at the stroke centre, and given as line-centre levels."""
    x_lo, x_hi = p.x0 + p.lw + 1, p.x1
    y_lo, y_hi = p.y0, p.y1 - p.lw
    width = x_hi - x_lo
    runs = {x: _closed_runs(mask[y_lo:y_hi, x], 2, y_lo) for x in range(x_lo, x_hi + 1)}
    cap = max(8, int(0.025 * (y_hi - y_lo)))
    lens = np.bincount([b - a + 1 for x in runs for a, b in runs[x] if b - a + 1 <= cap], minlength=cap + 1)
    if lens.sum() < 0.2 * width:
        if why is not None:
            why.append(("edges: too few thin runs", int(lens.sum())))
        return None
    lw = float(np.searchsorted(np.cumsum(lens), lens.sum() / 2))
    tmax = lw * 1.25 + 2.5
    x = next((xx for xx in range(x_lo, x_hi + 1) if any(b - a + 1 <= tmax for a, b in runs[xx])), None)
    if x is None:
        if why is not None:
            why.append(("edges: no thin start",))
        return None
    a0, b0 = min((r for r in runs[x] if r[1] - r[0] + 1 <= tmax), key=lambda r: r[0])
    xs, T, B, tprev, bprev = [x], [float(a0)], [float(b0)], a0, b0
    last_known, x_start = x, x
    max_gap = max(12, int(0.12 * width))
    while x < x_hi:
        x += 1
        # continuity is overlap with the previous column's run; a censor mark poking above or
        # below is part of the run here and is removed by the edge envelopes afterwards
        cand = [r for r in runs[x] if r[0] <= bprev + 2 and r[1] >= tprev - 2]
        col_other = other_ink[:, max(0, x - 1): x + 2].any(axis=1)
        if not cand:
            # hidden under another curve (also just below it: the drop happened underneath)
            if col_other[max(0, tprev - 2): int(bprev + tmax + 4)].any() and x - last_known <= max_gap:
                xs.append(x); T.append(np.nan); B.append(np.nan)
                continue  # under another curve
            if x - last_known <= 2:
                xs.append(x); T.append(np.nan); B.append(np.nan)
                continue  # JPEG hole
            # reacquire after occlusion: the highest thin run at or below the old top edge
            nxt = [r for r in runs[x] if r[0] >= tprev - 1 and r[1] - r[0] + 1 <= tmax]
            if nxt and x - last_known <= max_gap and np.isnan(T[-1]):
                cand = [min(nxt, key=lambda r: r[0])]
            else:
                break
        a, b = min(r[0] for r in cand), max(r[1] for r in cand)
        # an edge another curve touches is not this curve's edge
        ta = np.nan if col_other[max(0, a - 2): a].any() else float(a)
        tb = np.nan if col_other[b + 1: b + 3].any() else float(b)
        xs.append(x); T.append(ta); B.append(tb)
        tprev, bprev, last_known = a, b, x
    n = len(xs)
    T, B = np.array(T), np.array(B)
    while n and np.isnan(T[n - 1]) and np.isnan(B[n - 1]):
        n -= 1
    T, B, xs = T[:n], B[:n], xs[:n]
    if n < 0.3 * width:
        if why is not None:
            why.append(("edges: short", x_start, xs[-1] if xs else x_start))
        return None

    def envelope(v, top):
        """Monotone non-decreasing envelope.  Censor marks (| or +) poke ABOVE the top edge and
        BELOW the bottom edge, so the top edge takes the running max from the left and the bottom
        edge the running min from the right; each filter only ever discards a mark, never a drop.
        Anything else that bends one edge (a hole, a touching object) is caught by the double entry."""
        v = v.copy()
        idx = np.flatnonzero(~np.isnan(v))
        vals = v[idx]
        v[idx] = np.maximum.accumulate(vals) if top else np.minimum.accumulate(vals[::-1])[::-1]
        return v

    T2, B2 = envelope(T, True), envelope(B, False)
    both = ~np.isnan(T2) & ~np.isnan(B2)
    if both.sum() < 0.25 * width:
        if why is not None:
            why.append(("edges: too few readable columns", int(both.sum())))
        return None
    def agree(tc):
        k = tc + 1
        n_ok = 0
        for i in np.flatnonzero(~np.isnan(T2)):
            if any(0 <= i - kk < n and abs(B2[i - kk] - T2[i] - tc) <= 1.5 for kk in (k - 1, k, k + 1)):
                n_ok += 1
        return n_ok

    # stroke thickness: the value under which the two edges agree best (the median thickness is
    # wrong when drops are denser than the pen, because riser columns then outnumber plateaus)
    t = float(max(range(0, int(tmax) + 1), key=agree))
    k = int(round(t + 1))
    # double entry: a top-edge column counts only if the bottom edge, k columns earlier,
    # confirms it; unconfirmed columns become unknown (their width widens eps_t like an occlusion)
    ok = bad = longest = cur = 0
    for i in range(n):
        if np.isnan(T2[i]):
            continue
        best = min((abs(B2[i - kk] - T2[i] - t) for kk in (k - 1, k, k + 1) if 0 <= i - kk < n and not np.isnan(B2[i - kk])),
                   default=None)
        if best is not None and best <= 1.5:
            ok += 1
            cur = 0
        else:
            T2[i] = np.nan
            bad += 1  # no bottom-edge candidate at all is unconfirmed too
            cur += 1
            longest = max(longest, cur)
    if ok < 0.25 * width or bad > 0.1 * (ok + bad) or longest > max_gap:
        if why is not None:
            why.append(("edges: top and bottom edges disagree", ok, bad, longest))
        return None
    # steps from the top edge, from the first confirmed column, forward-filled over unknown ones
    known = np.flatnonzero(~np.isnan(T2))
    i0, i1 = int(known[0]), int(known[-1])
    Tf = T2.copy()
    for i in range(i0 + 1, n):
        if np.isnan(Tf[i]):
            Tf[i] = Tf[i - 1]
    x_start = xs[i0]
    n = i1 + 1  # nothing after the last confirmed column is read
    steps, gaps, gap_max, run_unknown = [], [], 0.0, 0
    i = i0 + 1
    while i < n:
        if np.isnan(T2[i]):
            run_unknown += 1
            i += 1
            continue
        if Tf[i] > Tf[i - 1]:
            # a drop first seen after r unknown columns happened somewhere in them: its time is
            # uncertain by r columns (unknown stretches with no drop across them cost nothing)
            gap_max = max(gap_max, float(run_unknown))
            j = i
            while j + 1 < n and Tf[j + 1] > Tf[j]:
                j += 1  # one drop spread over adjacent columns by anti-aliasing
            if Tf[j] - Tf[i - 1] >= 1.0:
                steps.append((float(xs[i]) - (t + 1) / 2, Tf[i - 1] + t / 2, Tf[j] + t / 2))
                gaps.append(float(run_unknown) + (j - i))
            i = j + 1
            run_unknown = 0
            continue
        run_unknown = 0
        i += 1
    if len(steps) < 3:
        if why is not None:
            why.append(("edges: fewer than 3 drops", len(steps)))
        return None
    # placement: the drop is the top-edge change moved back by half the stroke (+-1 px of AA),
    # a constant part of every drop's time tolerance, added in raster_claims via lw
    return Trace((0, 0, 0), steps, float(x_start), float(xs[n - 1]), max(gaps, default=0.0), t + 1,
                 method="edges", gaps=gaps)


# ------------------------------------------------------------------ figure -> claims


def _page_axis(ax: Axis, fig: FigureImage) -> Axis:
    """Pixel x-axis -> page-point x-axis."""
    sx = (fig.rect[2] - fig.rect[0]) / fig.width
    b = ax.b / sx
    a = ax.a - b * fig.rect[0]
    return Axis(a, b, ax.residual * sx, [(fig.rect[0] + c * sx, v) for c, v in ax.labels])


def ocr_risk_rows(words, p: Panel, xaxis: Axis, bottom: float | None = None) -> list[tuple[str, list[tuple[float, int]]]]:
    """Integer rows under the panel aligned to the time grid (OCR: leads only).  ``bottom`` is
    the top of the next panel below, so stacked panels never share candidate rows."""
    lab = [w for w in words if (RISK_LABEL_RE.search(w[4]) or w[4].lower() in ("risk", "at"))
           and w[1] >= p.y1 - 4 and (bottom is None or w[1] < bottom)]
    if not lab:
        return []
    top = min(w[1] for w in lab) - 4
    ints = [w for w in words if re.fullmatch(r"\d[\d,]*", w[4]) and w[1] >= top and w[5] >= 60
            and (bottom is None or w[3] <= bottom) and p.x0 - 15 <= (w[0] + w[2]) / 2 <= p.x1 + 15]
    if not ints:
        return []
    # rows: cluster word centres with a tolerance from the glyph height (resolution independent)
    gh = float(np.median([w[3] - w[1] for w in ints]))
    rows: list[list] = []
    for w in sorted(ints, key=lambda w: (w[1] + w[3]) / 2):
        yc = (w[1] + w[3]) / 2
        if rows and yc - np.mean([(v[1] + v[3]) / 2 for v in rows[-1]]) <= 0.5 * gh:
            rows[-1].append(w)
        else:
            rows.append([w])
    ticks = sorted(v for _, v in xaxis.labels)
    step = min((b - a for a, b in zip(ticks, ticks[1:]) if b > a), default=None)
    out = []
    for row in rows:
        r = sorted(row, key=lambda w: w[0])
        if len(r) < 2 or not step:
            continue
        if any(len(w[4]) > 1 and w[4][0] == "0" for w in r) or any(w[4].endswith(",") for w in r):
            continue  # "024" / "1,": a thousands number split by OCR; the row cannot be trusted
        ts = [xaxis.value((w[0] + w[2]) / 2) for w in r]
        snapped = [round(t / step) * step for t in ts]
        if any(abs(t - s) > 0.25 * step for t, s in zip(ts, snapped)) or len(set(snapped)) != len(snapped):
            continue
        vals = [int(w[4].replace(",", "")) for w in r]
        if any(b > a for a, b in zip(vals, vals[1:])):
            continue  # an increasing count is a misread far more often than a typo
        out.append((f"ocr row {len(out) + 1}", list(zip(snapped, vals))))
    return out


def digitize(doc, fig: FigureImage) -> tuple[list[RasterKM], list[str]]:
    notes: list[str] = []
    img = _rgb(doc, fig.xref)
    if img is None:
        return [], [f"page {fig.page + 1}: image {fig.xref} could not be decoded"]
    while max(img.shape[:2]) > 3000:  # 2x2 box average; every tolerance is in the reduced pixels
        h2, w2 = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
        img = (img[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2, 3).sum(axis=(1, 3), dtype=np.uint16) // 4).astype(np.uint8)
        fig = FigureImage(fig.page, fig.rect, fig.xref, w2 // 2, h2 // 2, fig.page_text, fig.km_candidate)
    if ADJUSTED_RE.search(fig.page_text):
        return [], [f"page {fig.page + 1}: raster figure skipped (adjusted / competing-risk wording)"]
    panels = find_panels(img)
    if not panels:
        return [], [f"page {fig.page + 1}: raster figure without a detectable plot frame"]
    cleaned = clean_for_ocr(img, panels)
    words = ocr_words(cleaned)
    if not words:
        why = "OCR returned no text" if shutil.which("tesseract") else "needs OCR for its axis labels (install tesseract)"
        return [], [f"page {fig.page + 1}: raster figure: {why}"]
    text_mask = np.zeros(img.shape[:2], dtype=bool)
    for x0, y0, x1, y1, t, c in words:
        if c >= 30 and len(t) >= 1:
            text_mask[max(0, int(y0) - 1): int(y1) + 2, max(0, int(x0) - 1): int(x1) + 2] = True
    out = []
    page = doc[fig.page]
    for p in panels:
        xt = x_ticks(img, p)
        xaxis, yaxis, n = calibrate_panel(p, words, xt)
        if xaxis is None and (xaxis := tick_axis(cleaned, p, xt, "x")) is not None:
            n = [m for m in n if not m.startswith("x tick")]
        if yaxis is None and (yaxis := tick_axis(cleaned, p, y_ticks(img, p), "y")) is not None:
            n = [m for m in n if not m.startswith("y tick")]
        if not xaxis or not yaxis:
            notes += [f"page {fig.page + 1}: raster panel: {m}" for m in n]
            continue
        colors = curve_colors(img, p, text_mask)
        hs = _hue_sat(img)
        masks = [ink_mask(img, c, hs) & ~text_mask for c in colors]
        del hs
        colors, masks = _drop_tints(colors, masks)
        colors, masks = _drop_halos(colors, masks)
        masks = _exclusive(img, colors, masks)
        if not masks:
            continue
        # a rising curve is read as 1 - KM only with Kaplan-Meier wording, and never when the
        # page may describe a cumulative-incidence function (competing risks do not follow KM)
        may_rise = bool(KAPLAN_RE.search(fig.page_text)) and not RISING_REFUSE_RE.search(fig.page_text)
        if not may_rise:
            notes.append(f"page {fig.page + 1}: rising raster curves not read (no Kaplan-Meier wording, or a "
                         "cumulative-incidence note on the page)")
        traces, whys = [], []
        for ci, (c, m) in enumerate(zip(colors, masks)):
            other = np.logical_or.reduce([o for j, o in enumerate(masks) if j != ci]) if len(masks) > 1 else np.zeros_like(m)
            why: list = []
            t = trace(m, p, other, why) or trace_edges(m, p, other, why)
            whys.append((c, why[:3]))
            if t is None and may_rise:
                mf, pf, of = m[::-1].copy(), _flip(p, img.shape[0]), other[::-1].copy()
                tf = trace(mf, pf, of) or trace_edges(mf, pf, of)
                if tf is not None:
                    t = _unflip(tf, img.shape[0])
                    t.rising = True
            if t is not None and _bars_on_curve(t, m, p) >= 2:
                why.append("own-colour ink far above the plateaus (error bars or a band)")
                notes.append(f"page {fig.page + 1}: raster curve with error bars or a band in its own colour refused")
                t = None
            if t is not None:
                t.color = c
                traces.append(t)
        traces = _drop_shadows(traces)
        if len({t.rising for t in traces}) > 1:
            # one panel is falling or rising; arms read in both orientations mean one of the
            # readings is wrong, and 1 - v applied to the wrong arms would invent contradictions
            notes.append(f"page {fig.page + 1}: raster panel with arms read in opposite orientations refused")
            continue
        rising = bool(traces) and traces[0].rising
        if traces and (y_scale := axis_scale(max(v for _, v in yaxis.labels), rising)) is None:
            notes.append(f"page {fig.page + 1}: raster y axis could be probability or percent (rising curve, labels <= 1.5): refused")
            continue
        if not traces:
            notes.append(f"page {fig.page + 1}: raster panel without a traceable staircase")
            if os.environ.get("TRIALGEIST_RASTER_DEBUG"):
                notes.append(f"debug: lw={p.lw} colours={colors} why={whys}")
            continue
        # numbers at risk: PDF text under the image first (proof grade), then OCR inside the image (leads)
        px_axis = _page_axis(xaxis, fig)
        sx = (fig.rect[2] - fig.rect[0]) / fig.width
        sy = (fig.rect[3] - fig.rect[1]) / fig.height
        rr = risk_rows(page, px_axis, fig.rect[3] - 2 if p.y1 > 0.75 * fig.height else fig.rect[1] + p.y1 * sy,
                       fig.rect[0] + p.x0 * sx, fig.rect[0] + p.x1 * sx)
        ocr = False
        if not rr:
            below = [q.y0 for q in panels if q.y0 > p.y1 and q.x0 < p.x1 and p.x0 < q.x1]
            rr, ocr = ocr_risk_rows(words, p, xaxis, min(below, default=None)), True
        if not rr:
            notes.append(f"page {fig.page + 1}: raster KM panel without numbers at risk")
            continue
        labels = legend_labels(img, words, [t.color for t in traces])
        for ci, ri in pair_rows(labels, rr).items():
            traces[ci].label, traces[ci].row = labels[ci], ri
        # one event moves the curve by at least 1/N0 of the y-range (largest N0 among the rows the
        # curve may belong to): px of the smallest possible single-event drop
        px_per_unit = 1.0 / max(abs(yaxis.b) * y_scale, 1e-12)
        kept = []
        for t in traces:
            cand = [rr[t.row]] if t.row is not None else rr
            n0 = max((max((n for _, n in row), default=0) for _, row in cand), default=0)
            if n0 > 0 and _sloped(t, px_per_unit / n0):
                notes.append(f"page {fig.page + 1}: raster curve drawn with sloping segments (not a KM staircase) refused")
                continue
            kept.append(t)
        if not kept:
            continue
        out.append(RasterKM(fig, p, xaxis, yaxis, y_scale, rising, kept, rr, ocr))
    return out, notes


def _sloped(t: Trace, event_px: float) -> bool:
    """A curve drawn as points joined by sloping lines (not a staircase) is read as runs of drops
    far smaller than one event, and its decreases are smeared across time. Every KM drop is at
    least S(t)/n(t) >= 1/N0 of the y-range, so when one event is clearly visible (event_px >= 3),
    a run of three or more sub-event drops adding up to most of an event is a slope, not steps."""
    if event_px < 3:
        return False
    small, near = 0.4 * event_px, max(3 * t.lw, 12)
    run_n, run_sum, last_x = 0, 0.0, None
    for x, b, a in t.steps:
        d = abs(a - b)
        if d < small:
            if last_x is not None and x - last_x > near:  # a slope is continuous; lone 1 px jitter is not
                run_n, run_sum = 0, 0.0
            run_n, run_sum, last_x = run_n + 1, run_sum + d, x
            if run_n >= 3 and run_sum >= max(0.6 * event_px, 3.0):
                return True
        else:
            run_n, run_sum, last_x = 0, 0.0, None
    return False


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    """Square dilation by r px (separable, no wrap-around)."""
    h = m.copy()
    for k in range(1, r + 1):
        h[:, k:] |= m[:, :-k]
        h[:, :-k] |= m[:, k:]
    out = h.copy()
    for k in range(1, r + 1):
        out[k:] |= h[:-k]
        out[:-k] |= h[k:]
    return out


def _exclusive(img: np.ndarray, colors, masks) -> list[np.ndarray]:
    """Remove from each colour's mask the edge pixels of the other curves: pixels within
    2 px of another colour's core (tight match) are that curve's anti-aliasing halo,
    which can look like a lighter curve (grey around black).  Crossings lose a few pixels
    and are bridged as occlusions."""
    cores = [color_mask(img, c, tol=45) for c in colors]
    out = []
    for i, m in enumerate(masks):
        others = np.zeros_like(m)
        for j, c in enumerate(cores):
            if j != i:
                others |= c
        out.append(m & ~(_dilate(others, 3) & ~cores[i]))
    return out


def _drop_tints(colors, masks):
    """A chromatic colour that is a clearly lighter shade of the same hue as a darker,
    more saturated curve colour is that curve's confidence band, band edge or error bars,
    not another arm.  Its stepped edge would otherwise be digitised as a survival curve.
    (Two arms drawn in light and dark shades of one hue are refused, never misread.)"""
    hsv = [_hsv(c) for c in colors]
    keep = []
    for i, c in enumerate(colors):
        hi, si, _ = hsv[i]
        tint = False
        for j, d in enumerate(colors):
            hj, sj, _ = hsv[j]
            dh = min(abs(hi - hj), 1 - abs(hi - hj))
            if j != i and si > 0.2 and sj > 0.3 and dh < 0.07 and min(c) > min(d) + 50 and sj >= si:
                tint = True
                break
        if not tint:
            keep.append(i)
    return [colors[i] for i in keep], [masks[i] for i in keep]


def _drop_halos(colors, masks):
    """A colour whose pixels mostly sit within 2 px of a more frequent colour's pixels is the
    anti-aliased edge of that curve (grey around black, pink around red), not a curve."""
    sizes = [int(m.sum()) for m in masks]
    keep = []
    for i, m in enumerate(masks):
        halo = False
        for j, o in enumerate(masks):
            if j != i and sizes[j] > sizes[i] and sizes[i]:
                near = (m & _dilate(o, 2)).sum() / sizes[i]
                if near > 0.6:
                    halo = True
                    break
        if not halo:
            keep.append(i)
    return [colors[i] for i in keep], [masks[i] for i in keep]


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9+]", "", t.lower())


def legend_labels(img: np.ndarray, words, colors) -> dict[int, str]:
    """Curve index -> legend label.  A swatch is a short horizontal run (8-80 px, at most
    4 px thick) of the curve colour with an OCR'd word starting within 20 px to its right on
    the same line; the label is that line's words.  Only unique, distinct labels are kept."""
    found: dict[int, set[str]] = {}
    good = [w for w in words if w[5] >= 70 and re.search(r"[A-Za-z0-9]", w[4])]
    for ci, c in enumerate(colors):
        m = color_mask(img, c, tol=50)
        for w in good:
            yc = int((w[1] + w[3]) / 2)
            x0 = int(w[0])
            band = m[max(0, yc - 5): yc + 6, max(0, x0 - 100): max(0, x0 - 2)]
            if band.size == 0:
                continue
            for row in band:
                runs = [(a, b) for a, b in _runs(row) if 8 <= b - a + 1 <= 80 and band.shape[1] - 1 - b <= 20]
                if runs:
                    break
            else:
                continue
            thick = int(band.any(axis=1).sum())
            if thick > 6:
                continue  # part of a curve, not a swatch
            line = sorted((v for v in good if abs((v[1] + v[3]) / 2 - yc) <= 4 and v[0] >= w[0] - 1 and v[0] - w[2] < 200),
                          key=lambda v: v[0])
            label, last = [], None
            for v in line:
                if last is not None and v[0] - last > 25:
                    break
                label.append(v[4])
                last = v[2]
            found.setdefault(ci, set()).add(" ".join(label))
    out = {ci: next(iter(ls)) for ci, ls in found.items() if len(ls) == 1}
    norms = [_norm(v) for v in out.values()]
    dup = {n for n in norms if norms.count(n) > 1}
    return {ci: v for ci, v in out.items() if _norm(v) not in dup}  # ambiguous labels only


def pair_rows(labels: dict[int, str], risk) -> dict[int, int]:
    """Curve index -> risk-row index when the legend label equals the row label exactly
    (normalised) and the pairing is one-to-one."""
    rows = [_norm(lab) for lab, _ in risk]
    out = {}
    for ci, lab in labels.items():
        hits = [i for i, r in enumerate(rows) if r and r == _norm(lab)]
        if len(hits) == 1:
            out[ci] = hits[0]
    if len(set(out.values())) != len(out):
        return {}
    return out


def _drop_shadows(traces: list[Trace]) -> list[Trace]:
    """A trace whose drops mostly coincide with another trace's drops is a shadow (halo,
    edge pixels, or a colour that hops between curves).  Keep the trace with more drops."""
    def same(s, t):
        return abs(s[0] - t[0]) <= 4 and abs(s[1] - t[1]) <= 4 and abs(s[2] - t[2]) <= 4

    keep = []
    for i, t in enumerate(traces):
        shared = {j: sum(1 for s in t.steps if any(same(s, v) for v in u.steps)) for j, u in enumerate(traces) if j != i}
        # the trace is mostly another, longer curve's drops
        shadow = any(n >= 0.5 * len(t.steps) and (len(traces[j].steps), j) > (len(t.steps), i) for j, n in shared.items())
        # a trace sharing many of its drops with two different curves hops between them
        # (a single shared drop is coincidence on dense curves that start together)
        if sum(1 for n in shared.values() if n >= max(2, 0.25 * len(t.steps))) >= 2:
            shadow = True
        if not shadow:
            keep.append(t)
    return keep


def _bars_on_curve(t: Trace, mask: np.ndarray, p: Panel) -> int:
    """Places where the curve's own ink runs far above its plateau, away from any drop: error bars
    or a band in the curve's own colour (KM steps only fall; a censor tick is a short mark).
    A reader could take such ink for risers or plateaus, so a curve that has them is refused."""
    if not t.steps:
        return 0
    lim = max(5 * p.lw, 0.035 * (p.y1 - p.y0), 8)
    near = max(p.lw + 3, 4)
    gaps = t.gaps if t.gaps is not None else [t.gap_px] * len(t.steps)
    hits, last = 0, -10 ** 9
    for x in range(int(t.x_start) + near, int(t.x_end) - near):
        if any(abs(x - sx) <= near + g for (sx, _, _), g in zip(t.steps, gaps)):
            continue
        k = sum(1 for sx, _, _ in t.steps if sx <= x)
        level = t.steps[k - 1][2] if k else t.steps[0][1]
        y = int(round(min(level, t.steps[k - 1][1] if k else level) - p.lw / 2 - 1))
        run, miss = 0, 0
        while y - run - miss > p.y0 and miss <= 1:
            if mask[y - run - miss, x]:
                run += 1 + miss
                miss = 0
            else:
                miss += 1
        if run > lim:
            if x - last > near:
                hits += 1
            last = x
    return hits


def _flip(p: Panel, h: int) -> Panel:
    """The panel in the vertically mirrored image (row y -> h - 1 - y).

    trace() reads rows [y0, y1 - lw): start just below the mirrored x-axis line
    and end at the mirrored top of the y-axis (inclusive)."""
    return Panel(x0=p.x0, y1=h - p.y0 + p.lw, x1=p.x1, y0=h - 1 - p.y1 + p.lw + 1, lw=p.lw)


def _unflip(t: Trace, h: int) -> Trace:
    return replace(t, steps=[(x, h - 1 - b, h - 1 - a) for x, b, a in t.steps])


def raster_claims(rk: RasterKM) -> list:
    from ..claims import KMArmClaim, KMRiskRow, KMStepIn

    fig = rk.figure
    sx = (fig.rect[2] - fig.rect[0]) / fig.width
    sy = (fig.rect[3] - fig.rect[1]) / fig.height
    rows = [KMRiskRow(label=lab, risk_times=[f"{t:.6f}" for t, _ in row], at_risk=[n for _, n in row]) for lab, row in rk.risk]
    out = []
    for i, tr in enumerate(rk.traces):
        eps_px = tr.lw / 2 + 2.0 + 3 * rk.yaxis.residual
        eps = abs(rk.yaxis.b) * rk.y_scale * eps_px
        # time tolerance: 3 px + axis residual (+ half the stroke for edge reading) for every drop,
        # plus, per drop, the occlusion it was hidden behind (a drop seen late may lie anywhere in it)
        base_px = 3.0 + 3 * rk.xaxis.residual + (tr.lw / 2 if tr.method == "edges" else 0.0)
        eps_t = abs(rk.xaxis.b) * base_px
        gaps = tr.gaps if tr.gaps is not None else [tr.gap_px] * len(tr.steps)
        steps = []
        for (x, b, a), g in zip(tr.steps, gaps):
            vb, va = rk.yaxis.value(b) * rk.y_scale, rk.yaxis.value(a) * rk.y_scale
            if rk.rising:
                vb, va = 1 - vb, 1 - va
            steps.append(KMStepIn(t=f"{rk.xaxis.value(x):.6f}", before=f"{min(max(vb, 0), 1):.6f}", after=f"{min(max(va, 0), 1):.6f}",
                                  dt=f"{abs(rk.xaxis.b) * g:.6f}"))
        bbox = (fig.rect[0] + rk.panel.x0 * sx, fig.rect[1] + rk.panel.y0 * sy, fig.rect[0] + rk.panel.x1 * sx, fig.rect[1] + rk.panel.y1 * sy)
        prov = [Provenance(source="raster", page=fig.page, bbox=bbox,
                           text=f"raster KM curve {i + 1} (colour {tr.color}){' plotted as 1 - S(t)' if rk.rising else ''}",
                           note=(f"image xref {fig.xref} {fig.width}x{fig.height}px; " +
                                 ("numbers at risk read by OCR from the image; " if rk.ocr_numbers else "") +
                                 f"read by {tr.method}; {len(tr.steps)} drops" + (" (first 200 listed)" if len(tr.steps) > 200 else "") + "; drops (px): " + ";".join(f"{x:.0f}:{b:.1f}->{a:.1f}" for x, b, a in tr.steps[:200]))),
                # the studio draws these on the page so a reviewer sees exactly what was digitised
                Provenance(source="raster", page=fig.page, text=f"digitised drops of curve {i + 1}",
                           note="vertices: " + ";".join(f"{fig.rect[0] + x * sx:.2f},{fig.rect[1] + y * sy:.2f}"
                                                        for x, bb, aa in tr.steps[:200] for y in (bb, aa)))]
        alts = [rows[tr.row]] if tr.row is not None else rows
        name = f"page {fig.page + 1} raster curve {i + 1}" + (f" ({tr.label}, paired by legend)" if tr.row is not None else "")
        out.append(KMArmClaim(label=name, alternatives=alts, steps=steps,
                              eps=f"{eps:.6f}", eps_t=f"{eps_t:.6f}", provenance=prov, ocr_numbers=rk.ocr_numbers))
    return out


def find_raster_km(path: str) -> tuple[list[RasterKM], list[str], list[FigureImage]]:
    with pymupdf.open(path) as doc:
        figs = figure_images(doc)
        plots, notes = [], []
        for fig in figs:
            if not fig.km_candidate:
                continue
            ps, ns = digitize(doc, fig)
            plots += ps
            notes += ns
    pymupdf.TOOLS.store_shrink(100)  # MuPDF caches decoded images; a corpus scan must not accumulate them
    return plots, notes, figs
