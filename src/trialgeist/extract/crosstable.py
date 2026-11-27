"""Cross-cell and cross-column claims from a table grid.

Works on a neutral ``Grid`` (built from JATS XML markup or from PDF word
positions) so both input paths share one set of conservative rules:

* Total column: a column whose header says Total/Overall/All and whose N is
  exactly the sum of the other columns' Ns is their union.  Row counts must
  add up exactly; a pooled mean must lie between the arm means (or equal the
  N-weighted mean when the paper states there is no missing data).
* Categories of one variable (rows under one section header, n (%) cells,
  percentages summing to 100 within rounding) share one denominator.
* Median (IQR) and median (range) cells: Q1 <= median <= Q3, min <= median <= max.

Every claim carries the cell provenance; the engines decide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

from ..claims import ArmTotalClaim, CommonDenominatorClaim, MeanPartitionClaim, MedianRangeClaim
from ..verdict import Provenance
from .pdf_text import MEANSD_RE, NPCT_RE, npct_count, OF_BEFORE_RE, RISK_CONTEXT_RE, UNIT_LABEL_RE

TOTAL_RE = re.compile(r"\b(total|overall|all (patients|participants|subjects|children|women|men)|combined|whole (cohort|sample))\b", re.I)
SIGNED = r"[\-\u2212\u2013]?\d+(?:[.\u00b7]\d+)?"
SPREAD_RE = re.compile(r"^\s*(" + SIGNED + r")\s*[\(\[]\s*(" + SIGNED + r")\s*(?:\u2013|\u2014|-|to|,|;|\u2212)\s*(" + SIGNED + r")\s*[\)\]]\s*$")
IQR_CTX_RE = re.compile(r"\bIQR\b|interquartile|quartile|\bQ1\b|25th", re.I)
RANGE_CTX_RE = re.compile(r"\brange\b|min(imum)?\s*[-\u2013,/]\s*max(imum)?", re.I)
CI_CTX_RE = re.compile(r"\bCI\b|confidence|credible|\bCrI\b|\bSE\b|standard error|percentile interval", re.I)
MEDIAN_RE = re.compile(r"\bmedian\b", re.I)
MEAN_RE = re.compile(r"\bmean\b", re.I)
MISSING_CTX_RE = re.compile(r"missing|unknown|not (reported|recorded|available)|n/?a\b", re.I)


@dataclass
class GridRow:
    label: str
    context: str  # own label + section header text (what the row's numbers are)
    section: int  # index of the section-header row above (-1: none)
    cells: dict[int, str]
    prov: dict[int, Provenance] = field(default_factory=dict)


@dataclass
class Grid:
    label: str
    caption: str
    col_n: dict[int, int]
    col_head: dict[int, str]
    rows: list[GridRow]
    complete: bool = False


def _num(s: str) -> str:
    return s.replace("\u00b7", ".").replace("\u2212", "-").replace("\u2013", "-")


def _total_column(g: Grid) -> tuple[int, list[int]] | None:
    tot = [c for c, h in g.col_head.items() if c in g.col_n and TOTAL_RE.search(h)]
    if len(tot) != 1:
        return None
    t = tot[0]
    arms = sorted(c for c in g.col_n if c != t)
    if len(arms) < 2 or sum(g.col_n[c] for c in arms) != g.col_n[t]:
        return None
    return t, arms


def _one(rx: re.Pattern, cell: str):
    ms = list(rx.finditer(cell))
    return ms[0] if len(ms) == 1 else None


def _all_one(rx: re.Pattern, cells: list[str]) -> list[re.Match] | None:
    """The single match in every cell, or None if any cell has zero or several."""
    out = [m for m in (_one(rx, c) for c in cells) if m is not None]
    return out if len(out) == len(cells) else None


def total_column_claims(g: Grid) -> list:
    tc = _total_column(g)
    if not tc:
        return []
    t, arms = tc
    cols = arms + [t]
    out: list = []
    for row in g.rows:
        if RISK_CONTEXT_RE.search(row.context) or not all(c in row.cells for c in cols):
            continue
        prov = [row.prov[c] for c in cols if c in row.prov]
        mean_row = MEAN_RE.search(row.context) and not MEDIAN_RE.search(row.context)
        # counts
        ms = _all_one(NPCT_RE, [row.cells[c] for c in cols])
        if not mean_row and ms and not any(OF_BEFORE_RE.search(row.cells[c][: m.start()]) for c, m in zip(cols, ms)):
            counts = [npct_count(m) for m in ms]
            if not UNIT_LABEL_RE.search(row.label):
                out.append(ArmTotalClaim(label=row.label[:80], total=counts[-1], counts=counts[:-1], total_n=g.col_n[t],
                                         arm_ns=[g.col_n[c] for c in arms], provenance=prov))
            continue
        # means
        if mean_row:
            mm = _all_one(MEANSD_RE, [row.cells[c] for c in cols])
            if mm:
                means = [_num(m.group(1)) for m in mm]
                out.append(MeanPartitionClaim(label=row.label[:80], overall_mean=means[-1],
                                              parts=[(g.col_n[c], v) for c, v in zip(arms, means[:-1])],
                                              weights_exact=g.complete, provenance=prov))
    return out


def common_denominator_claims(g: Grid) -> list:
    """Rows under one section header whose n (%) cells in a column sum to ~100%."""
    out: list = []
    blocks: dict[int, list[GridRow]] = {}
    for row in g.rows:
        if row.section >= 0:
            blocks.setdefault(row.section, []).append(row)
    for sec, rows in blocks.items():
        if len(rows) < 2 or any(RISK_CONTEXT_RE.search(r.context) or MEAN_RE.search(r.context) or MEDIAN_RE.search(r.context)
                                for r in rows):
            continue
        cols = set.intersection(*(set(r.cells) for r in rows))
        for c in sorted(cols):
            ms = _all_one(NPCT_RE, [r.cells[c] for r in rows])
            if not ms or any(OF_BEFORE_RE.search(r.cells[c][: m.start()]) for r, m in zip(rows, ms)):
                continue
            pcts = [m.group(2) for m in ms]
            vals = [Fraction(_num(p)) for p in pcts]
            dec = max((len(_num(p).split(".")[1]) if "." in _num(p) else 0) for p in pcts)
            if abs(sum(vals) - 100) > Fraction(len(vals), 2 * 10 ** dec):  # not a partition of 100%
                continue
            counts = [npct_count(m) for m in ms]
            out.append(CommonDenominatorClaim(label=f"{g.label}: {rows[0].context[:60]}", counts=counts, percents=pcts,
                                              d_max=_confirmed_n(g, c, {id(r) for r in rows}),
                                              provenance=[r.prov[c] for r in rows if c in r.prov]))
    return out


def _confirmed_n(g: Grid, c: int, skip: set[int]) -> int | None:
    """Header N bounds the denominator only if every other n (%) cell in the
    column (at least 5, outside the block) is exactly count/N."""
    from .pdf_text import _exact
    from ..claims import PercentClaim

    n = g.col_n.get(c)
    if n is None:
        return None
    cells = [m for r in g.rows if id(r) not in skip and c in r.cells and not RISK_CONTEXT_RE.search(r.context)
             for m in [_one(NPCT_RE, r.cells[c])] if m]
    if len(cells) < 5:
        return None
    ok = all(_exact(PercentClaim(count=npct_count(m), percent=m.group(2), n=n)) for m in cells)
    return n if ok else None


def spread_claims(g: Grid) -> list:
    out: list = []
    for row in g.rows:
        ctx = row.context + " " + g.caption
        if CI_CTX_RE.search(row.context) or not MEDIAN_RE.search(row.context):
            continue
        bounds = "iqr" if IQR_CTX_RE.search(row.context) else "range" if RANGE_CTX_RE.search(row.context) else None
        if bounds is None or (IQR_CTX_RE.search(row.context) and RANGE_CTX_RE.search(row.context)):
            continue  # ambiguous: "median (IQR or range)"
        if CI_CTX_RE.search(ctx) and not (IQR_CTX_RE.search(row.context) or RANGE_CTX_RE.search(row.context)):
            continue
        for c, cell in row.cells.items():
            m = SPREAD_RE.match(cell)
            if not m:
                continue
            med, lo, hi = (_num(x) for x in m.groups())
            if Fraction(lo) > Fraction(hi):
                continue  # unreadable as an interval (or a misparse)
            out.append(MedianRangeClaim(label=row.label[:80], median=med, min=lo, max=hi, bounds=bounds,
                                        provenance=[row.prov[c]] if c in row.prov else []))
    return out


TABLE_CAPTION_RE = re.compile(r"^\s*Table\s*\d+\b", re.I)
SECTION_LABEL_RE = re.compile(r"(\bn\s*\(\s*%\s*\)|\bno\.\s*\(\s*%\s*\)|:\s*$)", re.I)


def pdf_grids(pages) -> list[Grid]:
    """Grids from PDF word positions, anchored on a header row holding >= 2 "n = N".

    Columns are the header-N centres; a word belongs to the nearest centre
    within half the column spacing, words left of the first column form the
    row label.  A label-only row opens a section only if it looks like a
    variable header (ends with ':' or says n (%)) or the next row is indented
    under it; otherwise it may be a wrapped label, and it closes any section.
    """
    from .pdf_text import COMPLETE_RE

    grids: list[Grid] = []
    for pt in pages:
        rows = pt.rows
        caps = [i for i, r in enumerate(rows) if TABLE_CAPTION_RE.match(r.text)]
        complete = bool(COMPLETE_RE.search(pt.text))
        for k, ci in enumerate(caps):
            end = caps[k + 1] if k + 1 < len(caps) else len(rows)
            hdr = next((j for j in range(ci, min(end, ci + 12))
                        if sum(1 for x, y, n in pt.header_ns if abs(y - rows[j].y) < 0.1) >= 2), None)
            if hdr is None:
                continue
            ns = sorted((x, n) for x, y, n in pt.header_ns if abs(y - rows[hdr].y) < 0.1)
            xs = [x for x, _ in ns]
            gap = min((b - a for a, b in zip(xs, xs[1:])), default=80.0)
            half = gap / 2
            left = xs[0] - half

            def col_of(w) -> int | None:
                xc = (w.x0 + w.x1) / 2
                j = min(range(len(xs)), key=lambda i: abs(xs[i] - xc))
                return j + 1 if abs(xs[j] - xc) <= half else None

            col_head: dict[int, str] = {}
            for r in rows[ci + 1: hdr + 1]:
                for w in r.words:
                    c = col_of(w)
                    if c is not None and w.x0 >= left:
                        col_head[c] = (col_head.get(c, "") + " " + w.text).strip()
            grid_rows: list[GridRow] = []
            sec, sec_label, sec_x = -1, "", 0.0
            body = rows[hdr + 1: end]
            for j, r in enumerate(body):
                lab_words = [w for w in r.words if w.x1 <= left + 1]
                cells_w: dict[int, list] = {}
                for w in r.words:
                    if w.x1 > left + 1:
                        c = col_of(w)
                        if c is not None:
                            cells_w.setdefault(c, []).append(w)
                label = " ".join(w.text for w in lab_words)
                if not cells_w:
                    if not label:
                        continue
                    x0 = min(w.x0 for w in lab_words)
                    nxt = next((b for b in body[j + 1:] if b.words), None)
                    indented = bool(nxt and nxt.words[0].x0 > x0 + 3)
                    if SECTION_LABEL_RE.search(label) or indented:
                        sec, sec_label, sec_x = hdr + 1 + j, label, x0
                    else:
                        sec = -1
                    continue
                if sec >= 0 and lab_words and min(w.x0 for w in lab_words) < sec_x - 1:
                    sec = -1  # outdented past the section header: the section is over
                cells = {c: " ".join(w.text for w in ws) for c, ws in cells_w.items()}
                prov = {c: Provenance(source="pdf", page=pt.index,
                                      bbox=(min(w.x0 for w in ws), min(w.y0 for w in ws), max(w.x1 for w in ws), max(w.y1 for w in ws)),
                                      text=cells[c], note=f"{rows[ci].text[:60]} | {label[:60]}")
                        for c, ws in cells_w.items()}
                ctx = label + (" | " + sec_label if sec >= 0 else "")
                grid_rows.append(GridRow(label=label, context=ctx, section=sec, cells=cells, prov=prov))
            grids.append(Grid(label=rows[ci].text[:40], caption=rows[ci].text, col_n={i + 1: n for i, (_, n) in enumerate(ns)},
                              col_head=col_head, rows=grid_rows, complete=complete))
    return grids


def grid_claims(g: Grid) -> list:
    return total_column_claims(g) + common_denominator_claims(g) + spread_claims(g)
