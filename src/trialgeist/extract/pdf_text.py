"""Deterministic PDF text/table extraction with bbox provenance.

Only patterns whose meaning is unambiguous from the printed context become
claims.  Everything is conservative on purpose (PLAN.md decision 5/6):

* ``n (pct%)`` cells become claims only when the cell, its row label, its
  section header or the column header says the numbers are percentages.
  Without a reliable denominator, only the "no whole-number denominator at
  all" check is used; a column header ``N = x`` gives an upper bound.
* ``mean (SD)`` cells become claims only for rows naming a bounded integer
  instrument from a curated list (NIHSS, GCS, mRS, ...), with that
  instrument's scale.  Exact-N checks (GRIM/SPRITE) are used only when the
  same page states that data were complete; otherwise only checks valid for
  every N' <= N are used.
* Footnotes mentioning weighting or imputation switch percentage claims off
  for that page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

from ..claims import MeanClaim, PercentClaim
from ..verdict import Provenance

NUM = r"\d+(?:[.\u00b7]\d+)?"
DEC = r"\d+[.\u00b7]\d+"
NPCT_RE = re.compile(r"(?<![\d.,\u00b7/\-\u2013])(\d{1,3}(?:,\d{3})+|\d{1,6})\s*\(\s*(" + NUM + r")\s*(%?)\s*\)")
MEANSD_RE = re.compile(r"(?<![\d.,\u00b7])([\-\u2212]?" + DEC + r")\s*(?:\(\s*(" + DEC + r")\s*\)|\u00b1\s*(" + DEC + r"))")
def npct_count(m: re.Match) -> int:
    """The count of an NPCT_RE match ("1,075 (51.1)" -> 1075)."""
    return int(m.group(1).replace(",", ""))


HEADER_N_RE = re.compile(r"\(?\s*[nN]\s*=\s*(\d[\d,]*)\s*\)?")
PCT_CONTEXT_RE = re.compile(r"\b(no|n|number)\.?\s*\(\s*%\s*\)|\(\s*%\s*\)|\bnumbers?\s*\(\s*percent(age)?s?\s*\)", re.I)
PCTN_RE = re.compile(r"(?<![\d.])(" + NUM + r")\s*%?\s*\(\s*n\s*=\s*(\d{1,6})\s*\)")  # "36.6 (n=11)": percent first
MEDIAN_CONTEXT_RE = re.compile(r"median|IQR|interquartile|range", re.I)
UNIT_LABEL_RE = re.compile(r"\b(mean|SD|median|IQR|yr|years?|kg|cm|mg|mmol|ml|score)\b", re.I)
MEAN_CONTEXT_RE = re.compile(r"mean\s*(\(|\u00b1|\+/-)\s*(SD|s\.d\.)|mean\s*\(\s*SD\s*\)", re.I)
UNSAFE_PCT_RE = re.compile(r"weighted|imput|estimated percentage|survey", re.I)
BASELINE_CAPTION_RE = re.compile(r"^\s*Table\s*\d+\b.{0,80}(baseline|characteristics of (the )?(patients|participants))", re.I)
SKIP_BASELINE_ROW_RE = re.compile(r"geometric|median|IQR|interquartile", re.I)
OF_BEFORE_RE = re.compile(r"(\bof|\bout of|/|\bin)\s*$", re.I)
RISK_CONTEXT_RE = re.compile(r"\bat[- ]risk\b|\bcensored\b|\bno\.? at risk", re.I)
COMPLETE_RE = re.compile(r"no missing (baseline )?data|complete data (were|was) available for all", re.I)

# Bounded integer instruments: (pattern, min, max, granularity)
INSTRUMENTS: list[tuple[re.Pattern, str, str, str]] = [
    (re.compile(r"\bNIHSS\b|national institutes of health stroke scale", re.I), "0", "42", "1"),
    (re.compile(r"\bGCS\b|glasgow coma scale", re.I), "3", "15", "1"),
    (re.compile(r"\bmRS\b|modified rankin", re.I), "0", "6", "1"),
    (re.compile(r"\bMMSE\b|mini-mental", re.I), "0", "30", "1"),
    (re.compile(r"\bMoCA\b|montreal cognitive", re.I), "0", "30", "1"),
    (re.compile(r"\bSOFA\b", re.I), "0", "24", "1"),
    (re.compile(r"\bAPACHE\s*II\b", re.I), "0", "71", "1"),
    (re.compile(r"\bECOG\b", re.I), "0", "5", "1"),
    (re.compile(r"\bHunt[- ]and[- ]Hess\b|\bHunt[- ]Hess\b", re.I), "1", "5", "1"),
    (re.compile(r"\bWFNS\b", re.I), "1", "5", "1"),
    (re.compile(r"\bLikert\b.*\b1\s*[-\u2013]\s*5\b|\b1\s*[-\u2013]\s*5\b.*\bLikert\b", re.I), "1", "5", "1"),
    (re.compile(r"\bLikert\b.*\b1\s*[-\u2013]\s*7\b", re.I), "1", "7", "1"),
    (re.compile(r"numeric(al)? rating scale|\bNRS\b", re.I), "0", "10", "1"),
]


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass
class Row:
    words: list[Word]
    y: float

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    def span_bbox(self, start: int, end: int) -> tuple[float, float, float, float]:
        """bbox of the words overlapping character range [start, end) of self.text."""
        pos, boxes = 0, []
        for w in self.words:
            a, b = pos, pos + len(w.text)
            if a < end and b > start:
                boxes.append(w)
            pos = b + 1
        if not boxes:
            boxes = self.words
        return (min(w.x0 for w in boxes), min(w.y0 for w in boxes), max(w.x1 for w in boxes), max(w.y1 for w in boxes))

    def label_before(self, x: float) -> str:
        return " ".join(w.text for w in self.words
                        if w.x1 <= x + 0.5 and not (re.fullmatch(r"[\d.,()%\u00b7\u00b1\-\u2212]+", w.text) and re.search(r"\d", w.text)))


@dataclass
class PageText:
    index: int
    rows: list[Row]
    text: str
    header_ns: list[tuple[float, float, int]] = field(default_factory=list)  # (x_center, y, N)


def rows_from_page(page: pymupdf.Page, tol: float = 2.5) -> list[Row]:
    words = [Word(*w[:5]) for w in page.get_text("words")]
    words.sort(key=lambda w: ((w.y0 + w.y1) / 2, w.x0))
    rows: list[Row] = []
    for w in words:
        yc = (w.y0 + w.y1) / 2
        if rows and abs(rows[-1].y - yc) <= tol:
            rows[-1].words.append(w)
        else:
            rows.append(Row([w], yc))
    for r in rows:
        r.words.sort(key=lambda w: w.x0)
    return rows


def load_pages(path: str) -> list[PageText]:
    with pymupdf.open(path) as doc:
        return _load_pages(doc)


def _load_pages(doc) -> list[PageText]:
    out = []
    for i, page in enumerate(doc):
        rows = rows_from_page(page)
        pt = PageText(i, rows, page.get_text())
        for r in rows:
            for m in HEADER_N_RE.finditer(r.text):
                bb = r.span_bbox(m.start(), m.end())
                pt.header_ns.append(((bb[0] + bb[2]) / 2, r.y, int(m.group(1).replace(",", ""))))
        out.append(pt)
    return out


def _context(rows: list[Row], idx: int, x: float, back: int = 40) -> tuple[str | None, str]:
    """Nearest statistic marker walking up from the cell's row label.

    Returns ('pct' | 'mean' | 'median' | None, context text).  The nearest
    marker wins, so a "mean (SD)" row inside a "No. (%)" table is a mean row.
    """
    seen = []
    for j in range(idx, max(-1, idx - back - 1), -1):
        t = rows[j].label_before(x) if j == idx else rows[j].text
        seen.append(t)
        if RISK_CONTEXT_RE.search(rows[j].text):  # numbers-at-risk rows: "111 (0)" is at-risk (censored)
            return "risk", " | ".join(seen)
        kinds = [(MEAN_CONTEXT_RE, "mean"), (MEDIAN_CONTEXT_RE, "median"), (PCT_CONTEXT_RE, "pct")]
        hits = [(m.start(), k) for rx, k in kinds for m in [rx.search(t)] if m]
        if hits:
            if j == idx:
                return min(hits)[1], " | ".join(seen)
            kinds_here = {k for _, k in hits}
            return (kinds_here.pop() if len(kinds_here) == 1 else "mixed"), " | ".join(seen)
    return None, " | ".join(seen)


CAPTION_RE = re.compile(r"^\s*(Table|Figure|Fig\.)\s*\d", re.I)


def _header_n(pt: PageText, x: float, y: float) -> tuple[int | None, int | None]:
    """(column N nearest above, max N above) for a cell at (x, y), within the same table."""
    caps = [r.y for r in pt.rows if r.y < y and CAPTION_RE.match(r.text)]
    top = max(caps) if caps else -1.0
    above = [(xc, yy, n) for xc, yy, n in pt.header_ns if top < yy < y]
    if not above:
        return None, None
    col = [h for h in above if abs(h[0] - x) < 45]
    col_n = min(col, key=lambda h: (y - h[1]) + abs(h[0] - x))[2] if col else None
    return col_n, max(n for _, _, n in above)


def _exact(c) -> bool:
    from ..engines.counts import percent_check
    from ..verdict import Tier

    return c.count <= c.n and percent_check(c.n, c.percent, c.count, denominator_exact=True).tier is Tier.CONSISTENT


def _confirmed_cells(cells: list[tuple[int, object]]) -> list:
    """Cells the column header N can convict.

    Header N bounds a cell's denominator only if the rows share the column's
    population and unit of analysis.  The column has to prove that itself:
    every OTHER cell (at least 5 of them) must be exactly count/N.  So a
    column convicts at most one cell, an isolated outlier in an otherwise
    perfect column.  Two or more misfits mean the percentages use some other
    denominator (randomised vs analysed population, events, safety set) and
    nothing is claimed.
    """
    ok = [_exact(c) for _, c in cells]
    bad = [j for j, v in enumerate(ok) if not v]
    if len(bad) != 1 or len(cells) - 1 < 5:
        return []
    c = cells[bad[0]][1]
    return [c] if c.count <= c.n else []


def extract_claims(path: str) -> list:
    from .crosstable import grid_claims, pdf_grids

    claims: list = []
    column_cells: dict[tuple[int, int], list] = {}
    pages = load_pages(path)
    for g in pdf_grids(pages):
        if not UNSAFE_PCT_RE.search(g.caption):
            claims.extend(grid_claims(g))
    for pt in pages:
        pct_unsafe = bool(UNSAFE_PCT_RE.search(pt.text))
        complete = bool(COMPLETE_RE.search(pt.text))
        for i, row in enumerate(pt.rows):
            text = row.text
            # ---- n (%)
            if not pct_unsafe:
                for m in NPCT_RE.finditer(text):
                    bb = row.span_bbox(m.start(), m.end())
                    kind, ctx = _context(pt.rows, i, bb[0])
                    label_txt = row.label_before(bb[0])
                    if not ((m.group(3) and kind in (None, "pct", "mixed")) or kind == "pct"):
                        continue
                    if UNIT_LABEL_RE.search(label_txt):
                        continue
                    count, pct = npct_count(m), m.group(2)
                    if float(pct.replace("\u00b7", ".")) > 100:
                        continue
                    if OF_BEFORE_RE.search(text[: m.start()]):
                        continue  # "18 of 254 (7.1%)": the printed number is a denominator
                    col_n, _ = _header_n(pt, (bb[0] + bb[2]) / 2, row.y)
                    prov = [Provenance(source="pdf", page=pt.index, bbox=bb, text=m.group(0), note=ctx[:200])]
                    label = row.label_before(bb[0])[:80]
                    claims.append(PercentClaim(label=label, count=count, percent=pct, n=None, provenance=prov))
                    if col_n is not None:  # every cell votes, including ones that exceed N
                        column_cells.setdefault((pt.index, col_n), []).append(
                            (i, PercentClaim(label=label, count=count, percent=pct, n=col_n, provenance=prov)))
            # ---- mean (SD) on bounded integer instruments
            for m in MEANSD_RE.finditer(text):
                bb = row.span_bbox(m.start(), m.end())
                label = row.label_before(bb[0])
                inst = next(((lo, hi, g) for rx, lo, hi, g in INSTRUMENTS if rx.search(label)), None)
                if inst is None:
                    continue
                kind, ctx = _context(pt.rows, i, bb[0])
                if kind not in ("mean", "mixed") and "\u00b1" not in m.group(0):
                    continue  # could be median (IQR)
                mean, sd = m.group(1).replace("\u2212", "-"), m.group(2) or m.group(3)
                col_n, _ = _header_n(pt, (bb[0] + bb[2]) / 2, row.y)
                prov = [Provenance(source="pdf", page=pt.index, bbox=bb, text=m.group(0), note=ctx[:200])]
                lo, hi, g = inst
                if col_n and complete:
                    claims.append(MeanClaim(label=label[:80], n=col_n, mean=mean, sd=sd, granularity=g, scale_min=lo,
                                            scale_max=hi, provenance=prov))
                else:
                    claims.append(MeanClaim(label=label[:80], n=col_n or 2, n_min=2, n_exact=False, mean=mean, sd=sd,
                                            granularity=g, scale_min=lo, scale_max=hi, provenance=prov))
    for cells in column_cells.values():
        claims.extend(_confirmed_cells(cells))
    from .consort import allocation_claims

    doc_text = "\n".join(pt.text for pt in pages)
    for pt in pages:
        claims.extend(allocation_claims(pt.text, "pdf", pt.index, doc_text))
    return claims


def extract_baseline(path: str):
    """Table 1 style rows with one summary per arm -> BaselineTableClaim per page (for Carlisle only)."""
    from ..claims import BaselineTableClaim, BaselineVar
    from ..engines.carlisle import detect_design

    doc_text = ""
    pages = load_pages(path)
    for pt in pages:
        doc_text += pt.text + "\n"
    design = detect_design(re.sub(r"\s+", " ", doc_text))
    out = []
    for pt in pages:
        cap = next((i for i, r in enumerate(pt.rows) if BASELINE_CAPTION_RE.search(r.text)), None)
        if cap is None:
            continue
        end = next((i for i in range(cap + 1, len(pt.rows)) if re.match(r"\s*Table\s*\d", pt.rows[i].text)), len(pt.rows))
        y0, y1 = pt.rows[cap].y, (pt.rows[end].y if end < len(pt.rows) else 1e9)
        arm_ns = sorted({n for x, y, n in pt.header_ns if y0 <= y < y1})
        if len(arm_ns) == 3 and arm_ns[0] + arm_ns[1] == arm_ns[2]:
            arm_ns = arm_ns[:2]  # drop a Total column
        if len([h for h in pt.header_ns if y0 <= h[1] < y1 and h[2] in arm_ns]) != 2:
            continue  # Carlisle needs exactly two clearly labelled randomized arms
        vars_: list[BaselineVar] = []
        boxes = []
        for i, row in enumerate(pt.rows[cap + 1:end], start=cap + 1):
            if SKIP_BASELINE_ROW_RE.search(row.text):
                continue
            ms = list(MEANSD_RE.finditer(row.text))
            nps = list(NPCT_RE.finditer(row.text))
            if len(ms) == 2 and not nps:
                bb0 = row.span_bbox(ms[0].start(), ms[0].end())
                kind, _ = _context(pt.rows, i, bb0[0])
                if kind not in ("mean", "mixed") and "\u00b1" not in ms[0].group(0):
                    continue
                ns = [_header_n(pt, (bb[0] + bb[2]) / 2, row.y)[0] for bb in (row.span_bbox(m.start(), m.end()) for m in ms)]
                if None in ns:
                    continue
                vars_.append(BaselineVar(name=row.label_before(bb0[0])[:60], means=[m.group(1).replace("\u2212", "-") for m in ms],
                                         sds=[m.group(2) or m.group(3) for m in ms], ns=ns))
                boxes.append(bb0)
            elif len(nps) == 2 and not ms:
                bbs = [row.span_bbox(m.start(), m.end()) for m in nps]
                kind, _ = _context(pt.rows, i, bbs[0][0])
                if kind not in ("pct", "mixed") and not nps[0].group(3):
                    continue
                ns = [_header_n(pt, (bb[0] + bb[2]) / 2, row.y)[0] for bb in bbs]
                ks = [npct_count(m) for m in nps]
                if None in ns or any(k > n for k, n in zip(ks, ns)):  # type: ignore[operator]
                    continue
                vars_.append(BaselineVar(name=row.label_before(bbs[0][0])[:60],
                                         counts=[[k, n - k] for k, n in zip(ks, ns)]))  # type: ignore[operator]
                boxes.append(bbs[0])
        if len(vars_) >= 3:
            prov = [Provenance(source="pdf", page=pt.index, bbox=(min(b[0] for b in boxes), min(b[1] for b in boxes),
                                                                   max(b[2] for b in boxes), max(b[3] for b in boxes)),
                               text=f"{len(vars_)} baseline rows")]
            out.append(BaselineTableClaim(label=f"Table on page {pt.index + 1}", variables=vars_,
                                          randomization=design.randomization, strata=design.strata, provenance=prov))
    return out
