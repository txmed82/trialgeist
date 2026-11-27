"""PMC JATS XML extraction: exact table cells, no layout guessing.

Same claim policy as the PDF extractor (conservative, provenance on every
claim), but the cell grid comes from <table> markup, so column membership is
certain.  Header N is still only trusted after the column confirms it.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from ..claims import MeanClaim, PercentClaim, PercentPartitionClaim
from ..verdict import Provenance
from .crosstable import Grid, GridRow, grid_claims
from .pdf_text import (COMPLETE_RE, HEADER_N_RE, INSTRUMENTS, MEAN_CONTEXT_RE, MEANSD_RE, MEDIAN_CONTEXT_RE, NPCT_RE,
                       OF_BEFORE_RE, PCTN_RE, npct_count, RISK_CONTEXT_RE, PCT_CONTEXT_RE, UNIT_LABEL_RE, UNSAFE_PCT_RE, _confirmed_cells)


def _text(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def table_grid(tbl) -> tuple[list[list[str]], list[list[str]]]:
    """(header rows, body rows) with colspans expanded and rowspans filled down."""
    head, body = table_cells(tbl)
    return [[t for t, _ in r] for r in head], [[t for t, _ in r] for r in body]


def table_cells(tbl) -> tuple[list[list[tuple[str, bool]]], list[list[tuple[str, bool]]]]:
    """Like table_grid, but each slot is (text, own) where own is False for slots
    filled by another cell's colspan/rowspan."""

    def rows_of(sec_tags):
        out = []
        pending: dict[int, tuple[str, int]] = {}
        for tr in [tr for tag in sec_tags for sec in tbl.iter(tag) for tr in sec.iter("tr")]:
            row, col = [], 0
            cells = list(tr)
            ci = 0
            while ci < len(cells) or col in pending:
                if col in pending:
                    txt, left = pending.pop(col)
                    row.append((txt, False))
                    if left > 1:
                        pending[col] = (txt, left - 1)
                    col += 1
                    continue
                c = cells[ci]
                ci += 1
                txt = _text(c)
                cs, rs = int(c.get("colspan", "1") or 1), int(c.get("rowspan", "1") or 1)
                for k in range(cs):
                    row.append((txt, k == 0 and cs == 1))
                    if rs > 1:
                        pending[col] = (txt, rs - 1)
                    col += 1
            out.append(row)
        return out

    head = rows_of(["thead"])
    body = rows_of(["tbody"])
    if not head and not body:
        body = [[(_text(c), True) for c in tr] for tr in tbl.iter("tr")]
    return head, body


def _is_section(row: list[str]) -> bool:
    return bool(row) and all(not x or x == row[0] for x in row[1:])


def _ctx_kind(body: list[list[str]], r: int, header_text: str) -> str | None:
    """Own row label first, then the nearest section-header row above (a row whose
    other cells are empty/spanned); a data row like 'Mean (SD) age' only describes itself."""
    for j in range(r, -1, -1):
        if RISK_CONTEXT_RE.search(body[j][0] if body[j] else ""):
            return "risk"
        if j != r and not _is_section(body[j]):
            continue
        lab = body[j][0] if body[j] else ""
        for rx, k in ((MEAN_CONTEXT_RE, "mean"), (MEDIAN_CONTEXT_RE, "median"), (PCT_CONTEXT_RE, "pct")):
            if rx.search(lab):
                return k
    for rx, k in ((MEAN_CONTEXT_RE, "mean"), (MEDIAN_CONTEXT_RE, "median"), (PCT_CONTEXT_RE, "pct")):
        if rx.search(header_text):
            return k
    return None


def extract_claims_xml(path: str) -> list:
    root = ET.parse(path).getroot()
    full = _text(root.find(".//body") if root.find(".//body") is not None else root)
    complete = bool(COMPLETE_RE.search(full))
    claims: list = []
    for tw in root.iter("table-wrap"):
        label = _text(tw.find("label")) if tw.find("label") is not None else "Table"
        caption = _text(tw.find("caption")) if tw.find("caption") is not None else ""
        foot = _text(tw.find("table-wrap-foot")) if tw.find("table-wrap-foot") is not None else ""
        tbl = tw.find(".//table")
        if tbl is None:
            continue
        head_c, body_c = table_cells(tbl)
        if not head_c:  # header rows coded as <tbody> rows: leading rows carrying "n=" and no data cells
            while body_c and any(HEADER_N_RE.search(x) for x, _ in body_c[0][1:]) and not any(NPCT_RE.search(x) for x, _ in body_c[0]):
                head_c.append(body_c.pop(0))
        head = [[t for t, _ in r] for r in head_c]
        body = [[t for t, _ in r] for r in body_c]
        ncol = max((len(r) for r in head + body), default=0)
        col_n: dict[int, int] = {}
        for hr in head:
            for c, txt in enumerate(hr):
                m = HEADER_N_RE.search(txt)
                if m and c > 0:
                    col_n[c] = int(m.group(1).replace(",", ""))
        header_text = " ".join(" ".join(h) for h in head) + " " + caption
        pct_unsafe = bool(UNSAFE_PCT_RE.search(caption + " " + foot))
        column_cells: dict[int, list] = {}
        for r, row in enumerate(body):
            if not row:
                continue
            rlab = row[0]
            kind = _ctx_kind(body, r, header_text)
            for c in range(1, min(len(row), ncol)):
                cell = row[c]
                if c > 1 and row[c] == row[c - 1]:
                    continue  # colspan duplicate
                prov = [Provenance(source="xml", text=cell, note=f"{label} row {r + 1} col {c + 1}: {rlab[:80]} | {caption[:80]}")]
                # n (%)
                if not pct_unsafe and not UNIT_LABEL_RE.search(rlab):
                    for m in NPCT_RE.finditer(cell):
                        if OF_BEFORE_RE.search(cell[: m.start()]):
                            continue
                        if not ((m.group(3) and kind in (None, "pct")) or kind == "pct"):
                            continue
                        count, pct = npct_count(m), m.group(2)
                        if float(pct.replace("\u00b7", ".")) <= 100:  # >100 only counts once the column confirms N
                            claims.append(PercentClaim(label=rlab[:80], count=count, percent=pct, n=None, provenance=prov))
                        if c in col_n:
                            column_cells.setdefault(c, []).append(
                                (r, PercentClaim(label=rlab[:80], count=count, percent=pct, n=col_n[c], provenance=prov)))
                    for m in PCTN_RE.finditer(cell):  # "36.6 (n=11)"
                        pct, count = m.group(1), int(m.group(2))
                        if float(pct.replace("\u00b7", ".")) <= 100:
                            claims.append(PercentClaim(label=rlab[:80], count=count, percent=pct, n=None, provenance=prov))
                        if c in col_n:
                            column_cells.setdefault(c, []).append(
                                (r, PercentClaim(label=rlab[:80], count=count, percent=pct, n=col_n[c], provenance=prov)))
                # mean (SD) on bounded instruments
                inst = next(((lo, hi, g) for rx, lo, hi, g in INSTRUMENTS if rx.search(rlab)), None)
                if inst is None:
                    continue
                for m in MEANSD_RE.finditer(cell):
                    if kind != "mean" and "\u00b1" not in m.group(0):
                        continue
                    mean, sd = m.group(1).replace("\u2212", "-"), m.group(2) or m.group(3)
                    lo, hi, g = inst
                    n = col_n.get(c)
                    if n and complete:
                        claims.append(MeanClaim(label=rlab[:80], n=n, mean=mean, sd=sd, granularity=g, scale_min=lo,
                                                scale_max=hi, provenance=prov))
                    else:
                        claims.append(MeanClaim(label=rlab[:80], n=n or 2, n_min=2, n_exact=False, mean=mean, sd=sd,
                                                granularity=g, scale_min=lo, scale_max=hi, provenance=prov))
        for cells in column_cells.values():
            claims.extend(_confirmed_cells(cells))
        if not pct_unsafe:
            claims.extend(_partition_blocks(body, col_n, label, caption))
        claims.extend(grid_claims(_grid(head_c, body_c, col_n, label, caption, complete, pct_unsafe)))
    from .consort import allocation_claims

    seen = set()
    for c in allocation_claims(paper_text_xml(path), "xml"):
        if (c.total, tuple(c.counts)) not in seen:
            seen.add((c.total, tuple(c.counts)))
            claims.append(c)
    return claims


def _grid(head_c, body_c, col_n: dict[int, int], label: str, caption: str, complete: bool, pct_unsafe: bool) -> Grid:
    col_head: dict[int, str] = {}
    for hr in head_c:
        for c, (txt, _) in enumerate(hr):
            if c > 0 and txt and txt not in col_head.get(c, ""):
                col_head[c] = (col_head.get(c, "") + " " + txt).strip()
    rows: list[GridRow] = []
    section, section_label = -1, ""
    for r, row in enumerate(body_c):
        texts = [t for t, _ in row]
        if not texts:
            continue
        if _is_section(texts) or not any(t for t in texts[1:]):
            section, section_label = r, texts[0]
            continue
        cells = {c: t for c, (t, own) in enumerate(row) if c > 0 and own and t}
        if pct_unsafe:
            cells = {c: t for c, t in cells.items() if not NPCT_RE.search(t)}
        prov = {c: Provenance(source="xml", text=t, note=f"{label} row {r + 1} col {c + 1}: {texts[0][:80]} | {caption[:80]}")
                for c, t in cells.items()}
        ctx = texts[0] + (" | " + section_label if section >= 0 else "")
        rows.append(GridRow(label=texts[0], context=ctx, section=section, cells=cells, prov=prov))
    return Grid(label=label, caption=caption, col_n=dict(col_n), col_head=col_head, rows=rows, complete=complete)


BARE_PCT_RE = re.compile(r"\d{1,3}(?:[.\u00b7]\d+)?")


def _partition_blocks(body: list[list[str]], col_n: dict[int, int], label: str, caption: str) -> list:
    """Percent-only category blocks, e.g. severity None/Mild/Moderate/Severe at one time point.

    A block is a run of >= 2 consecutive rows sharing the first cell (or
    sharing a section) whose bare-number cells in a column with header N add
    up to 100 within rounding.  That sum is what marks the rows as one
    partition; each block becomes a partition claim over any D <= N.
    """
    out = []
    if MEAN_CONTEXT_RE.search(caption) or MEDIAN_CONTEXT_RE.search(caption):
        return out
    r = 0
    while r < len(body):
        key = body[r][0] if body[r] else ""
        s = r
        while r < len(body) and body[r] and body[r][0] == key:
            r += 1
        block = body[s:r] if r > s else [body[s]]
        r = max(r, s + 1)
        if len(block) < 2 or not key:
            continue
        for c, n in col_n.items():
            vals = [row[c] if c < len(row) else "" for row in block]
            if not all(BARE_PCT_RE.fullmatch(v) for v in vals):
                continue
            fl = [float(v.replace("\u00b7", ".")) for v in vals]
            dec = max((len(v.split(".")[1]) if "." in v else 0) for v in vals)
            if any(x > 100 for x in fl) or abs(sum(fl) - 100) > len(vals) * 0.5 * 10 ** -dec + 1e-9:
                continue
            prov = [Provenance(source="xml", text=" / ".join(vals),
                               note=f"{label} rows {s + 1}-{s + len(block)} col {c + 1}: {key[:40]} | {caption[:80]}")]
            out.append(PercentPartitionClaim(label=f"{key} ({', '.join(row[1] if len(row) > 1 else '' for row in block)[:60]})",
                                             n=n, n_exact=False, percents=vals, provenance=prov))
    return out


def paper_text_xml(path: str) -> str:
    root = ET.parse(path).getroot()
    return " ".join(_text(x) for x in root.iter() if x.tag in ("abstract", "body"))
