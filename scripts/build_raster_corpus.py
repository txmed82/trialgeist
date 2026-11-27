"""Assemble the raster-KM negative corpus: open-access RCTs whose Kaplan-Meier figures
are embedded bitmaps with a numbers-at-risk table.

    uv run python scripts/build_raster_corpus.py --want 25

Candidates come from Europe PMC (open access, randomized controlled trial, "Kaplan-Meier"
and "at risk" in the full text).  Each candidate is downloaded from the PMC Open Access
bucket and kept when a page that mentions Kaplan-Meier / numbers at risk carries an
embedded image of at least 300 x 200 px.  Retraction status is checked with Europe PMC
publication types and Crossref ``updated-by``; any retraction, expression of concern or
withdrawal excludes the paper.  Results go to tests/corpus/raster/manifest.json, which
is what gets committed (papers stay out of git; scripts/fetch_corpus.py refetches them).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1] / "tests" / "corpus"
OUT = ROOT / "raster"
S3 = "https://pmc-oa-opendata.s3.amazonaws.com"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
UA = {"User-Agent": "trialgeist-corpus/0.1 (research integrity tool; https://github.com/txmed82/trialgeist)"}
KM_PAGE = re.compile(r"Kaplan[\s\-\u2010-\u2014]*Meier|(no\.?|number)\s*(of patients\s*)?at\s*risk", re.I)
BAD_UPDATE = re.compile(r"retract|withdraw|concern|removal", re.I)


def get(url: str, tries: int = 4, binary: bool = False):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                data = r.read()
                return data if binary else data.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429:
                time.sleep(60)
                continue
            time.sleep(2 * (k + 1))
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def candidates(page_size: int = 100, cursor: str = "*"):
    q = ('OPEN_ACCESS:y AND PUB_TYPE:"Randomized Controlled Trial" AND "Kaplan-Meier" AND "at risk" '
         'AND HAS_PDF:y AND (LICENSE:"cc by" OR LICENSE:"cc0" OR LICENSE:"cc by-nc")')
    while True:
        url = EPMC + "?" + urllib.parse.urlencode({"query": q, "format": "json", "pageSize": page_size,
                                                    "cursorMark": cursor, "resultType": "core"})
        js = json.loads(get(url) or "{}")
        for r in js.get("resultList", {}).get("result", []):
            yield r
        nxt = js.get("nextCursorMark")
        if not nxt or nxt == cursor:
            return
        cursor = nxt
        time.sleep(0.5)


def raster_km_figures(pdf: Path) -> list[dict]:
    out = []
    with pymupdf.open(pdf) as doc:
        for pno, page in enumerate(doc):
            if not KM_PAGE.search(page.get_text()):
                continue
            for info in page.get_images(full=True):
                xref, w, h = info[0], info[2], info[3]
                if w < 300 or h < 200:
                    continue
                rects = page.get_image_rects(xref)
                if rects and rects[0].width >= 150 and rects[0].height >= 100:
                    out.append({"page": pno + 1, "xref": xref, "px": [w, h], "rect_pt": [round(v, 1) for v in rects[0]]})
    pymupdf.TOOLS.store_shrink(100)
    return out


def retraction_check(rec: dict) -> dict:
    types = rec.get("pubTypeList", {}).get("pubType", [])
    doi = rec.get("doi")
    updates = []
    if doi:
        js = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)) or "{}")
        msg = js.get("message", {})
        updates = [u.get("type", "") for u in msg.get("updated-by", []) + msg.get("update-to", [])]
        time.sleep(0.3)
    return {"europepmc_pubtypes": types, "crossref_updates": updates, "checked": dt.date.today().isoformat()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--want", type=int, default=25)
    ap.add_argument("--scan", type=int, default=600, help="max candidates to inspect")
    a = ap.parse_args()
    taken = {p.name for d in ("negative", "positive", "retracted") for p in (ROOT / d).glob("PMC*")}
    man_path = OUT / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else []
    have = {e["pmcid"] for e in manifest}
    seen = 0
    for rec in candidates():
        if len(have) >= a.want or seen >= a.scan:
            break
        pmc = rec.get("pmcid")
        if not pmc or pmc in have or pmc in taken:
            continue
        seen += 1
        rc = retraction_check(rec)
        if any(re.search(r"retract|concern|withdraw", t, re.I) for t in rc["europepmc_pubtypes"]) or \
                any(BAD_UPDATE.search(u) for u in rc["crossref_updates"]):
            print(pmc, "excluded: retraction/concern/withdrawal")
            continue
        d = OUT / pmc
        d.mkdir(parents=True, exist_ok=True)
        pdf = get(f"{S3}/{pmc}.1/{pmc}.1.pdf", binary=True)
        if not pdf or not pdf.startswith(b"%PDF"):
            d.rmdir() if not any(d.iterdir()) else None
            continue
        (d / "paper.pdf").write_bytes(pdf)
        figs = raster_km_figures(d / "paper.pdf")
        if not figs:
            (d / "paper.pdf").unlink()
            d.rmdir() if not any(d.iterdir()) else None
            continue
        xml = get(f"{S3}/{pmc}.1/{pmc}.1.xml", binary=True)
        if xml:
            (d / "paper.nxml").write_bytes(xml)
        lic = (rec.get("license") or "").lower()
        manifest.append({
            "pmcid": pmc, "pmid": rec.get("pmid"), "doi": rec.get("doi"), "title": rec.get("title"),
            "journal": rec.get("journalInfo", {}).get("journal", {}).get("title"), "year": rec.get("pubYear"),
            "license": lic, "pdf_url": f"{S3}/{pmc}.1/{pmc}.1.pdf", "xml_url": f"{S3}/{pmc}.1/{pmc}.1.xml",
            "pdf_sha256": hashlib.sha256(pdf).hexdigest(), "raster_figures_on_km_pages": figs,
            "retraction_check": rc, "visually_confirmed_km": None,
        })
        have.add(pmc)
        man_path.write_text(json.dumps(manifest, indent=1) + "\n")
        print(pmc, rec.get("journalInfo", {}).get("journal", {}).get("medlineAbbreviation"), rec.get("pubYear"), len(figs), "raster figure(s)", flush=True)
        time.sleep(0.5)
    print(f"{len(manifest)} papers in {man_path}")


if __name__ == "__main__":
    main()
