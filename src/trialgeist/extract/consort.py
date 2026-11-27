"""Randomisation arithmetic stated in prose.

    "A total of 605 patients were randomly assigned to drug (n = 300) or placebo (n = 302)."

When one sentence states how many participants were randomised and then enumerates the
arms they were assigned to, closed by "or"/"and", the arm counts are an exhaustive
partition of that total: they must add up.  The extractor is deliberately narrow, because
a single misparse would be a false accusation:

* the subject must be people (patients, participants, ...) or omitted, never clusters,
  sites, eyes or lesions;
* arm counts are read only in the explicit ``(n = k)`` form;
* the enumeration has to be closed by "or"/"and" before the last arm;
* sentences with qualifiers that break the partition ("of whom", "respectively",
  "subgroup", "first", "then", ...) are skipped;
* documents that describe cluster, factorial or crossover randomisation are skipped
  entirely (arm counts there are not a partition of one randomised total).
"""

from __future__ import annotations

import re

from ..claims import CountSumClaim
from ..verdict import Provenance

PEOPLE = r"(?:patients|participants|subjects|individuals|persons|people|adults|children|adolescents|infants|neonates|newborns|women|men|volunteers|smokers|survivors|mothers|pregnant women|residents|students|veterans|outpatients|inpatients)"
NUM = r"(\d{1,3}(?:,\d{3})+|\d+)"
TOTAL_RE = re.compile(
    NUM + r"\s+(?:(?:eligible|consenting|consecutive|enrolled|adult|\w+-\w+)\s+)?(?:" + PEOPLE + r"\s+)?"
    r"(?:(?:(?:were|was)\s+(?:eligible|enrolled|included|confirmed)|qualified)\s+and\s+)?"
    r"(?:were|was|had been)\s+(?:then\s+)?(?:randomly\s+(?:assigned|allocated)|randomi[sz]ed|allocated\s+at\s+random)"
    r"(?:\s*\([^)]{0,40}\))?(?:\s+(?:in|at)\s+a\s+\d+\s*:\s*\d+(?:\s*:\s*\d+)?\s+ratio)?\s*,?\s+(?:to|into|between)\b",
    re.I,
)
# form B: "605 patients were randomised, 300 to drug and 305 to placebo"
TOTAL_B_RE = re.compile(
    NUM + r"\s+(?:(?:eligible|consenting|consecutive|enrolled|adult|\w+-\w+)\s+)?(?:" + PEOPLE + r"\s+)?"
    r"(?:(?:(?:were|was)\s+(?:eligible|enrolled|included|confirmed)|qualified)\s+and\s+)?"
    r"(?:were|was|had been)\s+(?:then\s+)?(?:randomly\s+(?:assigned|allocated)|randomi[sz]ed)"
    r"(?:\s+(?:in|at)\s+a\s+\d+\s*:\s*\d+(?:\s*:\s*\d+)?\s+ratio)?\s*[,:]\s*(?:with\s+)?(?=\d)",
    re.I,
)
ARM_B_RE = re.compile(NUM + r"\s+(?:" + PEOPLE + r"\s+)?(?:(?:were\s+)?(?:assigned|allocated|randomi[sz]ed)\s+)?(?:to|into)\b(?!\s*\d)", re.I)
# form C: "511 were eligible and were randomised (258 to placebo, 253 to nilvadipine)"
TOTAL_C_RE = re.compile(
    NUM + r"\s+(?:(?:eligible|consenting|consecutive|enrolled|adult|\w+-\w+)\s+)?(?:" + PEOPLE + r"\s+)?"
    r"(?:(?:(?:were|was)\s+(?:eligible|enrolled|included|confirmed)|qualified)\s+and\s+)?"
    r"(?:were|was|had been)\s+(?:randomly\s+(?:assigned|allocated)|randomi[sz]ed)\s*\(([^()]{5,200})\)",
    re.I,
)
_NAME = r"[A-Za-z][^,;()=\d]*?(?:\s+\d+(?:\.\d+)?\s*(?:mg|µg|μg|mcg|g|ml|mL|IU|mg/kg)\b[^,;()=\d]*?)?"
ITEM_C_RE = re.compile(
    r"\s*(?:and\s+)?(?:n\s*=\s*" + NUM + r"\s+(?:in|to|for)\s+(?:the\s+)?" + _NAME +
    r"|" + NUM + r"\s+(?:" + PEOPLE + r"\s+)?(?:to|in)\s+(?:the\s+)?" + _NAME +
    r"|(?:the\s+)?" + _NAME + r"[,:]?\s*(?:\(\s*)?n\s*=\s*" + NUM + r")\s*$",
    re.I,
)
TRAILING_RE = re.compile(
    r"(?:\s*\((?:see\s+)?(?:Fig(?:ure)?s?\.?|Supplementary|Table|eFigure|eTable)[^)]*\))"
    r"|(?:,?\s+(?:at|in|across|from)\s+(?:more\s+than\s+)?\d+\s+(?:centres|centers|sites|hospitals|countries)"
    r"(?:\s+(?:in|across)\s+\d+\s+countries)?)",
    re.I,
)
DOSE_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:mg|µg|μg|mcg|g|ml|mL|IU|units?|mg/kg|mg/m2|Gy)\b", re.I)
ARM_N_RE = re.compile(r"\(\s*n\s*=\s*" + NUM + r"\s*\)", re.I)
REFUSE_TAIL_RE = re.compile(
    r"of whom|of these|of which|among|including|respectively|subset|sub-?group|cohort|strat(?:um|a)|\bpart\b|stage|phase|"
    r"period|cross-?over|\bfirst\b|\bsecond\b|\bthen\b|subsequently|\beach\b|\bper\b|cluster|%|\beyes?\b|lesions?|"
    r"sites?\b|centres?\b|centers?\b|hospitals?|practices|wards?|schools?|villages?|households?|teams?",
    re.I,
)
REFUSE_DOC_RE = re.compile(r"cluster[\s-]+randomi|factorial|cross-?over|stepped[\s-]+wedge|re-?randomi", re.I)
SENT_END_RE = re.compile(r";|\.(?=\s+[A-Z(\[]|\s*$)")


def _int(s: str) -> int:
    return int(s.replace(",", ""))


def flatten(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = text.replace("\u2009", " ").replace("\u00a0", " ").replace("\u202f", " ")
    return re.sub(r"\s+", " ", text)


def allocation_claims(text: str, source: str = "pdf", page: int | None = None,
                      doc_text: str | None = None) -> list[CountSumClaim]:
    """CountSum claims (exhaustive) for every randomisation sentence that enumerates its arms.
    ``doc_text`` (the whole paper) is screened for cluster / factorial / crossover designs."""
    flat = flatten(text)
    if REFUSE_DOC_RE.search(flatten(doc_text) if doc_text is not None else flat):
        return []
    out: list[CountSumClaim] = []
    for m in TOTAL_RE.finditer(flat):
        # the number must start the clause: not "...of 12 600 patients", not a decimal
        pre = flat[max(0, m.start() - 2): m.start()]
        if re.search(r"[\d.,]\s?$", pre) and not pre.endswith(", "):
            continue
        end = SENT_END_RE.search(flat, m.end())
        tail = TRAILING_RE.sub("", flat[m.end(): end.start() if end else min(len(flat), m.end() + 400)])
        if len(tail) > 400:
            continue
        arms = list(ARM_N_RE.finditer(tail))
        if len(arms) < 2 or REFUSE_TAIL_RE.search(tail):
            continue
        # every count follows an arm name, and the list is closed by "or"/"and"
        names = [tail[(arms[i - 1].end() if i else 0): a.start()] for i, a in enumerate(arms)]
        if any(not re.search(r"[A-Za-z]{2,}", nm) for nm in names):
            continue
        if not re.search(r"\b(?:or|and|versus|vs\.?)\b", names[-1]):
            continue
        if any(re.search(r"\b(?:or|and)\b", nm) for nm in names[1:-1]):
            continue  # "A (n) and B (n), or C (n)": nested grouping, not a flat list
        if ARM_N_RE.search(tail[arms[-1].end():]) or re.search(r"\d", tail[arms[-1].end():]):
            continue  # more numbers after the list: not a clean enumeration
        total, counts = _int(m.group(1)), [_int(a.group(1)) for a in arms]
        if total < 2 or any(c <= 0 for c in counts):
            continue
        out.append(_claim(flat, m, end, tail, total, counts, source, page))
    for m in TOTAL_B_RE.finditer(flat):
        pre = flat[max(0, m.start() - 2): m.start()]
        if re.search(r"[\d.,]\s?$", pre) and not pre.endswith(", "):
            continue
        end = SENT_END_RE.search(flat, m.end())
        tail = TRAILING_RE.sub("", flat[m.end(): end.start() if end else min(len(flat), m.end() + 400)])
        if len(tail) > 400 or REFUSE_TAIL_RE.search(tail):
            continue
        arms = list(ARM_B_RE.finditer(tail))
        if len(arms) < 2 or arms[0].start() != 0:
            continue
        # every digit in the clause belongs to an arm count or a dose
        rest = tail
        for a in reversed(arms):
            rest = rest[:a.start()] + " " * (a.end() - a.start()) + rest[a.end():]
        if re.search(r"\d", DOSE_RE.sub(" ", rest)):
            continue
        names = [tail[a.end(): (arms[i + 1].start() if i + 1 < len(arms) else len(tail))] for i, a in enumerate(arms)]
        if any(not re.search(r"[A-Za-z]{2,}", nm) for nm in names):
            continue
        if not re.search(r"\b(?:or|and)$", names[-2].rstrip(" ,")):
            continue  # the list must be closed: "..., 299 to X, and 300 to Y"
        if any(re.search(r"\b(?:or|and)\s*$", nm.rstrip(" ,")) for nm in names[:-2]):
            continue
        total, counts = _int(m.group(1)), [_int(a.group(1)) for a in arms]
        if total < 2 or any(c <= 0 for c in counts):
            continue
        out.append(_claim(flat, m, end, tail, total, counts, source, page))
    for m in TOTAL_C_RE.finditer(flat):
        pre = flat[max(0, m.start() - 2): m.start()]
        if re.search(r"[\d.,]\s?$", pre) and not pre.endswith(", "):
            continue
        inner = m.group(2)
        if REFUSE_TAIL_RE.search(inner) or "%" in inner:
            continue
        parts = re.split(r"\s*[;,]\s*", inner.strip())
        counts = []
        for part in parts:
            im = ITEM_C_RE.match(part)
            if not im:
                counts = []
                break
            counts.append(_int(next(g for g in im.groups() if g)))
        if len(counts) < 2:
            continue
        total = _int(m.group(1))
        end = SENT_END_RE.search(flat, m.end())
        out.append(_claim(flat, m, end, "", total, counts, source, page))
    return out


def _claim(flat, m, end, tail, total, counts, source, page) -> CountSumClaim:
    sentence = flat[m.start(): end.start() + 1 if end else m.end() + len(tail)]
    return CountSumClaim(
        label="randomised allocation stated in the text", total=total, counts=counts, exhaustive=True,
        provenance=[Provenance(source=source, page=page, text=sentence[:400],
                               note="randomisation sentence: total and per-arm allocation")])
