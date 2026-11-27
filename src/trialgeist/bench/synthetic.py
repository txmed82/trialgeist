"""Synthetic RCT generator for Gate 1 (mutation torture suite).

Each trial is rendered to a real PDF (vector KM plot with numbers-at-risk
table, Table 1 with instrument means and n (%) rows) plus structured claims
for the binary outcome.  Corruptions are impossible *by construction*, and the
generator checks that with its own brute force (never with the engines).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pymupdf

MODES = [ROUND_HALF_UP, ROUND_HALF_EVEN, ROUND_DOWN, "float"]
CORRUPTIONS = ["grim_mean", "sd_ceiling", "percent", "km_risk_table", "km_big_drop", "rr_crude"]

COLORS = [(0.8, 0.1, 0.1), (0.1, 0.2, 0.8)]


def fmt(x: Fraction, d: int, mode) -> str:
    if mode == "float":
        return f"{float(x):.{d}f}"
    with localcontext() as ctx:
        ctx.prec = 50
        v = Decimal(x.numerator) / Decimal(x.denominator)
        return str(v.quantize(Decimal(1).scaleb(-d), rounding=mode))


def fmt_sqrt(var: Fraction, d: int, mode) -> str:
    if mode == "float":
        return f"{math.sqrt(var):.{d}f}"
    with localcontext() as ctx:
        ctx.prec = 50
        v = (Decimal(var.numerator) / Decimal(var.denominator)).sqrt()
        return str(v.quantize(Decimal(1).scaleb(-d), rounding=mode))


@dataclass
class Trial:
    seed: int
    corruption: str | None
    pdf: Path
    claims: list[dict] = field(default_factory=list)  # structured (binary outcome)
    corrupted_label: str | None = None  # substring identifying the corrupted item
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------- data


def simulate_survival(rnd: random.Random, n: int, lam: float, horizon: float = 36.0):
    obs = []
    for _ in range(n):
        t = rnd.expovariate(lam)
        c = rnd.uniform(3, horizon * 1.3)
        tt = min(t, c, horizon)
        tt = Fraction(round(tt * 10), 10)
        obs.append((tt, t <= min(c, horizon)))
    return obs


def km(obs):
    s = Fraction(1)
    steps = []
    for t in sorted({t for t, e in obs if e}):
        n = sum(1 for u, _ in obs if u >= t)
        d = sum(1 for u, e in obs if e and u == t)
        new = s * (n - d) / n
        steps.append([t, s, new])
        s = new
    return steps


def grim_feasible_wide(n: int, v: Fraction, d: int) -> bool:
    """Independent brute force with a hull wider than the engine's."""
    h = Fraction(1, 2 * 10**d)
    lo, hi = v - 3 * h - Fraction(1, 10**6), v + 3 * h + Fraction(1, 10**6)
    return any(lo <= Fraction(s, n) <= hi for s in range(math.floor(lo * n) - 1, math.ceil(hi * n) + 2))


def pct_any_denominator_wide(k: int, p: Fraction, d: int) -> bool:
    h = Fraction(1, 2 * 10**d)
    lo, hi = p - 3 * h - Fraction(1, 10**6), p + 3 * h + Fraction(1, 10**6)
    if lo <= 0:
        return True
    return any(lo <= Fraction(100 * k, D) <= hi for D in range(max(k, 1), math.ceil(100 * k / lo) + 2))


# --------------------------------------------------------------- render


class Canvas:
    def __init__(self, page: pymupdf.Page):
        self.page = page

    def text(self, x, y, s, size=8.0, align="left"):
        w = pymupdf.get_text_length(s, fontname="helv", fontsize=size)
        if align == "center":
            x -= w / 2
        elif align == "right":
            x -= w
        self.page.insert_text((x, y), s, fontsize=size, fontname="helv")

    def polyline(self, pts, color, width=1.0):
        sh = self.page.new_shape()
        for p, q in zip(pts, pts[1:]):
            sh.draw_line(p, q)
        sh.finish(color=color, width=width, closePath=False)
        sh.commit()

    def line(self, p, q, color=(0, 0, 0), width=0.6):
        self.polyline([p, q], color, width)


def render_table1(page, rows, arms, footnote):
    c = Canvas(page)
    c.text(50, 60, "Table 1. Baseline characteristics", 10)
    c.text(50, 80, "Characteristic")
    xs = [330, 450]
    for x, (name, n) in zip(xs, arms):
        c.text(x, 80, f"{name} (n={n})", align="center")
    y = 100
    for label, cells in rows:
        c.text(50, y, label)
        for x, cell in zip(xs, cells):
            if cell:
                c.text(x, y, cell, align="center")
        y += 15
    c.text(50, y + 10, footnote, 7)


def render_km(page, arms_data, horizon=36, step=6, rising=False, colors=None, width=1.0):
    """arms_data: list of (name, steps[[t, before, after]], risk[(tau, n)], censor_times).

    rising=True plots 1 - S(t) (cumulative mortality, as in many Lancet/NEJM figures)."""
    c = Canvas(page)
    L, R, T, B = 90.0, 500.0, 120.0, 380.0
    X = lambda t: L + (R - L) * float(t) / horizon
    Y = (lambda s: B - (B - T) * float(1 - s)) if rising else (lambda s: B - (B - T) * float(s))
    c.text(50, 90, "Figure 2. Kaplan-Meier estimates of cumulative mortality" if rising else
           "Figure 2. Kaplan-Meier estimates of overall survival", 10)
    c.line((L, B), (R, B))
    c.line((L, B), (L, T))
    for k in range(6):
        v = k / 5
        c.line((L - 4, Y(v)), (L, Y(v)))
        c.text(L - 6, Y(v) + 2.8, f"{(1 - v) * 100:.0f}" if rising else f"{v:.1f}", align="right")
    for t in range(0, horizon + 1, step):
        c.line((X(t), B), (X(t), B + 4))
        c.text(X(t), B + 14, str(t), align="center")
    c.text((L + R) / 2, B + 28, "Months since randomization", align="center")
    for i, (name, steps, risk, cens) in enumerate(arms_data):
        col = (colors or COLORS)[i]
        pts = [(X(0), Y(1))]
        for t, b, a in steps:
            pts.append((X(t), Y(b)))
            pts.append((X(t), Y(a)))
        last_t = max([t for t, _, _ in steps] + [0])
        end_t = min(horizon, max(cens + [last_t]))
        pts.append((X(end_t), pts[-1][1]))
        c.polyline(pts, col, width)
        for ct in cens:
            lvl = Fraction(1)
            for t, b, a in steps:
                if t <= ct:
                    lvl = a
            c.line((X(ct), Y(lvl) - 2.5), (X(ct), Y(lvl) + 2.5), col, 0.6)
        c.line((R - 120, T + 15 + 12 * i), (R - 100, T + 15 + 12 * i), col, 1.0)
        c.text(R - 95, T + 18 + 12 * i, name)
    c.text(20, B + 50, "No. at risk")
    for i, (name, steps, risk, cens) in enumerate(arms_data):
        y = B + 64 + 13 * i
        c.text(20, y, name)
        for tau, n in risk:
            c.text(X(tau), y, str(n), align="center")


# ---------------------------------------------------------------- trial


def make_trial(seed: int, out_dir: Path, corruption: str | None = None) -> Trial:
    rnd = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    small = corruption in ("grim_mean", "km_risk_table", "km_big_drop") or rnd.random() < 0.4
    n1, n2 = (rnd.randint(20, 90), rnd.randint(20, 90)) if small else (rnd.randint(60, 350), rnd.randint(60, 350))
    if corruption == "grim_mean":  # GRIM has power only when N < ~45 at 2 decimals
        n1, n2 = rnd.randint(12, 30), rnd.randint(12, 30)
    arms = [("Treatment", n1), ("Control", n2)]
    complete = small or rnd.random() < 0.5
    trial = Trial(seed, corruption, out_dir / f"trial_{seed:04d}_{corruption or 'clean'}.pdf")

    # ---------------- Table 1
    rows = []
    instruments = [("NIHSS score, mean (SD)", 0, 42, 2, 12), ("Glasgow Coma Scale, mean (SD)", 3, 15, 1, 14),
                   ("mRS score, mean (SD)", 0, 6, 1, 3), ("Pain NRS (0-10), mean (SD)", 0, 10, 1, 5)]
    rnd.shuffle(instruments)
    corrupt_arm = rnd.randrange(2)
    for label, lo, hi, _, centre in instruments[:3]:
        cells = []
        for ai, (_, n) in enumerate(arms):
            data = [min(hi, max(lo, round(rnd.gauss(centre, (hi - lo) / 5)))) for _ in range(n)]
            n_obs = n if complete else n - rnd.randint(0, max(1, n // 10))
            data = data[:n_obs]
            mean = Fraction(sum(data), n_obs)
            ss = sum((Fraction(x) - mean) ** 2 for x in data)
            var = ss / (n_obs - 1) if rnd.random() < 0.8 else ss / n_obs
            d, mode = rnd.choice([1, 2, 2]), rnd.choice(MODES)
            m_txt, s_txt = fmt(mean, d, mode), fmt_sqrt(var, d, mode)
            if corruption == "grim_mean" and ai == corrupt_arm and trial.corrupted_label is None:
                d = 2
                for delta in range(1, 200):
                    v = Fraction(round(mean * 100) + rnd.choice([-1, 1]) * delta, 100)
                    if lo < v < hi and not grim_feasible_wide(n, v, 2):
                        m_txt, s_txt = fmt(v, 2, ROUND_HALF_UP), fmt_sqrt(var, 2, ROUND_HALF_UP)
                        trial.corrupted_label = label.split(",")[0].split(" (")[0]
                        break
            if corruption == "sd_ceiling" and ai == corrupt_arm and trial.corrupted_label is None:
                mu = Fraction(m_txt)
                bound = math.sqrt(float((mu - lo) * (hi - mu)) * 2) if lo < mu < hi else 1.0
                s_txt = f"{bound * 1.25 + 0.3:.{d}f}"
                trial.corrupted_label = label.split(",")[0].split(" (")[0]
            cells.append(f"{m_txt} ({s_txt})")
        rows.append((label, cells))
    rows.append(("Age, mean (SD), yr", [f"{rnd.gauss(64, 2):.1f} ({rnd.gauss(11, 1):.1f})" for _ in arms]))
    for label, prob in [("Male sex, no. (%)", 0.55), ("Diabetes, no. (%)", 0.25), ("Current smoker, no. (%)", 0.2)]:
        cells = []
        for ai, (_, n) in enumerate(arms):
            denom = n if complete else n - rnd.randint(0, 3)
            k = sum(rnd.random() < prob for _ in range(denom))
            d, mode = rnd.choice([0, 1, 1]), rnd.choice(MODES)
            p_txt = fmt(Fraction(100 * k, denom), d, mode)
            if corruption == "percent" and ai == corrupt_arm and trial.corrupted_label is None and k >= 3:
                for delta in range(1, 400):
                    p = Fraction(round(Fraction(1000 * k, denom)) + delta * rnd.choice([-1, 1]), 10)
                    if 0 < p < 100 and not pct_any_denominator_wide(k, p, 1):
                        p_txt = fmt(p, 1, ROUND_HALF_UP)
                        trial.corrupted_label = label.split(",")[0].split(" (")[0]
                        break
            cells.append(f"{k} ({p_txt})")
        rows.append((label, cells))
    foot = ("There were no missing baseline data." if complete else
            "Percentages are based on participants with available data.")

    # ---------------- KM
    km_arms = []
    lam0 = rnd.uniform(0.015, 0.05)
    for ai, (name, n) in enumerate(arms):
        obs = simulate_survival(rnd, n, lam0 * (1 if ai == 0 else rnd.uniform(0.7, 1.4)))
        steps = km(obs)
        risk = [(tau, sum(1 for u, _ in obs if u >= tau)) for tau in range(0, 37, 6)]
        cens = sorted({t for t, e in obs if not e and t < 36})[:25]
        km_arms.append([name, steps, risk, cens])
    eps = Fraction(1, 260)  # 1 pt on a 260 pt survival axis (extractor default)
    if corruption in ("km_risk_table", "km_big_drop"):
        ai = corrupt_arm
        name, steps, risk, cens = km_arms[ai]
        for m in range(len(risk) - 1):
            t0, t1 = risk[m][0], risk[m + 1][0]
            inside = [s for s in steps if t0 + Fraction(1, 5) < s[0] < t1 - Fraction(1, 5) and s[1] - s[2] > 3 * eps]
            near_edge = [s for s in steps if abs(s[0] - t0) <= Fraction(1, 5) or abs(s[0] - t1) <= Fraction(1, 5)]
            if corruption == "km_risk_table" and len(inside) >= 2 and not near_edge and m + 1 < len(risk) - 0:
                risk[m + 1] = (t1, risk[m][1] - len(inside) + 1)
                # keep later checkpoints monotone so only this interval is wrong
                for j in range(m + 2, len(risk)):
                    risk[j] = (risk[j][0], min(risk[j][1], risk[m + 1][1]))
                trial.corrupted_label = f"km:{ai}"
                break
            if corruption == "km_big_drop" and inside and not near_edge and risk[m + 1][1] >= 15:
                # a drop of 40% of the curve height in one step, with at most n_m - n_m+1 people leaving
                s = inside[0]
                need = s[1] * Fraction(2, 5)
                leaving = risk[m][1] - risk[m + 1][1]
                # factor a/(a+D) >= n_end/(n_end+leaving) must stay above the drawn factor
                if Fraction(risk[m + 1][1], risk[m + 1][1] + leaving) > 1 - Fraction(2, 5) + 4 * eps / s[1]:
                    delta = need
                    for st in steps:
                        if st[0] > s[0]:
                            st[1] -= delta
                            st[2] -= delta
                    s[2] -= delta
                    if all(st[2] > 0 for st in steps):
                        trial.corrupted_label = f"km:{ai}"
                        break
                    for st in steps:  # undo
                        if st[0] > s[0]:
                            st[1] += delta
                            st[2] += delta
                    s[2] += delta

    # ---------------- binary outcome (structured claim)
    a = sum(rnd.random() < 0.3 for _ in range(n1))
    cc = sum(rnd.random() < 0.3 for _ in range(n2))
    a, cc = max(a, 1), max(cc, 1)
    rr = Fraction(a * n2, n1 * cc)
    rr_txt = fmt(rr, 2, rnd.choice(MODES))
    if corruption == "rr_crude":
        cand = rr * Fraction(13, 10) + Fraction(5, 100)
        rr_txt = fmt(cand, 2, ROUND_HALF_UP)
        trial.corrupted_label = "Primary outcome"
    trial.claims.append({"kind": "two_by_two", "label": "Primary outcome", "n1": n1, "n2": n2, "events1": a,
                         "events2": cc, "rr": rr_txt, "estimate_is_crude": True})

    if corruption and trial.corrupted_label is None:
        raise ValueError(f"corruption {corruption} not applicable for seed {seed}")

    with pymupdf.open() as doc:
        render_table1(doc.new_page(width=612, height=792), rows, arms, foot)
        render_km(doc.new_page(width=612, height=792), km_arms)
        doc.save(trial.pdf)
    trial.meta = {"n": [n1, n2], "complete": complete}
    return trial


# ------------------------------------------------------------- registry

OUTCOME_POOL = [
    ("Modified Rankin Scale score", "90 days", "the distribution of modified Rankin Scale scores at 90 days"),
    ("All-cause mortality", "28 days", "death from any cause at 28 days"),
    ("Time to recovery", "28 days", "the time to recovery within 28 days after enrollment"),
    ("Change in HbA1c", "26 weeks", "the change from baseline in glycated hemoglobin (HbA1c) at 26 weeks"),
    ("Progression-free survival", "36 months", "progression-free survival, assessed by blinded central review"),
    ("Major adverse cardiovascular events", "12 months", "a composite of cardiovascular death, myocardial infarction, or stroke (major adverse cardiovascular events) at 12 months"),
    ("Oswestry Disability Index", "24 months", "the change in Oswestry Disability Index score at 24 months"),
    ("Ventilator-free days", "28 days", "the number of ventilator-free days through day 28"),
    ("Surgical site infection", "30 days", "surgical site infection within 30 days after surgery"),
    ("Pain intensity on numeric rating scale", "6 weeks", "pain intensity on the numeric rating scale at 6 weeks"),
]


def make_registry_case(seed: int, corruption: str | None = None):
    """Return (fixtures dict path->json, nct, paper text, corrupted?) for an offline fetcher."""
    rnd = random.Random(seed)
    nct = f"NCT{90000000 + seed:08d}"
    picks = rnd.sample(OUTCOME_POOL, 4)
    prim, secs = picks[0], picks[1:]
    n_plan = rnd.randint(80, 900)

    def ps(primary, enroll):
        return {"protocolSection": {
            "statusModule": {"startDateStruct": {"date": "2021-03-01", "type": "ACTUAL"},
                             "primaryCompletionDateStruct": {"date": "2023-06-30", "type": "ACTUAL"}},
            "designModule": {"enrollmentInfo": {"count": enroll, "type": "ESTIMATED"}},
            "outcomesModule": {"primaryOutcomes": [{"measure": primary[0], "timeFrame": primary[1]}],
                               "secondaryOutcomes": [{"measure": m, "timeFrame": t} for m, t, _ in secs]}}}

    n_actual = round(n_plan * rnd.uniform(0.9, 1.1))
    v0 = ps(prim, n_plan)
    later = ps(prim, n_actual)
    paper_primary = prim[2]
    if corruption == "endpoint_swap":
        paper_primary = secs[0][2]
    fx = {
        f"/api/v2/studies/{nct}": later,
        f"/api/int/studies/{nct}/history": {"changes": [
            {"version": 0, "date": "2021-01-15", "moduleLabels": []},
            {"version": 1, "date": "2023-09-01", "moduleLabels": ["Study Design"]}]},
        f"/api/int/studies/{nct}/history/0": {"studyVersion": 0, "study": v0},
        f"/api/int/studies/{nct}/history/1": {"studyVersion": 1, "study": later},
    }
    text = (f"Methods. This was a randomized trial (ClinicalTrials.gov {nct}). The primary outcome was {paper_primary}. "
            f"Results. A total of {n_actual} patients underwent randomization.")
    return fx, nct, text
