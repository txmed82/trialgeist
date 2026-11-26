"""Engine 3: Carlisle-Stouffer baseline randomization forensics.

Never produces PROVEN_IMPOSSIBLE.  Output is STATISTICALLY_EXTREME (combined
p < 1e-5 in either tail) or CONSISTENT / INCONCLUSIVE, with a standalone
simulation script so the calibration can be re-run independently.

Guardrails:
* cluster randomization -> INCONCLUSIVE (individual-level tests invalid);
* stratified / minimized / blocked designs: variables named as stratification
  factors are dropped, and the under-dispersion tail (arms "too similar") is
  only reported when at least 5 unstratified continuous variables remain;
* rounding: per-variable p-values come from the printed summaries, and the
  attached script calibrates the combined statistic by simulating trials
  whose summaries are rounded to the printed precision.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from scipy import stats

from ..verdict import Finding, Provenance, Tier, make_finding

ENGINE = "carlisle"
THRESHOLD = 1e-5


@dataclass
class ContinuousVar:
    name: str
    means: list[str]
    sds: list[str]
    ns: list[int]


@dataclass
class CategoricalVar:
    name: str
    counts: list[list[int]]  # per arm, per category (a binary row is [[k1, n1-k1], [k2, n2-k2]])


@dataclass
class Design:
    randomization: str = "unknown"  # simple | block | stratified | minimized | cluster | unknown
    strata: list[str] = field(default_factory=list)


DESIGN_PATTERNS = [
    ("cluster", re.compile(r"cluster[- ]randomi[sz]|randomi[sz]ed by (site|cluster|ward|practice|hospital)", re.I)),
    ("minimized", re.compile(r"minimi[sz]ation|minimi[sz]ed", re.I)),
    ("stratified", re.compile(r"stratified (by|according|for|on)|stratification (factor|variable)s?", re.I)),
    ("block", re.compile(r"(permuted|random(ly)?[- ]sized)? ?blocks? (of|size)|block randomi[sz]", re.I)),
    ("simple", re.compile(r"simple randomi[sz]ation|unrestricted randomi[sz]ation", re.I)),
]
STRATA_RE = re.compile(r"stratified (?:by|according to|for|on)\s+([^.;]{3,200})", re.I)


def detect_design(methods_text: str) -> Design:
    for kind, rx in DESIGN_PATTERNS:
        if rx.search(methods_text):
            d = Design(kind)
            m = STRATA_RE.search(methods_text)
            if m:
                d.strata = [s.strip(" ,") for s in re.split(r",|\band\b|\bor\b", m.group(1)) if len(s.strip()) > 2]
            return d
    return Design()


def _dec(s: str) -> int:
    s = s.replace("\u00b7", ".")
    return len(s.split(".")[1]) if "." in s else 0


def _num(x: str) -> float:
    return float(x.replace("\u00b7", ".").replace("\u2212", "-"))


def rounded_diff_midp(m1: float, m2: float, se1: float, se2: float, mu: float, h: float) -> float:
    """Two-sided mid-p for the difference of two means printed on a grid of step h.

    Under the null each true mean is N(mu, se_i^2) and is then rounded to the
    grid; R = round(m1) - round(m2) is discrete, so a naive p-value piles up
    at 1 (identical printed means) and fakes under-dispersion.  The mid-p
    P(|R| > |r|) + P(|R| = |r|)/2 is close to uniform under the null.
    """
    import numpy as np
    from scipy.signal import fftconvolve

    js = np.arange(math.floor((mu - 9 * max(se1, se2)) / h) - 1, math.ceil((mu + 9 * max(se1, se2)) / h) + 2)
    p1, p2 = (stats.norm.cdf((js * h + h / 2 - mu) / se) - stats.norm.cdf((js * h - h / 2 - mu) / se) for se in (se1, se2))
    # distribution of R = J1 - J2 (index G-1 is R = 0): a convolution, O(G) memory instead of a G x G grid
    pr = np.convolve(p1, p2[::-1]) if len(js) <= 4000 else np.clip(fftconvolve(p1, p2[::-1]), 0, None)
    k = np.abs(np.arange(pr.size) - (len(js) - 1))
    r = abs(round(m1 / h) - round(m2 / h))
    gt, eq, tot = float(pr[k > r].sum()), float(pr[k == r].sum()), float(pr.sum())
    return min(1.0, max(0.0, (gt + 0.5 * eq) / tot))


def continuous_p(v: ContinuousVar) -> float:
    """Two arms: rounding-aware mid-p.  More arms: one-way ANOVA F (needs fine rounding)."""
    m = [_num(x) for x in v.means]
    s = [_num(x) for x in v.sds]
    n, k, N = v.ns, len(v.ns), sum(v.ns)
    sp = math.sqrt(sum((ni - 1) * si**2 for ni, si in zip(n, s)) / (N - k)) if N > k else 0.0
    h = 10.0 ** -max(_dec(x) for x in v.means)
    if sp <= 0:
        return 0.5
    if k == 2:
        mu = (n[0] * m[0] + n[1] * m[1]) / N
        return rounded_diff_midp(m[0], m[1], sp / math.sqrt(n[0]), sp / math.sqrt(n[1]), mu, h)
    grand = sum(ni * mi for ni, mi in zip(n, m)) / N
    ssb = sum(ni * (mi - grand) ** 2 for ni, mi in zip(n, m))
    F = (ssb / (k - 1)) / sp**2
    return float(stats.f.sf(F, k - 1, N - k))


def continuous_usable(v: ContinuousVar) -> bool:
    if len(v.ns) == 2:
        return True
    s = [_num(x) for x in v.sds]
    h = 10.0 ** -max(_dec(x) for x in v.means)
    return h < 0.1 * min(s) / math.sqrt(max(v.ns))  # ANOVA only when rounding is negligible


def fisher_midp(t: list[list[int]]) -> float:
    (a, b), (c, d) = t
    r1, c1, N = a + b, a + c, a + b + c + d
    lo, hi = max(0, r1 + c1 - N), min(r1, c1)
    pm = {x: float(stats.hypergeom.pmf(x, N, c1, r1)) for x in range(lo, hi + 1)}
    po = pm[a]
    less = sum(p for p in pm.values() if p < po * (1 - 1e-7))
    eq = sum(p for p in pm.values() if abs(p - po) <= po * 1e-7)
    return min(1.0, less + 0.5 * eq)


def categorical_p(v: CategoricalVar) -> float:
    table = [row for row in v.counts]
    if len(table) == 2 and len(table[0]) == 2:
        return fisher_midp(table)
    return float(stats.chi2_contingency(table, correction=False).pvalue)


def stouffer(ps: list[float]) -> tuple[float, float, float]:
    """(Z, p_low, p_high): Z = sum(Phi^-1(p))/sqrt(k); p_low small = arms too different."""
    eps = 1e-300
    z = sum(stats.norm.ppf(min(max(p, eps), 1 - 1e-16)) for p in ps) / math.sqrt(len(ps))
    return z, float(stats.norm.cdf(z)), float(stats.norm.sf(z))


STRATA_STOP = {"and", "the", "for", "site", "status", "center", "centre", "group", "level", "presence", "history"}


def _is_stratum(name: str, strata: list[str]) -> bool:
    words = set(re.findall(r"[a-z]{3,}", name.lower()))
    return any(w in words for s in strata for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in STRATA_STOP)


def carlisle_check(cont: list[ContinuousVar], cat: list[CategoricalVar], design: Design, label: str = "",
                   provenance: list[Provenance] | None = None) -> Finding:
    used_c = [v for v in cont if not _is_stratum(v.name, design.strata) and continuous_usable(v)]
    used_k = [v for v in cat if not _is_stratum(v.name, design.strata)]
    cluster = design.randomization == "cluster"  # individual-level tests are invalid; don't compute them
    pvals = {} if cluster else {v.name: continuous_p(v) for v in used_c}
    pvals.update({} if cluster else {v.name: categorical_p(v) for v in used_k})
    inputs: dict[str, Any] = {
        "label": label, "randomization": design.randomization, "strata": design.strata,
        "continuous": [v.__dict__ for v in used_c], "categorical": [v.__dict__ for v in used_k],
        "dropped_as_strata": [v.name for v in cont + cat if _is_stratum(v.name, design.strata)],
    }
    assumptions = [
        "baseline variables are approximately normal (continuous) or exchangeable counts (categorical) within arms",
        "variables are treated as independent; correlated variables inflate both tails",
        f"randomization: {design.randomization}" + (f", strata {design.strata} excluded" if design.strata else ""),
    ]
    common: dict[str, Any] = dict(engine=ENGINE, check="carlisle_stouffer", inputs=inputs, assumptions=assumptions,
                                  provenance=provenance, per_variable_p={k: round(p, 6) for k, p in pvals.items()})
    if design.randomization == "cluster":
        return make_finding(**{k: v for k, v in common.items() if k != "per_variable_p"}, tier=Tier.INCONCLUSIVE,
                            title="Baseline test skipped: cluster randomization",
                            detail="Individual-level baseline tests are invalid under cluster randomization.")
    if len(pvals) < 3:
        return make_finding(**{k: v for k, v in common.items() if k != "per_variable_p"}, tier=Tier.INCONCLUSIVE,
                            title="Too few baseline variables for a combined test", detail=f"{len(pvals)} usable variables")
    z, p_low, p_high = stouffer(list(pvals.values()))
    stats_ = {"stouffer_z": round(z, 4), "p_overdispersed": p_low, "p_underdispersed": p_high, "k": len(pvals)}
    common.pop("per_variable_p")
    stats_["per_variable_p"] = {k: round(p, 6) for k, p in pvals.items()}
    under_ok = len(used_c) >= 5 and design.randomization in ("simple", "block", "unknown", "stratified", "minimized")
    tier, title = Tier.CONSISTENT, "Baseline distribution compatible with randomization"
    if p_low < THRESHOLD:
        tier, title = Tier.STATISTICALLY_EXTREME, f"Baseline arms more different than chance (p = {p_low:.2g})"
    elif p_high < THRESHOLD and under_ok:
        tier, title = Tier.STATISTICALLY_EXTREME, f"Baseline arms more similar than chance (p = {p_high:.2g})"
    detail = (f"Stouffer Z = {z:.3f} over {len(pvals)} variables; P(arms this different) = {p_low:.3g}, "
              f"P(arms this similar) = {p_high:.3g}.")
    f = make_finding(**common, tier=tier, title=title + (f" ({label})" if label else ""), detail=detail, stats=stats_)
    if tier is Tier.STATISTICALLY_EXTREME:
        f.certificate = simulation_script(f.id, used_c, used_k, z)
    return f


def simulation_script(fid: str, cont: list[ContinuousVar], cat: list[CategoricalVar], z_obs: float) -> str:
    cv = [(v.name, v.means, v.sds, v.ns, max(_dec(x) for x in v.means), max(_dec(x) for x in v.sds)) for v in cont]
    kv = [(v.name, v.counts) for v in cat]
    return f'''#!/usr/bin/env python3
"""TRIALGEIST STATISTICALLY_EXTREME simulation, finding {fid}.
Recomputes the Carlisle-Stouffer statistic from the printed Table 1 and
calibrates it by simulating randomized trials whose summaries are rounded to
the printed precision.  This is a probability statement, not a proof.
Requires numpy + scipy.
"""
import math, numpy as np
from scipy import stats
from scipy.signal import fftconvolve
CONT = {cv!r}  # name, means, sds, ns, mean decimals, sd decimals
CAT = {kv!r}
Z_OBS = {z_obs!r}

def p_cont(m, s, n, h):  # rounding-aware two-sided mid-p (2 arms) or ANOVA F
    N, k = sum(n), len(n); sp = math.sqrt(sum((a - 1) * b * b for a, b in zip(n, s)) / (N - k))
    if sp <= 0: return 0.5
    if k > 2:
        g = sum(a * b for a, b in zip(n, m)) / N
        return stats.f.sf(sum(a * (b - g) ** 2 for a, b in zip(n, m)) / (k - 1) / sp ** 2, k - 1, N - k)
    mu, se = (n[0] * m[0] + n[1] * m[1]) / N, [sp / math.sqrt(x) for x in n]
    js = np.arange(math.floor((mu - 9 * max(se)) / h) - 1, math.ceil((mu + 9 * max(se)) / h) + 2)
    p1, p2 = [stats.norm.cdf((js * h + h / 2 - mu) / e) - stats.norm.cdf((js * h - h / 2 - mu) / e) for e in se]
    pr = np.clip(fftconvolve(p1, p2[::-1]), 0, None); k = np.abs(np.arange(pr.size) - (len(js) - 1))  # law of J1 - J2
    r = abs(round(m[0] / h) - round(m[1] / h))
    return float((pr[k > r].sum() + 0.5 * pr[k == r].sum()) / pr.sum())

def p_cat(t):
    if len(t) == 2 and len(t[0]) == 2:
        (a, b), (c, d) = t; pm = stats.hypergeom.pmf(np.arange(0, a + b + 1), a + b + c + d, a + c, a + b); po = pm[a]
        return float(pm[pm < po * (1 - 1e-7)].sum() + 0.5 * pm[np.abs(pm - po) <= po * 1e-7].sum())
    return stats.chi2_contingency(t, correction=False).pvalue

def z_of(ps):
    return sum(stats.norm.ppf(min(max(p, 1e-300), 1 - 1e-16)) for p in ps) / math.sqrt(len(ps))

ps = [p_cont([float(x) for x in m], [float(x) for x in s], n, 10.0 ** -dm) for _, m, s, n, dm, _ in CONT] + [p_cat(t) for _, t in CAT]
z = z_of(ps)
print(f"observed Stouffer Z = {{z:.4f}} (report: {{Z_OBS:.4f}})")
rng, sims = np.random.default_rng(1), []
for _ in range(1000):
    q = []
    for _, m, s, n, dm, ds in CONT:
        mu = sum(float(a) * b for a, b in zip(m, n)) / sum(n); sd = math.sqrt(sum((b - 1) * float(a) ** 2 for a, b in zip(s, n)) / (sum(n) - len(n)))
        xs = [rng.normal(mu, sd, b) for b in n]
        q.append(p_cont([round(x.mean(), dm) for x in xs], [round(x.std(ddof=1), ds) for x in xs], n, 10.0 ** -dm))
    for _, t in CAT:
        tot = [sum(c) for c in zip(*t)]; ns = [sum(r) for r in t]
        lab = np.repeat(np.arange(len(tot)), tot); rng.shuffle(lab); cut = np.cumsum([0] + ns)
        q.append(p_cat([[int((lab[cut[i]:cut[i + 1]] == j).sum()) for j in range(len(tot))] for i in range(len(ns))]))
    sims.append(z_of(q))
sims = np.array(sims)
print(f"simulated null Z: mean {{sims.mean():.3f}}, sd {{sims.std():.3f}} (rounding-aware)")
lo = stats.norm.cdf((z - sims.mean()) / sims.std()); hi = stats.norm.sf((z - sims.mean()) / sims.std())
print(f"calibrated P(this different) = {{lo:.3g}}, P(this similar) = {{hi:.3g}}")
print("EXTREME" if min(lo, hi) < 1e-5 else "NOT EXTREME after calibration")
'''
