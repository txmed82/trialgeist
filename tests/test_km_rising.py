"""Rising (1 - S) Kaplan-Meier plots: extracted, converted, and judged like falling ones."""

from __future__ import annotations

from fractions import Fraction as F

import pymupdf

from trialgeist.bench.synthetic import render_km
from trialgeist.claims import check_claim
from trialgeist.extract.km_vector import find_km_plots, plot_to_claims
from trialgeist.verdict import Tier


def km_from_times(n, deaths):
    """Exact KM steps for n patients with the given death times (others censored at the end)."""
    s, at, steps = F(1), n, []
    for t in sorted(set(deaths)):
        d = deaths.count(t)
        a = s * (at - d) / at
        steps.append([t, s, a])
        s, at = a, at - d
    return steps


def risk(n, deaths, taus):
    return [(tau, n - sum(1 for t in deaths if t < tau)) for tau in taus]


def make(tmp_path, rising, edit=None):
    taus = list(range(0, 37, 6))
    arms = []
    for name, n, deaths in [("Drug", 40, [2, 3, 5, 8, 8, 13, 17, 22, 29]), ("Placebo", 38, [1, 2, 4, 4, 7, 9, 11, 15, 20, 25, 31])]:
        rk = risk(n, deaths, taus)
        if edit and name == "Drug":
            rk = [(t, k + edit if t == 12 else k) for t, k in rk]
        arms.append((name, km_from_times(n, deaths), rk, [36]))
    path = tmp_path / f"km_{rising}_{edit}.pdf"
    with pymupdf.open() as doc:
        render_km(doc.new_page(width=612, height=792), arms, rising=rising)
        doc.save(path)
    return path


def verdicts(path):
    plots, notes = find_km_plots(str(path))
    assert plots, notes
    return [f.tier for pl in plots for c in plot_to_claims(pl) for f in check_claim(c)], plots


def test_rising_plot_is_read_as_one_minus_survival(tmp_path):
    tiers, plots = verdicts(make(tmp_path, rising=True))
    assert plots[0].curves[0].rising
    assert tiers and all(t is Tier.CONSISTENT for t in tiers)
    claim = plot_to_claims(plots[0])[0]
    first = claim.steps[0]
    assert float(first.before) > 0.95 and float(first.after) < float(first.before)  # survival, not mortality


def test_rising_and_falling_agree(tmp_path):
    up, _ = verdicts(make(tmp_path, rising=True, edit=4))
    down, _ = verdicts(make(tmp_path, rising=False, edit=4))
    assert Tier.PROVEN_IMPOSSIBLE in up and Tier.PROVEN_IMPOSSIBLE in down


def test_rising_requires_kaplan_meier_wording(tmp_path):
    path = make(tmp_path, rising=True)
    with pymupdf.open(path) as doc:
        for inst in doc[0].search_for("Kaplan-Meier estimates of"):
            doc[0].add_redact_annot(inst)
        doc[0].apply_redactions()
        doc.save(tmp_path / "no_km_words.pdf")
    plots, _ = find_km_plots(str(tmp_path / "no_km_words.pdf"))
    assert not plots  # a rising curve might be cumulative incidence with competing risks
