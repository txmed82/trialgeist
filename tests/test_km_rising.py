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


def _retitle(tmp_path, path, title, name):
    with pymupdf.open(path) as doc:
        page = doc[0]
        for inst in page.search_for("Kaplan-Meier estimates of cumulative mortality"):
            page.add_redact_annot(inst)
        page.apply_redactions()
        page.insert_text((50, 90), title, fontsize=10)
        doc.save(tmp_path / name)
    return tmp_path / name


def test_rising_refused_when_caption_may_be_cumulative_incidence(tmp_path):
    path = make(tmp_path, rising=True)
    for i, title in enumerate(["Kaplan-Meier estimates of the cumulative risk of recurrence, death as a competing event",
                               "Kaplan-Meier cumulative probability of relapse"]):
        plots, notes = find_km_plots(str(_retitle(tmp_path, path, title, f"t{i}.pdf")))
        assert not plots and any("cumulative incidence" in n or "competing" in n or "adjusted" in n for n in notes), notes


def test_same_style_arms_sharing_the_start_point_are_both_read_whole(tmp_path):
    from trialgeist.bench.synthetic import render_km as rk

    taus = list(range(0, 37, 6))
    arms = [(nm, km_from_times(n, d), risk(n, d, taus), [36]) for nm, n, d in
            [("Drug", 40, [2, 3, 5, 8, 8, 13, 17, 22, 29]), ("Placebo", 38, [1, 2, 4, 4, 7, 9, 11, 15, 20, 25, 31])]]
    path = tmp_path / "same_style.pdf"
    with pymupdf.open() as doc:
        rk(doc.new_page(width=612, height=792), arms, colors=[(0, 0, 0), (0, 0, 0)])
        doc.save(path)
    plots, notes = find_km_plots(str(path))
    assert plots, notes
    curves = [c for p in plots for c in p.curves]
    drops = sorted(sum(1 for a, b in zip(c.points, c.points[1:]) if b[1] - a[1] > 1e-3) for c in curves)
    assert drops == [8, 10], drops  # distinct death times per arm: 8 and 10 drops
    tiers = [f.tier for p in plots for c in plot_to_claims(p) for f in check_claim(c)]
    assert tiers and all(t is Tier.CONSISTENT for t in tiers)
