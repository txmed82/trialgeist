"""A0: registry time-machine on saved ClinicalTrials.gov fixtures (offline)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from trialgeist.extract.paper_outcomes import paper_from_text
from trialgeist.registry.diff import (PaperClaims, durations_days, fixture_fetcher, load_registry, parse_date,
                                      registry_diff, select_version, similarity)
from trialgeist.verdict import Tier

FIX = Path(__file__).parent / "fixtures" / "ctgov"


def test_select_version_rules():
    ch = [{"version": 0, "date": "2020-02-20"}, {"version": 10, "date": "2020-03-24"}, {"version": 11, "date": "2020-03-24"}]
    assert select_version(ch, date(2020, 3, 24)) == (11, False)  # same-day: max version
    assert select_version(ch, date(2020, 1, 1)) == (0, True)  # retrospective fallback, flagged
    assert parse_date("2014-01", end_of_period=True) == date(2014, 1, 31)


def test_actt1_time_machine():
    view = load_registry("NCT04280705", fixture_fetcher(FIX))
    assert view.at_start.version == 1  # 2020-02-21 start
    assert view.at_primary_completion.version == 17
    assert "ordinal" in view.at_start.primary[0].measure.lower()
    assert "recovery" in view.at_primary_completion.primary[0].measure.lower()
    assert "&#x2F;" not in view.at_primary_completion.primary[0].description  # html unescaped
    paper = paper_from_text(
        "Methods ... The primary outcome was the time to recovery, defined by the first day, during the 28 days after "
        "enrollment, on which a patient satisfied categories 1, 2, or 3 on the eight-category ordinal scale. "
        "A total of 1062 patients underwent randomization.")
    assert paper.randomized_n == 1062
    fs = registry_diff(view, paper)
    checks = {f.check for f in fs}
    assert "primary_changed_after_start" in checks
    assert "sample_size_changed" in checks  # 394 planned at start
    assert all(f.tier is Tier.REGISTRY_DISCORDANT and f.certificate is None for f in fs)
    assert any("tab=history&a=2" in (p.text or "") for f in fs for p in f.provenance)


def test_no_discordance_when_paper_matches_registry():
    view = load_registry("NCT04280705", fixture_fetcher(FIX))
    view.at_start = view.at_primary_completion
    paper = PaperClaims(primary_outcomes=["time to recovery within 28 days of enrollment"], randomized_n=600)
    view.at_start.enrollment = 572
    assert registry_diff(view, paper) == []


def test_text_helpers():
    assert similarity("Time to recovery", "the time to clinical recovery by day 28") >= 0.6
    assert durations_days("at 12 months") == {365.2}
    assert durations_days("three-month mRS") == {91.3}
