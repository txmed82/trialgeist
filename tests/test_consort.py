"""Randomisation arithmetic in prose (extract/consort.py)."""

from trialgeist.claims import check_claim
from trialgeist.extract.consort import allocation_claims
from trialgeist.verdict import Tier


def one(text):
    cs = allocation_claims(text)
    return [(c.total, c.counts) for c in cs]


def test_reads_enumerated_allocation():
    assert one("A total of 605 patients were randomly assigned to receive drug (n = 300) or placebo (n = 305).") == [(605, [300, 305])]
    assert one("Overall, 1,204 participants were randomized in a 1:1:1 ratio to 10 mg (n = 401), 20 mg (n = 402), or placebo (n = 401). Next.") == [(1204, [401, 402, 401])]
    assert one("Of these, 90 were randomised to surgery (n=45) and observation (n=45).") == [(90, [45, 45])]


def test_mismatch_is_proven_with_a_certificate(tmp_path):
    import subprocess, sys

    (c,) = allocation_claims("In total, 605 patients were randomly assigned to drug (n = 300) or placebo (n = 302).")
    (f,) = check_claim(c)
    assert f.tier is Tier.PROVEN_IMPOSSIBLE and f.certificate
    path = tmp_path / "cert.py"
    path.write_text(f.certificate)
    r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=10)
    assert r.returncode == 0 and "VERIFIED" in r.stdout


def test_refusals():
    refuse = [
        "12 practices were randomised to intervention (n = 750) or control (n = 750).",                # clusters
        "600 patients were randomized to drug (n = 300) or placebo (n = 300); 150 of them to C (n = 75).",  # clause ends at ;
        "300 eyes were randomized to laser (n = 150) or sham (n = 160).",                              # eyes
        "400 patients were randomized to A (n = 200) or B (n = 190), respectively, after screening.",  # respectively
        "400 patients were randomized to A (n = 200) or B (n = 190), of whom 10 withdrew.",            # of whom
        "400 patients were randomized to A (n = 210, 52%) or B (n = 190).",                            # not the (n = k) form
        "400 patients were randomized to A (n = 210), B (n = 100).",                                   # list not closed
    ]
    got = [one(t) for t in refuse]
    assert got[0] == [] and got[2] == [] and got[3] == [] and got[4] == [] and got[5] == [] and got[6] == []
    assert got[1] == [(600, [300, 300])]  # the clause before ';' is a complete partition
    assert allocation_claims("200 patients were randomized to A (n = 100) or B (n = 90).",
                             doc_text="This 2x2 factorial trial ...") == []
    assert allocation_claims("200 patients were randomized to A (n = 100) or B (n = 90).",
                             doc_text="a cluster-randomised trial") == []


def test_form_b_counts_before_arms():
    assert one("In total, 605 patients were randomised, 300 to drug and 305 to placebo.") == [(605, [300, 305])]
    assert one("900 participants were randomized: 301 to semaglutide 2.4 mg, 299 to semaglutide 1.0 mg, and 300 to placebo.") == [(900, [301, 299, 300])]
    (c,) = allocation_claims("In total, 605 patients were randomised, 300 to drug and 302 to placebo.")
    assert check_claim(c)[0].tier is Tier.PROVEN_IMPOSSIBLE
    for bad in ["605 patients were randomised, 300 to drug and 302 to placebo, and 3 to an open-label extension.",
                "605 patients were randomised, 300 to drug and 302 to placebo, of whom 3 withdrew.",
                "605 patients were randomised, 300 to drug, 302 to placebo.",
                "605 patients were randomised, 300 to drug and 302 to placebo between 2019 and 2021."]:
        assert one(bad) == [], bad


def test_form_c_parenthetical_and_joined_totals():
    assert one("Of 577 participants screened, 511 were eligible and were randomised (258 to placebo, 253 to nilvadipine).") == [(511, [258, 253])]
    assert one("A total of 302 were randomized (n = 101 in BA; n = 99 in CT; n = 102 in MM).") == [(302, [101, 99, 102])]
    assert one("Of 598 screened, 324 qualified and were randomized: 162 to dapagliflozin and 162 to placebo.") == [(324, [162, 162])]
    assert one("In all, 530 patients were randomly assigned to empagliflozin (n = 265) or placebo (n = 265) at 118 centers in 15 countries (Fig. 1).") == [(530, [265, 265])]
    for bad in ["A total of 520 participants were randomized (260 per group).",
                "A total of 520 participants were randomized (260 [50%] to drug, 260 to placebo).",
                "In all, 300 patients were randomized (150 to drug, 150 to placebo, of whom 3 withdrew).",
                "The 78 were randomised to TAU and 78 to RCBT."]:
        assert one(bad) == [], bad
