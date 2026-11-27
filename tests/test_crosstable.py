"""Cross-table extraction + engines: detect planted contradictions, stay silent on real tables."""

from __future__ import annotations

import random
from fractions import Fraction as F

from trialgeist.claims import check_claim
from trialgeist.engines.consistency import arm_total_check, common_denominator_check, mean_partition_check
from trialgeist.extract.crosstable import Grid, GridRow, grid_claims
from trialgeist.certificates import run_certificate
from trialgeist.verdict import Tier


def fmt(x: F, d: int = 1) -> str:
    q = round(x * 10 ** d)
    return f"{q / 10 ** d:.{d}f}"


def grid(rows, cols=("Drug (n=60)", "Placebo (n=58)", "Total (n=118)"), ns=(60, 58, 118), complete=False):
    g = Grid(label="Table 1", caption="Baseline characteristics", col_n={i + 1: n for i, n in enumerate(ns)},
             col_head={i + 1: h for i, h in enumerate(cols)}, rows=[], complete=complete)
    sec = -1
    for r, (label, cells) in enumerate(rows):
        if cells is None:
            sec, sec_label = r, label
            continue
        ctx = label + (" | " + sec_label if sec >= 0 else "")
        g.rows.append(GridRow(label=label, context=ctx, section=sec, cells={i + 1: c for i, c in enumerate(cells)}))
    return g


def proven(claims):
    return [f for c in claims for f in check_claim(c) if f.tier is Tier.PROVEN_IMPOSSIBLE]


def test_detects_planted_contradictions():
    g = grid([
        ("Male sex, n (%)", ["30 (50.0)", "29 (50.0)", "60 (50.8)"]),        # 30 + 29 != 60
        ("Age, mean (SD), y", ["54.1 (9.2)", "55.3 (8.7)", "56.0 (9.0)"]),   # 56.0 outside [54.1, 55.3]
        ("Race, n (%)", None),
        ("White", ["40 (66.7)", "38 (65.5)", "78 (66.1)"]),
        ("Black", ["12 (20.0)", "13 (22.4)", "25 (21.2)"]),
        ("Other", ["8 (13.3)", "7 (12.1)", "15 (12.7)"]),
        ("Length of stay, median (IQR), d", ["6 (8-10)", "5 (3-8)", "5 (3-9)"]),  # 6 (< 7) below Q1 8 (>= 7.5)
    ])
    titles = [f.check for f in proven(grid_claims(g))]
    assert titles.count("arm_total") == 1 and "mean_hull" in titles and "median_range" in titles, titles


def test_common_denominator_planted(tmp_path):
    g = grid([("Smoking", None), ("Never", ["20 (40.0)", "", ""]), ("Former", ["20 (33.3)", "", ""]),
              ("Current", ["20 (26.7)", "", ""])])  # 20/D cannot be both 40.0 and 33.3; sum 100.0
    fs = proven(grid_claims(g))
    assert [f.check for f in fs] == ["common_denominator"]
    ok, _, out = run_certificate(fs[0].certificate, tmp_path, fs[0].id)
    assert ok, out


def test_silent_on_consistent_random_tables():
    rnd = random.Random(11)
    for _ in range(300):
        na, nb = rnd.randint(10, 300), rnd.randint(10, 300)
        miss_a, miss_b = rnd.randint(0, 3), rnd.randint(0, 3)  # per-row missing data
        da, db = na - miss_a, nb - miss_b
        def split(d, k):
            cuts = sorted(rnd.randint(0, d) for _ in range(k - 1))
            return [b - a for a, b in zip([0] + cuts, cuts + [d])]
        ka, kb = split(da, 3), split(db, 3)
        dec = rnd.choice([0, 1])
        rows = [("Category, n (%)", None)]
        for i in range(3):
            rows.append((f"level {i}", [f"{ka[i]} ({fmt(F(100 * ka[i], da), dec)})", f"{kb[i]} ({fmt(F(100 * kb[i], db), dec)})",
                                        f"{ka[i] + kb[i]} ({fmt(F(100 * (ka[i] + kb[i]), da + db), dec)})"]))
        xa = [rnd.randint(200, 900) for _ in range(da)]
        xb = [rnd.randint(200, 900) for _ in range(db)]
        ma, mb, mt = F(sum(xa), 10 * da), F(sum(xb), 10 * db), F(sum(xa) + sum(xb), 10 * (da + db))
        rows.append(("Weight, mean (SD), kg", [f"{fmt(ma)} (1.0)", f"{fmt(mb)} (1.0)", f"{fmt(mt)} (1.0)"]))
        sa = sorted(xa)
        med = F(sa[(da - 1) // 2] + sa[da // 2], 20)
        q1, q3 = F(sa[da // 4], 10), F(sa[(3 * da) // 4], 10)
        rows.append(("Score, median (IQR)", [f"{fmt(med)} ({fmt(q1)}-{fmt(q3)})", "", ""]))
        g = grid(rows, ns=(na, nb, na + nb))
        bad = proven(grid_claims(g))
        assert not bad, (bad[0].detail, rows)


def test_no_total_claims_when_ns_do_not_add():
    g = grid([("Male, n (%)", ["30 (50.0)", "29 (50.0)", "60 (50.8)"])], ns=(60, 58, 120))
    assert not [c for c in grid_claims(g) if c.kind == "arm_total"]


def test_ci_rows_are_not_spreads():
    g = grid([("Median survival (95% CI), mo", ["14 (15-20)", "10 (8-12)", "12 (10-14)"])])
    assert not grid_claims(g)


def test_engines_certificates(tmp_path):
    for f in [arm_total_check(60, [30, 29], 118, [60, 58]), mean_partition_check("56.0", [(60, "54.1"), (58, "55.3")], weights_exact=False),
              common_denominator_check([20, 20, 20], ["40.0", "33.3", "26.7"])]:
        assert f.tier is Tier.PROVEN_IMPOSSIBLE
        ok, secs, out = run_certificate(f.certificate, tmp_path, f.id)
        assert ok and "VERIFIED" in out and secs < 2, out
        assert len(f.certificate.splitlines()) < 80


# ---------------------------------------------------------------- PDF grids


def _table_pdf(tmp_path, rows, name="t.pdf", heads=("Drug", "Placebo", "Total"), ns=(60, 58, 118)):
    """Typeset a baseline table the way journals do: caption, two-line header, indented categories."""
    import pymupdf

    path = tmp_path / name
    with pymupdf.open() as doc:
        page = doc.new_page(width=612, height=792)
        page.insert_text((50, 60), "Table 1. Baseline characteristics", fontsize=10)
        xs = (260, 360, 460)
        for x, h, n in zip(xs, heads, ns):
            page.insert_text((x, 84), h, fontsize=9)
            page.insert_text((x, 96), f"(n = {n})", fontsize=9)
        y = 114
        for label, cells, indent in rows:
            page.insert_text((50 + 10 * indent, y), label, fontsize=9)
            for x, c in zip(xs, cells or ()):
                page.insert_text((x, y), c, fontsize=9)
            y += 13
        doc.save(path)
    return path


ROWS = [
    ("Age, years, mean (SD)", ("61.2 (8.1)", "60.4 (7.9)", "60.8 (8.0)"), 0),
    ("Sex, n (%)", None, 0),
    ("Female", ("27 (45.0)", "25 (43.1)", "52 (44.1)"), 1),
    ("Male", ("33 (55.0)", "33 (56.9)", "66 (55.9)"), 1),
    ("Smoking status", None, 0),
    ("Never", ("30 (50.0)", "29 (50.0)", "59 (50.0)"), 1),
    ("Former", ("20 (33.3)", "19 (32.8)", "39 (33.1)"), 1),
    ("Current", ("10 (16.7)", "10 (17.2)", "20 (16.9)"), 1),
    ("BMI, kg/m2, median (IQR)", ("27 (24-30)", "28 (25-31)", "27 (25-31)"), 0),
]


def test_pdf_grid_structure(tmp_path):
    from trialgeist.extract.crosstable import pdf_grids
    from trialgeist.extract.pdf_text import load_pages

    grids = pdf_grids(load_pages(str(_table_pdf(tmp_path, ROWS))))
    assert len(grids) == 1
    g = grids[0]
    assert g.col_n == {1: 60, 2: 58, 3: 118}
    assert "Total" in g.col_head[3]
    rows = {r.label: r for r in g.rows}
    assert rows["Female"].cells == {1: "27 (45.0)", 2: "25 (43.1)", 3: "52 (44.1)"}
    assert rows["Female"].section == rows["Male"].section >= 0
    assert rows["Never"].section == rows["Current"].section != rows["Female"].section
    assert rows["Age, years, mean (SD)"].section == -1


def test_pdf_grid_clean_table_is_silent(tmp_path):
    from trialgeist.extract.crosstable import pdf_grids
    from trialgeist.extract.pdf_text import load_pages

    claims = [c for g in pdf_grids(load_pages(str(_table_pdf(tmp_path, ROWS)))) for c in grid_claims(g)]
    assert {c.kind for c in claims} >= {"arm_total", "common_denominator", "mean_partition"}
    assert not proven(claims)


def test_pdf_grid_catches_bad_total_and_bad_category(tmp_path):
    from trialgeist.extract.crosstable import pdf_grids
    from trialgeist.extract.pdf_text import load_pages

    bad = list(ROWS)
    bad[2] = ("Female", ("27 (45.0)", "25 (43.1)", "55 (46.6)"), 1)  # 27 + 25 != 55 in the Total column
    bad[7] = ("Current", ("10 (16.7)", "12 (17.2)", "20 (16.9)"), 1)  # 12/D = 17.2% needs D ~ 70 > 58
    claims = [c for g in pdf_grids(load_pages(str(_table_pdf(tmp_path, bad, "bad.pdf")))) for c in grid_claims(g)]
    found = proven(claims)
    assert any("Female" in (p.note or "") + (p.text or "") for f in found for p in f.provenance)
    assert any("Current" in (p.note or "") for f in found for p in f.provenance)
