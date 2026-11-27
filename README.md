# TRIALGEIST

Machine-verified mathematical forensics for clinical trials.

TRIALGEIST reads a trial PDF, PMC JATS XML, or ClinicalTrials.gov ID and looks
for reported numbers that cannot all be true. It only accuses a paper when it
can hand you a proof. Every `PROVEN_IMPOSSIBLE` finding comes with a standalone
`certificate_<id>.py`: under 80 lines, using exact `fractions` arithmetic (and
Z3 for multiset and Kaplan-Meier problems), finishing in under 2 seconds, and
re-deriving the contradiction from the printed digits. The certificate never
imports TRIALGEIST.

```
pipx install git+https://github.com/txmed82/trialgeist   # or, from a checkout: uv tool install .
trialgeist scan paper.pdf --nct NCT01234567 --out report/
trialgeist scan paper.nxml --out report/        # PMC JATS XML: exact table cells
trialgeist check claims.json --out report/      # hand-entered or reviewed claims
trialgeist verify report/certificates/          # re-run every certificate
trialgeist serve                                # local studio on http://127.0.0.1:8765
```

## Verdict tiers

| Tier | Meaning | Evidence shipped |
|---|---|---|
| `PROVEN_IMPOSSIBLE` | No assignment of integer patients, under every supported rounding convention and both the N and N-1 SD formulas, reproduces the printed numbers | runnable certificate, already self-checked before it is reported |
| `STATISTICALLY_EXTREME` | Possible in integers, but p < 1e-5 under the stated randomization model (Carlisle-Stouffer baseline test) | rounding-aware simulation script with the caveats spelled out |
| `REGISTRY_DISCORDANT` | The published primary outcome, time horizon, or sample size differs from the ClinicalTrials.gov version active when enrollment started | registry version numbers, posting dates, history links, and the compared text |
| `CONSISTENT` / `INCONCLUSIVE` | Checked and fine, or not decidable (solver limit, cluster randomization, too few variables) | witness data set where one exists |

LLMs are never in the trust path. Extraction is deterministic (regex plus PDF
geometry, or JATS markup), and every claim keeps page and bbox provenance.

## Engines

1. **Summary statistics** (`engines/grim_grimmer.py`, `sprite_sat.py`, `counts.py`): GRIM,
   GRIMMER (S, Q and parity), Bhatia-Davis SD ceilings on bounded scales, exact SPRITE-SAT
   over value counts (CP-SAT finds witnesses, Z3 proves UNSAT in the certificate),
   n (%) feasibility against any denominator or a confirmed N, count conservation, and
   percentage partitions.
2. **2x2 tables** (`contingency_exact.py`): every table consistent with the printed counts or
   percentages, checked against crude RR/OR/RD and Fisher (exact rational), chi-square and
   Wald p-values with generous floating-point margins. Adjusted estimates are never used as
   evidence.
3. **KM-SAT** (`km_vector_sat.py`, `extract/km_vector.py`): survival staircases are read from
   PDF drawing operators, axes are calibrated from tick labels, and the numbers-at-risk
   table is read. Each visible drop is modelled as a cluster of events with arbitrary
   censoring inside it. CP-SAT then asks whether any integer events and censorings reproduce
   every drop within the plotting tolerance and every printed risk count. If the
   curve-to-row pairing is ambiguous, every row is tried. Rising 1 - S(t) curves are read
   when the page says Kaplan-Meier and nothing suggests cumulative incidence; multi-panel
   pages and broken y-axes are handled per panel.
4. **Raster KM** (`extract/raster.py`): Kaplan-Meier figures embedded as bitmaps are
   digitised: L-frame panel detection, tick labels OCR'd with tesseract and accepted only if
   they lie on an exact linear scale, curves separated by colour (halo and band tints
   removed), plateaus joined only through drawn risers, and curves paired to at-risk rows by
   their legend swatches. Thick lines whose drops are smaller than the line width are read
   from both stroke edges, and a column counts only when the two edges agree. A drop first
   seen after an occlusion gets its own wider time window. When the numbers at risk come from PDF text, a raster finding can
   be proven like a vector one. **When they were read by OCR, the finding is capped at
   `INCONCLUSIVE` and labelled a lead**: OCR digits are never evidence.
5. **Cross-table consistency** (`engines/consistency.py`, `extract/crosstable.py`): baseline
   and outcome tables (JATS cells or PDF geometry grids) are checked for Total-column sums,
   arm totals, pooled means lying inside the arm means, a shared denominator across the
   categories of one variable, and medians inside their IQR or range.
6. **Carlisle-Stouffer** (`carlisle.py`): rounding-aware two-arm mid-p values for continuous
   variables (so identical printed means do not fake under-dispersion), mid-p Fisher for
   binary rows, and stratification and cluster guardrails. It never proves anything.
7. **Registry time machine** (`registry/diff.py`): ClinicalTrials.gov v2 plus the history API.
   It selects the versions active at the actual start and primary-completion dates and diffs
   primary outcomes, outcome swaps, time horizons and sample size against the paper.
6. **Randomisation arithmetic** (`extract/consort.py`): a sentence that states how many people
   were randomised and lists every arm, such as "530 patients were randomly assigned to
   empagliflozin (n = 265) or placebo (n = 265)", "324 were randomized: 162 to dapagliflozin and
   162 to placebo" or "511 were randomised (258 to placebo, 253 to nilvadipine)", must add up.
   - It reads only closed lists (the last arm comes after "or"/"and") with people as the
     subject, and sentences across page breaks are included.
   - It skips anything qualified: "of whom", "respectively", subgroups, eyes, sites, or a list
     that carries on with more counts after a semicolon.
   - Cluster, factorial, crossover, stepped-wedge and re-randomised designs skip the whole
     document.

## Soundness rules

* A printed value names an interval: the hull of half-up, half-even, half-down, truncation
  and cascaded rounding (53.846 -> 53.85 -> 53.9), plus 1e-9 relative float slack.
* SD claims are impossible only if they fail under both the N-1 and N formulas.
* Engines use only necessary conditions when they claim infeasibility.
* A column header N is trusted as a denominator bound only after the column itself confirms
  it: at least 60% of its cells, and every neighbouring cell, must be exactly count/N. This
  keeps rows with another unit of analysis (vessels, events, safety population) from being
  read against the wrong N.
* KM curves are refused when the page mentions adjusted, weighted, competing-risk or
  cumulative-incidence estimates; a rising curve is only read as 1 - KM under Kaplan-Meier
  wording.
* Raster axes need at least three OCR'd tick labels on an exact linear scale (one interior
  misread may be dropped, never an end label, so an axis break is not bridged).

## Acceptance gates (`pytest`)

| Gate | File | Condition |
|---|---|---|
| A0 | `tests/test_*_core.py` | brute-force cross-checks agree; zero false impossibles across rounding and SD conventions |
| G1 | `tests/test_gate1_synthetic.py` | 200 clean simulated RCTs rendered to real PDFs (instrument means, n (%), binary, vector KM plus risk table) and 100 registry cases give 0 false `PROVEN_IMPOSSIBLE` and 0 false registry flags; recall on injected corruptions >= 0.9 |
| G2 | `tests/test_gate2_positive.py` | documented impossibilities in retracted trials are proven, and certificates exit 0 |
| G3 | `tests/test_gate3_negative.py` | 30+ open-access RCTs (PDF and XML) give 0 `PROVEN_IMPOSSIBLE` (four genuine typos found in them are recorded in the manifest) |
| R | `tests/test_raster_gate.py` | synthetic KM figures rasterised at 110-300 dpi, JPEG quality 60-95, falling and rising, 0.6-1.4 pt lines: >= 60% digitised with 0 false proofs; >= 50% of planted risk-table edits caught where the edited curve was read |
| RC | `tests/test_gate_raster_corpus.py` | 25 retraction-checked OA RCTs with bitmap KM figures give 0 `PROVEN_IMPOSSIBLE` |
| C | `tests/test_consort.py` | prose allocation sentences: total vs enumerated arms, N-1/N convention pairs, two-word crossover spellings, page-break sentences read across the seam |

Raster figures need `tesseract` on PATH (`brew install tesseract`, `apt install tesseract-ocr`);
without it they are inventoried and skipped with a note.

Fetch the third-party corpora first with `uv run python scripts/fetch_corpus.py`, then run `uv run pytest -q` (set `TG_GATE1_CLEAN` / `TG_GATE1_CORRUPT` for a quick G1).
