# TRIALGEIST build plan (living checklist)

North star: a local scanner that only ever *accuses* with a runnable proof.
Every `PROVEN_IMPOSSIBLE` ships a standalone certificate (<80 lines, <2 s) that
re-derives the contradiction from the printed digits.

## Acceptance gates

| Gate | Test | Pass condition |
|---|---|---|
| A0 | Exact cores unit tests (`tests/test_*_core.py`) | brute-force cross-checks agree; zero false impossibles under HALF_UP / HALF_EVEN / HALF_DOWN / TRUNCATE and N vs N-1 |
| G1 | Synthetic mutation torture (`tests/test_gate1_synthetic.py`) | 200 clean simulated RCTs (continuous, Likert, binary, KM vector PDF + risk table, registry) -> 0 false `PROVEN_IMPOSSIBLE`; recall on injected, by-construction-impossible corruptions reported (target >= 0.9) |
| G2 | Real retracted positive controls (`tests/test_gate2_positive.py`) | each documented impossibility rediscovered from printed numbers, certificate runs and exits 0 |
| G3 | Clean OA negative controls (`tests/test_gate3_negative.py`) | 20 PMC OA RCT PDFs end to end -> 0 `PROVEN_IMPOSSIBLE` |

## Phases

- [x] P0 repo scaffold (uv, py3.13, z3, ortools, pymupdf)
- [x] P1 typed schemas + verdict tiers + certificate emitter
- [x] P2 exact cores: rounding, GRIM/GRIMMER, Bhatia-Davis, SPRITE-SAT, percentages + sums + partitions, 2x2 contingency, KM-SAT, pooled-mean and median/range consistency
- [x] P3 PDF text/table extraction with bbox provenance; vector KM extraction; PMC JATS XML tables
- [x] P4 ClinicalTrials.gov v2 + history differ
- [x] P5 Carlisle-Stouffer baseline engine (STATISTICALLY_EXTREME only)
- [x] P6 CLI `trialgeist scan|check|verify|serve` + report writer
- [x] P7 Gate 1 synthetic torture
- [x] P8 Gate 2 positive controls: 23 documented impossibilities across 8 retracted/withdrawn papers + 4 conservative refusals; end-to-end rediscovery from PMC XML for 4 papers
- [x] P9 Gate 3 negative controls: 31 OA RCTs (NEJM, Lancet, Lancet Oncol/Neurol, BMJ, Nat Med, PLoS Med, JNNP...), PDF + XML
- [x] P10 `trialgeist serve` local UI (PDF overlay + export certificate)

## Open items

- Cross-table checks (subgroup pooling across Table 2 vs Table 4, exhaustive category sums) are engines but not auto-extracted.
- Raster KM figures are refused (only vector paths are read). Most corpus KM figures are embedded images, so KM-SAT is exercised mainly by Gate 1.
- Registry discordance needs network; Gate 1 covers it with fixtures.

## Soundness rules (decisions log)

1. Rounding: a printed value is mapped to the closed hull of every supported
   convention (half-up either direction, half-even, half-down, truncation
   toward zero), then widened by a relative float slack of 1e-9 so binary
   floating point pipelines (Excel, R, numpy) can never create a false
   impossibility. Per-convention results are reported for explanation only.
2. Variance: impossible only if infeasible under BOTH sample (N-1) and
   population (N) SD.
3. Engines may only use *necessary* conditions (relaxations) when claiming
   infeasibility. Anything that is merely "unusual" is never PROVEN.
4. KM: each visible vertical drop is modelled as a *cluster* of one or more
   event times with arbitrary censoring inside it, so merged ties or
   near-coincident events cannot produce a false proof.
5. Assumptions the proof depends on (denominator = N, scale bounds,
   exhaustive categories, crude/unadjusted estimate) are written into the
   finding and the certificate header. Missing denominators are searched
   (D in [n, N]) before a percentage mismatch is called impossible.
6. LLMs are never in the trust path. Extraction is deterministic regex /
   geometry with page+bbox provenance.

7. Column header N convicts a cell only when every other cell in that column
   (>= 5) is exactly count/N. Two or more misfits mean another denominator is
   in use, so nothing is claimed. Numbers-at-risk rows ("111 (0)") are never
   read as n (%).
8. Gate 3 tolerates a proof on a clean paper only if it is a hand-verified
   real typo recorded in the manifest (`known_true_impossibilities`), e.g.
   BMJ 2021;372:n84 Table 1 "5 (5)" of n=65.
