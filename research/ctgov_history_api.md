# ClinicalTrials.gov Study & Record History API Specification

Empirical specification for TRIALGEIST Engine 4 (protocol-version diffing). All endpoints, headers, status codes, and response shapes below were verified live against `clinicaltrials.gov` on 2026-11-26 using `httpx` (Python 3.13) and Chrome DevTools network capture.

---

## 1. Confirmed Endpoints

### 1.1 Current Study Record (Public v2 API — Documented)

| Property | Value |
|---|---|
| URL template | `GET https://clinicaltrials.gov/api/v2/studies/{nctId}` |
| Query params | `format` (`json` default \| `csv` \| `json.zip` \| `fhir.json` \| `ris`), `markupFormat` (`markdown` default \| `legacy`), `fields` (comma- or pipe-separated area/module/field names, e.g. `NCTId,BriefTitle,PrimaryOutcome`) |
| HTTP statuses | `200` OK; `301` redirect if `nctId` is an `NCTIdAlias`; `400` malformed `nctId` (must match `^[Nn][Cc][Tt]0*[1-9]\d{0,7}$`) or unknown param; `404` unknown study |
| Auth / Cookie | None required |
| Top-level JSON keys | `protocolSection`, `resultsSection` *(only if results posted)*, `documentSection` *(optional)*, `derivedSection`, `hasResults` (`bool`) |

Also useful for lightweight cookie bootstrap & freshness checks:
- `GET https://clinicaltrials.gov/api/v2/version` -> `200` `{"apiVersion": "2.0.5", "dataTimestamp": "2026-11-25T10:00:05"}` (sets `ncbi_sid` cookie).

---

### 1.2 Study History Index (Internal SPA API — Undocumented)

| Property | Value |
|---|---|
| URL template | `GET https://clinicaltrials.gov/api/int/studies/{nctId}/history` |
| Query params | None |
| HTTP statuses | `200` OK; `400` malformed `nctId`; `403` missing `ncbi_sid` cookie (edge WAF); `404` study not found |
| Auth / Cookie | **Requires `Cookie: ncbi_sid=...`** (see Section 2) |
| Top-level JSON keys | `changes` (`list[dict]`), `originalData` (`dict`), `lastUpdateVersions` (`dict`), `outcomesUpdateCount` (`int`) |

**Versions list location:** `changes[]` (ordered chronologically from version `0` to `N-1`).

**Per-version fields in `changes[i]`:**
- `version` (`int`, **0-indexed**): `0` is the first posted version (shown as Version `1` / `?tab=history&a=1` in the web UI).
- `date` (`str`, `YYYY-MM-DD`): Date this version was posted on ClinicalTrials.gov.
- `lastUpdateSubmitQcDate` (`str`, `YYYY-MM-DD`): QC-meeting submission date (often equals `date`).
- `status` (`str`): `overallStatus` enum at this version (e.g. `NOT_YET_RECRUITING`, `RECRUITING`, `COMPLETED`).
- `studyType` (`str`): `INTERVENTIONAL` \| `OBSERVATIONAL` \| `EXPANDED_ACCESS`.
- `moduleLabels` (`list[str]`): Human-readable modules changed relative to `version - 1` (empty `[]` on `version: 0`; e.g. `["Study Status", "Outcome Measures", "Study Design"]`).
- `unpostedEvents` (`list[dict]`, *optional*): Present only on versions with QC-returned/unposted events, e.g. `[{"type": "RELEASE", "date": "2020-09-16"}, {"type": "UNRELEASE", "date": "2020-09-17"}]`.

**Other top-level summary fields:**
- `originalData`: Snapshot summary of earliest posted `primaryOutcomes`, `secondaryOutcomes`, `enrollmentInfo`, and sponsor/org identity flags (`orgFullNameSame`, `responsiblePartySame`, `leadSponsorSame`). On older pre-2005 trials, `primaryOutcomes`/`enrollmentInfo` may be omitted if absent at `v0`.
- `lastUpdateVersions`: 0-indexed version numbers of the last change per category, e.g. `{"outcomes": 24, "primaryOutcomes": 24, "secondaryOutcomes": 20, "otherOutcomes": 7, "enrollmentInfo": 18}`.
- `outcomesUpdateCount`: Integer count of historical versions that modified outcome measures.

---

### 1.3 Historical Version Snapshot & Server-Side Diff (Internal SPA API — Undocumented)

| Property | Value |
|---|---|
| URL template | `GET https://clinicaltrials.gov/api/int/studies/{nctId}/history/{version}` |
| Query params | Optional `patchToVersion={otherVersion}` (0-indexed int). **Rejects** v2 params (`fields`, `markupFormat`) with `400` (`"<param> is unknown parameter"`). |
| HTTP statuses | `200` OK; `400` malformed `nctId`, `version < 0`, or unknown query param; `403` missing `ncbi_sid` cookie; `404` study or version out of range (`"Study NCT... has no version: 999"`) |
| Auth / Cookie | **Requires `Cookie: ncbi_sid=...`** |
| Top-level JSON keys (snapshot) | `studyVersion` (`int`, 0-indexed), `study` (`dict`) |
| Top-level JSON keys (with `patchToVersion`) | `studyVersion` (`int`), `study` (`dict`), `patchVersion` (`int`), `patch` (`list[dict]`, RFC 6902 JSON Patch ops: `{"op": "replace"\|"add"\|"remove", "path": "/protocolSection/...", "value": ...}`) |

Inside `study`: keys are `protocolSection`, `hasResults`, and (only on versions where present) `resultsSection` and `documentSection`. `derivedSection` is **not** included in historical snapshots.

---

### 1.4 Combined Study + History Endpoint (Internal SPA API — Undocumented)

| Property | Value |
|---|---|
| URL template | `GET https://clinicaltrials.gov/api/int/studies/{nctId}?history=true` |
| HTTP statuses | `200` OK; `403` missing `ncbi_sid` cookie |
| Top-level JSON keys | `study` (current record, serialized with internal HTML encoding), `topics` (MedlinePlus/DailyMed links), `history` (byte-identical object to `/api/int/studies/{nctId}/history`) |

For Engine 4, prefer `/api/v2/studies/{nctId}` + `/api/int/studies/{nctId}/history` + `/api/int/studies/{nctId}/history/{version}` so the current-record fetch uses the documented public v2 contract and bootstraps `ncbi_sid` automatically.

---

## 2. Cookies, Headers, Rate Limits & Caching

- **WAF / Cookie gate on `/api/int/*`:** Requests to `/api/int/*` without an `ncbi_sid` cookie are rejected at the Google edge proxy with `403 Forbidden` (134-byte HTML body, no `Server: istio-envoy` header).
  - Verified empirically: `Cookie: foo=bar` -> `403`; `Cookie: ctg-app=...` -> `403`; `Cookie: ncbi_sid=x` -> `200`; `Cookie: ncbi_sid=` -> `200`.
  - **Recommended pattern in `httpx`:** Reuse a single `httpx.Client` instance and call `GET /api/v2/studies/{nctId}` (or `GET /api/v2/version`) first. That response emits `Set-Cookie: ncbi_sid=<token>; Max-Age=31536000; Domain=clinicaltrials.gov; Path=/`, which `httpx.Client` stores and sends automatically on all subsequent `/api/int/*` requests.
- **`User-Agent`:** Not gated on either `/api/v2/*` or `/api/int/*` (default `python-httpx`, custom UA, and empty UA all return `200` once `ncbi_sid` is present). Set a descriptive `User-Agent` (e.g. `trialgeist/0.1`) as good citizenship.
- **Compression & Caching:** Both APIs support `gzip` (`Vary: Accept-Encoding`) and emit `etag` headers (e.g. `"0/0.34.1/mwvycwaw-gzip"`). Passing `If-None-Match: <etag>` returns `304 Not Modified`.
- **Rate limits:** Neither `/api/v2/*` nor `/api/int/*` exposes `X-RateLimit-*` headers. Official CT.gov guidance recommends keeping automated traffic polite (~50 req/min / <= 3 req/s per IP) with exponential backoff on `429`/`5xx`.

---

## 3. Target Field JSON Paths & Schema Comparison

Within a version snapshot (`/history/{v}`), fields live under `study.protocolSection.*`. Within `/api/v2/studies/{nctId}`, the same fields live under `protocolSection.*`.

| Engine 4 Field | JSON Path (relative to `protocolSection`) | Value Shape & Notes |
|---|---|---|
| Primary outcomes | `outcomesModule.primaryOutcomes[]` | List of `{"measure": str, "timeFrame": str, "description"?: str}` |
| Secondary outcomes | `outcomesModule.secondaryOutcomes[]` | List of `{"measure": str, "timeFrame": str, "description"?: str}` |
| Other outcomes | `outcomesModule.otherOutcomes[]` | Optional list with same item shape (seen e.g. in `NCT04381936` v7) |
| Enrollment count & type | `designModule.enrollmentInfo` | `{"count": int, "type"?: "ESTIMATED" \| "ACTUAL"}` (`type` may be absent on pre-2017 legacy records) |
| Start date | `statusModule.startDateStruct` | `{"date": "YYYY-MM-DD" \| "YYYY-MM", "type"?: "ESTIMATED" \| "ACTUAL"}` |
| Primary completion date | `statusModule.primaryCompletionDateStruct` | `{"date": "YYYY-MM-DD" \| "YYYY-MM", "type"?: "ESTIMATED" \| "ACTUAL"}` |
| Completion date | `statusModule.completionDateStruct` | `{"date": "YYYY-MM-DD" \| "YYYY-MM", "type"?: "ESTIMATED" \| "ACTUAL"}` |
| Overall status | `statusModule.overallStatus` | Enum string (`NOT_YET_RECRUITING`, `RECRUITING`, `ACTIVE_NOT_RECRUITING`, `COMPLETED`, `TERMINATED`, `WITHDRAWN`, `SUSPENDED`, `UNKNOWN`, etc.) |

### Schema comparison: `/history/{v}` vs `/api/v2/studies/{nctId}`

`study.protocolSection` in `/history/{v}` follows the **exact same v2 module and field schema** as `protocolSection` in `/api/v2/studies/{nctId}`, with **three critical serialization differences**:

1. **Envelope & `derivedSection`:** `/history/{v}` wraps the record in `{"studyVersion": N, "study": {...}}` and omits `derivedSection`.
2. **HTML entity escaping in `/api/int/*`:** All `/api/int/*` endpoints (including `/history` `originalData` and `/history/{v}`) HTML-escape characters inside JSON strings (`&#x2F;` for `/`, `&#x27;` for `'`, `&lt;`, `&gt;`, `&amp;`, `&quot;`). Example in `NCT04280705` `v24`: `"activities and&#x2F;or requiring"` vs v2 `"activities and/or requiring"`. Always run `html.unescape()` on strings read from `/api/int/*`.
3. **HTML markup vs Markdown in rich-text fields:** `/api/int/*` serializes `markup`-typed fields (`detailedDescription`, `eligibilityCriteria`, `briefSummary`) as HTML (`<p>...</p><ol><li>...</li></ol>`) and preserves single `\n` newlines inside some descriptions, whereas `/api/v2/*` returns CommonMark (`markupFormat=markdown`). When diffing two historical versions, diff `/history/{vA}` against `/history/{vB}` (identical serializer) rather than mixing `/history/{vA}` with `/api/v2/studies/{nctId}`.

---

## 4. Probed Trials Summary & Implementation Pitfalls

| Trial | Versions | History Span | Authoritative Start / Primary Completion | Key Edge Case Demonstrated |
|---|---|---|---|---|
| `NCT04280705` (ACTT-1) | 25 (`v0`–`v24`) | `2020-02-20` .. `2022-03-09` | `2020-02-21` / `2020-05-21` | Primary outcome swap (`v14`), enrollment escalations, duplicate posting date (`v10` & `v11` on `2020-03-24`), `unpostedEvents` (`v19`) |
| `NCT04381936` (RECOVERY) | 18 (`v0`–`v17`) | `2020-05-07` .. `2026-04-15` | `2020-03-19` / `2028-09-30` (est.) | Retrospective registration (`v0` posted 49 days *after* `startDate`); `otherOutcomes` present |
| `NCT02008227` (OAK) | 93 (`v0`–`v92`) | `2013-12-06` .. `2019-12-06` | `2014-03-11` / `2016-07-07` | High-version modern oncology trial; `v0` had `YYYY-MM` start date (`2014-01`) without `type` |
| `NCT00045162` (S0124) | 95 (`v0`–`v94`) | `2005-06-23` .. `2018-07-03` | `2002-11` / `2008-03` | Pre-2005 archive cutoff (`v0` dated `2005-06-23`); `YYYY-MM` partial dates; `v0` omits `outcomesModule`, `startDateStruct`, `primaryCompletionDateStruct`, `enrollmentInfo` |
| `NCT00001372` | 406 (`v0`–`v405`) | `2005-06-23` .. `2026-10-28` | `1994-02-10` / `None` | Very high version count; pre-2005 archive cutoff; `v0` has `enrollmentInfo: {"count": 9999}` with no `type` and no `outcomesModule` |

### Pitfalls for Engine 4

1. **0-indexed API vs 1-indexed web UI:** API `version: 0` corresponds to UI Version `1` (`?tab=history&a=1`). Store and log 0-indexed `version` explicitly (or label UI version as `version + 1`).
2. **Use authoritative dates from the latest record to pick historical versions:** On `2020-05-21`, `NCT04280705` `v17` still listed `primaryCompletionDateStruct: {"date": "2023-04-01", "type": "ESTIMATED"}`; the actual `primaryCompletionDate` (`2020-05-21`) was only entered retrospectively in `v18` (`2020-07-07`). Always read the trial's actual `startDate` and `primaryCompletionDate` from the current record first, then select the active version from `changes[]`.
3. **Same-day multiple versions:** Multiple versions can share the same `date` (e.g. `NCT04280705` `v10` and `v11` both have `date: "2020-03-24"`). Select the **maximum `version`** among entries with `date <= target_date`.
4. **Partial `YYYY-MM` dates:** Normalize `YYYY-MM` consistently when comparing against `YYYY-MM-DD` version dates (e.g. end-of-month inclusive bound when finding versions active during that month, and flag precision as `MONTH`).
5. **No version active on `startDate`:** Occurs on both retrospectively registered trials (`NCT04381936`) and pre-archive trials started before `2005-06-23` (`NCT00045162`, `NCT00001372`). Fall back explicitly to `v0` with a `retrospective_registration=True` / `pre_archive_cutoff=True` flag rather than silently returning `None`.
6. **Sparse legacy snapshots:** Never assume `outcomesModule`, `primaryOutcomes`, `secondaryOutcomes`, `enrollmentInfo`, `enrollmentInfo.type`, or date structs exist in old `v0` payloads (`NCT00045162` `v0` lacks all of them).
7. **HTML entities:** Always `html.unescape()` strings from `/api/int/*` before text comparison or LLM/NLP diffing.
8. **Alias redirects & undocumented stability:** Pass `follow_redirects=True` (or resolve canonical `nctId` from `/api/v2/studies/{nctId}` first) to handle `NCTIdAlias` 301s. Because `/api/int/*` is internal to the CT.gov web app, guard against contract drift with schema assertions and fixture tests.

---

## 5. Worked Example: `NCT04280705` (ACTT-1 Remdesivir)

Authoritative dates from `/api/v2/studies/NCT04280705`:
- `startDateStruct`: `{"date": "2020-02-21", "type": "ACTUAL"}` -> Active version (`max version` with `date <= 2020-02-21`): **`v1`** (`2020-02-21`; UI Version 2).
- `primaryCompletionDateStruct`: `{"date": "2020-05-21", "type": "ACTUAL"}` -> Active version (`max version` with `date <= 2020-05-21`): **`v17`** (`2020-05-06`; UI Version 18).
- First posted version: **`v0`** (`2020-02-20`; UI Version 1).
- Latest version: **`v24`** (`2022-03-09`; UI Version 25).

### Version Progression Summary

| Version (UI #) | Posted Date | Status | Enrollment | Primary Outcomes | Secondary Outcomes |
|---|---|---|---|---|---|
| `v0` (#1) — *First posted* | `2020-02-20` | `NOT_YET_RECRUITING` | `394` (`ESTIMATED`) | 1 (7-point ordinal scale, Day 15) | 24 |
| `v1` (#2) — *Active at `startDate`* | `2020-02-21` | `RECRUITING` | `394` (`ESTIMATED`) | 1 (7-point ordinal scale, Day 15) | 24 |
| `v9` (#10) — *Scale expanded* | `2020-03-20` | `RECRUITING` | `440` (`ESTIMATED`) | 1 (8-point ordinal scale, Day 15) | 30 |
| `v14` (#15) — *Primary outcome swap* | `2020-04-16` | `RECRUITING` | `572` (`ESTIMATED`) | 1 (Time to recovery, Day 1–29) | 28 |
| `v17` (#18) — *Active at `primaryCompletionDate`* | `2020-05-06` | `RECRUITING` | `800` (`ESTIMATED`) | 1 (Time to recovery, Day 1–29) | 28 |
| `v18` (#19) — *Completion recorded* | `2020-07-07` | `COMPLETED` | `1062` (`ACTUAL`) | 1 (Time to recovery, Day 1–29) | 28 |
| `v24` (#25) — *Latest* | `2022-03-09` | `COMPLETED` | `1062` (`ACTUAL`) | 4 (Time to Recovery + Race/Ethnicity/Sex) | 39 |

### Verbatim Primary Outcome Comparison: First Posted (`v0`) vs Active on Primary Completion Date (`v17`)

**First posted version — `v0` (`2020-02-20`)** *(identical in `v1`, active at `startDate` `2020-02-21`)*:
```json
{
  "measure": "Percentage of subjects reporting each severity rating on the 7-point ordinal scale",
  "description": "The ordinal scale is an assessment of the clinical status at the first assessment of a given study day.\nThe scale is as follows: 1) Death; 2) Hospitalized, on invasive mechanical ventilation or extracorporeal membrane oxygenation (ECMO); 3) Hospitalized, on non-invasive ventilation or high flow oxygen devices; 4) Hospitalized, requiring supplemental oxygen; 5) Hospitalized, not requiring supplemental oxygen; 6) Not hospitalized, limitation on activities; 7) Not hospitalized, no limitations on activities.",
  "timeFrame": "Day 15"
}
```

**Version active on `primaryCompletionDate` (`2020-05-21`) — `v17` (`2020-05-06`)** *(first introduced in `v14` on `2020-04-16`; note raw API escapes `/` as `&#x2F;` in `and&#x2F;or`)*:
```json
{
  "measure": "Time to recovery",
  "description": "Day of recovery is defined as the first day on which the subject satisfies one of the following three categories from the ordinal scale: 1) Hospitalized, not requiring supplemental oxygen - no longer requires ongoing medical care; 2) Not hospitalized, limitation on activities and&#x2F;or requiring home oxygen; 3) Not hospitalized, no limitations on activities.",
  "timeFrame": "Day 1 through Day 29"
}
```

**Forensic diff detected by Engine 4 (`v0`/`v1` -> `v17`):**
1. **Primary outcome swapped & demoted:** Replaced fixed-day ordinal severity distribution (`Day 15`, 7-point -> 8-point) with time-to-event `"Time to recovery"` (`Day 1 through Day 29`) mid-enrollment on `2020-04-16` (`v14`). The prior primary outcome (`"Percentage of subjects reporting each severity rating on an 8-point ordinal scale"`, `timeFrame: "Day 15"`) appears verbatim in `v14` and `v17` `outcomesModule.secondaryOutcomes[]`.
2. **Follow-up horizon changed:** `Day 15` -> `Day 1 through Day 29`.
3. **Sample size escalated:** `394` (`ESTIMATED`, `v0`/`v1`) -> `440` (`v9`) -> `572` (`v14`) -> `800` (`ESTIMATED`, `v17`) -> `1062` (`ACTUAL`, `v18`–`v24`).
4. **Post-completion primary outcome expansion:** `v24` (`2022-03-09`) added three subgroup primary outcomes (`Time to Recovery by Race`, `by Ethnicity`, `by Sex`) after results submission.

---

## 6. Saved Test Fixtures

All fixtures live in `/Users/colin/src/trialgeist/tests/fixtures/ctgov/` as unmodified compacted JSON responses (< 215 KB each):

- `NCT04280705_current.json` — `/api/v2/studies/NCT04280705`
- `NCT04280705_history.json` — `/api/int/studies/NCT04280705/history`
- `NCT04280705_v0.json` — `/api/int/studies/NCT04280705/history/0` (first posted)
- `NCT04280705_v1.json` — `/api/int/studies/NCT04280705/history/1` (active at `startDate`)
- `NCT04280705_v14.json` — `/api/int/studies/NCT04280705/history/14` (mid-trial primary outcome swap)
- `NCT04280705_v17.json` — `/api/int/studies/NCT04280705/history/17` (active at `primaryCompletionDate`)
- `NCT04280705_v24.json` — `/api/int/studies/NCT04280705/history/24` (latest)
- `NCT04381936_history.json` — `/api/int/studies/NCT04381936/history` (retrospective registration)
- `NCT02008227_history.json` — `/api/int/studies/NCT02008227/history` (93-version modern trial)
- `NCT00045162_history.json` — `/api/int/studies/NCT00045162/history` (95-version pre-2005 trial)
- `NCT00045162_v0.json` — `/api/int/studies/NCT00045162/history/0` (sparse 2005 archive snapshot)
