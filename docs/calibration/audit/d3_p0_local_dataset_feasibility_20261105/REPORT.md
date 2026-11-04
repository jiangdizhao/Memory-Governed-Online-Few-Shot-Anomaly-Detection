# D3-P0 Local Dataset Audit: VisA and MVTec AD 2 (2026-11-05)

**Record type:** research evidence summary; metadata and image-header audit, not a scientific experiment.  
**Reported audit status:** `COMPLETE_METADATA_AND_HEADER_AUDIT`.  
**Audit run:** `d3_dataset_audit_20261105_6c1d3aa4`.  
**Source of this document:** the researcher's 2026-11-05 verbatim summary of the completed Codex D3-P0 audit. The *full local audit artifacts* are not included in this Git commit. Aggregate counts below were arithmetically cross-checked when writing this summary; individual source files, full duplicate hashes, and the local receipts were **not independently re-examined** for this Git document.  
**Code baseline at audit:** branch `calibration-stage0-contract-20261029`, HEAD `46a9cd634015dd190ee1a2910beeedceb378ea30`.  
**Decision:** dataset inventory and arithmetic budget feasibility only. **No D3 GO, no query-rate selection, no confirmation-eligibility certification, no model result.**

This record supplements, and does not revise, the earlier [2026-11-01 P0 MVTec AD 2 audit](../p0_benchmark_integrity_and_exposure_20261101/P0_AUDIT_DECISION.md).

## 1. Scope and audit limitations

The authorized local roots were `G:\dataset\VisA` and `G:\dataset\mvtec_ad_2`. Both were readable. The auditor reported no directory reparse points or enumeration errors. The audit used folder/manifest metadata and image headers, with bounded exact-duplicate hashing. It **did not** run CUDA, a feature extractor, anomaly scoring, a query controller, or Stage B outcome evaluation. It made no dataset or tracked-source changes and no original Git commit.

The audit found 20 categories overall (12 VisA, 8 MVTec AD 2), and a query-budget planning CSV with 320 rows: 20 categories × 2 candidate stream definitions × 8 nominal query rates. The CSV is a *planning resource*, not a query tape or an executed-results table.

## 2. VisA inventory — 1-class split

**Total:** 12 categories, **10,821 image files**, **1,200 masks**. The reported one-class split consists of **8,659 labeled train normals** and **2,162 TEST images** (**962 normal, 1,200 anomalous**). All 1,200 defect masks were reportedly paired with their defect images; no missing or orphan masks were found.

`S1` denotes the official TEST-only event count. `S2` is the **arithmetic-only** candidate `train_normal - 68 + S1` assuming 4 Supports and 64 disjoint C64 normals; it was **NOT RECOMMENDED**, not instantiated, and must not be passed off as the official one-class benchmark.

| Category | Train normal | TEST normal | TEST anomalous | S1 events | S2 arithmetic candidate (not approved) |
|---|---:|---:|---:|---:|---:|
| candle | 900 | 100 | 100 | 200 | 1032 |
| capsules | 542 | 60 | 100 | 160 | 634 |
| cashew | 450 | 50 | 100 | 150 | 532 |
| chewinggum | 453 | 50 | 100 | 150 | 535 |
| fryum | 450 | 50 | 100 | 150 | 532 |
| macaroni1 | 900 | 100 | 100 | 200 | 1032 |
| macaroni2 | 900 | 100 | 100 | 200 | 1032 |
| pcb1 | 904 | 100 | 100 | 200 | 1036 |
| pcb2 | 901 | 100 | 100 | 200 | 1033 |
| pcb3 | 905 | 101 | 100 | 201 | 1038 |
| pcb4 | 904 | 101 | 100 | 201 | 1037 |
| pipe_fryum | 450 | 50 | 100 | 150 | 532 |
| **Total** | **8,659** | **962** | **1,200** | **2,162** | **10,005** |

The VisA dataset documentation notes multiple object instances in some subsets. Image count is not a valid proxy for independent physical objects or feedback groups. **Physical-group independence is unverified.**

## 3. MVTec AD 2 inventory — public/private access distinction

**Total:** 8 categories, **8,004 image files**, **705 public masks**. Public TEST-only S1 consists of **1,084 images: 379 normal and 705 anomalous**. The local private and private-mixed image files exist, but their outcomes are not locally available; they are **not** usable for ordinary local label-feedback simulation.

MVTec AD 2's private-mixed scenes reuse private scenes with lighting changes; their file counts must not be interpreted as independent physical group counts.

`S2` below is again `train_normal - 68 + S1`, purely arithmetic and **NOT RECOMMENDED** as an online evaluation protocol.

| Category | Train normal | Validation normal | Public normal | Public anomalous | S1 events | S2 candidate | Private files | Private-mixed files |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| can | 412 | 46 | 72 | 90 | 162 | 506 | 321 | 321 |
| fabric | 387 | 43 | 66 | 90 | 156 | 475 | 314 | 314 |
| fruit_jelly | 263 | 37 | 20 | 60 | 80 | 275 | 255 | 255 |
| rice | 313 | 35 | 42 | 90 | 132 | 377 | 277 | 277 |
| sheet_metal | 137 | 19 | 24 | 90 | 114 | 183 | 142 | 142 |
| vial | 291 | 41 | 35 | 105 | 140 | 363 | 276 | 276 |
| wallplugs | 293 | 33 | 60 | 90 | 150 | 375 | 232 | 232 |
| walnuts | 432 | 48 | 60 | 90 | 150 | 514 | 228 | 228 |
| **Total** | **2,528** | **302** | **379** | **705** | **1,084** | **3,068** | **2,045** | **2,045** |

All 705 public defect images reportedly paired correctly with masks after correctly handling the nested `ground_truth/bad` layout. No public masks were missing or orphaned.

Official reference: [MVTec AD 2 dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) and [paper](https://doi.org/10.1007/s11263-026-02743-0). This record does not independently verify download origin or license compliance; see the earlier P0 audit for those separate caveats.

## 4. Physical groups, duplicate candidates, and data-access eligibility

- **Regime A (4 normals)** and **Regime B (4 Supports + 64 disjoint commissioning normals)** each pass simple *image-count* checks in every inspected category. However both remain `STRUCTURALLY_UNVERIFIED` for group-disjoint scientific deployment because verified physical sample IDs were not established.
- Exact duplicate detection was **partial**: the auditor capped source-byte reads at **64 MiB per dataset** because of slow source access. Among MVTec AD 2 candidates actually hashed, **39 cross-partition duplicate groups were confirmed**. Their individual split pairs and causes must be checked in the local inventory before claiming train/test leakage; some cross-partition repetitions may be expected (e.g. related private/mixed scenes). The remaining same-size candidates were *not* fully checked.
- No cross-partition exact duplicates were found among the **hashed VisA candidates**, but the partial test does **not** establish that none exist across the full dataset.
- Exact identical-file hashing does not prove physical independence when different images depict the same object.
- No official five-view Source Group structure is established for either dataset. The Real-IAD five-camera query exclusion rule must **not** be transplanted without a dataset-specific grouping contract.
- Supports, C64, and evaluation groups need explicit documented, group-disjoint access rules before any D3 Stage A. Image-folder labels, defect names, masks, and TEST outcomes cannot enter online controller inputs.

## 5. Short-stream feedback-budget feasibility

The audit's illustrative target-rate grid was `[1%,3%,5%,8%,10%,12%,15%,20%]`. For a stream of `T` eligible events, `E[Q] = rho * T` is only an arithmetic target-scale estimate, not guaranteed executed feedback.

| Dataset | S1 events/category | Expected 3% labels/category | Expected 10% labels/category | Can 256-event utility history fill? |
|---|---:|---:|---:|---|
| VisA | 150–201 | 4.50–6.03 | 15.0–20.1 | **No; every S1 stream is under 256** |
| MVTec AD 2 public | 80–162 | 2.40–4.86 | 8.0–16.2 | **No; every S1 stream is under 256** |

A 3% policy yields just a handful of expected labels; feedback may be too sparse to demonstrate online adaptation. A larger target rate is a *possible design response*, not a rate selected or validated by this audit. In particular:

- **Do not impose an identical target rate across datasets** with very different horizons.
- For comparisons within the same dataset/split/draw/order, compare realized query expenditure or report the quality–budget tradeoff explicitly.
- Query-budget forecasts do not account automatically for physical-group eligibility, correlated images, adaptive query-price behavior, ties, or cold-start effects.
- The existing 256-event rolling query-history assumption cannot reach its full window on either official S1 public TEST stream; that limitation must be disclosed or a different **predeclared** short-stream protocol designed.
- No forced initial querying, knowledge of future outcomes, after-the-fact labels, or future-length-aware catch-up is approved.
- The 64 C64 normal commissioning labels, if legitimately available, are **additional predeployment supervision** and separate from online query counts; this is not strict four-shot.

The arithmetic S2 expansion increases event counts, but also converts previously labeled training examples into online events, changes the stream class composition and supervision provenance, and is **NOT RECOMMENDED** by this audit. No S2 stream or query identity manifest was generated.

## 6. Historical outcome exposure and D3 confirmation status

| Dataset | What was found | Appropriate current status |
|---|---|---|
| VisA | No VisA experiment or result artifact found in the searched records; this does **not** prove nonexposure | `CONFIRMATION_ELIGIBLE_PENDING_RESEARCHER_ATTESTATION` **only conditionally**; physical grouping and access protocol remain unverified |
| MVTec family | Previous rejected-manuscript MVTec-AD study used a **10% target query rate** | `PREVIOUSLY_EXPOSED` for the MVTec family at least; do not claim an independent confirmation set merely by renaming/repartitioning |
| MVTec AD 2 exact release | Prior use for method tuning or evaluation not certified from the available records | `CONFIRMATION_STATUS_UNVERIFIED`; at most consider disclosed development use until assessed |

The VisA and MVTec AD 2 inventories do not override older Real-IAD exposure decisions, nor do they authorize access to other protected categories. **No dataset is currently certified as an independent D3 confirmation stream.**

## 7. Compute / provenance / follow-up

The audit used D2 B1 as a *volume reference*: **15,295 event encodings and 321,438,823 output bytes**. A comparable elapsed-time receipt was not available, so no seconds-per-image extrapolation was made. Candidate repetition counts (1×1, 3×2, 5×3) remain illustrative, not selected.

The Codex audit reported:
- `py_compile`: PASS; synthetic tests: **8 passed**; `git diff --check`: PASS.
- 320 planning CSV rows with size/hash receipts checked **locally by Codex**.
- No GPU, CUDA, model, anomaly inference, controller, or scientific metric calculation.
- No changes to original data, masks, old runs, or tracked source.
- Two superseded audit-attempt directories were preserved; **only the `6c1d3aa4` directory is the final result**.

**Final local audit artifacts (outside tracked Git):**

```text
outputs/calibration/d3_dataset_audit_20261105_6c1d3aa4/
  dataset_audit.py
  test_dataset_audit.py
  dataset_inventory.json
  query_budget_feasibility.csv
  D3_dataset_audit_report.md
  audit_provenance.json
```

The full local artifacts and hashes are the source of truth for granular split-pair duplicate records, inventory warnings, and the 320 budget rows. **Only this curated summary is version-controlled here.** When those receipts become available, verify them directly rather than treating the prose as an independent full-file audit.

**Unresolved before any D3 approval:** (i) researcher attestation of VisA nonexposure; (ii) group-level Support/C64/TEST independence; (iii) complete or explicitly bounded duplicate assessment, particularly MVTec AD 2 cross-partition repeats; (iv) legitimate public TEST-only short-stream supervision design and per-dataset target-query budget; (v) separate resource and statistical precision plan; (vi) frozen model/preprocessing and label firewalls.

**Current research priority:** complete the supervisor's Recovery E1/E2/E3 evidence and Decision Memos. D3 remains **NOT AUTHORIZED**. This documentation update is not a modification of E123 or of any frozen E6/D2 scientific artifact.
