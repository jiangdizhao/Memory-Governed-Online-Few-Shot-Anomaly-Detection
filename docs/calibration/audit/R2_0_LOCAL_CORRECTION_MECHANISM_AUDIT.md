# R2-0 Label-Blind Local Correction Mechanism Audit

## 1. Executive Summary

**Classification: `R2_0_MECHANISM_OBSERVABLE`.** Integrity passed. In the frozen 512-event R1-S4B tape, both C1 and C2 produced correction vectors above `EPS_PATCH = 1e-7` on 472/512 events (92.19%) and changed image scores above `EPS_SCORE = 1e-6` on the same 472/512 events. Score differences were zero through event 35, as required by the query timing; the first later score position was 36.

The corrections also changed many label-free sample rankings. C1 reversed 19,052/130,816 comparable pairs (14.56%); C2 reversed 39,779/130,816 (30.41%). C2 caused larger score shifts in this engineering fixture. These observations establish mechanism activity only; they do not show that either method improves anomaly detection.

The clearest observed limits were 40 events with no eligible anchors in each arm and C1’s broad feature-space far-field observations. Maximum aggregation did not erase any event-level score change when a tolerance-active patch correction was present. Some C2 corrected maxima moved to unchanged patches, but those image scores still changed.

## 2. Immutable Evidence Identity

| Item | Verified value |
|---|---|
| Repository branch | `calibration-stage0-contract-20261029` |
| Repository HEAD at audit start | `606844407dc5c356fafcabd2adddb0dc294bde5e` |
| Frozen R1-S4B source commit | `c86d5b89fa0062b87d80adb324eaa6bc3c94448e` |
| R1 Run ID | `r1s4b-switch-s0-512-20261031-a1` |
| R1 evidence directory | `outputs/calibration/r1_digest_contract_v1_engineering_r1s4b-switch-s0-512-20261031-a1` |
| R1 manifest status | `ENGINEERING_PASS` |
| Stream prefix | `switch`, support seed 0, positions 0–511 |
| Support regime | Four deployed supports plus separate 64-image C64 calibration evidence |

The audit opened the R1 manifest, public sealed-score JSONL, 512 event NPZ shards, the transition tape through a structural-field projection, the R1 execution report’s fixed-bank digest lines, and source files for SHA verification. It did not open `private_feedback.jsonl` or `correction_anchor_events.jsonl`. No images, labels, masks, confirmation data, or accuracy metrics were accessed. No DINO model was loaded and no GPU was used. The R1 directory was used read-only; all R2-0 outputs are in a separate new directory.

## 3. Evidence Integrity Results

| Check | Result |
|---|---:|
| Sealed / finalized positions | 512 / 512; exactly 0–511 |
| Recorded shard hashes verified | 512/512 |
| Recorded array hashes verified | 512/512 |
| C0/C1/C2 reconstructed scores exactly matched | 512/512 events |
| Frozen query positions | 17, unique and in range |
| Query-position SHA-256 | `7f4aedf49f3e3bffd2482bd00d9e16512553b7cef8c0af7e002b47b60ddcb17f` |
| Query-binding SHA-256 | `dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183` |
| Query-verification SHA-256 | `ffbedc32edbd8f6c9b79d0d5b0fe3663d4cfb315d20273815405b84a486cc991` |
| Structured fixed-bank SHA-256 | `8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558` |
| R1 source-file hashes matching current source files | 15/15 |
| Structural transition continuity | Passed; only allow-listed position/query/completion/state fields were decoded |

The structured fixed-bank digest matched the R1 execution report and manifest receipt. Score reconstruction used the frozen R1 float32 update `result[ids] = result[ids] + delta` and the frozen patch-map maximum. No alternate interpolation or aggregation was used.

## 4. Patch-level Correction Activation

Each arm has 512 × 32 = 16,384 selected-patch correction entries.

| Measure | C1 | C2 |
|---|---:|---:|
| Exact float32 nonzero entries | 15,104 / 16,384 (92.19%) | 15,104 / 16,384 (92.19%) |
| Nonzero entries at `EPS_PATCH` | 15,102 / 16,384 (92.18%) | 15,104 / 16,384 (92.19%) |
| Events with a tolerance-active patch correction | 472 / 512 | 472 / 512 |
| Positive / negative exact entries | 9,899 / 5,205 | 9,024 / 6,080 |
| Positive / negative entries at `EPS_PATCH` | 9,899 / 5,203 | 9,024 / 6,080 |
| Mean of per-event signed correction means | +0.008691 | +0.019678 |
| Mean of per-event mean absolute correction | 0.018007 | 0.120508 |
| Median / p90 of per-event mean absolute correction | 0.018863 / 0.027461 | 0.127500 / 0.177500 |
| Maximum per-event maximum absolute correction | 0.048163 | 0.200000 |

Both arms had 40 events with no eligible anchors. There were **zero** events with eligible anchors but an entirely zero correction vector at `EPS_PATCH`. C1 had two exact nonzero entries below the tolerance; C2 had none. Entry counts at other patch tolerances are recorded in `mechanism_summary.json`; the per-event counts are in the CSV.

## 5. Image-level Score Changes

`delta = corrected image score − C0 image score`; score-change classification uses `EPS_SCORE = 1e-6`.

| Measure | C1 | C2 |
|---|---:|---:|
| Events changed at `EPS_SCORE` | 472/512 (92.19%) | 472/512 (92.19%) |
| Positive / negative score changes | 390 / 82 | 418 / 54 |
| Median / p90 absolute score change | 0.020086 / 0.033443 | 0.161313 / 0.200000 |
| Maximum absolute score change | 0.046511 | 0.200000 |
| Tolerance-active patch correction but unchanged image score | 0 | 0 |
| Positions 0–35 changed | 0/36 | 0/36 |
| Positions 36–511 changed | 472/476 (99.16%) | 472/476 (99.16%) |

Position 35’s score was sealed before its queried label could be disclosed. Therefore it is included in the pre-feedback window; event 36 is the first later score that could reflect that feedback. The four unchanged scores after position 35 coincide with events without eligible anchors. At score tolerance `1e-4`, changed-event counts were 457 for C1 and 472 for C2; at `1e-5`, they were 468 and 472.

## 6. Image Max Masking Analysis

`np.argmax`’s deterministic first-index tie rule was preserved. The frozen `max_score` uses the dense-map maximum. Exact ties and near ties are reported separately; a near tie means within `EPS_SCORE` of the maximum.

| Measure | C1 | C2 |
|---|---:|---:|
| Original maximizing patch in Top-32 | 512/512 | 512/512 |
| Corrected maximizing patch in Top-32 | 512/512 | 488/512 |
| Corrected argmax identity changed | 158/512 | 302/512 |
| Events with an exact maximum tie | 0 | 0 |
| Events with a near-maximum tie at `EPS_SCORE` | 0 | 0 |
| Corrected selected maximum wins over unchanged patches | 512 | 488 |
| Unchanged patch wins | 0 | 24 |
| Exact selected/unchanged maximum ties | 0 | 0 |
| Tolerance-active patch correction but unchanged image score | 0 | 0 |

For C2, 24 corrected maxima moved to unchanged patches outside Top-32. The image scores still changed in those events, so this is an aggregation/location effect, not a complete masking of the correction’s image-level effect. The evidence does not support treating each nonzero patch delta as an expected score change in general.

## 7. Ranking Changes and Source-group Dependence

There were 103 unique source-group hashes. Of 130,816 all-event pairs, 1,021 same-source-group pairs were excluded from the cross-group analysis, leaving 129,795 cross-group pairs. All pairs were comparable: neither arm had score ties, and no pairs became newly tied or untied.

| Arm and pair scope | Spearman statistic | Kendall tau-b | Strict reversals / comparable pairs | Reversal fraction |
|---|---:|---:|---:|---:|
| C1 vs C0, all pairs | 0.881503 | 0.708721 | 19,052 / 130,816 | 14.56% |
| C1 vs C0, cross-group pairs | 0.881730* | 0.708525 | 18,916 / 129,795 | 14.57% |
| C2 vs C0, all pairs | 0.546255 | 0.391833 | 39,779 / 130,816 | 30.41% |
| C2 vs C0, cross-group pairs | 0.546533* | 0.391178 | 39,511 / 129,795 | 30.44% |

The all-pair statistic is conventional Spearman rho, computed as Pearson correlation of average ranks. `*` For the filtered cross-group pair scope, the reported Spearman-style statistic is Pearson correlation of global average-rank differences over eligible cross-group pairs; it is explicitly pair-weighted and is not presented as a conventional correlation over 103 group aggregates. Strict reversals never count tied pairs. These rank changes are structural and say nothing about detection quality.

## 8. C1/C2 Anchor Eligibility

| Eligibility measure | C1 eligible slots | C2 eligible anchors |
|---|---:|---:|
| Zero-eligible events | 40/512 | 40/512 |
| Eligible-count minimum | 0 | 0 |
| Median eligible count | 128 | 224 |
| p90 / maximum | 256 / 256 | 256 / 256 |
| Eligible but zero patch correction at `EPS_PATCH` | 0 | 0 |

The allow-listed C1/C2 state hashes changed at all 17 public query positions and at no unqueried position. This establishes correction-state transitions, not the number of anchors newly inserted versus merged or replaced. Exact per-event admissions are `NOT_IDENTIFIABLE_FROM_R1_TAPE` without the label-bearing correction ledger, which this audit did not read.

Under the frozen C2 writer contract (32 selected patch rows per queried event, capacity 256, FIFO), the eighth query at position 268 fills the history; position 269 is the first later score with a full history. Before that score there are 269 events, 229 with tolerance-active patch corrections and score changes. From position 269 onward, all 243 events had active corrections and score changes.

## 9. Feature-space Locality and Far-field Limitations

The R1 far-field diagnostic counts selected-patch observations whose feature distance exceeds three times the nearest anchor radius. It recorded 10,415 observations across the 512 events (median 21.5 and maximum 32 observations per event). The largest recorded per-event far-field absolute correction was 0.047888.

This is a feature-space distance rule, not pixel-space locality. The tape does not expose the correction-anchor contents needed to count exact nonzero far-field correction entries, so that quantity is `NOT_IDENTIFIABLE_FROM_R1_TAPE`. The C1 same-source-excluded slot count had median 0, p90 0, and maximum 32; because native Codebook replacement history can cause conservative overexclusion, the number of falsely excluded slots is also `NOT_IDENTIFIABLE_FROM_R1_TAPE`.

## 10. Mechanism Attribution and Failure Modes

| Funnel stage (512-event denominator) | C1 | C2 |
|---|---:|---:|
| Events scored | 512 | 512 |
| Events with eligible anchors | 472 | 472 |
| Events with nonzero patch corrections at `EPS_PATCH` | 472 | 472 |
| Events with image-score change at `EPS_SCORE` | 472 | 472 |
| Events participating in at least one strict reversal | 511 | 510 |

For both mechanisms, missing eligibility explains the 40 inactive events. Where anchors were eligible, correction vectors were tolerance-active and image-level scores changed. Maximum aggregation masked some corrected C2 argmax locations but did not suppress an image-score change in any tolerance-active event. C1 shows substantial feature-space far-field activity. C2 produces larger score shifts and more pair reversals in this tape, but this is not evidence that C2 is the better detector.

## 11. Implications for R2-1 Scientific Evaluation

The mechanism is observable, so an independently authorized R2-1 evaluation could assess scientific outcomes under a preregistered protocol. It should preserve the current identity and distinguish adapted C1 from a faithful historical method; C2 remains an engineering fixture with unapproved scientific hyperparameters. Freeze outcomes, budgets, class/source-group handling, and evaluation boundaries before any label access. This audit did not begin R2-1 or tune either mechanism.

## 12. Unidentifiable Quantities

- Exact C1 anchor insertion, merge, replacement, and per-event admission counts.
- Exact number of falsely excluded C1 slots due to conservative replacement history.
- Exact count of nonzero far-field patch corrections and any pixel-space locality measure.
- Whether rank reversals improve or worsen detection of defects.
- AUROC, AUPRC, balanced accuracy, FPR, FNR, accuracy, precision, or recall.

The first three are `NOT_IDENTIFIABLE_FROM_R1_TAPE` under the authorized label-blind inputs. The last two are outside this audit’s authorization and purpose.

## 13. Reproducibility and Source Hashes

- Audit implementation: `experiments/calibration/r2_0_label_blind_mechanism_audit.py`.
- Audit implementation SHA-256: `e2c9d9c6d22dba6a2222b84698d712dc70727d3f45d698e017d558b807b9a727`.
- Synthetic tests: `tests/calibration/test_r2_0_label_blind_mechanism.py` (**16 passed**).
- Synthetic test source SHA-256: `b13003255601ac5c57b6003a656345cb53ab0abb72fd5dd246dff0742c9ce1cd`.
- R1 manifest SHA-256: `af2423c89d28d865550127a27173d3c37ddd7b3cd003f26da52d560fd92868e2`.
- R1 execution report SHA-256: `3eda09bcf4b6118fb7bae81e179fb9b45a94f3722aa390c93ae0340170bda5a1`.
- Public sealed-score file SHA-256: `7fd0a920202609d0b242c7de4b8a77128e03b4acdd5f7df7915cbbb2269e3eb9`.
- Structural transition projection SHA-256: `ad77084a13fcf11311eb90a6326371bcf2db53b4d957c6e0a2d4cedfaa9277c1` (allow-listed fields only; no full transition-file hash was calculated).
- R1 manifest source hash inventory: 15/15 files matched current source files.
- Per-shard SHA-256 and per-array SHA-256 receipts for all 512 positions are in `outputs/calibration/r2_0_label_blind_r1s4b_switch_s0_512_a1/integrity_receipt.json`.
- Machine-readable results: `mechanism_summary.json`, `event_mechanism_diagnostics.csv`, `rank_change_summary.json`, and `integrity_receipt.json` in that same directory.

Reproduction command from repository root:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& 'D:\anaconda3\envs\torch\python.exe' -m experiments.calibration.r2_0_label_blind_mechanism_audit
```

CPU-only source and test verification command (result: `16 passed in 0.40s`):

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& 'D:\anaconda3\envs\torch\python.exe' -m py_compile experiments/calibration/r2_0_label_blind_mechanism_audit.py tests/calibration/test_r2_0_label_blind_mechanism.py
& 'D:\anaconda3\envs\torch\python.exe' -m pytest -q tests/calibration/test_r2_0_label_blind_mechanism.py
```

## 14. Final Research Recommendation

Retain the Local Correction research direction. The R1 tape shows that C1 and C2 measurably alter patch scores, image scores, and rankings, while also exposing an eligibility gap and a C1 far-field behavior worth diagnosing. Do not infer performance from these effects. A future R2-1 study requires its own authorization and a frozen, scientifically appropriate evaluation protocol.
