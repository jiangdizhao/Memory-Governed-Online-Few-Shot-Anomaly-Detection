# R2-PERF-1 — Switch 512-Event Development Evaluation

## Execution status

Completed the authorized GPU-free, outcome-aware development evaluation on the existing R1-S4B switch / support-seed-0 / 512-event tape. The result is conditional on this historically exposed trajectory and is not independent confirmation.

| Item | Result |
|---|---|
| Branch at launch | `calibration-stage0-contract-20261029` |
| HEAD at launch | `31e20db54eef22b12232915e7f147497914f1362` |
| Run ID | `20261101T025600_70c8a20b` |
| Output | `outputs/calibration/r2_perf1_switch_s0_512_20261101T025600_70c8a20b` |
| Interpreter | `D:\anaconda3\envs\torch\python.exe` |
| GPU / DINO / image decoding | Not used |
| Event tape | 512 events; 103 verified Source Groups |
| Frozen query tape | 17 query positions; query-position SHA-256 `7f4aedf49f3e3bffd2482bd00d9e16512553b7cef8c0af7e002b47b60ddcb17f` |
| App output bound | 2 GiB, pre-write accounting, with 64 KiB failure reserve |
| Output written | 1,001,586 bytes |
| Free space before launch | 299,132,694,528 bytes; 10 GiB minimum required |
| Supervised wall time | 31.421 s of the 7,200 s limit; Windows Job Object reported exit confirmed and zero active descendants |
| Stage B evaluator time | 13.906 s wall, 13.766 s process CPU |

`run_manifest.json` records the immutable launch identity and is marked `INCOMPLETE` as the initial manifest. Completion is recorded separately in `orchestrator_complete.json`, `stage_a_sealed.json`, `stage_b_report.json`, and `supervisor_status.json`. No incomplete run was resumed.

## Evidence and causal checks

The exact authorized R1 directory was used; no other category or private all-category receipt was opened. R2-0 verified all 512 event shards, the sealed C0/C1/C2 scores, and the structural transition tape before replay. The verified R1 receipt contains 512 events, 17 queries, 103 Source Groups, and fixed-support digest `8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558`.

Stage A is a separate process with no TEST JSON path or outcome parser. It reconstructed C2 from the 17 query disclosures, each event's native Top-32 features, the official Source Group digest, and the original 32-anchor queried writer. At each query position it flushed the candidate score row before reading the next private-feedback and correction-ledger records. It did not read unqueried feedback. All 512 reconstructed C2 delta vectors and scores matched exactly:

| C2 parity check | Result |
|---|---:|
| Maximum patch-delta absolute error | 0.0 |
| Maximum image-score absolute error | 0.0 |
| Pre-update state digest mismatches | 0 |
| Post-update state digest mismatches | 0 |
| Final FIFO anchors retained | 256 |
| Final C2 state SHA-256 | `44064de2bdfb1a06445a8e4030a1a4c4365015ad89fc8bca28ed705d1549b72a` |
| Queried labels | 10 anomalous, 7 normal |

Only after the Stage A score file was sealed and its SHA-256 verified did a separate Stage B process open the local `switch.json`. The projection matched all 512 first TEST identities and Source Groups to the sealed event positions before decoding each corresponding `anomaly_class`. TRAIN labels, later TEST labels, and mask fields were not materialized. No image or mask files were opened. A full-file JSON digest was not recomputed because that would hash TRAIN and non-target TEST label bytes; the R1 manifest's historical `class_json_sha256` is `b863b819fe7734a9c83acc26b25d419c8d8897a9ae93d5b8fd9d0e424ecf42d8`, but this evaluation does not claim a fresh whole-file source-hash verification.

Stage B found 153 normal and 359 anomalous images (70.117% defect prevalence) across the 103 groups. It wrote only aggregate outcomes and metrics; the raw outcome vector was not written.

## Primary results

The fixed operating threshold remained the R1 C64 `theta0_higher_q90`, `0.5165398120880127`. It was not recalibrated on these 512 outcomes. AUROC/AUPRC are ranking metrics; FPR/FNR/Balanced Accuracy use that frozen threshold.

| Arm | AUROC (95% Source Group cluster interval) | AUPRC | Δ AUROC vs C0 (95% paired cluster interval) | FPR | FNR | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.8070 (0.7760, 0.8376) | 0.9152 | 0.0000 | 0.0261 | 0.6017 | 0.6861 |
| C1 | 0.8431 (0.8102, 0.8746) | 0.9322 | +0.0360 (0.0063, 0.0694) | 0.0654 | 0.4318 | 0.7514 |
| C2 | 0.7522 (0.6994, 0.8062) | 0.8929 | −0.0548 (−0.1118, 0.0076) | 0.7255 | 0.1226 | 0.5760 |
| C2-Gated, R=.10, λ=.10 | 0.7123 (0.6623, 0.7630) | 0.8081 | −0.0947 (−0.1437, −0.0458) | 0.2288 | 0.4735 | 0.6489 |
| C3, R=.10, λ=.10 | 0.7123 (0.6623, 0.7630) | 0.8081 | −0.0947 (−0.1437, −0.0458) | 0.2288 | 0.4735 | 0.6489 |

C3 minus C2-Gated AUROC was `0.0000`, with a paired cluster interval of `(0.0000, 0.0000)`. Their 512 primary image-score vectors were exactly equal (maximum absolute score difference `0.0`). This says there was no image-level distinction at the primary setting; it does not establish equality of every intermediate patch correction. C1 is the adapted native R1 Codebook system-level reference, not a pure kernel ablation.

Intervals use 1,000 paired Source Group cluster bootstrap replicates with seed `20261101`. They are conditional on the fixed historical stream and do not describe variation across independently regenerated online trajectories. Pair-level rank diagnostics below are descriptive and are not treated as independent samples.

## Full predeclared sensitivity grid

Every declared `(R, λ)` cell is reported for both methods. Primary `(0.10, 0.10)` is included in the grid and also reported separately above. No configuration was selected using outcomes.

| Method | R | λ | AUROC | AUPRC | Δ AUROC vs C0 | FPR | FNR | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C2-Gated | .02 | .10 | 0.8070 | 0.9152 | 0.0000 | 0.0261 | 0.6017 | 0.6861 |
| C3 | .02 | .10 | 0.8070 | 0.9152 | 0.0000 | 0.0261 | 0.6017 | 0.6861 |
| C2-Gated | .02 | 1.00 | 0.8070 | 0.9152 | 0.0000 | 0.0261 | 0.6017 | 0.6861 |
| C3 | .02 | 1.00 | 0.8070 | 0.9152 | 0.0000 | 0.0261 | 0.6017 | 0.6861 |
| C2-Gated | .05 | .10 | 0.7915 | 0.8987 | −0.0155 | 0.0458 | 0.5933 | 0.6805 |
| C3 | .05 | .10 | 0.7915 | 0.8987 | −0.0155 | 0.0458 | 0.5933 | 0.6805 |
| C2-Gated | .05 | 1.00 | 0.8044 | 0.9142 | −0.0026 | 0.0261 | 0.6017 | 0.6861 |
| C3 | .05 | 1.00 | 0.8044 | 0.9142 | −0.0026 | 0.0261 | 0.6017 | 0.6861 |
| C2-Gated | .10 | .10 | 0.7123 | 0.8081 | −0.0947 | 0.2288 | 0.4735 | 0.6489 |
| C3 | .10 | .10 | 0.7123 | 0.8081 | −0.0947 | 0.2288 | 0.4735 | 0.6489 |
| C2-Gated | .10 | 1.00 | 0.7842 | 0.9017 | −0.0228 | 0.0654 | 0.5905 | 0.6721 |
| C3 | .10 | 1.00 | 0.7842 | 0.9017 | −0.0228 | 0.0654 | 0.5905 | 0.6721 |
| C2-Gated | .20 | .10 | 0.6746 | 0.7756 | −0.1324 | 0.3791 | 0.3092 | 0.6559 |
| C3 | .20 | .10 | 0.6746 | 0.7756 | −0.1324 | 0.3791 | 0.3092 | 0.6559 |
| C2-Gated | .20 | 1.00 | 0.7554 | 0.8725 | −0.0516 | 0.2222 | 0.4540 | 0.6619 |
| C3 | .20 | 1.00 | 0.7554 | 0.8725 | −0.0516 | 0.2222 | 0.4540 | 0.6619 |
| C2-Gated | .40 | .10 | 0.6912 | 0.8070 | −0.1159 | 0.4444 | 0.2368 | 0.6594 |
| C3 | .40 | .10 | 0.6916 | 0.8080 | −0.1155 | 0.4444 | 0.2368 | 0.6594 |
| C2-Gated | .40 | 1.00 | 0.7388 | 0.8297 | −0.0682 | 0.2941 | 0.3259 | 0.6900 |
| C3 | .40 | 1.00 | 0.7393 | 0.8302 | −0.0678 | 0.2941 | 0.3259 | 0.6900 |

The full-precision values and cluster intervals for every grid cell are in `stage_b_sensitivity.json`. At `R=.40`, all-local C3 was slightly above the top-five C2-Gated AUROC (by 0.0004 at λ=.10 and 0.0005 at λ=1.00); both remained below C0. At all smaller radii, the two arms had identical image scores in this run.

## Mechanism diagnostics at the primary setting

| Diagnostic | C2-Gated and C3 result |
|---|---:|
| Events with a strictly past, source-disjoint anchor pool | 472 / 512 |
| Events with at least one local anchor among the 32 selected patches | 240 / 512 |
| Events with no local anchor | 272 / 512 |
| Positive / negative local-anchor observations across selected patches | 418 / 432 |
| Events with local positive/negative conflict | 11 |
| Selected patches with both signs locally available | 14 / 16,384 |
| Nonzero corrected selected-patch fraction | 4.83% |
| Image scores changed from C0 | 104 / 512 (20.31%) |
| Changed image argmax patch | 103 / 512 |
| Nonzero far-field corrections | 0 |
| Absolute patch-delta median / p90 / p99 / maximum | 0 / 0 / 0.1607 / 0.1897 |
| Nearest eligible-anchor distance median | 0.3512 |
| Nearest local-anchor distance median | 0.0777 |
| Effective local-neighbor count median / p99 / maximum | 0 / 1 / 1.9984 |
| Same-source anchors excluded | Up to 32 at an event; no same-source anchor entered the eligible set |

Across the 512 offline outcomes, primary C3 moved 70 scores in the correct anomaly-score direction, 34 in the harmful direction, and left 408 image scores unchanged. Mean score change was `+0.0142` for anomalies and `+0.0232` for normal images. Relative to C0, 22,072 of 129,795 strictly comparable cross-Source-Group pairs reversed order (17.01%); the 1,021 within-group pairs were excluded from that descriptive diagnostic.

The fixed primary setting therefore showed strong spatial restraint but weak average ranking performance. Local anchor availability was limited at the selected-patch level, while sign conflict was uncommon. C3 did not separate from the strong gated baseline at the primary image-score level. These observations do not identify a single causal bottleneck and do not falsify Local Correction generally.

## Integrity digests and artifacts

| Receipt | SHA-256 |
|---|---|
| Historical R1 run manifest | `af2423c89d28d865550127a27173d3c37ddd7b3cd003f26da52d560fd92868e2` |
| R1 sealed-score file | `7fd0a920202609d0b242c7de4b8a77128e03b4acdd5f7df7915cbbb2269e3eb9` |
| R1 transition structural projection | `ad77084a13fcf11311eb90a6326371bcf2db53b4d957c6e0a2d4cedfaa9277c1` |
| Fixed support bank | `8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558` |
| Frozen parameter configuration | `464e2c128ff0b42ea96b6b9f03227c7a9d5debab1877e33fcd672109db3a7f16` |
| Stage A candidate score table | `513ad88c627b1abc3ae8bba62a2f79c6185f1c1fc810bd4357b981752d2d6b05` |
| Stage A final C2 state | `44064de2bdfb1a06445a8e4030a1a4c4365015ad89fc8bca28ed705d1549b72a` |
| Historical R1-recorded `class_json_sha256` | `b863b819fe7734a9c83acc26b25d419c8d8897a9ae93d5b8fd9d0e424ecf42d8` |

Execution source fingerprints (the run manifest and Stage A seal contain the same complete 11-file map):

| Source | SHA-256 |
|---|---|
| `experiments/calibration/r2_perf1_common.py` | `0fbffdacc85ec6d04ce810fe80165a47b15f0db6c5a690e7ad9faca2b56d00d6` |
| `experiments/calibration/r2_perf1_local_correction_eval.py` | `9c806ea63ca6407b90353b6d77e1ebd3bd5d73c8e2fbf3147b19b943aea8ac6f` |
| `experiments/calibration/r2_perf1_stage_a.py` | `722b7fd58cf0aa6c44abe6936d13ead47b78a2b514ec7f903dd6116538cfb8b2` |
| `experiments/calibration/r2_perf1_stage_b.py` | `aaac635b5d52ba06ada7a9f719f4f8e0722310ccc359563d9d9da8d3994a90db` |
| `experiments/calibration/r1_matched_correction_pilot.py` | `3208e2c52277ad76351eb86968e362aa315effe9e3c734bfe2b6ff7bab0db448` |
| `experiments/calibration/r1_runtime_supervisor.py` | `e6254a82d8dd24156cea65380b41b04cb1372744da64a085043aa6dd74df5bab` |
| `experiments/calibration/r2_0_label_blind_mechanism_audit.py` | `e2c9d9c6d22dba6a2222b84698d712dc70727d3f45d698e017d558b807b9a727` |
| `experiments/calibration/r2_l1_locality_kernels.py` | `44b7f57e5542aff647d64b3a62efcc130ddce8d8c6840d250508476aee39d64f` |
| `experiments/calibration/query_tape.py` | `ed6f77c2a4de1bd05969ae3c09f2c98c13a744481ca436f80e4633bd6c412c6d` |
| `experiments/calibration/stage05.py` | `4b8aae252f5f1e096b127906794064da269aa79f7cd5514f6f738cda585c017e` |
| `experiments/risk_safe/preflight.py` | `2070ba770ac36af06f688b0450c711d67b7bca526e3939178d559216535c25ba` |

The output directory contains the frozen run manifest, Stage A score table and seal, C2 parity report, mechanism diagnostics, primary metrics, all 20 sensitivity rows, aggregate Stage B report, and supervisor status. It contains no raw TEST outcome list. The OS does not enforce a hard storage quota; the run used the audited application-level pre-write limit and free-space check.

## Regression and limitations

Focused tests: **19 passed** under the exact interpreter with `CUDA_VISIBLE_DEVICES` set empty before imports. Tests cover vectorized/scalar distance agreement, compact support, sign conflicts, top-k equivalence, FIFO capacity, source/time exclusion, score-seal-before-query order, TEST-label projection isolation, outcome metrics, paired cluster bootstrap, score-table tampering/alignment failure, output pre-write limits, partial-output retention, and no output reuse. `py_compile` passed for all four new modules and the test module. No unrelated suite was run.

Peak process working set was not retained in the output: the Stage A worker computed an in-process RSS sample but the orchestration layer discarded its stdout payload. The combined supervised wall time, Stage B CPU/wall time, disk free-space check, and exact output-byte count are retained. This does not affect score or outcome calculations.

## Scientific conclusion

At the predeclared primary `(R=.10, λ=.10)`, C2-Gated and C3 both underperformed C0 by 0.0947 AUROC, and neither differed from the other at image-score level. C1 was higher than C0 on this exposed development stream but remains a system-level adapted reference. The sensitivity grid shows the primary setting was not representative of every engineering setting; the results support a negative finding for the fixed primary configuration and a locality/coverage limitation, not rejection of the broader Local Correction idea. No radius was selected from these outcomes, and no expansion to another category or dataset was performed.

The local commit SHA is reported in the completion message. It is not embedded here because adding a commit's own hash to its contents would change that hash.
