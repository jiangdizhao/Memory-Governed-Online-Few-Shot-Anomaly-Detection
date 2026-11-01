# R2-EXP1 — C1 Mechanism Discovery

**Status: COMPLETE_DEVELOPMENT_ONLY.** This is an outcome-aware analysis of the already exposed switch, support-seed-0, 512-event development stream. All 512 outcomes had already been used by R2-PERF-1. The results are descriptive and conditional on one frozen historical trajectory; they do not establish independent confirmation or causal mechanism effects.

## Integrated finding

C1 image AUROC is 0.843083 versus 0.752198 for historical C2, a difference of 0.090884. The predeclared C2 amplitude sweep tested uniform downscaling on the same queried-anchor trajectory. At most 97.7% of that AUROC gap was recovered by a predeclared amplitude below 0.20. The strongest evidence for a follow-up is summarized under the Experiment 2 priorities below; no mechanism is causally confirmed.

## A. Performance and correction movement

All threshold metrics use the frozen C64 threshold 0.5165398120880127 and the historical `score >= threshold` rule. Directionally favorable movement is upward for defects and downward for normals; those event counts are separate from ranking-pair changes.

| Method | Image AUROC | AP | FPR | FNR | Balanced accuracy | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.807035 | 0.915183 | 0.0261 | 0.6017 | 0.6861 | 143 | 149 | 4 | 216 |
| C1 | 0.843083 | 0.932168 | 0.0654 | 0.4318 | 0.7514 | 204 | 143 | 10 | 155 |
| C2 | 0.752198 | 0.892874 | 0.7255 | 0.1226 | 0.5760 | 315 | 42 | 111 | 44 |

| Correction | Mean absolute image-score change | P90 absolute image-score change | Positive | Negative | Zero | Favorable | Harmful | Image score changed | Prediction flips | New FP | Repaired FP | New FN | Repaired FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 0.018219 | 0.033443 | 390 | 82 | 40 | 337 | 135 | 472 | 73 | 8 | 2 | 1 | 62 |
| C2 | 0.140132 | 0.200000 | 418 | 54 | 40 | 331 | 141 | 472 | 293 | 107 | 0 | 7 | 179 |

### C1 versus C2 magnitude

| Method | Mean absolute image delta | Median absolute image delta | P90 absolute image delta | Mean absolute Top-32 patch delta | Median absolute Top-32 patch delta | P90 absolute Top-32 patch delta | Nonzero Top-32 fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | 0.0182185 | 0.0200857 | 0.0334435 | 0.0180066 | 0.0204597 | 0.0325961 | 0.9219 |
| C2 | 0.1401319 | 0.1613128 | 0.2000000 | 0.1205078 | 0.1200000 | 0.2000000 | 0.9219 |

| Method | Defect mean absolute image delta | Normal mean absolute image delta | C0-margin <0.02: events / favorable / harmful / flips | Corrected argmax changed | Unchanged score masked by unchanged baseline argmax |
|---|---:|---:|---:|---:|---:|
| C1 | 0.0180754 | 0.0185544 | 140 / 99 / 30 / 59 | 158 | 0 |
| C2 | 0.1491987 | 0.1188575 | 140 / 99 / 30 / 77 | 302 | 0 |

Top-32 patch-level magnitude and image-score movement are not interchangeable: image scoring takes the maximum over the corrected dense map, so changes on non-maximal patches can have no image-score effect. The argmax/masking counts and normal/anomaly magnitude quantiles are in `aggregate_results.json`.

## B. Exact AUROC pairwise decomposition

The verified labels contain 359 defects and 153 normals, giving exactly 54,927 anomaly-normal pairs. AUROC gives ties weight 0.5. Strict repairs/damage exclude ties; tie creation and resolution are reported separately.

| Candidate vs C0 | Strict incorrect→correct | Strict correct→incorrect | C0 incorrect→tie | C0 correct→tie | C0 tie→correct | C0 tie→incorrect | Net concordance change | AUROC difference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 4951 | 2971 | 0 | 0 | 0 | 0 | 1980.0 | 0.036048 |
| C2 | 6223 | 9235 | 0 | 0 | 0 | 0 | -3012.0 | -0.054836 |

Direct C1-versus-C2 pairwise comparison: C1 strictly repairs 8144 C2-incorrect pairs and strictly damages 3152 C2-correct pairs; net concordance change is 4992.0 across 54,927 pairs (AUROC difference 0.090884).

For each arm, `AUROC_m − AUROC_C0 = net concordance change / 54,927` was verified against the evaluator within 2e-15. The decomposition by original C0 margin, favorable/harmful normal and anomaly movement, pair time interval, and Source Group participation is in `aggregate_results.json`. Source Group bootstrap intervals are paired and conditional on this fixed trajectory; the 54,927 pairs are not treated as independent samples.

| Arm | Source Group bootstrap 95% AUROC interval | Paired AUROC difference vs C0 95% interval |
|---|---:|---:|
| C1 | [0.810182, 0.874590] | [0.006349, 0.069419] |
| C2 | [0.699434, 0.806167] | [-0.111756, 0.007601] |

## C. Magnitude-controlled C2

C2 a=0.20 reproduced the historical C2 patch vectors and all 512 image scores exactly (0.752198 AUROC). Each other row rescales the stored Top-32 patch delta before the exact `corrected_dense_map` and max aggregation; no image-score delta was scaled.

| Amplitude | AUROC | AP | ΔAUROC vs C0 | ΔAUROC vs C1 | Gap recovered | FPR | FNR | Balanced accuracy | FP | FN | Strict pairs repaired | Strict pairs damaged |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.841025 | 0.930454 | 0.033991 | -0.002057 | 97.7% | 0.0588 | 0.4958 | 0.7227 | 9 | 178 | 3813 | 1946 |
| 0.04 | 0.831485 | 0.927710 | 0.024451 | -0.011597 | 87.2% | 0.1307 | 0.3816 | 0.7438 | 20 | 137 | 5165 | 3822 |
| 0.08 | 0.803175 | 0.916539 | -0.003860 | -0.039908 | 56.1% | 0.4314 | 0.1978 | 0.6854 | 66 | 71 | 5953 | 6165 |
| 0.20 | 0.752198 | 0.892874 | -0.054836 | -0.090884 | 0.0% | 0.7255 | 0.1226 | 0.5760 | 111 | 44 | 6223 | 9235 |

**Can reducing C2 magnitude explain most of C1’s observed AUROC advantage?** Yes, on this fixed trajectory: at least half of the C1−C2 gap was recovered by a predeclared amplitude below 0.20. This diagnostic does not rule out C1’s outcome-aware or margin-dependent strength policy, because its per-anchor beta values were not recorded.

## D. Query timing and memory evidence

The query ledger confirms 17 feedback events: 10 defect and 7 normal. Across the 544 Top-32 candidates, C1’s native gate kept 544 (1.0000), with 256 inserts, 0 merges, 34 replacements and 254 skipped updates. C2 added 544 patch anchors and ended at its FIFO capacity of 256.


| Queried C0 outcome | Queries | Top-32 kept | Inserted | Merged | Replaced | Skipped |
|---|---:|---:|---:|---:|---:|---:|
| TP | 4 | 128 | 39 | 0 | 3 | 86 |
| TN | 6 | 192 | 36 | 0 | 31 | 125 |
| FP | 1 | 32 | 26 | 0 | 0 | 6 |
| FN | 6 | 192 | 155 | 0 | 0 | 37 |

Per-query positions, feedback labels, identities and ledger rows are stored only in the private run output. Query-interval AUROCs are descriptive; an interval without both outcomes is reported as unavailable rather than imputed. The query-age buckets and early/late position quartiles are also descriptive and do not establish an update effect.

| Query interval | Position range | Events (normal/anomaly) | C0 AUROC | C1 AUROC | C2 AUROC |
|---:|---|---:|---:|---:|---:|
| 0 | 0–35 | 36 (12/24) | 0.7917 | 0.7917 | 0.7917 |
| 1 | 36–55 | 20 (9/11) | 0.8586 | 0.8384 | 0.2727 |
| 2 | 56–72 | 17 (0/17) | NA | NA | NA |
| 3 | 73–116 | 44 (10/34) | 0.8088 | 0.7647 | 0.8588 |
| 4 | 117–159 | 43 (12/31) | 0.7796 | 0.8737 | 0.8441 |
| 5 | 160–162 | 3 (0/3) | NA | NA | NA |
| 6 | 163–189 | 27 (7/20) | 0.7857 | 0.8429 | 0.7643 |
| 7 | 190–268 | 79 (21/58) | 0.7430 | 0.8046 | 0.7307 |
| 8 | 269–297 | 29 (10/19) | 0.8316 | 0.8053 | 0.8263 |
| 9 | 298–302 | 5 (0/5) | NA | NA | NA |
| 10 | 303–349 | 47 (17/30) | 0.8235 | 0.8784 | 0.7667 |
| 11 | 350–367 | 18 (3/15) | 0.6222 | 0.8667 | 0.8889 |
| 12 | 368–375 | 8 (2/6) | 1.0000 | 1.0000 | 0.9167 |
| 13 | 376–395 | 20 (8/12) | 0.8125 | 0.9271 | 0.7917 |
| 14 | 396–413 | 18 (10/8) | 0.7875 | 0.9125 | 0.8625 |
| 15 | 414–488 | 75 (25/50) | 0.8360 | 0.8696 | 0.7736 |
| 16 | 489–501 | 13 (3/10) | 0.8667 | 0.8333 | 0.7667 |
| 17 | 502–511 | 10 (4/6) | 0.7083 | 0.9583 | 1.0000 |

Query age and early/late position breakdowns are available in `aggregate_results.json`; the query event itself is assigned to the interval before its feedback disclosure.

The tape does not identify per-anchor beta, kappa, radius, hidden Codebook prototypes, or unrecorded state changes: `NOT_IDENTIFIABLE_FROM_EXISTING_TAPE`. The R1 manifest reports final C1 positive/negative anchor counts, while the correction-event ledger provides only its recorded aggregate update statistics.

## E. Mechanism ranking and Experiment 2 priorities

| Rank | Hypothesis | Evidence label | Directly observed | Causal limitation | Experiment 2 priority |
|---:|---|---|---|---|---|
| 1 | H2 — Outcome-aware and margin-dependent Correction Magnitudes | `SUPPORTED_BUT_CONFOUNDED` | Uniformly reducing C2 amplitude recovered at least half of the C1-minus-C2 AUROC gap on this fixed trajectory. | Uniform scaling is not the same intervention as C1's outcome- and margin-dependent beta policy; per-anchor beta is not recorded. | PRIMARY |
| 2 | H3 — Sign-aware Merging and Retention | `SUPPORTED_BUT_CONFOUNDED` | The correction ledger records sign-aware merges/replacements during a trajectory where C1 exceeds C2 AUROC. | The tape does not include matched C1 FIFO or C2 sign-aware-retention counterfactuals. | LOWER |
| 3 | H4 — Broader Gaussian Influence Propagation | `SUPPORTED_BUT_CONFOUNDED` | The historical tape records nonzero C1 far-field patch activity while C1 exceeds C2 AUROC. | Activation coverage and AUROC are trajectory-level associations, not a propagation-only ablation. | PRIMARY |
| 4 | H1 — Selective Anchor Admission | `WEAK_EVIDENCE` | The tape does not show a selective gate associated with a C1 AUROC advantage. | Admission counts do not reveal whether rejected or retained anchors were useful; C1 also differs in magnitude and retention. | LOWER |

**Recommended primary Experiment 2 mechanisms:** H2, H4. These priorities follow the observed admission, amplitude, retention and far-field records plus interpretability. All require randomized or otherwise controlled component intervention; none is causally confirmed here.

## F. Reproducibility and limits

- Repository: `calibration-stage0-contract-20261029@4a718090790812c91a90c8ab6d6c1a347593b8de`; task code SHA-256: `eeee3efbe137c00fa75d3b24b26fbe83eca4d21fb39acd2dd77fcb7308437052`.
- R1 manifest SHA-256: `af2423c89d28d865550127a27173d3c37ddd7b3cd003f26da52d560fd92868e2`; sealed scores SHA-256: `7fd0a920202609d0b242c7de4b8a77128e03b4acdd5f7df7915cbbb2269e3eb9`; structural transition projection SHA-256: `ad77084a13fcf11311eb90a6326371bcf2db53b4d957c6e0a2d4cedfaa9277c1`.
- Prior R2-PERF-1 score table SHA-256: `513ad88c627b1abc3ae8bba62a2f79c6185f1c1fc810bd4357b981752d2d6b05`; prior Stage A seal SHA-256: `f4029a2a79552d0362e0435a654f5f5ec63a576837c5026fecd6949f2f3fb1db`.
- R1 public tape check: 512 event NPZ shard and array hashes verified; all recorded C0/C1/C2 scores reconstructed exactly; 512-entry shard receipt-list SHA-256: `7f230261851417fbbd897909394c14afedb05ab751785d0ae6bbe25b858dc95a`.
- EXP1 Stage A: 512 scores sealed with score-table SHA-256 `c6023436ee7cedc3f2eca49734ca7f30cae656dc78e675dcdb1f9fd45b0e6763`; seal SHA-256 `160d50844addcfec9c774674e3b03144dd7134ee88ac71ae47c8d70ce7b66b6e`; query positions SHA-256 `7f4aedf49f3e3bffd2482bd00d9e16512553b7cef8c0af7e002b47b60ddcb17f`; event/identity alignment with R1 and R2-PERF-1 was exact.
- Stage B identity mapping: `512_TEST_OUTCOMES_IDENTITY_POSITION_VERIFIED` for TEST positions 0–511; 359 defects and 153 normals. The full-file source SHA remains `NOT_COMPUTED`.
- New private run: `outputs\calibration\r2_perf1_switch_s0_512_20261101T040718_5ec10eee`; unique run ID `20261101T040718_5ec10eee`; new outputs are separate from all historical artifacts.
- Supervisor: `COMPLETED`, elapsed 27.75 seconds, confirmed termination `True`; output byte count is enforced by the existing pre-write `OutputBudget`.
- New output size: 1364841 bytes under the 2 GiB application limit.
- Focused CPU regression: `D:\anaconda3\envs\torch\python.exe -m pytest -q tests/calibration/test_r2_exp1_c1_mechanism_discovery.py`; 10 passed; test source SHA-256: `c341c70e60eb98170e9ac909b66c41df8541a9846048c5d7a9c7a2283a32e5bd`.
- Stage workers: Stage A 21.86 seconds; Stage B 4.82 wall seconds and 4.30 CPU seconds; peak Stage B working set NA bytes.
- Source Group bootstrap: 1000 paired resamples, seed 20261101, 103 groups, 1000 valid replicates; intervals describe uncertainty conditional on the historical trajectory.
- The existing R2-PERF-1 projection read authorized switch.json as UTF-8 text, identity-checked the first 512 TEST records, and parsed only those outcome fields. It structurally skipped TRAIN and non-prefix TEST outcomes and did not open mask files or other categories. The full-file source SHA was not computed, so no independent full-JSON fingerprint is claimed.
- No DINO load, feature extraction, image decoding, GPU/CUDA inference, new category, mask, independent confirmation set, or algorithm modification was performed.
- The committed aggregate was re-emitted from the sealed Stage B report using recursive removal of event-level fields; sanitizer SHA-256: `65fb1d610784ecea24988b3b97b7ff45860d5876d67dfca5cb115f6d2cf06b0d`. The analysis worker source SHA above remains the run-time fingerprint.
- Raw per-event labels, event/image identities and queried feedback are retained only in the private untracked run directory. The committed report and JSON contain aggregate results only.
- This development-set analysis does not validate a final hyperparameter and does not prove that any component caused C1’s observed advantage. Experiment 2 must isolate components on an appropriately authorized and frozen design.
