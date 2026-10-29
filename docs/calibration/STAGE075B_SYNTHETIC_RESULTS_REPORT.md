# Stage 0.75-B full synthetic results

**Primary run:** `outputs/calibration/stage075b_synthetic_v3_blocked_final_20261029/`
**Contract:** candidate v3, frozen before the full run.
**Status:** synthetic score/memory mechanics only; not Real-IAD performance or native scorer/Greedy parity.

## Protocol and execution

S0–S14 and control A each ran at 1,024 events and seeds `37`, `101`, `2029`: 48 rows. The label-blind Q-A-TS tape used seed 37 with 31/1,024 queries per trajectory (3.027%). Scores, query positions, and queried-normal memory writes were controller-independent. The three-layer, capacity-256 synthetic memory is a mechanics fixture. The runner repeated all 48 trajectories with identical digests. Four checkpoint cuts (1, 129, 401, 733) reproduced the uninterrupted trace and state. Compilation passed; 133 calibration tests passed; the direct equation regression and independent artifact verification passed. G1–G15 passed, but G16/Control E is BLOCKED, so overall Stage 0.75-B acceptance remains blocked.

## Overall means on defined trajectories

Means are unweighted over 42 trajectories with defined BA. S0 and S5 contain one class, so BA is null. Confusion totals include all 48 trajectories. No confidence intervals or inferential claims are made.

| Arm | Mean BA | Mean FPR | Mean FNR | Mean TV | Mean updates | Mean state bytes |
|---|---:|---:|---:|---:|---:|---:|
| B3 | 0.746193 | 0.112206 | 0.395408 | 0.032227 | 20.95 | 1391 |
| B7 | 0.750430 | 0.113965 | 0.385174 | 0.069307 | 7.93 | 5698 |
| B3+Offset | 0.760482 | 0.110061 | 0.368975 | 0.251637 | 25.62 | 955 |
| B8-M | 0.744957 | 0.083745 | 0.426340 | 0.057903 | 20.95 | 6079 |
| B8-S | 0.746193 | 0.112206 | 0.395408 | 0.032227 | 20.95 | 6493 |
| B8-P | 0.743525 | 0.104576 | 0.408374 | 0.224640 | 29.57 | 6280 |
| B8-Full | 0.746193 | 0.112206 | 0.395408 | 0.032227 | 20.95 | 6634 |

Confusion totals across all 48 trajectories (each 1,024 events):

| Arm | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| B3 | 29046 | 3376 | 6833 | 9897 |
| B7 | 28974 | 3448 | 6639 | 10091 |
| B3+Offset | 29117 | 3305 | 6452 | 10278 |
| B8-M | 29861 | 2561 | 7277 | 9453 |
| B8-S | 29046 | 3376 | 6833 | 9897 |
| B8-P | 29292 | 3130 | 7107 | 9623 |
| B8-Full | 29046 | 3376 | 6833 | 9897 |

B8-Full−B3 was exactly zero for all 42 defined pairs. Against B3+Offset, the mean paired difference was `-0.014289`; B3+Offset was higher in 35 rows, lower in 7, tied in 0.

## Per-scenario descriptive means

| Scenario | BA rows | B3 | B7 | B3+Offset | B8-M | B8-S | B8-P | B8-Full | Full−B3 | Full−Offset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0_exact_b3_fallback | 0/3 | — | — | — | — | — | — | — | — | — |
| S1_monotone_coordinates | 3/3 | 0.7919 | 0.7995 | 0.8108 | 0.7915 | 0.7919 | 0.7791 | 0.7919 | 0.0000 | -0.0189 |
| S2_sparse_c64_tail | 3/3 | 0.7839 | 0.7963 | 0.8146 | 0.7824 | 0.7839 | 0.7913 | 0.7839 | 0.0000 | -0.0307 |
| S3_asymmetric_classes | 3/3 | 0.8272 | 0.8300 | 0.8523 | 0.8228 | 0.8272 | 0.8306 | 0.8272 | 0.0000 | -0.0251 |
| S4_delayed_defects | 3/3 | 0.8123 | 0.8249 | 0.8374 | 0.8044 | 0.8123 | 0.8134 | 0.8123 | 0.0000 | -0.0251 |
| S5_defect_only | 0/3 | — | — | — | — | — | — | — | — | — |
| S6_scarce_correlated | 3/3 | 0.8176 | 0.8192 | 0.8234 | 0.8159 | 0.8176 | 0.8093 | 0.8176 | 0.0000 | -0.0058 |
| S7_score_vs_concept_staleness | 3/3 | 0.6976 | 0.6999 | 0.7165 | 0.6945 | 0.6976 | 0.6978 | 0.6976 | 0.0000 | -0.0190 |
| S8_c64_mismatch | 3/3 | 0.6123 | 0.6116 | 0.6200 | 0.6136 | 0.6123 | 0.6222 | 0.6123 | 0.0000 | -0.0077 |
| S9_rejoin_motion | 3/3 | 0.7038 | 0.7066 | 0.7109 | 0.7018 | 0.7038 | 0.6950 | 0.7038 | 0.0000 | -0.0071 |
| S10_numeric_edges | 3/3 | 0.6964 | 0.7025 | 0.7057 | 0.7089 | 0.6964 | 0.6964 | 0.6964 | 0.0000 | -0.0092 |
| S11_provenance_lso | 3/3 | 0.8036 | 0.8056 | 0.8190 | 0.8023 | 0.8036 | 0.7924 | 0.8036 | 0.0000 | -0.0154 |
| S12_shared_score_invariance | 3/3 | 0.8080 | 0.8109 | 0.8215 | 0.8040 | 0.8080 | 0.7941 | 0.8080 | 0.0000 | -0.0135 |
| S13_checkpoint_resume | 3/3 | 0.7865 | 0.7866 | 0.7977 | 0.7842 | 0.7865 | 0.7866 | 0.7865 | 0.0000 | -0.0112 |
| S14_nonidentifiability | 3/3 | 0.8068 | 0.8135 | 0.8174 | 0.8036 | 0.8068 | 0.8017 | 0.8068 | 0.0000 | -0.0106 |
| A_no_gain_control | 3/3 | 0.4987 | 0.4991 | 0.4996 | 0.4993 | 0.4987 | 0.4996 | 0.4987 | 0.0000 | -0.0009 |

## Per-seed paired results

`—` denotes undefined BA.

| Scenario | Seed | B3 | B7 | B3+Offset | B8-M | B8-S | B8-P | B8-Full | Full−B3 | Full−Offset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0_exact_b3_fallback | 37 | — | — | — | — | — | — | — | — | — |
| S0_exact_b3_fallback | 101 | — | — | — | — | — | — | — | — | — |
| S0_exact_b3_fallback | 2029 | — | — | — | — | — | — | — | — | — |
| S1_monotone_coordinates | 37 | 0.8216 | 0.8242 | 0.8334 | 0.8223 | 0.8216 | 0.8277 | 0.8216 | 0.0000 | -0.0119 |
| S1_monotone_coordinates | 101 | 0.7568 | 0.7687 | 0.7869 | 0.7575 | 0.7568 | 0.7233 | 0.7568 | 0.0000 | -0.0302 |
| S1_monotone_coordinates | 2029 | 0.7974 | 0.8055 | 0.8120 | 0.7948 | 0.7974 | 0.7861 | 0.7974 | 0.0000 | -0.0146 |
| S2_sparse_c64_tail | 37 | 0.7972 | 0.7975 | 0.8331 | 0.7959 | 0.7972 | 0.7844 | 0.7972 | 0.0000 | -0.0358 |
| S2_sparse_c64_tail | 101 | 0.7401 | 0.7647 | 0.7897 | 0.7362 | 0.7401 | 0.7845 | 0.7401 | 0.0000 | -0.0496 |
| S2_sparse_c64_tail | 2029 | 0.8144 | 0.8268 | 0.8210 | 0.8152 | 0.8144 | 0.8049 | 0.8144 | 0.0000 | -0.0066 |
| S3_asymmetric_classes | 37 | 0.8413 | 0.8389 | 0.8575 | 0.8371 | 0.8413 | 0.8202 | 0.8413 | 0.0000 | -0.0162 |
| S3_asymmetric_classes | 101 | 0.7683 | 0.7768 | 0.8260 | 0.7626 | 0.7683 | 0.8199 | 0.7683 | 0.0000 | -0.0577 |
| S3_asymmetric_classes | 2029 | 0.8720 | 0.8742 | 0.8735 | 0.8688 | 0.8720 | 0.8518 | 0.8720 | 0.0000 | -0.0015 |
| S4_delayed_defects | 37 | 0.8356 | 0.8422 | 0.8571 | 0.8254 | 0.8356 | 0.8338 | 0.8356 | 0.0000 | -0.0215 |
| S4_delayed_defects | 101 | 0.7729 | 0.7917 | 0.8010 | 0.7622 | 0.7729 | 0.7879 | 0.7729 | 0.0000 | -0.0282 |
| S4_delayed_defects | 2029 | 0.8285 | 0.8408 | 0.8541 | 0.8255 | 0.8285 | 0.8186 | 0.8285 | 0.0000 | -0.0256 |
| S5_defect_only | 37 | — | — | — | — | — | — | — | — | — |
| S5_defect_only | 101 | — | — | — | — | — | — | — | — | — |
| S5_defect_only | 2029 | — | — | — | — | — | — | — | — | — |
| S6_scarce_correlated | 37 | 0.8071 | 0.8109 | 0.8109 | 0.7996 | 0.8071 | 0.8071 | 0.8071 | 0.0000 | -0.0038 |
| S6_scarce_correlated | 101 | 0.8066 | 0.8055 | 0.8288 | 0.8071 | 0.8066 | 0.8076 | 0.8066 | 0.0000 | -0.0223 |
| S6_scarce_correlated | 2029 | 0.8390 | 0.8412 | 0.8304 | 0.8411 | 0.8390 | 0.8132 | 0.8390 | 0.0000 | 0.0087 |
| S7_score_vs_concept_staleness | 37 | 0.7095 | 0.7108 | 0.7194 | 0.7060 | 0.7095 | 0.6937 | 0.7095 | 0.0000 | -0.0099 |
| S7_score_vs_concept_staleness | 101 | 0.6652 | 0.6679 | 0.6988 | 0.6624 | 0.6652 | 0.6804 | 0.6652 | 0.0000 | -0.0336 |
| S7_score_vs_concept_staleness | 2029 | 0.7180 | 0.7210 | 0.7315 | 0.7150 | 0.7180 | 0.7194 | 0.7180 | 0.0000 | -0.0135 |
| S8_c64_mismatch | 37 | 0.6337 | 0.6330 | 0.6456 | 0.6334 | 0.6337 | 0.6518 | 0.6337 | 0.0000 | -0.0119 |
| S8_c64_mismatch | 101 | 0.6293 | 0.6237 | 0.6260 | 0.6273 | 0.6293 | 0.6252 | 0.6293 | 0.0000 | 0.0033 |
| S8_c64_mismatch | 2029 | 0.5740 | 0.5780 | 0.5885 | 0.5801 | 0.5740 | 0.5895 | 0.5740 | 0.0000 | -0.0145 |
| S9_rejoin_motion | 37 | 0.6944 | 0.6929 | 0.7072 | 0.6909 | 0.6944 | 0.6886 | 0.6944 | 0.0000 | -0.0128 |
| S9_rejoin_motion | 101 | 0.6955 | 0.7052 | 0.7124 | 0.6928 | 0.6955 | 0.6814 | 0.6955 | 0.0000 | -0.0169 |
| S9_rejoin_motion | 2029 | 0.7216 | 0.7216 | 0.7132 | 0.7217 | 0.7216 | 0.7149 | 0.7216 | 0.0000 | 0.0084 |
| S10_numeric_edges | 37 | 0.6890 | 0.6916 | 0.7056 | 0.7073 | 0.6890 | 0.6890 | 0.6890 | 0.0000 | -0.0165 |
| S10_numeric_edges | 101 | 0.7089 | 0.7099 | 0.7123 | 0.7007 | 0.7089 | 0.7089 | 0.7089 | 0.0000 | -0.0034 |
| S10_numeric_edges | 2029 | 0.6914 | 0.7061 | 0.6992 | 0.7188 | 0.6914 | 0.6914 | 0.6914 | 0.0000 | -0.0078 |
| S11_provenance_lso | 37 | 0.8168 | 0.8170 | 0.8338 | 0.8126 | 0.8168 | 0.8134 | 0.8168 | 0.0000 | -0.0169 |
| S11_provenance_lso | 101 | 0.7565 | 0.7651 | 0.7875 | 0.7551 | 0.7565 | 0.7523 | 0.7565 | 0.0000 | -0.0309 |
| S11_provenance_lso | 2029 | 0.8375 | 0.8348 | 0.8357 | 0.8392 | 0.8375 | 0.8115 | 0.8375 | 0.0000 | 0.0017 |
| S12_shared_score_invariance | 37 | 0.8158 | 0.8196 | 0.8319 | 0.8076 | 0.8158 | 0.8143 | 0.8158 | 0.0000 | -0.0160 |
| S12_shared_score_invariance | 101 | 0.7745 | 0.7796 | 0.7853 | 0.7716 | 0.7745 | 0.7485 | 0.7745 | 0.0000 | -0.0108 |
| S12_shared_score_invariance | 2029 | 0.8336 | 0.8336 | 0.8473 | 0.8329 | 0.8336 | 0.8194 | 0.8336 | 0.0000 | -0.0137 |
| S13_checkpoint_resume | 37 | 0.7990 | 0.8000 | 0.8015 | 0.7954 | 0.7990 | 0.7778 | 0.7990 | 0.0000 | -0.0026 |
| S13_checkpoint_resume | 101 | 0.7661 | 0.7678 | 0.7981 | 0.7635 | 0.7661 | 0.7955 | 0.7661 | 0.0000 | -0.0320 |
| S13_checkpoint_resume | 2029 | 0.7944 | 0.7920 | 0.7933 | 0.7937 | 0.7944 | 0.7863 | 0.7944 | 0.0000 | 0.0011 |
| S14_nonidentifiability | 37 | 0.8169 | 0.8333 | 0.8309 | 0.8157 | 0.8169 | 0.7888 | 0.8169 | 0.0000 | -0.0140 |
| S14_nonidentifiability | 101 | 0.7785 | 0.7777 | 0.7939 | 0.7713 | 0.7785 | 0.7901 | 0.7785 | 0.0000 | -0.0154 |
| S14_nonidentifiability | 2029 | 0.8251 | 0.8295 | 0.8274 | 0.8238 | 0.8251 | 0.8261 | 0.8251 | 0.0000 | -0.0023 |
| A_no_gain_control | 37 | 0.4989 | 0.4979 | 0.4987 | 0.5018 | 0.4989 | 0.4986 | 0.4989 | 0.0000 | 0.0003 |
| A_no_gain_control | 101 | 0.4985 | 0.4986 | 0.4934 | 0.4985 | 0.4985 | 0.4934 | 0.4985 | 0.0000 | 0.0052 |
| A_no_gain_control | 2029 | 0.4986 | 0.5007 | 0.5067 | 0.4977 | 0.4986 | 0.5067 | 0.4986 | 0.0000 | -0.0081 |

## Runtime and interpretation

The runner took 487.10s wall time including tests, focused checks, both matrices and artifact generation. The first matrix pass summed to 70.74s (1.474s per trajectory). A representative traced run took 6.075s with 16,129,661 Python-tracked peak bytes; this is not RSS or GPU memory.

The Full raw threshold digest matched B3 in all 48 trajectories. The reliability term is a heuristic, not a confidence bound. Three fixed synthetic seeds are descriptive. These results establish neither Real-IAD superiority nor population-risk guarantees. No Stage 0.75-C, Stage 1/2, Real-IAD performance, mask, Mint/PCB outcome, or tuning access occurred.
