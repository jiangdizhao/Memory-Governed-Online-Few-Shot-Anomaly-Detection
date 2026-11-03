# R8 Unified Correction Bank Mechanism Attribution

## 1. Research question and hypotheses

R8 tests Correction capacity, feedback order, and feature-address association in one matched retrospective development diagnostic. H1 is capacity retention; H2 is feedback timing; H3 is feature/prototype address association beyond a state-level scalar shift.

## 2. Frozen protocol and causal permissions

Run `r8_correction_mechanism_20261103_a5` used the four previously exposed categories, one Support Seed 0, the fixed four-image Support plus separate 64-normal C64 commissioning set, shared Greedy K=1024, Correction K=256/1024, and identity-matched R7 Seed 37 queries. This is RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC, not blind or independent confirmation. Stage A sealed all eight full conditions before Stage B read image-level outcomes; online labels were disclosed only for registered queries after score fsync; masks were not opened.

## 3. R7 numerical reproduction

The immutable R7 run was launched from parent `276746b01fba47a7d41fa3ceb01b40e961fee341` with runner SHA `73a0d0c564500b565e58ea6c45e052f4b944b6a0ba38f24ad8ded8310c5545e9`. The R7 report records that only trailing blank lines were later removed before commit; the committed runner SHA is `0e2b0bc468cdcc1a8d72082487b1618817a45342fe93b99a00052ecb50f85b3a`. R8 did not rerun R7 and binds to its sealed event rows and identity receipt.

The fresh 512-event GPU integration smoke passed in 101.4 seconds under the Windows Job Object. It reproduced the R7 prefix with zero query, score, threshold, and decision mismatches; exercised 12 queried normals and 5 queried defects, 12 Greedy state changes, 17 updates in each Correction bank, current-state B3 refresh, and all registered read-only control scores.

{
  "audiojack": {
    "atol": 1e-05,
    "decision_mismatches": 0,
    "event_count": 3815,
    "identity_bijection_verified": true,
    "maximum_absolute_score_error": {
      "G1024": 0.0,
      "GC1024": 0.0
    },
    "maximum_absolute_threshold_error": {
      "G1024-B1": 0.0,
      "G1024-B3": 0.0,
      "G1024-Frozen": 0.0,
      "GC1024-B1": 0.0,
      "GC1024-B3": 0.0,
      "GC1024-Frozen": 0.0
    },
    "query_mismatches": 0,
    "rtol": 0.0001
  },
  "regulator": {
    "atol": 1e-05,
    "decision_mismatches": 0,
    "event_count": 3855,
    "identity_bijection_verified": true,
    "maximum_absolute_score_error": {
      "G1024": 0.0,
      "GC1024": 0.0
    },
    "maximum_absolute_threshold_error": {
      "G1024-B1": 0.0,
      "G1024-B3": 0.0,
      "G1024-Frozen": 0.0,
      "GC1024-B1": 0.0,
      "GC1024-B3": 0.0,
      "GC1024-Frozen": 0.0
    },
    "query_mismatches": 0,
    "rtol": 0.0001
  },
  "rolled_strip_base": {
    "atol": 1e-05,
    "decision_mismatches": 0,
    "event_count": 3805,
    "identity_bijection_verified": true,
    "maximum_absolute_score_error": {
      "G1024": 0.0,
      "GC1024": 0.0
    },
    "maximum_absolute_threshold_error": {
      "G1024-B1": 0.0,
      "G1024-B3": 0.0,
      "G1024-Frozen": 0.0,
      "GC1024-B1": 0.0,
      "GC1024-B3": 0.0,
      "GC1024-Frozen": 0.0
    },
    "query_mismatches": 0,
    "rtol": 0.0001
  },
  "switch": {
    "atol": 1e-05,
    "decision_mismatches": 0,
    "event_count": 3820,
    "identity_bijection_verified": true,
    "maximum_absolute_score_error": {
      "G1024": 0.0,
      "GC1024": 0.0
    },
    "maximum_absolute_threshold_error": {
      "G1024-B1": 0.0,
      "G1024-B3": 0.0,
      "G1024-Frozen": 0.0,
      "GC1024-B1": 0.0,
      "GC1024-B3": 0.0,
      "GC1024-Frozen": 0.0
    },
    "query_mismatches": 0,
    "rtol": 0.0001
  }
}

## 4. Full Capacity × Order results

Per-category scorer ranking and controller confusion results are in `aggregate_results.json`; the public aggregate contains no event rows, paths, feature vectors, or label vectors.

| Category | Order | G AUROC | GC256 AUROC | GC1024 AUROC | G AP | GC256 AP | GC1024 AP |
|---|---|---:|---:|---:|---:|---:|---:|
| audiojack | RAW | 0.669268 | 0.655362 | 0.809880 | 0.542194 | 0.539430 | 0.729966 |
| audiojack | SHUFFLE0 | 0.739300 | 0.746678 | 0.772655 | 0.606677 | 0.629664 | 0.660450 |
| regulator | RAW | 0.826580 | 0.838981 | 0.869072 | 0.674421 | 0.738497 | 0.774471 |
| regulator | SHUFFLE0 | 0.766741 | 0.794824 | 0.813097 | 0.585251 | 0.694008 | 0.710714 |
| switch | RAW | 0.896100 | 0.889871 | 0.896257 | 0.904727 | 0.902871 | 0.906112 |
| switch | SHUFFLE0 | 0.868575 | 0.878165 | 0.884892 | 0.890019 | 0.901189 | 0.906164 |
| rolled_strip_base | RAW | 0.953469 | 0.955423 | 0.951711 | 0.973761 | 0.975148 | 0.973055 |
| rolled_strip_base | SHUFFLE0 | 0.953357 | 0.957173 | 0.956829 | 0.973409 | 0.976207 | 0.975962 |

## 5. Capacity and Order main effects and interaction

Per-category primary contrasts and the equal-category macro are recorded in the aggregate. Paired whole-Source-Group bootstrap intervals are descriptive and conditional on this fixed trajectory.

{
  "D_RAW_1024": 0.045375710472416314,
  "D_RAW_256": -0.0014452022126615893,
  "D_SHUFFLE0_1024": 0.024875021670619668,
  "D_SHUFFLE0_256": 0.012216640745831203,
  "capacity_RAW": 0.0468209126850779,
  "capacity_SHUFFLE0": 0.012658380924788465,
  "capacity_by_order_interaction": -0.03416253176028944,
  "order_C1024": -0.020500688801796646,
  "order_C256": 0.013661842958492793
}

{
  "D_RAW_1024": {
    "auroc_difference_ci_95_percentile": [
      0.039335786921443475,
      0.051241405813750814
    ],
    "valid_replicates": 1000
  },
  "D_RAW_256": {
    "auroc_difference_ci_95_percentile": [
      -0.004976657861292397,
      0.0021472031553294636
    ],
    "valid_replicates": 1000
  },
  "D_SHUFFLE0_1024": {
    "auroc_difference_ci_95_percentile": [
      0.020361316120484047,
      0.02914276486470891
    ],
    "valid_replicates": 1000
  },
  "D_SHUFFLE0_256": {
    "auroc_difference_ci_95_percentile": [
      0.006542232175796252,
      0.017801163627992552
    ],
    "valid_replicates": 1000
  },
  "capacity_RAW": {
    "auroc_difference_ci_95_percentile": [
      0.041041973899150486,
      0.05260332632543348
    ],
    "valid_replicates": 1000
  },
  "capacity_SHUFFLE0": {
    "auroc_difference_ci_95_percentile": [
      0.009216289663656712,
      0.016161391974207382
    ],
    "valid_replicates": 1000
  },
  "capacity_by_order_interaction": {
    "auroc_difference_ci_95_percentile": [
      -0.04098737558077133,
      -0.02754313314411985
    ],
    "valid_replicates": 1000
  },
  "order_C1024": {
    "auroc_difference_ci_95_percentile": [
      -0.026814783839312933,
      -0.013592866335048604
    ],
    "valid_replicates": 1000
  },
  "order_C256": {
    "auroc_difference_ci_95_percentile": [
      0.007270637597506279,
      0.01992126519239086
    ],
    "valid_replicates": 1000
  }
}

## 6. Full versus Scalar and Address-shuffled results

{
  "Address17": {
    "auroc": 0.7978565033056972,
    "average_precision": 0.7308055649902606
  },
  "Address53": {
    "auroc": 0.796194448070108,
    "average_precision": 0.7284179319153854
  },
  "Address97": {
    "auroc": 0.8013319332465899,
    "average_precision": 0.7348523990885691
  },
  "GC1024": {
    "auroc": 0.8568682585950181,
    "average_precision": 0.8133226728781693
  },
  "Scalar": {
    "auroc": 0.8408974900142203,
    "average_precision": 0.7839187955924349
  }
}

## 7. Early/middle/late feedback diagnostics

Early, middle, and late image-level ranking metrics are reported for the main arms and for Full, Scalar, and all three Address controls. Cumulative query timing is summarized at the 25th, 50th, 75th, and 100th percentile of stream progress for each category and order.

## 8. Calibration and FPR/FNR trade-offs

Frozen C64 q90, B1 queried-normal rolling q90, and B3 current-state q90 results include confusion counts, BA, FPR, FNR, accuracy, coverage, and worst complete 512-event windows in the aggregate.

## 9. Memory and compute costs

Across the eight full conditions, measured per-condition wall time ranged from 311.4 to 533.8 seconds (3,143.8 seconds summed), while supervisor elapsed time ranged from 318.5 to 540.5 seconds; every child stayed below the 7,200-second limit. Maximum measured CPU RSS was 3,976,339,456 bytes, peak CUDA allocation was 1,573,543,424 bytes, and peak CUDA reservation was 14,285,799,424 bytes. Maximum combined mutable tensor storage was 10,042,368 bytes for C256 and 22,671,360 bytes for C1024. The eight conditions made 140,800 C64 score calls in 755.6 seconds total. Per-condition runtime, Greedy occupancy/writes, C1 occupancy, and C64 rescoring details remain in the public aggregate and private seals.

## 10. Hypotheses supported, challenged or unresolved

- H1: **MIXED**; category SHUFFLE0 capacity contrasts: `[0.025977417307475492, 0.018272939606272853, 0.006726916785405601, -0.00034375000000008704]`; capacity-pressure categories: `['audiojack', 'regulator', 'switch', 'rolled_strip_base']`.
- H2: **MIXED**; category matched-query order contrasts: `[-0.1072564558736322, 0.003863697030363822, 0.01615965091330951, 0.00523035272277228]`.
- H3: **SUPPORTED_WITHIN_DEVELOPMENT_CONDITIONS**; Full-minus-Scalar: `[0.01760769387224992, 0.03358391725058396, 0.006337193398376906, 0.006354269801980061]`; Full-minus-Address AUROC by seed: `{'17': [0.06408655330598556, 0.12131931931931927, 0.03332601858148376, 0.017315129950495], '53': [0.05865223682727316, 0.13499749749749745, 0.03398269836892942, 0.015062809405940625], '97': [0.053326661814653065, 0.11927677677677673, 0.032087500178520245, 0.01745436262376232]}`.

## 11. Limitations and interpretation boundaries

There is one Support Seed and one Shuffle Seed. Source Group bootstrap is not independent trajectory uncertainty. The categories and development outcomes were previously exposed. Results do not establish broad seed robustness, blind confirmation, strict Gaussian locality, semantic correctness, conformal coverage, or factory-chronology generalization.

## 12. Final recommendation for Correction Bank

Treat Correction capacity and address controls as condition-dependent development evidence. Do not infer a universal preferred capacity from this single matched run; use the category effects, operating-point trade-offs, memory cost, and conditional intervals together.

## 13. Source commits, hashes, runtime and artifacts

- Branch: `calibration-stage0-contract-20261029`; source parent: `579fb7f5f4c590b95fda2b1c1c417f372aaf013f`; run ID: `r8_correction_mechanism_20261103_a5`.
- Executed R8 runner SHA-256: `0662ef3ba79ef6dcf15b64c3c720dd1d0cba3c1e2b1e8bb9e0b6dd59423c5a6f`; committed runner SHA-256: `73fda414c90e8effcfcc8af994ae4062e51da41d25b9c81f02581459411ba704`.
- Pre-run focused test source SHA-256: `bde404cef5750e90a07bb1f7da6004deed2ec6a2113756b41c723043a840e3fa`; committed test SHA-256: `18c7ed212ff8df9f59e02fe44c0398d92c448e686aa9ed74e6821ed0038906f1`.
- Execution-to-commit audit: after a5 completed, only the extra blank line at EOF was removed from the runner and test file to satisfy `git diff --cached --check`. A direct diff against the executed files shows no logic or test changes; the whitespace-clean committed files were compiled and tested again before commit.
- Focused CPU regression command: `D:\anaconda3\envs\torch\python.exe -m pytest -q tests/calibration/test_r8_correction_mechanism_attribution.py`; result: 30 passed.
- R7 score rows SHA-256: `08a558060e03616a82fcc5b5516ee9fc9ef919acef86625c5b47cb460aca38d1`.
- R8 Stage A seal SHA-256: `88c1e59885c58ee5e3d841ee3938f0f2d64e4d1356556e408affc1a3e969cc3a`; Stage B seal SHA-256: `27594444651b451b479530098e43c4584e748cb81af57f38420423723c36b1b6`.
- Private artifacts: `outputs/calibration/r8_correction_mechanism_20261103_a5/`; public aggregate: `aggregate_results.json`.
- Preserved prior attempts (none resumed or overwritten): `r8_correction_mechanism_20261103_a1` INITIALIZATION_FAILED_BEFORE_GPU: parent pre-created manifest directory before OutputBudget; no experiment child launched; `r8_correction_mechanism_20261103_a2` SUPERVISED_SMOKE_FAILED_BEFORE_TEST_ENCODING: the fixed-bank receipt evidence object was compared as if it were a digest string; `r8_correction_mechanism_20261103_a3` SMOKE_CHILD_COMPLETE_PARENT_GATE_FAILED: the parent parity verifier was defined after the `__main__` call; full conditions were not launched; `r8_correction_mechanism_20261103_a4` ALL_STAGE_A_SEALED_STAGE_B_FAILED: Stage B attempted to score RAW rows with SHUFFLE0-only mechanism controls; `KeyError: Scalar`. Artifacts preserved; no automatic restart.
- CUDA numerical equivalence is verified event by event against the sealed R7 G/GC1024 trajectories and Frozen/B1/B3 decisions.
