# R4-FPSA-AUDIOJACK - Single-Class Causal Mechanism Validation

## Classification

`RETROSPECTIVE_SINGLE_EXPOSED_DEVELOPMENT_CLASS_MECHANISM_TEST` - one frozen audiojack stream; this is not independent confirmation or generalization evidence.

## Protocol and provenance

- Run: `r4_fpsa_audiojack_20261102_a1`; source parent: `90d15a8d3aced956fe50a43c3671b9c4ca870397`.
- Runner SHA-256: `c7878683439e50a2ffe9f1bd386fa2c29d1b93f01b214e3d3bd8fbc00053a610`; focused test SHA-256: `9f308f39cdf4b76bbe77cb2070b4b8b136502b35c8aed69365f723ae8e3904a2`.
- Audiojack class JSON SHA-256: `03333ade4def8710688fd8f23d568a1aaad55629115c36cf23662b0aa57fd267`.
- Q-A-TS Seed 37 SHA-256: `cb1b8b4342b23dc520a45ef72c014ceb3ffa1e976046aef726fcc0aaa45504d5`; C64 score tape SHA-256: `ad6edc2677809f1f0c455f80ea8aa002e9123fd9f19711936b5c8501696fdd93`.
- Structured fixed Support SHA-256: `837fa5153e852b617e1d0ecf563509bca037f5c0cab29c48b06befd21ce02673`.
- Stage A seal SHA-256: `6be6e9aac4daa7a5b931e8fe88d5ed731d44f9fbd4b787dc371f0a19a59d3af5`; score ledger SHA-256: `815be4d306f698b4789c2695729ce5885e2e3a9c198e2890c7d415d728bce9dd`; queried feedback ledger SHA-256: `ac828b9a71c0a1f418165071df19e491c262f831bd1b6715f2f7d77d4bd07b23`.
- Focused CPU tests: `16 passed, 4 warnings in 6.72s`. Smoke: 256 events under Windows Job Object, with queried feedback/update path exercised; no Stage B.
- DINO source, checkpoint and loaded model-state fingerprints are bound in the local run manifest and Stage A receipt.
- Stage A accessed only frozen audiojack metadata, Support/C64, the fixed query tape, and legal feedback after each score seal. Stage B projected audiojack image-level outcomes after the complete Stage A seal; masks were not opened.
- The historical R3 manifest retains its prior top-level `INCOMPLETE` status, but its audiojack category score digest, Stage A seal, Stage B seal, and complete seal were validated for this comparison.

## R3 numerical parity

- G: 3815 events; max absolute difference `0`.
- GC: 3815 events; max absolute difference `0`.
- Source-group/query identities and Greedy write/accepted-patch events: source identity parity True; write-event parity True; accepted-patch parity True.
- GC-FPSA vs GC through position 1305: 1306 events; max absolute difference `0`.

## Prequential results

Full stream (3,815 events), at the frozen C64 q90 threshold:

| Arm | BA | FPR | FNR | TP | TN | FP | FN | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G | 0.541820 | 0.002779 | 0.913580 | 112 | 2512 | 7 | 1184 | 0.687811 |
| GC | 0.537400 | 0.001588 | 0.923611 | 99 | 2515 | 4 | 1197 | 0.685190 |
| GC-FPSA | 0.593792 | 0.010719 | 0.801698 | 257 | 2492 | 27 | 1039 | 0.720577 |

GC-FPSA's full-stream BA was 0.056391 higher than GC. Its FPR rose from 0.001588 to 0.010719 while FNR fell from 0.923611 to 0.801698. This is the observed operating-point trade-off at the frozen threshold.

Post-anomalous-feedback window (positions 1306-3814):

| Arm | Events | BA | FPR | FNR | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G | 2509 | 0.541764 | 0.003223 | 0.913249 | 110 | 1237 | 4 | 1158 |
| GC | 2509 | 0.537838 | 0.001612 | 0.922713 | 98 | 1239 | 2 | 1170 |
| GC-FPSA | 2509 | 0.590874 | 0.020145 | 0.798107 | 256 | 1216 | 25 | 1012 |

## Sign allocation and borrowing

- Final GC occupancy: positive 1, negative 255 (K=256).
- Final GC-FPSA occupancy: positive 91, negative 165 (K=256).
- Final FPSA target: positive 90, negative 166; targets are not guaranteed occupancy.
- Borrowing opportunities: 365; successful cross-sign replacements: 93; retention-rejected borrowings: 272.
- Mechanism criterion: PASS for quota activation, successful cross-sign replacements, and a delayed positive-sign occupancy trajectory. Performance criterion: PASS for the prespecified positive full-stream BA contrast over GC; this is a single-stream descriptive result.
- Trajectory is reported in fixed 512-event bins to avoid publishing individual queried labels.

## Interpretation and limitations

BA contrasts (GC-FPSA-GC, GC-FPSA-G, GC-G): `{'GC-FPSA_minus_G': 0.051971528727351224, 'GC-FPSA_minus_GC': 0.05639148643151559, 'GC_minus_G': -0.004419957704164368}`.
Mechanism success requires successful cross-sign borrowing and a changed delayed-feedback occupancy trajectory. Performance success requires positive full-stream BA difference over GC. Any FPR/FNR trade-off is reported as measured; this is one exposed audiojack development stream and does not establish transfer, independent significance, or safety.
The 4 GiB application OutputBudget is pre-write accounting; no OS-level hard storage isolation is provided. The Windows Job Object enforces the 120-minute process-tree deadline. The run is not resumable and was launched in a fresh namespace.

## Source-controlled artifact scope

Only this report, its redacted aggregate, the isolated R4 runner, and its focused tests are committed. Private score/feedback ledgers remain in the local output namespace.
