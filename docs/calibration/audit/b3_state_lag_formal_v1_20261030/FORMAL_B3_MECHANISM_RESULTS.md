# Formal B3 state-consistency mechanism ablation

## Table 1 — Overall comparison

Equal-class macro means are computed from four class means; seed dispersion is reported separately in the JSON artifact. Worst-256 metrics use only complete, non-overlapping 256-image windows. A positive FPR/FNR paired difference means A2 has more errors than the lagged arm.

| Scope | Arm | BA | FPR | FNR | Worst-256 BA | Coverage | Defect recall |
|---|---|---:|---:|---:|---:|---:|---:|
| Full stream | A2 | 0.7362 | 0.1042 | 0.4235 | 0.6530 | 1.0000 | 0.5765 |
| Full stream | A3 | 0.7324 | 0.0991 | 0.4360 | 0.6491 | 1.0000 | 0.5640 |
| Full stream | A4 | 0.7162 | 0.0818 | 0.4859 | 0.6407 | 1.0000 | 0.5141 |
| Common horizon t ≥ 1050 | A2 | 0.7413 | 0.1137 | 0.4038 | 0.6956 | 1.0000 | 0.5962 |
| Common horizon t ≥ 1050 | A3 | 0.7385 | 0.1109 | 0.4122 | 0.6943 | 1.0000 | 0.5878 |
| Common horizon t ≥ 1050 | A4 | 0.7233 | 0.0977 | 0.4557 | 0.6831 | 1.0000 | 0.5443 |

## Table 2 — Per-class paired effects over five support seeds

Each cell is the mean per-seed raw difference. BA and Worst-256 BA use A2 minus lagged arm; FPR/FNR differences are costs when positive.

| Class | BA A2−A3 | BA A2−A4 | FPR A2−A3 | FPR A2−A4 | FNR A2−A3 | FNR A2−A4 | Worst-256 BA A2−A3 | Worst-256 BA A2−A4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| audiojack | -0.0002 | 0.0018 | 0.0033 | 0.0160 | -0.0029 | -0.0196 | 0.0039 | 0.0018 |
| regulator | 0.0021 | 0.0126 | 0.0038 | 0.0172 | -0.0080 | -0.0424 | 0.0008 | 0.0249 |
| switch | 0.0049 | 0.0255 | 0.0035 | 0.0232 | -0.0133 | -0.0741 | 0.0061 | 0.0051 |
| rolled_strip_base | 0.0081 | 0.0401 | 0.0097 | 0.0330 | -0.0258 | -0.1132 | 0.0046 | 0.0174 |

## Table 3 — Mechanism evidence

| Contrast | Mean signed q divergence | Mean absolute q divergence | Negligible q fraction | Mean signed post-update theta divergence | Transitions with equal theta | Decisions changed | Next-event error difference after transitions (A2−lag) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 memory version | -0.0010 | 0.0011 | 0.7084 | -0.0010 | 815/1190 (0.6849) | 784 | 0 |
| 5 memory versions | -0.0049 | 0.0053 | 0.2378 | -0.0049 | 274/1190 (0.2303) | 3544 | -1 |

Reference-version gap counts (current version minus reference version): A2: {"0": 1190}; A3: {"1": 1190}; A4: {"1": 20, "2": 20, "3": 20, "4": 20, "5": 1110}.
Threshold movement/clipping: A2: 47/1190 clipped, 47 at the ±0.01 cap; A3: 47/1190 clipped, 47 at the ±0.01 cap; A4: 47/1190 clipped, 47 at the ±0.01 cap.
A changed q90 that leaves the clipped theta unchanged is counted separately from a changed binary decision. Since scores and Memory histories are shared, every inter-arm classification change is caused by the threshold positions.

## Table 4 — Integrity

| Item | Result |
|---|---|
| GPU calls | 0 |
| formal unit tests | 9/9 PASS; A1 suite 14/14 PASS including A0 verification |
| historical a2 parity | PASS (pre/post thresholds, predictions, event identity) |
| input identity sha256 | 3ffa3305a7876d4ce352b5db7467cc2463049e6f7ba9f6368a6f39be846ce840 |
| label access ledger sha256 | e17f4440c1b7eedb70c09895a30f9626891e8ff03e84e66a97fb21b7975cfd65 |
| label access phase | after prediction seal verification |
| masks and confirmation access | none |
| memory content changes | 1190 |
| official label provider calls | 8 across 2 evaluator attempts; 4 distinct class vectors |
| prediction seal | 8e1b4a78cbc4037386ff1bda83341b26695448a1d6f75bea9d8ddbad6953ff92 |
| score events | 76475 |
| source commit | a0e444215e2cfcef9dfeb796bd95cf85e9d2a0f2 |
| source sha256 set digest | 4a7f573aaf76733ef448221145d28da82989a639cfcdc648a51eeeb874a4c492 |
| trajectories | 20/20 |
| versioned c64 states | 1210 |

## Research interpretation

**H1_WEAK_OR_MIXED** — Effects are smaller than the frozen meaningful margin, heterogeneous, or accompanied by operating-point/robustness costs.

Current-state quantile differences can leave the clipped threshold unchanged; threshold differences can also leave every binary decision unchanged. The diagnostics report both stages separately. No monotonic deterioration with longer lag is assumed.

### Operating-point tradeoff

FPR and FNR differences are reported alongside BA. If A2's BA gain comes with a higher FPR or worse complete-window FNR, that cost remains part of the conclusion.

### Negative-result record

A null, mixed, or lag-favoring result is retained as valid development evidence. The lag manipulation's quantile, threshold, and decision divergence counts are included even when the primary BA contrast is null. No statistical significance, population guarantee, or confirmation-class claim is made.

### Limitations

This is a four-class development study with five correlated support-seed trajectories per class. It uses four deployed normal supports plus 64 additional protected C64 calibration normals, not a strict four-shot-total setting. Historical Stage 0.75-A/G2 source-byte parity remains blocked; A1's frozen-input A0 parity is a separate verification. No images were rescored, no GPU inference was run, and no masks or confirmation outcomes were accessed.

Machine-readable details, per-trajectory outcomes, paired effects, transition diagnostics, source hashes, and artifact checksums are in the versioned run directory recorded by the audit.
