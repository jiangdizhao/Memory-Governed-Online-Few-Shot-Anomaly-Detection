# Stage 0.75-B full synthetic results

**Run:** `outputs/calibration/stage075b_synthetic_20261029/`
**Primary contract/method:** candidate v2, frozen before the full matrix.
**Scope:** synthetic score streams only; this is not Real-IAD evidence or native memory/scorer parity.

## Design and execution

The frozen matrix contains S0–S14 plus control A, each with 1,024 observations and seeds `37`, `101`, and `2029` (48 complete scenario/seed trajectories). Every trajectory used the label-blind 3% Q-A-TS tape at seed 37: 31 queried image groups out of 1,024 (3.027%). All threshold-only arms received one shared score path, identical memory writes and query positions. Only queried verified normals wrote the synthetic shared memory. The memory adapter has three source-attributed layers and capacity 256 per layer; it is a protocol fixture, not a Greedy or VisionAD implementation.

The runner executed the complete matrix twice. All 48 canonical trajectory digests matched. It also ran four exact checkpoint/resume cuts at events `1`, `129`, `401`, and `733`, matching the uninterrupted trace, metrics, and controller state. The complete calibration test suite passed 132 tests; compilation and independent artifact verification passed.

## Mean balanced accuracy by scenario

Values are unweighted descriptive means over three synthetic seeds. `—` means BA is undefined because the stream contains one class. No confidence intervals or significance claims are attached.

| Scenario | B3 | B7 | B3+Offset | B8-M | B8-S | B8-P | B8-Full | Full − B3 | Full − Offset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A no-gain control | 0.4987 | 0.4991 | 0.4996 | 0.4993 | 0.4987 | 0.4996 | 0.4987 | 0.0000 | −0.0009 |
| S0 exact B3 fallback | — | — | — | — | — | — | — | — | — |
| S1 monotone coordinates | 0.7919 | 0.7995 | 0.8108 | 0.7915 | 0.7919 | 0.7791 | 0.7919 | 0.0000 | −0.0189 |
| S2 sparse C64 tail | 0.7839 | 0.7963 | 0.8146 | 0.7824 | 0.7839 | 0.7913 | 0.7839 | 0.0000 | −0.0307 |
| S3 asymmetric classes | 0.8272 | 0.8300 | 0.8523 | 0.8228 | 0.8272 | 0.8306 | 0.8272 | 0.0000 | −0.0251 |
| S4 delayed defects | 0.8123 | 0.8249 | 0.8374 | 0.8044 | 0.8123 | 0.8134 | 0.8123 | 0.0000 | −0.0251 |
| S5 defect-only | — | — | — | — | — | — | — | — | — |
| S6 scarce/correlated | 0.8176 | 0.8192 | 0.8234 | 0.8159 | 0.8176 | 0.8093 | 0.8176 | 0.0000 | −0.0058 |
| S7 score/concept staleness | 0.6976 | 0.6999 | 0.7165 | 0.6945 | 0.6976 | 0.6978 | 0.6976 | 0.0000 | −0.0190 |
| S8 C64 mismatch | 0.6123 | 0.6116 | 0.6200 | 0.6136 | 0.6123 | 0.6222 | 0.6123 | 0.0000 | −0.0077 |
| S9 movement/rejoin | 0.7038 | 0.7066 | 0.7109 | 0.7018 | 0.7038 | 0.6950 | 0.7038 | 0.0000 | −0.0071 |
| S10 numerical edges | 0.6964 | 0.7025 | 0.7057 | 0.7089 | 0.6964 | 0.6964 | 0.6964 | 0.0000 | −0.0092 |
| S11 provenance/LSO | 0.8036 | 0.8056 | 0.8190 | 0.8023 | 0.8036 | 0.7924 | 0.8036 | 0.0000 | −0.0154 |
| S12 common-score invariance | 0.8080 | 0.8109 | 0.8215 | 0.8040 | 0.8080 | 0.7941 | 0.8080 | 0.0000 | −0.0135 |
| S13 checkpoint/resume | 0.7865 | 0.7866 | 0.7977 | 0.7842 | 0.7865 | 0.7866 | 0.7865 | 0.0000 | −0.0112 |
| S14 non-identifiability | 0.8068 | 0.8135 | 0.8174 | 0.8036 | 0.8068 | 0.8017 | 0.8068 | 0.0000 | −0.0106 |

Across the 42 scenario/seed rows with both classes, mean BA was `0.746193` for B3, `0.750430` for B7, `0.760482` for B3+Offset, `0.744957` for B8-M, `0.746193` for B8-S, `0.743525` for B8-P, and `0.746193` for B8-Full. B8-Full minus B3 averaged exactly `0.000000`; B8-Full minus B3+Offset averaged `−0.014289`. B3+Offset had higher BA in 35 rows and lower BA in 7 rows. These are descriptive results under this generator, not inferential comparisons.

The full-arm raw threshold trajectory digest matched B3 in all 48 rows; there were no full-arm threshold deviations. The reliability heuristic averaged `1.2222` over 41 final snapshots with both queried classes, ranging `1.0361` to `1.9595`. The best unpenalized empirical delta-J averaged `0.1989` and peaked at `0.6310`; no final snapshot had a positive penalized objective. B8-Full therefore followed its frozen fallback exactly. This is a negative mechanism result, not evidence of no possible useful B8 design.

## Scenario and control outcomes

| ID | Execution status | Result / falsification |
|---|---|---|
| S0 | `PASS` | Actual B3 and B8-Full thresholds and predictions matched at every event with no queried defects; BA remained null for the one-class stream. |
| S1 | `PASS` | Finite rank invariance held under common translation, positive affine, and strictly increasing nonlinear transforms, including ties. No BA invariance was inferred. |
| S2 | `PASS` | C64 median shift was exactly zero while higher-q90 moved by 0.5 in the fixed tail fixture. |
| S3 | `EXECUTED` | Asymmetric class transforms were included. B8-Full stayed at B3; normal-reference coordinates did not identify defect operating risk. |
| S4 | `EXECUTED` | The query tape produced a long normal-only prefix before queried defects. Predictions preceded disclosure; adaptation remained bounded and B8-Full stayed at B3. |
| S5 | `PASS` | Defect-only stream caused zero verified-normal writes and no memory-version change. BA was null; FPR was unavailable. |
| S6 | `PASS` | Effective source ladder was 1, 2, 3, 7, 8, 14, and 20. Rescoring did not add sources; repeated groups were rejected. |
| S7 | `EXECUTED` | Defect score concept shift and memory-score refresh were represented separately. Fresh scores did not imply fresh concept evidence; no age decay was used. |
| S8 | `PASS` | C64 mismatch was exposed; C64 remained a coordinate reference, not the queried-stream FPR denominator. |
| S9 | `PASS` | A direct adversarial evidence fixture forced B8 departure and confirmed gradual return to B3 under the 0.01 move cap. |
| S10 | `PASS` | Exact `>=` ties, rank saturation, zero-IQR floor, near-equal floats, and nonfinite rejection were checked. |
| S11 | `PASS` | Synthetic all-three-layer source exclusion, cross-view group duplication rejection, and defect duplicate rejection passed. This does not establish native source-level LSO parity. |
| S12 | `PASS` | Shared score/query/memory path was used across all arms; ranking metrics were computed from the one shared score vector. |
| S13 | `PASS` | Four checkpoint/resume cut points reproduced the uninterrupted trace and final state exactly. |
| S14 | `PASS` | Paired worlds were identical through event 512 and differed only in future defect outcomes; controller decisions were identical through the shared prefix. |
| A | `PASS` | Identical class-conditional score laws yielded BA near 0.5; B8-Full remained B3. |
| B | `PASS` | B3+Offset and B8-Full used the same tape, distinct evidence groups, LSO policy, and movement cap. |
| C | `PASS` | Implemented as the S14 paired-world prefix test. |
| D | `PASS` | Eight queried sources and six threshold candidates yielded training delta-J `+0.5000` but independent future delta-J `−0.2857`. |
| E | `PASS` | B8-M/S/P and Full were evaluated on the same frozen scenarios, evidence, query positions, and score trajectory. |

`EXECUTED` means the complete preregistered diagnostic ran and its results are reported, but the scenario does not have a binary implementation assertion that would justify calling the scientific expectation a PASS. All 15 mandatory S0–S14 scenarios ran for all three seeds; the matrix completeness gate passed.

## Runtime and memory

The first matrix pass took `70.36` seconds total, averaging `1.466` seconds per scenario/seed. A representative traced run took `6.328` seconds and peaked at `16,126,793` Python-tracked bytes (about 15.4 MiB). The traced runtime includes allocator instrumentation; this is not process RSS or GPU memory. The full release runner completed in about 488 seconds including 132 tests and two matrix passes.

## Data and interpretation boundary

Per-seed confusion counts, FPR/FNR, threshold movement, evidence counts, source digests, runtime, and checkpoint identities are in `stage075b_results.json`. One-class BA remains JSON `null`. No temporal-image confidence intervals were computed. The synthetic memory is not the native Greedy writer, and these metrics do not support Real-IAD superiority, deployment-risk claims, or authorization for Stage 0.75-C.
