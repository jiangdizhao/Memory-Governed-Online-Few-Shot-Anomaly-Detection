# R10 Targeted Signed kNN Baseline Comparison

## Status and scope

R10 completed as a RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC on the same exposed four-category development stream. It is not independent confirmation or a blind evaluation. The only new arms were Signed kNN with k=5, amplitude 0.02, and global FIFO capacities 256 and 1024. Support Seed 0, Shuffle Seed 0, the exact R9 query set, and the exact R9 event order were used.

Run ID: r10_signed_knn_baseline_20261103_a1. Parent: 22e6948915150e9e1b6492cca36b8385633b9ebf. Identity SHA-256: 4ed1d2bc4a59cf7dba98bb549da957135d52b7a1e7d940eb3794912b1f7d89ef.

Smoke: SMOKE_PASS_INCOMPLETE_NOT_EVALUATED for 16 events; it was incomplete and not evaluated. Full Stage A: SEALED_COMPLETE for 15,295 events and 459 legal queries. Stage B: SEALED_COMPLETE; independent second pass: PASS.

## Why Signed kNN was revisited

R2 previously reported switch-prefix AUROC 0.807035 for C0, 0.843083 for C1, 0.752198 for C2 at amplitude 0.20, and 0.841025 for C2 at amplitude 0.02. R2-EXP2A reported descriptive three-class means of 0.816168 for C1-Full and 0.816702 for C2 at 0.02. Those short development prefixes were already exposed; audiojack's prefix lacked defects, and amplitude 0.02 was selected after switch exposure. R10 asks whether this simple mechanism retains performance on the complete matched R9 stream.

R2-EXP1 included an earlier controlled amplitude comparison on an exposed switch trajectory. That historical comparison can support a local amplitude effect on that trajectory. R10 holds amplitude fixed at 0.02, so R10 does not identify an amplitude effect or validate 0.02 as an optimum.

## Reused baseline and execution

R9 run r9_correction_mechanism_20261103_a9 was preflighted from its outcome-independent Stage A seals and event score tapes. The reference supplied G1024 and GC1024 scores, decisions, thresholds, event identities, query decisions, and Greedy pre-event state digests. Full C1 was not rescored or updated in R10.

No compatible full TEST feature/map cache existed in R9: its private artifacts held score/feedback tapes and seals, not per-event Top-32 features or base maps. Audiojack's protected C64 feature cache was reused from the previously verified R5 cache. The other 192 C64 images were encoded once for this task; each TEST image was encoded once per full R10 stream. The 16-event smoke is counted separately.

Full R10 TEST DINO encodings: 15,295 across four full category passes. Smoke TEST encodings: 16. New C64 DINO encodings: 192. Support feature encoder invocations: 64. Full C1 reruns: 0.

Elapsed time, including smoke and two CPU Stage B reconciliations: 44.5 minutes. Minimum free space at launch: 269.4 GiB. Windows Job Object deadlines were 10 minutes for smoke and 60 minutes per full category child; no child was restarted.

## Parity

Exact event identity coverage: 15,295/15,295. G score parity against sealed R9: 0 mismatches, maximum absolute error 0.000000000 (tolerance atol=1e-05, rtol=0.0001). Query, identity, and Greedy-state mismatches: 0. The Greedy state digest was compared at every pre-event position, including each post-query transition before the next event.

## Ranking results

| Category | G AUROC | Full C1 AUROC | C2-256 AUROC | C2-1024 AUROC | G AP | Full AP | C2-256 AP | C2-1024 AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| audiojack | 0.739300 | 0.772655 | 0.756640 | 0.771739 | 0.606677 | 0.660450 | 0.629972 | 0.648955 |
| regulator | 0.766741 | 0.813097 | 0.780503 | 0.791218 | 0.585251 | 0.710714 | 0.607765 | 0.623320 |
| switch | 0.868575 | 0.884892 | 0.871896 | 0.884469 | 0.890019 | 0.906164 | 0.894494 | 0.906016 |
| rolled_strip_base | 0.953357 | 0.956829 | 0.955727 | 0.957529 | 0.973409 | 0.975962 | 0.974772 | 0.975651 |
| **Equal-category macro** | 0.831993 | 0.856868 | 0.841191 | 0.851239 | 0.763839 | 0.813323 | 0.776751 | 0.788485 |

Primary AUROC contrasts are equal-category macro differences. Positive Full−C2 favors Full C1; positive C2−G favors C2.

| Contrast | AUROC difference | AP difference |
|---|---:|---:|
| C2-256_minus_G1024 | 0.009198 | 0.012912 |
| C2-1024_minus_G1024 | 0.019245 | 0.024646 |
| FullC1_minus_C2-256 | 0.015677 | 0.036572 |
| FullC1_minus_C2-1024 | 0.005630 | 0.024837 |
| C2-1024_minus_C2-256 | 0.010047 | 0.011735 |

## Operating metrics

| Category / method | Controller | BA | FPR | FNR | TP | TN | FP | FN | Coverage | Worst 512 BA |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| audiojack / G1024 | Frozen | 0.542128 | 0.008337 | 0.907407 | 120 | 2498 | 21 | 1176 | 1.000000 | 0.518817 |
| audiojack / G1024 | B1 | 0.591564 | 0.018261 | 0.798611 | 261 | 2473 | 46 | 1035 | 1.000000 | 0.558016 |
| audiojack / G1024 | B3 | 0.674296 | 0.102025 | 0.549383 | 584 | 2262 | 257 | 712 | 1.000000 | 0.584166 |
| audiojack / FullC1-GC1024 | Frozen | 0.584698 | 0.012703 | 0.817901 | 236 | 2487 | 32 | 1060 | 1.000000 | 0.553860 |
| audiojack / FullC1-GC1024 | B1 | 0.615570 | 0.025804 | 0.743056 | 333 | 2454 | 65 | 963 | 1.000000 | 0.568807 |
| audiojack / FullC1-GC1024 | B3 | 0.689312 | 0.145296 | 0.476080 | 679 | 2153 | 366 | 617 | 1.000000 | 0.572512 |
| audiojack / C2-256 | Frozen | 0.541941 | 0.007940 | 0.908179 | 119 | 2499 | 20 | 1177 | 1.000000 | 0.521169 |
| audiojack / C2-256 | B1 | 0.608539 | 0.018261 | 0.764660 | 305 | 2473 | 46 | 991 | 1.000000 | 0.545552 |
| audiojack / C2-256 | B3 | 0.679972 | 0.109964 | 0.530093 | 609 | 2242 | 277 | 687 | 1.000000 | 0.572536 |
| audiojack / C2-1024 | Frozen | 0.550814 | 0.007940 | 0.890432 | 142 | 2499 | 20 | 1154 | 1.000000 | 0.528396 |
| audiojack / C2-1024 | B1 | 0.602763 | 0.017467 | 0.777006 | 289 | 2475 | 44 | 1007 | 1.000000 | 0.549905 |
| audiojack / C2-1024 | B3 | 0.693374 | 0.117110 | 0.496142 | 653 | 2224 | 295 | 643 | 1.000000 | 0.570258 |
| regulator / G1024 | Frozen | 0.511341 | 0.003243 | 0.974074 | 28 | 2766 | 9 | 1052 | 1.000000 | 0.497658 |
| regulator / G1024 | B1 | 0.604112 | 0.009369 | 0.782407 | 235 | 2749 | 26 | 845 | 1.000000 | 0.497658 |
| regulator / G1024 | B3 | 0.712748 | 0.091171 | 0.483333 | 558 | 2522 | 253 | 522 | 1.000000 | 0.559920 |
| regulator / FullC1-GC1024 | Frozen | 0.587858 | 0.005766 | 0.818519 | 196 | 2759 | 16 | 884 | 1.000000 | 0.538367 |
| regulator / FullC1-GC1024 | B1 | 0.691557 | 0.018739 | 0.598148 | 434 | 2723 | 52 | 646 | 1.000000 | 0.538367 |
| regulator / FullC1-GC1024 | B3 | 0.744875 | 0.084324 | 0.425926 | 620 | 2541 | 234 | 460 | 1.000000 | 0.575634 |
| regulator / C2-256 | Frozen | 0.510518 | 0.003964 | 0.975000 | 27 | 2764 | 11 | 1053 | 1.000000 | 0.494963 |
| regulator / C2-256 | B1 | 0.615480 | 0.007928 | 0.761111 | 258 | 2753 | 22 | 822 | 1.000000 | 0.494963 |
| regulator / C2-256 | B3 | 0.732112 | 0.098739 | 0.437037 | 608 | 2501 | 274 | 472 | 1.000000 | 0.551027 |
| regulator / C2-1024 | Frozen | 0.516819 | 0.004324 | 0.962037 | 41 | 2763 | 12 | 1039 | 1.000000 | 0.494963 |
| regulator / C2-1024 | B1 | 0.624945 | 0.009369 | 0.740741 | 280 | 2749 | 26 | 800 | 1.000000 | 0.494963 |
| regulator / C2-1024 | B3 | 0.738746 | 0.103063 | 0.419444 | 627 | 2489 | 286 | 453 | 1.000000 | 0.531703 |
| switch / G1024 | Frozen | 0.673310 | 0.010953 | 0.642427 | 713 | 1806 | 20 | 1281 | 1.000000 | 0.620619 |
| switch / G1024 | B1 | 0.742644 | 0.031763 | 0.482949 | 1031 | 1768 | 58 | 963 | 1.000000 | 0.670104 |
| switch / G1024 | B3 | 0.798252 | 0.104600 | 0.298897 | 1398 | 1635 | 191 | 596 | 1.000000 | 0.743135 |
| switch / FullC1-GC1024 | Frozen | 0.741687 | 0.030668 | 0.485958 | 1025 | 1770 | 56 | 969 | 1.000000 | 0.697428 |
| switch / FullC1-GC1024 | B1 | 0.775486 | 0.043812 | 0.405216 | 1186 | 1746 | 80 | 808 | 1.000000 | 0.710156 |
| switch / FullC1-GC1024 | B3 | 0.806282 | 0.092552 | 0.294885 | 1406 | 1657 | 169 | 588 | 1.000000 | 0.742791 |
| switch / C2-256 | Frozen | 0.696831 | 0.018072 | 0.588265 | 821 | 1793 | 33 | 1173 | 1.000000 | 0.630859 |
| switch / C2-256 | B1 | 0.739361 | 0.032311 | 0.488967 | 1019 | 1767 | 59 | 975 | 1.000000 | 0.697833 |
| switch / C2-256 | B3 | 0.802151 | 0.101314 | 0.294383 | 1407 | 1641 | 185 | 587 | 1.000000 | 0.740952 |
| switch / C2-1024 | Frozen | 0.711147 | 0.017525 | 0.560181 | 877 | 1794 | 32 | 1117 | 1.000000 | 0.668238 |
| switch / C2-1024 | B1 | 0.767191 | 0.038335 | 0.427282 | 1142 | 1756 | 70 | 852 | 1.000000 | 0.699678 |
| switch / C2-1024 | B3 | 0.816035 | 0.099124 | 0.268806 | 1458 | 1645 | 181 | 536 | 1.000000 | 0.740876 |
| rolled_strip_base / G1024 | Frozen | 0.817488 | 0.037500 | 0.327525 | 1698 | 1232 | 48 | 827 | 1.000000 | 0.781346 |
| rolled_strip_base / G1024 | B1 | 0.851194 | 0.043750 | 0.253861 | 1884 | 1224 | 56 | 641 | 1.000000 | 0.793446 |
| rolled_strip_base / G1024 | B3 | 0.886685 | 0.135937 | 0.090693 | 2296 | 1106 | 174 | 229 | 1.000000 | 0.804830 |
| rolled_strip_base / FullC1-GC1024 | Frozen | 0.877937 | 0.102344 | 0.141782 | 2167 | 1149 | 131 | 358 | 1.000000 | 0.777557 |
| rolled_strip_base / FullC1-GC1024 | B1 | 0.869368 | 0.094531 | 0.166733 | 2104 | 1159 | 121 | 421 | 1.000000 | 0.777557 |
| rolled_strip_base / FullC1-GC1024 | B3 | 0.894492 | 0.119531 | 0.091485 | 2294 | 1127 | 153 | 231 | 1.000000 | 0.803125 |
| rolled_strip_base / C2-256 | Frozen | 0.871105 | 0.059375 | 0.198416 | 2024 | 1204 | 76 | 501 | 1.000000 | 0.819281 |
| rolled_strip_base / C2-256 | B1 | 0.881621 | 0.062500 | 0.174257 | 2085 | 1200 | 80 | 440 | 1.000000 | 0.819281 |
| rolled_strip_base / C2-256 | B3 | 0.890227 | 0.132812 | 0.086733 | 2306 | 1110 | 170 | 219 | 1.000000 | 0.810136 |
| rolled_strip_base / C2-1024 | Frozen | 0.876842 | 0.058594 | 0.187723 | 2051 | 1205 | 75 | 474 | 1.000000 | 0.824716 |
| rolled_strip_base / C2-1024 | B1 | 0.872491 | 0.059375 | 0.195644 | 2031 | 1204 | 76 | 494 | 1.000000 | 0.824716 |
| rolled_strip_base / C2-1024 | B3 | 0.897698 | 0.125000 | 0.079604 | 2324 | 1120 | 160 | 201 | 1.000000 | 0.811364 |

Paired whole-physical-Source-Group bootstrap intervals are in the public aggregate for every category and registered contrast. They are descriptive conditional on these development streams; repeated camera views are not independent trials.

## FIFO and resource diagnostics

| Category | Capacity | Inserted patch rows | FIFO evictions | Final + / − | First saturation position | Corrected patch fraction | Mean absolute delta | Array bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| audiojack | 256 | 3648 | 3392 | 96 / 160 | 268 | 0.989515 | 0.01242162 | 4,204,800 |
| audiojack | 1024 | 3648 | 2624 | 320 / 704 | 1013 | 0.989515 | 0.01081815 | 16,819,200 |
| regulator | 256 | 3712 | 3456 | 32 / 224 | 268 | 0.989624 | 0.01223722 | 4,204,800 |
| regulator | 1024 | 3712 | 2688 | 256 / 768 | 1025 | 0.989624 | 0.01157471 | 16,819,200 |
| switch | 256 | 3680 | 3424 | 96 / 160 | 268 | 0.989529 | 0.01161426 | 4,204,800 |
| switch | 1024 | 3680 | 2656 | 448 / 576 | 1013 | 0.989529 | 0.00997775 | 16,819,200 |
| rolled_strip_base | 256 | 3648 | 3392 | 160 / 96 | 268 | 0.989488 | 0.01227700 | 4,204,800 |
| rolled_strip_base | 1024 | 3648 | 2624 | 640 / 384 | 1013 | 0.989488 | 0.01207615 | 16,819,200 |

Scoring and update times, score change from G, absolute correction quantiles, source exclusions, C64 rescoring counts, memory bytes, and B3 state transitions are in the private category seals.

## Interpretation

- Historical Signed kNN at amplitude 0.02 had equal-category macro AUROC 0.841191 at capacity 256 and 0.851239 at capacity 1024 on this stream.
- Capacity-matched Full C1 minus C2-1024 is 0.005630 AUROC; C2-1024 minus C2-256 is 0.010047.
- Whether Full C1 has a meaningful incremental advantage depends on these observed margins and the descriptive Source Group intervals; this is not independent evidence.
- R10 cannot attribute a difference to amplitude because amplitude was fixed. The earlier R2-EXP1 controlled amplitude comparison was limited to the already exposed switch development trajectory.
- The same development categories and query set were used in prior work. Results are retrospective and do not establish performance on confirmation classes.

## Source identity and sealed artifacts

R10 identity SHA-256: 4ed1d2bc4a59cf7dba98bb549da957135d52b7a1e7d940eb3794912b1f7d89ef. R9 identity SHA-256: f0a1145cb72aa917a7e34d7929c2ce98109aa2b147cdf1d5114f87547cb3ee20. R9 Stage A global seal SHA-256: c691fdfbc8b66c07aac5d4788efb85dbc6026d066c3118d827549504fc2e6ba0.

Stage B metrics payload SHA-256: 882cf24d27b263867cfed105ddb683896db957042d29152ed09de4a24bd77556. Private category score, feedback, and completion hashes are bound by the global seals. Public output excludes sample paths, raw labels, event-level scores, and feature vectors.

## Limitations

Amplitude 0.02 was chosen after exposure to switch development results. The 1024-anchor FIFO is a capacity-matched R10 extension, not the historical C2 configuration. This is one fixed shuffle and support seed. The paired Source Group intervals are descriptive conditional on the selected streams. No test-label-guided parameter selection occurred.
