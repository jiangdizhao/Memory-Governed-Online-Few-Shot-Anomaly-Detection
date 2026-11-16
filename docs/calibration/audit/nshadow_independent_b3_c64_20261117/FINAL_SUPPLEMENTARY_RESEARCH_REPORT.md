# N-shadow with independent B3/C64: supplementary results

## Status and method

Completed as a post-hoc exploratory supplementary analysis over the existing sealed streams: 90 Real-IAD category-condition trajectories (P0/P1/P2) and 12 VisA P0 trajectories, 345,917 image events total.

The N-shadow detector remains the original shared F-B Greedy memory plus its independent normal-only C1 correction memory. It inherits only F-B's already executed query events, updates C1 only from verified-normal feedback, ignores defects for shadow C1, and receives no independent query decisions. The supplementary controller is a separate R7 B3 StateConsistentQuantile initialized and refreshed from the 64 registered protected C64 images, q90 with method `higher`, minimum 16 normal anchors, and a maximum 0.01 threshold movement per source-state transition.

The sidecar reuses original sealed N-shadow scores for non-query events and reconstructs mutable state from the exact purchased F-B feedback records. At each inherited F-B query event, a native F-B and N-shadow score replay was computed and compared with the original sealed score within a predeclared 1e-5 tolerance. State digests and versions were checked on every event. Each Real-IAD five-view group (and each registered VisA atomic unit) was scored and fsynced before its corresponding private purchased feedback was consumed. Official image-level outcomes were opened only after all 102 supplementary seals; masks were not accessed.

The central exploratory contrast is Δ BER = BER(N-shadow+B3) − BER(F-B); positive values favor F-B. This is post hoc and does not replace the original AUROC/AP-only N-shadow analysis or establish causal attribution beyond this registered feedback history.

### Real-IAD: 30 categories, conditions averaged within category

| Method | AUROC | AP | BER | FPR | FNR | BA |
|---|---:|---:|---:|---:|---:|---:|
| N-shadow+B3 | 0.8681 | 0.8496 | 0.1899 | 0.1329 | 0.2469 | 0.8101 |
| F-B | 0.8935 | 0.8843 | 0.1779 | 0.1344 | 0.2215 | 0.8221 |
| F-M | 0.8814 | 0.8673 | 0.1876 | 0.1328 | 0.2424 | 0.8124 |

| Contrast | Mean difference | Paired category-bootstrap 95% CI |
|---|---:|---:|
| Δ BER (N-shadow+B3 − F-B) | 0.0119 | [0.0078, 0.0164] |
| Δ FPR (N-shadow+B3 − F-B) | -0.0015 | [-0.0075, 0.0041] |
| Δ FNR (N-shadow+B3 − F-B) | 0.0254 | [0.0171, 0.0346] |

Categories where N-shadow+B3 has lower BER than F-B: 3/30.
Categories where F-B has lower BER than N-shadow+B3: 27/30.

### VisA: 12 categories, P0

| Method | AUROC | AP | BER | FPR | FNR | BA |
|---|---:|---:|---:|---:|---:|---:|
| N-shadow+B3 | 0.9389 | 0.9396 | 0.1322 | 0.0951 | 0.1692 | 0.8678 |
| F-B | 0.9465 | 0.9487 | 0.1251 | 0.0893 | 0.1608 | 0.8749 |
| F-M | 0.9356 | 0.9395 | 0.1388 | 0.0918 | 0.1858 | 0.8612 |

| Contrast | Mean difference | Paired category-bootstrap 95% CI |
|---|---:|---:|
| Δ BER (N-shadow+B3 − F-B) | 0.0071 | [-0.0046, 0.0196] |
| Δ FPR (N-shadow+B3 − F-B) | 0.0058 | [-0.0075, 0.0183] |
| Δ FNR (N-shadow+B3 − F-B) | 0.0083 | [-0.0067, 0.0250] |

Categories where N-shadow+B3 has lower BER than F-B: 5/12.
Categories where F-B has lower BER than N-shadow+B3: 5/12.

## Interpretation

Real-IAD mean Δ BER was 0.0119 (paired 30-category bootstrap 95% CI [0.0078, 0.0164]).
VisA mean Δ BER was 0.0071 (paired 12-category bootstrap 95% CI [-0.0046, 0.0196]).

A positive Δ BER indicates the independent B3 threshold leaves N-shadow with higher operating BER than F-B; a negative value indicates lower N-shadow+B3 BER. FPR and FNR differences are reported separately because a BER change can conceal an operating-point trade-off. F-M is descriptive only because it follows a different query/feedback history.

This analysis identifies the realized post-hoc behavior of the independently calibrated normal-only shadow on these streams. It does not identify counterfactual learning value for unqueried observations, prove that defect-side correction alone caused any performance difference, establish a universal threshold policy, or establish VisA physical independence from Real-IAD.

## Computation and limitations

Measured Stage A replay time: 10087.2 s across 102 trajectories. End-to-end execution time through Stage B and aggregation: 10828.3 s. The sidecar avoided recomputing the four-arm stream: only the already purchased F-B query images were re-encoded for native score parity. Current-state C64 scoring ran when N-shadow's complete scoring-state digest changed.

Per-category results, temporal segments, state-transition records, Stage B metrics, confidence intervals, resource measures and fidelity checks are saved beside this report. Real-IAD and VisA were analyzed separately with categories as the bootstrap unit.
