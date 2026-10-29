# Stage 0.75-B B8 mechanism ablation report

**Status:** synthetic-only; no later-stage method registry change.
**Contract:** v3; see [`STAGE075B_SYNTHETIC_RESULTS_REPORT.md`](STAGE075B_SYNTHETIC_RESULTS_REPORT.md).

## Fair comparison

All seven arms used the same causal engine, score vector, label-blind tape, memory writes, and current-version evidence. B3+Offset and B8-Full used the same distinct queried source groups. Moving thresholds shared the raw `0.01` cap. Thresholds never controlled queries or memory. AUROC/AUPRC were computed once from the shared scores. Synthetic three-layer LSO does not establish native Greedy source exclusion.

## Aggregate ablation

Unweighted means over the 42 defined BA trajectories:

| Arm | Mean BA | Mean FPR | Mean FNR | Mean TV | Mean updates | Mean state bytes |
|---|---:|---:|---:|---:|---:|---:|
| B3 | 0.746193 | 0.112206 | 0.395408 | 0.032227 | 20.95 | 1391 |
| B7 | 0.750430 | 0.113965 | 0.385174 | 0.069307 | 7.93 | 5698 |
| B3+Offset | 0.760482 | 0.110061 | 0.368975 | 0.251637 | 25.62 | 955 |
| B8-M | 0.744957 | 0.083745 | 0.426340 | 0.057903 | 20.95 | 6079 |
| B8-S | 0.746193 | 0.112206 | 0.395408 | 0.032227 | 20.95 | 6493 |
| B8-P | 0.743525 | 0.104576 | 0.408374 | 0.224640 | 29.57 | 6280 |
| B8-Full | 0.746193 | 0.112206 | 0.395408 | 0.032227 | 20.95 | 6634 |

B8-Full matched B3 in every threshold digest and every defined BA. It was lower than B3+Offset in 35/42 paired rows, higher in 7, tied in 0; mean difference `-0.014289`. This is generator-specific descriptive evidence.

## Mechanism findings

- **B8-M:** S2 left the C64 median unchanged but moved higher-q90 from 57.0 to 57.5; median transport misses this tail change.
- **B8-S:** its mean BA equaled B3 and the full-matrix threshold traces matched B3.
- **B8-P:** changed some thresholds but its mean BA was below B3; finite rank coordinates do not identify unique raw decision boundaries.
- **B8-Full:** v3's percentile equation regression passed, but its matrix threshold trace still matched B3. Across 41 final two-class snapshots, mean heuristic `U` was 1.2222; final best penalized objectives ranged -1.4828–-0.5385. This supports a negative finding for these frozen settings, not a universal impossibility claim.
- **B3+Offset:** mean BA was 0.760482 versus B3 0.746193; this is not a deployment recommendation.

## Candidate decision

The preliminary synthetic outcome is `SYNTHETIC_MECHANISM_NOT_SUPPORTED` for the frozen Full arm, but the required one-component-at-a-time Control E is incomplete. The research decision is therefore `BLOCKED_BY_SCIENTIFIC_OR_ENGINEERING_FAILURE`, not a completed `SIMPLER_BASELINE_SUFFICIENT` decision. The findings do not justify proposing Stage 0.75-C. Any new B8 settings require researcher review and a separately versioned, preregistered experiment.
