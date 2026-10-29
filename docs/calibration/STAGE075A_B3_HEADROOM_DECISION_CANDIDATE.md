# Stage 0.75-A B3 Headroom — Research Decision Candidate

**Candidate classification:** `MIXED_BY_CATEGORY`

## Basis

- The source-faithful replay uses all 20 raw trajectories and matches the original frozen and current-q90 metrics. B3 and read-only current q90 differ by only -0.00097 equal-class macro BA on the full stream.
- Mean retrospective all-remaining `H_beyond_q90` is materially different by class: Audiojack 0.0188, Regulator 0.0650, Switch 0.0293, and Rolled Strip Base 0.0202 BA.
- Lower-threshold local actions are frequently best in the retrospective grid, while Rolled Strip Base also has more KEEP/RAISE cases. Regulator's headroom exceeds Rolled Strip Base's; neither result specifies an online action.
- The artifact set cannot identify current-snapshot, source-excluded scores for queried anchors. The central online-feasibility question is therefore unresolved even though replay and evaluator-only headroom are complete.

## Recommendation for researcher review

Retain actual B3 as the reference. Do not implement or tune B8 from these results. If further study is authorized, first preregister a measurement-only extension that stores legal, source-aware query-anchor rescores at each frozen memory version. Then evaluate whether sparse feedback can identify a bounded local correction on a separate held-out design. Keep percentile transport out of that first test unless new evidence shows B3-to-q90 lag is a material error source.

All retrospective Oracle values are in-sample and must not be called achievable or unbiased. This candidate does not approve Stage 0.75-C, Stage 1, Stage 2 or any confirmation-class analysis.

Gate tally at generation: `{'PASS': 17, 'FAIL': 0, 'BLOCKED': 0, 'SKIPPED': 0}`. Optional current-anchor identifiability is recorded separately as `BLOCKED` in the audit matrix; mandatory replay and integrity gates are reported individually.
