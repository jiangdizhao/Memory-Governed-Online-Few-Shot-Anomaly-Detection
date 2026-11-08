# Experiment C — Memory-Conditioned Evidence Acquisition

## Status and scope

This is the prospective four-category development protocol for separating a
predicted-normal asymmetric query band, its past-only average gate, and the
current sample's Correction Memory Evidence. It is not independent
confirmation. Implementation source lives in `exp_c_memory_evidence.py`,
`exp_c_historical_parity.py`, and `exp_c_full_run.py`.

This scoped authorization permits only Experiment C implementation, focused
synthetic CPU validation, and a local commit. It does not permit a Real-IAD
campaign, GPU execution, protected confirmation outcomes or masks, historical
R7/R9/R10/E6/D2/B/FS4 edits, or a push. No campaign was executed while creating
this implementation.

## Fixed registry

The arm order and formulas are:

| ID | Policy |
| --- | --- |
| `RAND37` | Historical D2 identity-hashed Random37 |
| `MARGIN37` | Historical D2 Margin37, including its 3.7% cold start, past-only q=.963 threshold and tie lottery |
| `B` | `B` |
| `BD` | `B + D` |
| `BN_BAND` | `B + N + D_band` |
| `BN_AVG_GATE` | `B + N + g_bar(t-1) * D_band` |
| `BND` | `B + N + D` |

All `BEAComponents` values come from the existing E6 `compute_bea_components`
implementation. C derives only:

```text
D_band = 0.35 * (0 if pred_defect else p_defect * gate_far * gate_near)
g_sample = 1 - correction_evidence
D = D_band * g_sample
```

The adapter checks the last equality to absolute tolerance `1e-12`. The
AverageGate is an arm-local deque of the preceding 256 `(D_band, 1-evidence)`
records, weighted by `D_band`. Empty history and zero total weight yield 1.0.
The event's record is appended only after all seven event decisions are fsynced
and all queried feedback processing for that event has finished.

BEA uses the historical D2 price controller: initial 0.80, target 0.03, 256
executed actions, learning rate 0.02, clipping `[0.05, 3.0]`, strict
`objective > current price`, and price update after the current action.
`RAND37` and `MARGIN37` are the existing D2 controllers without changes to
their 0.037 baseline semantics, identity hashing, cold start, percentile,
tie-breaking, or Source Group suppression.

## Online data and feedback protocol

The registered categories and event counts are the D2 four-category set:
audiojack 3815, regulator 3855, switch 3820, rolled_strip_base 3805. Inputs use
the existing fixed Support Seed 0, four normal Supports plus 64 disjoint C64
TRAIN normals, R7 Shuffle Seed 0, official Source Groups, frozen DINOv2 and
native R7 Full C1 K1024 / Greedy K1024 / C64-B3 / Coverage Memory semantics.
Every arm owns independent mutable scorer, correction, coverage, controller,
query-group and private-feedback state. A TEST event is encoded once and its
frozen features are shared across arms.

The public seven-arm event row is label-free and fsynced before any label is
requested. The on-demand reader bound to an arm can disclose only that arm's
executed query. Each arm can query a Source Group once. Other arms receive no
supervised feedback when one arm queries. Unqueried events follow the existing
D2/R7 unsupervised writer action. Stage A does not project the full TEST
outcome stream.

## Historical D2 parity and Stage B

Before Stage B, the future runner validates all four C seals and compares the
five unchanged arms (`RAND37`, `MARGIN37`, `B`, `BD`, `BND`) to sealed D2 public
artifacts: event order and identities, Support/C64 identities, pre-event
scores, thresholds, state versions/digests, components, proposals, group
suppression, executed positions, and query counts. Numeric score/threshold
comparisons use the D2 manifest's registered tolerances. The verifier reads
private D2 feedback only to validate its file digest; it never parses those
feedback rows.

Only after all C and D2 Stage A seals and public parity pass does the separate
CPU Stage B project the official JSON image-level labels. It recomputes D2's
five common-arm metrics from those same official labels and compares them with
the sealed D2 aggregate evaluation. It does not open historical label files.
If historical public artifacts or the sealed D2 Stage B evaluation are absent,
parity is unavailable and the future campaign cannot be reported complete.

Primary outcome is category-equal macro prequential BER. Secondary metrics
are per-category and macro AUROC, AP, BA, FPR, FNR, worst complete 256-event
BER windows, query rate/count, queried-label composition, four-segment query
timing, maximum query-free intervals, first queried defect, group suppression,
state changes, occupancy, runtime, memory, and persisted bytes. Registered
BER contrasts are `BN_AVG_GATE - BND`, `BN_BAND - BND`, `BD - BND`,
`MARGIN37 - BND`, and `B - BND`, each with query-rate differences and budget
classification. Budgets count as matched only when the absolute rate
difference is at most 0.003, per category and macro.

Stage B reports 1,000 paired Source Group bootstrap draws (seed 0) as
descriptive conditional-trajectory intervals. They do not establish
independence across fresh online training histories. A read-only diagnostic
also computes sample-gate, band, and average-gate proposals at the same BND
pre-event state and price. It executes no counterfactual query and changes no
state.

## Future execution controls and resources

The future `--full-run` entry point accepts only a separate
`exp-c.full-run-authorization.v1` JSON bound to the exact Git commit, this
document's SHA-256, immutable Run ID, independent-review `PASS`, researcher
identity, four categories, Stage A/B scope, and explicit denial of confirmation
data and masks. It refuses a reused output
namespace, automatic retry, resume, overwrite, or a modified source tree.
Stage A categories and CPU Stage B run as Windows Job Object children; startup
failures preserve exception type, message, and traceback. Stage B requires all
four validated C seals and D2 public parity first.

Resource contract: at least 20 GiB free disk at launch; at most 2 GiB across
score and private-feedback streams per category (8 GiB total upper bound); a
seven-day per-category child deadline and 24-hour CPU Stage B child deadline.
Host RSS and CUDA allocator high-water marks are recorded when available.
The existing Windows Job Object helper enforces process-tree lifetime, but it
does not set a hard RAM or GPU memory cap; the runner records start/end free
device memory, allocator high-water marks, and host RSS. A future authorized
preflight must stop if the measured device or disk headroom is materially
insufficient. These are guardrails, not measured Experiment C costs. Seven independent
native states and per-event native scoring make Stage A GPU-heavy; no runtime
or peak-memory estimate is claimed before authorized execution.

P1/P2 shuffle sensitivities, alternate budgets, SignedKNN transfer, and
confirmation datasets are out of scope and require separate authorization.
