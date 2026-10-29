# Stage 0 sixth-pass implementation report

**Release status: `WIP / STAGE0_BLOCKED`.** This report records engineering
and synthetic evidence for independent research review. It is not scientific
validation, and it does not authorize Stage 1 or Stage 2.

## Scope and protected boundaries

The implementation was completed on `calibration-stage0-contract-20261029`,
starting from `ced068258709841df11c59f6a4a1accafa24246d`. Historical v22,
VisionAD, P0/P1, risk-safe, and prior experiment files were left unchanged.
The protected-source check covers 13 files and reports byte equality with the
reviewed baseline, including `experiments/risk_safe/shadow_memory.py`.

Only the four development classes were included in metadata preflight:
`audiojack`, `regulator`, `switch`, and `rolled_strip_base`. Mint/PCB
confirmation outcomes were not opened. No Stage 1 or Stage 2 experiment,
Real-IAD performance tuning, image-mask evaluation, or confirmation-class
analysis was run.

The protocol preserves four deployed supports plus **64 additional** protected
C64 calibration normals. C64 is not a strict four-shot-total regime. The
native shared Greedy bank is verified-normal-only, controller-independent,
and K=256 per layer.

## Implemented work

### Sun-CS Algorithm 3

B6 is registered as
`PAPER_EQUATIONS_WITH_DECLARED_ENDPOINT_CONVENTIONS`. It implements the
causal score-quantile target, Equations (1) and (2), Algorithms 1–3, exact
forward-prefix/backward-suffix confidence-set intersection checks, strict
upper-bound anomaly decisions, benign lower-bound decisions, abstention,
change-event abstention/reset, and checkpoint restore.

The independent small-sequence reference and the optimized Algorithm 2
implementation agree event-by-event for both endpoint policies. Algorithm 3
traces and event-boundary resume tests pass. The explicit implementation
conventions are in [SUN_CS_A3_DECLARED_CONVENTIONS_v1.md](SUN_CS_A3_DECLARED_CONVENTIONS_v1.md):

- Primary `empirical_extrema`: q=0 selects the sample minimum and q=1 selects
  the sample maximum. Interior midpoint quantiles use the declared one-based
  floor/ceiling order-statistic ranks; a zero floor rank extends to the first
  order statistic.
- Empty past: emit `ABSTAIN_COLD_START` before appending the current score.
- Detected change: abstain, reset, then append the current score as the first
  event in the new segment.
- Secondary `conservative_extended_real`: explicit infinite endpoints are
  encoded as structured JSON values and are a separate sensitivity variant.

No verified author implementation was located, so author-code trace parity is
not claimed. Sun-CS targets a score-distribution quantile, not Real-IAD
physical defects. Dynamic-memory dependence, physical-label FPR/FNR, and
selective feedback do not inherit the source paper's guarantees.

The exact implementation has O(n log n) time and O(n log n) auxiliary space
for one Algorithm 2 invocation. Algorithm 3 recomputes the exact detector on
each past prefix, giving O(T^2 log T) worst-case cumulative time and
O(T log T) peak auxiliary space for a length-T no-reset segment. The measured
3,815-event constant-score synthetic no-reset run took **1,234.22 seconds**
with tracemalloc enabled and used **1,101,999 peak traced Python bytes**. It
emitted one cold-start abstention and 3,814 interval abstentions (0% binary
coverage), with no resets. This is a computational stress fixture, not a
Real-IAD timing or performance result. Process RSS/native allocations were
not measured.

On the registered 80-event synthetic stationary and piecewise-stationary
reference sequences, each had 40 binary predictions and 40 abstentions (50%
coverage), including one cold-start and 39 change-detection abstentions. The
piecewise trace detects the declared synthetic shift. These are trace
diagnostics, not claims about physical defects.

The registered Stage 0 comparison setting is p=0.99, delta=0.01; it is not
asserted as a universal source-paper default. For n=64, 256, 1,050, and 3,815,
the unclamped radius is respectively approximately 0.35610, 0.17993, 0.08960,
and 0.04731; the corresponding lower clipped levels are approximately 0.27779,
0.63014, 0.81081, and 0.89539, with upper level 1 in these examples.

### B7 full-engine integration

`B7_SYNTHETIC_FULL_ENGINE_ACTIVATION_PASS-v1.0.0` runs B7 inside the actual
`PrequentialEngine`, with the other 11 registered original/fallback rows, a
label-blind Q-A-TS-v1 tape, shared score and memory paths, synthetic native
Greedy operator, current-memory rescoring of all C64 anchors, queried-normal
source-aware LSO, event-boundary checkpoint/resume, and a B7-threshold
counterfactual.

| Scenario | Evidence and result |
|---|---|
| Sparse mechanism | 1,024 events; 31 fixed queries (3.027%); 30 queried normals and 1 queried defect. B7 remains `NORMAL_ONLY`, holds on all 1,024 events, evaluates no candidates, and makes no threshold move. Memory reaches version 30; all 64 C64 anchors are current; 60 queried-normal source rows are excluded by LSO. |
| Two-sided mechanism stress | 1,024 events; 31 fixed queries, deliberately containing 21 synthetic defects (67.7%) and 10 normals to exercise constraints. This is not a prevalence model. B7 visits `NORMAL_ONLY` (320), `BOTH` (495), and `CONFLICT` (209), finishes `BOTH`, evaluates 7,130 candidates and 14,302 bounds across 21 unique evidence revisions, and moves once by 0.0000641 against the 0.01 limit. Memory reaches version 10; all C64 anchors are current; 20 queried-normal source rows are excluded by LSO. |

Both scenarios pass uninterrupted versus resumed equality, including complete
engine checkpoint state, score vector, query positions, disclosures,
predictions, threshold path, memory state/hash, and anchor state. Changing B7's
starting threshold leaves shared scores, queries, and memory writes unchanged.
The stress-run Python traced peaks were 26,553,565 bytes (sparse) and
33,874,626 bytes (stress); native process RSS was not measured. B7's bounds
remain descriptive and do not certify online risk under selected feedback or
changing memory.

### Baselines, fallback, and protocol

SPOT and dSPOT preserve their registered 1,000-score and 50+1,000 residual
startup. Source-native startup is unavailable, and separate Frozen-Theta rows
report fallback decisions. In the 1,150-event synthetic activation:

- SPOT: 1,000 startup-unavailable events, 150 source-method decisions.
- dSPOT: 1,050 startup-unavailable events, 100 source-method decisions.
- Sun-CS: 1 cold-start and 574 change-detection abstentions among 1,150
  events; its separate fallback row records 575 abstention fallbacks and 575
  source-method decisions.

All original rows and three separate fallback variants pass synthetic
activation and checkpoint checks. [STAGE1_EVALUATION_PROTOCOL_FROZEN_CANDIDATE.md](STAGE1_EVALUATION_PROTOCOL_FROZEN_CANDIDATE.md)
defines the method groups, information access, startup/fallback policy,
full-stream and t=1050 common-horizon metrics, statistical assumptions, and
locked configuration. It remains `RESEARCHER_APPROVAL_REQUIRED`.

The candidate gate dimensions are in
[STAGE0_GATE_AMENDMENT_CANDIDATE.md](STAGE0_GATE_AMENDMENT_CANDIDATE.md).
They do not change or relax the operative gates pending researcher approval.

## Validation evidence

Environment: the existing Conda `torch` environment, Python 3.11.14,
PyTorch 2.0.1+cu117, CUDA available. No package or dependency changes were
made.

| Check | Result |
|---|---|
| `python -m compileall -q experiments/calibration tests/calibration` | PASS |
| Full `python -m unittest` calibration discovery | 85 run, 85 passed, 0 failed, 0 errors, 0 skipped; 63.18 seconds in the Runner record. |
| B7 full-engine activation | 3 tests passed; both full/resume scenarios and stress counterfactual passed. |
| Final focused Sun-CS, registry, audit, Runner integrity, and engine/ledger regression set | 38 passed, 0 failed. |
| Protected historical-source hashes | PASS; all 13 reviewed files match. |
| Native scorer parity smoke | PASS; live-v22/untouched-clone and adapter score deltas 0.0; native memory update equals shared Greedy update. |
| Native Greedy operator matrix | PASS; FP16 append/replace/skip, repeated-slot source attribution, LSO, and checkpoint restore compare with exact `torch.equal`. |
| Bounded native replay | PASS; 34 audiojack training-normal events, one queried normal, all 64 C64 anchors refreshed at current M, eight source rows excluded, and full/resumed trace and memory equal. All 12 controller rows participated. |
| Metadata preflight | PASS for the four development classes; no image or mask decoding. |
| B6 class-scale synthetic diagnostic | PASS, completed as reported above. |
| Runner artifact `--verify-only` | `ARTIFACT_INTEGRITY_VERIFIED`; source/config fingerprints match and protected-source status is PASS. |
| `git diff --check` | PASS; Git emitted only working-tree LF/CRLF normalization advisories. |

The first Runner report pass used stale test-name selectors for the B6
checkpoint and fallback tests and mixed-case duplicate G9 aliases. Those
reporting defects were corrected against the actual 85 passing test IDs; the
gate JSON was re-emitted with canonical G9 keys, source fingerprints and
checksums were refreshed, and `--verify-only` passed. The native payloads and
test result payload were not rerun or changed during this report reconciliation.
The class-scale B6 diagnostic was run separately after that selector issue
and linked into the final Runner artifact checksums.

## Actual and candidate gate status

| Gate | Actual sixth-pass result |
|---|---|
| G0 | PASS |
| G1 | PASS |
| G1 metadata preflight | PASS |
| G2 | PASS |
| G3 | PASS |
| G4 | PASS |
| G5 | PASS |
| G6 | `BLOCKED_FIDELITY` under the existing contract: external author-code parity is unavailable, B2 is adapted, and B7 remains provisional. B6 itself is an executable paper-equation implementation with declared conventions, not blocked solely for endpoint conventions. |
| G7 | BLOCKED under the existing source-fidelity contract; reference mathematics passes, but author-code trace parity and guarantee/applicability transfer are not established. |
| G8 | PASS |
| G9 | PASS for bounded Stage 0 engineering parity only; no empirical scientific validation is implied. |

Candidate dimensions: G6 core methods PASS; external paper-equation
implementations PASS; operational variants PASS; unresolved-method review
BLOCKED. G7 mathematical reference tests PASS; source implementation parity
BLOCKED/UNAVAILABLE; assumption compatibility and theoretical transfer NOT
ESTABLISHED. G9 native core, synthetic activation, full-engine integration,
and all registered method rows PASS within their recorded scopes; scientific
validation is `OUT_OF_SCOPE_NOT_RUN`.

Overall status remains `BLOCKED_WIP` because existing G6/G7 acceptance is not
met and candidate gate amendments require researcher approval. No external
comparator is removed. Empirical comparability does not imply theoretical
equivalence.

## Audit and remaining review decisions

The immutable compact package is
[`audit/stage0_sixthpass_final_v1/`](audit/stage0_sixthpass_final_v1/). It
contains the actual gate matrix, candidate dimensions, test identifiers,
method cards, native and synthetic evidence, the class-scale B6 result,
source/protocol fingerprints, and checksums.

Researcher review is required for the Stage 1 protocol and proposed gate
amendment, retention/role of external comparators without author-code parity,
B7's provisional parameters, and any claim about theoretical assumptions or
guarantee transfer. Independent scientific review remains separate from this
engineering implementation.
