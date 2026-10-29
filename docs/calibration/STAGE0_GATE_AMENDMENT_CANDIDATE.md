# Stage 0 gate amendment — versioned candidate with final protocol correction

**Revision:** sixth-pass gate dimensions plus the v2 Sun-CS endpoint and B7
feasibility correction.

**Status: `RESEARCHER_APPROVAL_REQUIRED`.** This document proposes reporting
dimensions for independent review. It does not change the current mandatory
gate definitions, remove a comparator, or authorize Stage 1. Until a researcher
approves or rejects it, report both the existing gate result and the candidate
dimensions. No candidate PASS is a Stage 1 authorization.

## Rationale

The original aggregate G6/G7/G9 labels combine different claims: code
execution, equation correctness, parity with an external implementation,
compatibility of source assumptions with our score process, and empirical
scientific validation. The candidate below exposes those separate claims while
retaining every existing method row and recording unavailable evidence as
BLOCKED or NOT CLAIMED.

## G6 — implementation availability and fidelity declarations

| Candidate dimension | Evidence required | Candidate reporting rule |
|---|---|---|
| `G6_CORE_METHODS` | B0-B4 and B7 executable with registered access controls, missing-side behavior, and checkpoint state. | PASS only for tested registered implementations; B7 remains labeled provisional. |
| `G6_EXTERNAL_PAPER_EQUATION_IMPLEMENTATIONS` | B5a, B5b, and B6 equations, information permissions, numerical conventions, independent reference tests, and differences recorded. | A paper-equation implementation with declared conventions is not labeled author-code parity. |
| `G6_OPERATIONAL_VARIANTS` | Separate wrapper rows, fixed fallback policy, source method continues updating, and fallback-mode accounting. | Report independently from each original method. |
| `G6_UNRESOLVED_METHODS` | Any method whose math/source/access rule remains unresolved. | BLOCKED; never removed from mandatory comparison silently. |

Candidate G6 completion does not imply empirical validity or theoretical
guarantee transfer. If a source equation remains materially ambiguous, that
method remains BLOCKED and the existing mandatory comparator requirement stays
in force pending researcher decision.

For B6, the v2 implementation makes `conservative_extended_real` the primary
declared endpoint convention and retains `empirical_extrema` as a separately
named sensitivity version. The two-observation stationary false-reset
pathology is reproduced and regression-tested. Both remain
`PAPER_EQUATIONS_WITH_DECLARED_CONVENTIONS`; no verified author-code parity is
claimed. The conservative policy's lack of finite anomaly decisions at
registered class-length sizes is an applicability result, not a gate failure
or reason to silently remove Sun-CS.

## G7 — four independent validity dimensions

| Candidate dimension | Evidence required | Candidate reporting rule |
|---|---|---|
| `G7_MATHEMATICAL_REFERENCE_TESTS` | Independent equation reference, edge cases, endpoint/tie behavior, synthetic traces, and exact optimized/reference parity. | PASS only for tested mathematics and disclosed conventions. |
| `G7_SOURCE_IMPLEMENTATION_PARITY` | Trace comparison against a verified author implementation and pinned source revision. | BLOCKED/UNAVAILABLE if author code cannot be verified; do not imply it from equation tests. |
| `G7_ASSUMPTION_COMPATIBILITY` | Evidence that the experiment's labels, scores, feedback, dependence, and drift meet source assumptions. | State limitations per method; incompatibility does not itself invalidate a clearly labeled empirical comparison. |
| `G7_THEORETICAL_GUARANTEE_TRANSFER` | A proof covering the actual score process, memory changes, feedback selection, and evaluated target. | NOT ESTABLISHED unless separately proved. No inherited ACI, Sun-CS, EVT, or CP guarantee by naming a method. |

Absence of verified author code does not automatically invalidate a
mathematically well-specified, independent-reference-tested method; it does
leave source implementation parity explicitly unverified. This distinction is
a candidate amendment and is not approved by this implementation.

Sun-CS's paper target remains a score-quantile anomaly, distinct from the
Real-IAD physical-defect target. Its source assumes an IID score process for
the cited coverage result, with additional conditions for change-detection
claims. Adaptive memory can make scores dependent and nonstationary. Empirical
physical-defect measurements may be reported under a frozen protocol, but
source guarantees do not transfer. `Empirical comparability does not imply
theoretical equivalence.`

## G9 — scope-specific integration evidence

| Candidate dimension | Evidence required | Candidate reporting rule |
|---|---|---|
| `G9_NATIVE_CORE` | Native v22/VisionAD score parity, shared native Greedy update parity, protected support, K=256, current-state anchor refresh, and bounded replay. | Report native core scope and reused/re-executed artifact provenance. |
| `G9_CONTROLLER_SYNTHETIC_ACTIVATION` | Deterministic causal and method-specific synthetic branches, including score/query/memory invariance. | Mechanism evidence only; not Real-IAD performance. |
| `G9_FULL_ENGINE_INTEGRATION` | Actual PrequentialEngine, B7, query-normal memory writes, queried-normal LSO, current C64 rescore, both feedback scenarios, and complete resume parity. | Name the result `B7_SYNTHETIC_FULL_ENGINE_ACTIVATION_PASS`; synthetic only. |
| `G9_ALL_REGISTERED_METHODS` | Every original and separately registered operational variant participates in the tested shared engine; availability, abstentions, fallbacks, and metrics are accounted. | Never infer from a subset or omit a blocked row. |
| `G9_SCIENTIFIC_VALIDATION_NOT_STAGE0` | Real-IAD precommitted evaluation, statistical analysis, independent classes/seeds, and later review. | OUT OF SCOPE / NOT RUN in Stage 0. |

The added seeded continuous-stream tests distinguish stationary IID uniform,
stationary lognormal, and a predeclared piecewise-uniform shift for both Sun-CS
endpoint policies. They provide endpoint-mechanism, coverage, reset, and
computational evidence only; they do not establish Real-IAD performance.
B7's separate query-budget probabilities are hypothetical Bernoulli
sensitivity scenarios from metadata-only query counts, not estimates of
development defect prevalence.

## Existing-gate preservation and approval

The existing G0-G9 gates remain the operative acceptance contract until
researcher approval. This candidate does not relax or remove G6, G7, G9, B6,
or B7. The sixth-pass audit must list (a) actual existing gate outcomes and
(b) candidate-dimension outcomes in separate columns, including every
BLOCKED, SKIPPED, and NOT CLAIMED item. No candidate PASS is a Stage 1
authorization.
