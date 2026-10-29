# Stage 0 final scientific protocol review

**Disposition:** Engineering evidence is bounded to Stage 0 verification.
`SCIENTIFIC_VALIDATION_NOT_RUN`. The proposed Stage 1 protocol and gate
amendment remain `RESEARCHER_APPROVAL_REQUIRED`; this report does not authorize
Stage 1 or Stage 2.

The machine-readable Runner artifacts and compact package are the authoritative
test, runtime, provenance, and gate record. The synthetic diagnostic values
below are mechanism/applicability observations, not Real-IAD performance.

## 1. Verified engineering components

The calibration package has a modular causal `PrequentialEngine`, fixed
label-blind query tapes, controller-isolated prediction/update interfaces,
verified-normal-only shared Greedy memory writes, current-memory anchor
refresh, queried-normal source-aware LSO handling, checkpoint/resume state, and
a Runner that writes checksummed, resumable, non-overwriting Stage 0 artifacts.
Native v22/VisionAD code is used through the native bridge; historical source
is protected by a byte-level baseline check. Synthetic score/memory parity is
separate from scientific validity.

Verdicts: `ENGINEERING_PASS` applies only to the bounded tests and Runner rows
marked PASS in the attached artifacts. The operative G6 and G7 gates remain
blocked as recorded in `gates.json`; no aggregate label overrides those rows.

## 2. Executable methods and fidelity

The executable registry covers the eight numbered families B0–B7, with B5
split into SPOT (B5a) and dSPOT (B5b), and separate Frozen-Theta operational
fallback rows for B5a, B5b, and B6. B2 is an explicitly adapted historical
sidecar; B4 is explicitly adapted ACI; B7 SCDAC-G remains provisional v0.1.

B5a, B5b, and B6 are `PAPER_EQUATIONS_WITH_DECLARED_CONVENTIONS`. Their
equations, information permissions, warm-up/update behavior, differences, and
independent references are declared. This is not author-code parity:
`SOURCE_PARITY_UNVERIFIED`. The SPOT/dSPOT/Sun-CS startup rules have not been
replaced by operational fallbacks in their source-native rows.

## 3. Sun-CS target, endpoint pathology, and correction

Sun-CS labels the score-quantile event
`Y_quantile,t = 1[S_t > Q_Ft(.99)]`. The evaluation outcome is the Real-IAD
physical-defect label `Y_defect,t`. These are distinct targets. The source
coverage result assumes IID scores; change results require their source
conditions. Dynamic memory can induce dependence and nonstationarity. No
physical-defect guarantee, adaptive-memory guarantee, or theorem transfer is
claimed: `THEORY_TRANSFER_NOT_ESTABLISHED`.

For p=.99 and delta=.01, the raw confidence radius is retained. At one- and
two-score prefixes the requested levels clip to q=0 and q=1. For distinct
finite scores `a < b`, empirical-extrema confidence sets are `[a,a]`, `[a,b]`,
and `[b,b]`; the prefix and suffix intersections are disjoint. Algorithm 2
therefore falsely resets on a two-observation stationary continuous trace,
and Algorithm 3 repeats that failure after each reset. The regression fixture
checks production optimized/reference Algorithm 2, independent reference
Algorithm 3, and both endpoint policies. The conservative primary maps clipped
q=0/1 to negative/positive infinity and prevents that minimal false reset.
The empirical-extrema implementation remains a separately versioned
sensitivity, not the primary.

At n=64, 256, 1,050, and 3,815, the upper requested level remains one, so the
primary upper endpoint is positive infinity and no finite score can be called
an anomaly. The first n with an unclipped upper level is 353,018. This makes
limited-sample binary coverage an applicability issue; it is not grounds for
removing Sun-CS. The source-native method may be reported as an unsupervised
reference with low or zero usable anomaly-prediction coverage.

Verdicts: `MATHEMATICAL_REFERENCE_PASS` for the tested equations and declared
conventions; `PAPER_EQUATIONS_WITH_DECLARED_CONVENTIONS`; `SOURCE_PARITY_UNVERIFIED`;
`THEORY_TRANSFER_NOT_ESTABLISHED`.

## 4. Synthetic evidence and computational cost

The seeded 512-event diagnostics use stationary IID Uniform, stationary
LogNormal, and a piecewise Uniform shift at zero-based t=256. They evaluate
the source score-quantile target offline after each prediction. In the
measured run, conservative-primary binary coverage was 55.47%, 55.27%, and
25.20%, respectively; reset counts were 0, 0, and 0. Its source-target
interval coverage was 1.0 in these finite synthetic runs, but the piecewise
stream produced no detected post-shift reset. A perfect interval-coverage
fraction here does not establish a guarantee.

The empirical-extrema sensitivity had 50% binary coverage and 255 resets on
each 512-event trace, with its first reset at t=2. Its stationary false-reset
count was 255 on both stationary streams. On the piecewise stream, the first
post-shift reset was at t=256, but this followed the same reset-every-two
pathology already present before the shift; it is not evidence of reliable
change localization.

The 3,815-event class-scale Uniform primary run took 191.8 seconds untraced.
The separate traced pass used 3,926,124 bytes peak Python memory; sampled
process RSS peak was 45,846,528 bytes. It made 3,087 benign binary predictions,
727 interval abstentions, one cold-start abstention, zero anomaly predictions,
and zero resets: 80.92% binary coverage. These values describe this synthetic
seed and implementation runtime only. The paired comparison JSON contains the
complete rows, counters, source hashes, coverage definitions, and measured
runtime/memory for the Runner's current execution.

## 5. SPOT/dSPOT startup and fallback

Source-native SPOT remains unavailable for its first 1,000 observations.
Source-native dSPOT retains 50 local-window observations followed by 1,000
residual initialization observations. Their startup-unavailable counts,
source-native coverage, and common-horizon metrics remain visible. The
precommitted common horizon is zero-based t=1050. Ordinary full-stream balanced
accuracy is undefined when source-native predictions are missing; startup
unavailability is not hidden. SPOT+Frozen, dSPOT+Frozen, and Sun-CS+Frozen are
separate deployment hybrids. Report the original-method and fallback decision
fractions; hybrid results do not establish source-method competitiveness.

The preregistered applicability rule is denominator-based: report actual
overall, normal-class, defect-class, and common-horizon coverage. If a source
method lacks binary predictions for any full-stream event, do not assign it
ordinary full-stream binary BA. Preserve selective metrics and mark
inappropriate metrics undefined. Do not impose a numerical coverage cutoff
after outcomes, and do not remove a weak comparator.

## 6. B7 activation and query-budget feasibility

Eight queried defect anchors satisfy the current minimum count for nominal
two-sided evidence processing; they do not imply the descriptive FNR target is
feasible. With one-sided Clopper–Pearson delta=.05, zero errors give upper
bounds .312344 at n=8, .205817 at n=13, and .192636 at n=14. Therefore at
least 14 zero-error defect anchors are required for this specific descriptive
FNR upper bound to be at most .20. Observed errors can increase that count.
Adaptive selection, time dependence, repeated rescoring, and
memory-dependent scores prevent interpreting these empirical bounds as
certified online population risk.

Metadata-only preflight counts are: Audiojack 3,815 images / 114 queries;
Regulator 3,855 / 116; Switch 3,820 / 115; Rolled Strip Base 3,805 / 114.
Totals are 15,295 images and 459 queries under the frozen 3% half-up Q-A-TS
rule. The label-blind tape selects one view per official sample group (459 of
3,059 groups, approximately 15.0%); no hidden test labels or prevalence were
read. These are metadata counts, not defect counts.

The separate feasibility JSON reports hypothetical independent Bernoulli
queried-label scenarios at 5%, 10%, 20%, and 30%; these are not prevalence
estimates. It gives per-class probabilities, expected number of development
classes reaching 8 and 14 defects, probability all four reach each count, and
probability a class remains below 8 and therefore stays `NORMAL_ONLY`/HOLD.
At 5%, expected classes reaching 8 are 0.866 and the chance all four reach 8 is
0.00219; expected classes reaching 14 are 0.00729 and the chance all four
reach 14 is 1.10e-11. At 10%, expected classes reaching 8 are 3.589 and all
four reach 8 with probability .6481, while expected classes reaching 14 are
1.025 and all four reach 14 with probability .00430. At 20%, those expected
counts are 4.000 and 3.959 (all four reach 14 with probability .9599). At
30%, expected counts are 4.000 for both thresholds (all four reach 14 with
probability .999993).

In Stage 1, never reaching BOTH or never finding a feasible threshold is a
valid result. Do not reduce the minimum defect count, change delta, increase
the query budget, use unqueried defects, or relabel a tuned SCQ threshold as
B7. Preserve all v0.1 HOLD and fallback behavior.

## 7. Interpretation of synthetic evidence

The current synthetic checks establish causal ordering, access restrictions,
deterministic reset behavior for the tested endpoint conventions, checkpoint
and resume parity, reproducible mechanism-level coverage/reset/runtime
measurements, and B7 engine-state behavior on its declared synthetic
scenarios. They establish that the two-observation sensitivity pathology is
reproducible and that the conservative convention avoids that specific
pathology.

They do not establish Real-IAD defect prevalence, accuracy, superiority,
robustness across arbitrary streams, population-risk control, source-code
parity, validity of IID/stationarity assumptions under adaptive memory, or
theoretical guarantee transfer. No empirical scientific validation was run.

## 8. Methodological assumptions still unresolved

- Whether the conservative extended-real endpoints are the preferred
  preregistered interpretation of the paper's quantile construction.
- Whether source assumptions for SPOT, dSPOT, and Sun-CS can be usefully
  interpreted under memory-driven score changes.
- Whether the permitted empirical comparison and coverage-based applicability
  rule meet the researcher's intended treatment of source-native methods.
- Whether B7's provisional descriptive CP diagnostics are acceptable given
  sparse selective labels, repeated memory-dependent rescoring, and candidate
  selection. They are not certified online risk bounds.
- Whether the proposed gate amendment should distinguish implementation,
  mathematical-reference, author-source parity, assumption compatibility,
  and guarantee transfer as separate approvals.

## 9. G6/G7/G9 readiness and protocol consistency

The revised G6/G7/G9 dimensions are technically ready for researcher review as
reporting proposals. They do not replace operative gate outcomes: existing
G6 remains `BLOCKED_FIDELITY`, G7 remains blocked by unavailable source parity
and unestablished guarantee transfer, and G9 is only bounded engineering
evidence. The actual and proposed matrices are separate in the audit package.

The Stage 1 candidate is internally consistent across the four development
classes, four deployed supports plus 64 *additional* C64 anchors, frozen native
scorer, shared K=256 Greedy memory, controller-independent memory writes,
fixed query tape and 3% budget, common t=1050 horizon, selective and
class-specific coverage, worst 256-image windows, and Mint/PCB confirmation
isolation. B7 v0.1 defaults and fallback/HOLD behavior are unchanged. The
candidate is not researcher-approved.

**Empirical comparability does not imply theoretical equivalence.**

## 10. Decisions requiring explicit researcher approval

1. Approve, revise, or reject the v2 conservative Sun-CS endpoint convention
   and retain/remove the empirical-extrema sensitivity.
2. Approve, revise, or reject the source-native coverage/applicability
   reporting rule and the separate Frozen-Theta operational hybrids.
3. Approve, revise, or reject the proposed G6/G7/G9 gate amendment while
   retaining current operative gate statuses until a decision is recorded.
4. Approve, revise, or reject the frozen Stage 1 evaluation candidate,
   including its B7 v0.1 descriptive feasibility reporting.
5. Separately authorize any Stage 1 execution after review; this report grants
   no such authorization.

## 11. Final verdicts

- `ENGINEERING_PASS` — only for bounded components whose current Runner/tests
  record PASS.
- `MATHEMATICAL_REFERENCE_PASS` — for the tested equations and explicit
  endpoint conventions.
- `PAPER_EQUATIONS_WITH_DECLARED_CONVENTIONS` — SPOT/dSPOT/Sun-CS
  implementations where declared in method cards.
- `SOURCE_PARITY_UNVERIFIED` — external author-code trace parity is not
  verified.
- `THEORY_TRANSFER_NOT_ESTABLISHED` — no adaptive-memory/physical-defect risk
  transfer is claimed.
- `SCIENTIFIC_VALIDATION_NOT_RUN` — no Stage 1/2 outcome evaluation occurred.
- `RESEARCHER_APPROVAL_REQUIRED` — the revised protocol and amendment remain
  candidates.

No Stage 1 or Stage 2 experiment was executed. No Real-IAD test outcome,
confirmation-class label, or defect-prevalence distribution was inspected.
