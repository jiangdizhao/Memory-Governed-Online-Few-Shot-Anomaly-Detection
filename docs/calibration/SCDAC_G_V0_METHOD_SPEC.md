# SCDAC-G v0 method specification

**Version:** `SCDAC-G-v0-PROVISIONAL_METHOD_SPEC`
**Status:** executable Stage 0 mechanism; numerical design is provisional and not scientifically validated.
**Authoritative conceptual source:** `docs/Designing and Prioritizing Memory-Dependent _ State-Consistent Calibration for Dynamic-Memory Indust.pdf`
**Source SHA-256:** `5fe07256384351c1170d67183eae5a80e5aa289a03586ba767671f062f4fb977`.

The proposal defines state-consistent rescoring, two-sided evidence, a balanced
upper-risk objective, bounded movement, and explicit missing/conflict behavior.
It does not freeze the error-bound family, confidence level, evidence minima,
movement penalty, score normalization, support-risk values, or cooldown. This
specification freezes those choices as provisional defaults so the mechanism is
deterministic and testable. They are not presented as the proposal's unique
interpretation or as deployment-safe values.

## Information and event timing

B7 receives current-memory scores only for legally eligible anchors:

- Protected C64 normals, always retained as read-only calibration evidence.
- Queried verified-normal anchors only when the ledger has rescored them with
  all-layer leave-source-out (LSO) for their official sample group.
- Queried verified defects, which are never written into normal memory.

Every anchor score must match the current `memory_version`. The controller
predicts before current-label disclosure. It receives no unqueried label, path,
image, mask, official sample key, or mutable memory object. New score evidence
is applied to the threshold for the next event only.

The initial threshold is the shared `theta0`: D2's higher q90 of the 64
additional C64 normals rescored under M0 (zero-based rank 57). The initial
evidence state is `NONE`; after loading C64 it is `NORMAL_ONLY`, and B7 holds
until it has both eligible classes. C64 remains additional to four deployed
supports, never “strict four-shot total.”

## Risks, ties, and uncertainty guard

The operational rule is `predict_defect = (score >= theta)`. For eligible
current-state normal scores `N` and defect scores `D`:

```text
k_FP(theta) = sum(1[n_i >= theta] for n_i in N)
k_FN(theta) = sum(1[d_j <  theta] for d_j in D)
FPR_hat(theta) = k_FP(theta) / |N|
FNR_hat(theta) = k_FN(theta) / |D|
```

An unavailable class risk is `null`/unavailable; it is never filled with zero.
Exact `>=` and `<` comparisons apply at ties. The candidate set includes every
score value, its finite `nextafter` neighbors (so tied and strictly-separated
decisions are both represented), midpoints between adjacent distinct scores,
and the current threshold. Candidates are sorted and deduplicated.

For each available class at a fixed candidate threshold, the code computes the
one-sided Clopper–Pearson upper endpoint:

```text
U(k,n;delta) = BetaInverse(1-delta; k+1, n-k),  k < n
U(n,n;delta) = 1
```

The registered per-side `delta` is `0.05`. This is named
`descriptive_upper`, not a confidence sequence or certified bound. Selective
label disclosure, temporal dependence, repeated rescoring under
label-dependent memory, and selecting a threshold after examining the same
anchors invalidate an automatic population-coverage interpretation. The
implementation uses the existing environment's SciPy beta quantile and fails
closed if that registered calculation is unavailable.

## Evidence states

Minimum counts are 16 eligible normals and 8 eligible defects. A class below
its minimum is unavailable to optimization, even if a few raw anchors exist.

| State | Eligible evidence | B7 action |
|---|---|---|
| `BOTH` | At least 16 normals and 8 defects | Optimize the primary two-sided objective. |
| `NORMAL_ONLY` | Normal minimum met; defect minimum not met | Hold. FNR is unavailable. |
| `DEFECT_ONLY` | Defect minimum met; normal minimum not met | Hold. FPR is unavailable. |
| `NONE` | Neither minimum met | Hold. Both risks are unavailable. |
| `CONFLICT` | Both minima met, but no candidate satisfies the descriptive support limits | Use the bounded minimum-descriptive-risk fallback; make no feasibility claim. |

The diagnostic state can change from `BOTH` to `CONFLICT` only after current
scores have been evaluated. Raw counts and eligible counts are both recorded.

## Primary BA-oriented mode

For `BOTH`, let `sigma_s` be the population standard deviation of all eligible
normal and defect anchor scores. `epsilon_s` is added to the denominator. For each candidate:

```text
J(theta) = 0.5 * U_FPR(theta)
         + 0.5 * U_FNR(theta)
         + lambda_move * abs(theta - theta_t) / (sigma_s + epsilon_s)
```

The candidate with the smallest objective is selected, with deterministic
ties broken by smallest movement and then smallest threshold. The selected
threshold is clipped to `[theta_t - Delta_theta, theta_t + Delta_theta]`.
This objective uses both class risks; the missing side is never optimized as a
zero. If the two-sided support set is empty, the same objective supplies the
proposal's bounded conflict compromise; this is the documented `CONFLICT`
fallback, not a claim that both risks are controlled.

## Secondary FPR-constrained diagnostic

For each predeclared `alpha in {0.05, 0.10, 0.20}`, report the candidate with
`U_FPR(theta) <= alpha` that minimizes `U_FNR(theta)` plus the same normalized
movement penalty. If no such candidate exists, report
`NO_DESCRIPTIVE_FEASIBLE_THRESHOLD`. These diagnostics do not change B7's
primary threshold trajectory and are not population risk guarantees.

## Supported interval and stale-evidence policy

For audit only, when the candidate set is nonempty under both descriptive
support limits `U_FPR <= 0.20` and `U_FNR <= 0.20`, store
`last_two_sided_supported_interval = [min(feasible), max(feasible)]`. The record
contains memory version, evidence revision and evidence digest, eligible
counts, limits, and `bound_label=descriptive_upper_not_certified`.

The interval is usable only for its exact memory version and evidence digest.
Any memory change, anchor-score change, source-attribution change, or ledger
revision marks it stale and inactive. Stale intervals remain in diagnostics
only; they never constrain threshold movement or imply safety in a new score
state. In missing-evidence states, B7 holds even if an old interval exists.

## Provisional defaults and serialization

| Parameter | Default | Rationale/status |
|---|---:|---|
| `min_normals` | 16 | Provisional guard; C64 normally supplies 64. |
| `min_defects` | 8 | Provisional minimum; sparse defects may delay B7 activation. |
| `confidence_delta` | 0.05 | Fixed-time one-sided CP descriptive endpoint. |
| `support_fpr`, `support_fnr` | 0.20, 0.20 | Provisional interval/conflict diagnostic limits. |
| FPR diagnostic targets | 0.05, 0.10, 0.20 | Predeclared per request; diagnostic only. |
| `lambda_move` | 0.05 | Provisional dimensionless movement penalty. |
| `epsilon_s` | 1e-12 | Positive score-scale floor. |
| `Delta_theta` | 0.01 | Provisional bounded move, aligned with Stage 0 SCQ step. |
| Cooldown | 0 | No cooldown is specified in the proposal; each fresh evidence revision is processed once. |
| Initial theta/state | `theta0` / `NONE` | D2 C64 higher-q90 at M0; C64 snapshot then yields `NORMAL_ONLY`. |

`state_dict()` includes version/config, threshold, evidence scores and ids,
memory/evidence versions, interval validity, diagnostics, pending update flag,
and deterministic operation counts. Restoring a checkpoint with another method
version or config fails closed. Runtime latency is measured by the engine and
is not used by B7.

The objective evaluates O(G(N+D)) anchor/candidate comparisons for `G`
candidate thresholds and N/D anchors; the implementation records candidate
and bound-evaluation counts. Native anchor rescoring cost is reported by the
engine/ledger separately.

## Research decisions still open

The independent research reviewer must assess the evidence minima, per-side
confidence level, simultaneous/post-selection interpretation, support limits,
movement penalty, score-scale normalization, and whether conflict should use
the same balanced-risk fallback. Stage 0 synthetic mechanism tests cannot
resolve these methodological choices. No empirical tuning or superiority
claim is made here.
