# R2-SGRM First-Version Implementation Review

**Algorithm:** `r2.sgrm.group_max_relevance.v1`
**Status:** `IMPLEMENTED_SYNTHETIC_TESTS_PASSED_AWAITING_REPOSITORY_REVIEW`

## Frozen equations

For query descriptor `h`, eligible anchor `j`, and Source Group `g`, let `d` be cosine distance, `beta` the existing signed patch correction, `tau = 0.10`, and `epsilon = 1e-6`:

```text
w_gj(h) = exp(-d(h, h_gj)^2 / (2 tau^2))
mu_g(h) = sum_j(w_gj(h) beta_gj) / sum_j(w_gj(h))
a_g(h) = max_j w_gj(h)
delta_SGRM(h) = sum_g(a_g(h) mu_g(h)) / (epsilon + sum_g a_g(h))
```

The within-group mean has no epsilon. The only epsilon is the absolute-scale term in the final denominator. Empty memory and numerically zero group mass yield zero correction. No class balancing, sign priors, compact support, adaptive bandwidth, outcome weights, confidence gates, or writer/replacement changes are implemented.

## Interfaces and numerical contract

```python
sgrm_delta_from_distances(
    distances, beta, group_ids, *, tau=0.10, epsilon=1e-6,
    return_diagnostics=False,
)

sgrm_delta_from_anchors(
    query_features, anchors, *, current_position, current_source_group,
    tau=0.10, epsilon=1e-6, return_diagnostics=False,
)
```

The distance core accepts floating-point `distances` of shape `(Q, A)`, `beta` of shape `(A,)`, and one nonempty string group identifier per anchor. It assumes temporal and same-source eligibility has already been enforced. Inputs must be finite; distances must be within `[0, 2]`, allowing and preserving only a `1e-6` roundoff margin. Invalid values are rejected.

Both functions return a float32 correction vector of shape `(Q,)`. With `return_diagnostics=True`, they also return small float64 aggregates: group patch counts, group means, maximum group relevance, outer weights, total group relevance, maximum correction magnitude, and zero-evidence flags. Diagnostic group arrays are ordered by sorted group identity, but the identities themselves are not returned or logged.

The anchor adapter accepts objects exposing `feature`, `beta`, `radius`, `kappa`, `position`, and `source_group`. It requires nonnegative integer positions strictly before `current_position`, validates every anchor, excludes all anchors matching `current_source_group`, checks dimensions and finiteness, and does not mutate features or anchor objects. It uses one category bank per call; callers must supply official Source Group identity and must not mix categories with colliding identifiers.

Feature arrays are copied to CPU float32. Normalization matches the existing v22 expression, `x / (norm(x) + 1e-12)`; cosine similarities are clipped to `[-1, 1]` before distance conversion. The initial adapter accepts only `kappa == 1` and `anchor.radius == tau`, each within absolute tolerance `1e-6`. Beta values retain the caller's existing eligible population and are not refit.

The core converts distances and beta to float64 for log-Gaussian weights and reductions. A within-group log shift preserves the group's absolute maximum relevance. The outer log-sum scaling retains epsilon at its original absolute scale. Final corrections are cast to float32; diagnostics remain float64. The module sets `CUDA_VISIBLE_DEVICES=""` before any optional Torch tensor conversion and does not import Torch, model-loading code, or dataset-access code at module import.

## Group and causal eligibility

The adapter rejects malformed or current/future anchors, then excludes every anchor from the query's Source Group. Every anchor needs one nonempty Source Group. Within a group, all nonzero beta values must have the same sign; zero beta is allowed. Conflicting signs raise `SGRMInputError` and are not split into artificial groups. Multi-source merged prototypes are unsupported: the future caller must reject them rather than assign their provenance to a latest source.

The adapter reads the bank as a snapshot and never writes it. A query's disclosed feedback can be appended by the existing caller only after that event's score. It can therefore affect a later eligible score, while same-source exclusion still applies. No labels are inferred for unqueried views.

## Mathematical properties and limits

- **Boundedness:** each `mu_g` is a weighted mean of beta values; the outer group weights sum to at most one because epsilon remains in the denominator. Thus `abs(delta) <= max(abs(beta))`, up to float32 rounding. This is an amplitude bound, not a risk guarantee.
- **Whole-group replication invariance:** uniformly duplicating every anchor in one group leaves its mean and maximum relevance unchanged. The result is invariant for a fixed snapshot, without eviction or writer effects.
- **Singleton-group equivalence:** when each group has one anchor, the equation equals flat Gaussian residual aggregation with the same tau, epsilon, and precision contract.
- **Permutation and renaming invariance:** anchor order and consistent group renaming do not change the result except for declared floating-point tolerance.
- **Selective-duplicate counterexample:** duplicating only one patch in a heterogeneous group can change its within-group weighted mean. Arbitrary duplicate invariance is not claimed.
- **Not count-independent:** adding genuinely different patches can change both the group mean and maximum relevance. The method removes direct sum-mass amplification by within-group patch count; it does not remove dependence on group contents or size.
- **No superiority or safety theorem:** this implementation says nothing about AUROC, calibrated probabilities, statistical independence, or FPR/FNR improvement. Grouping is a representation choice, not proof of independent evidence or correct patch labels.

## Synthetic test evidence

The tests use tiny hand-checkable and seeded randomized fixtures. An independent scalar float64 reference implements the equations separately from production code. Coverage includes empty memory, epsilon attenuation, singleton equivalence, whole-group replication, permutation/renaming, signed-group cancellation, boundedness, underflow stability, invalid inputs, causal/source exclusions, nonmutation, sign/radius/confidence contracts, score-before-update behavior, full 784-patch dense-map maximum, and 4096D CPU descriptors.

Command used the requested interpreter and a 300-second subprocess ceiling:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
& 'D:\anaconda3\envs\torch\python.exe' -m py_compile experiments/calibration/r2_sgrm.py tests/calibration/test_r2_sgrm.py
& 'D:\anaconda3\envs\torch\python.exe' -c "import subprocess,sys; r=subprocess.run([sys.executable,'-m','pytest','-q','tests/calibration/test_r2_sgrm.py'],timeout=300); sys.exit(r.returncode)"
```

Observed result: **47 passed in 2.93 seconds**; compilation succeeded. Whole-group replication invariance and singleton-group equivalence both passed their dedicated tests. The selective-duplication counterexample also passed. The wrapper's float32 feature normalization also matched an independent scalar reference. These are synthetic contract results only.

## Future benchmark integration

A later, separately authorized benchmark can pass the existing EXP2B matched FIFO `Anchor` snapshot and the current event's selected float32 descriptors to `sgrm_delta_from_anchors` before disclosing or applying the current query update. The returned patch corrections can be passed unchanged to the existing `corrected_dense_map(base_map, selected_ids, delta)` and `max_score(dense_map)`. Use only the same eligible C1-NoBoth anchors and beta values as the control; do not create a new SGRM bank. Do not supply Codebook merges or any multi-source anchor representation.

For a fair numerical comparison, the later flat Gaussian control should reuse the same distances and float64 reduction contract so that precision changes are not mistaken for an algorithmic effect.

**REAL-DATA VALIDATION NOT RUN.** No historical feature/output artifact, dataset, label, image, GPU, or DINO was accessed or evaluated for this implementation task.
