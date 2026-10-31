# R2-L1 Locality Mathematical Specification

## 1. Status and scope

**Status: `R2_L1_MATH_SYNTHETIC_PASS`.** The equations below have isolated NumPy references and deterministic synthetic CPU tests. This status is not evidence of anomaly-detection improvement, calibration, or production parity. No real image, feature cache, label, mask, private-feedback file, correction-anchor ledger, model, or GPU was accessed. No effective support radius, distance threshold, regularizer, or kNN cutoff was selected. Prepared 2026-10-31.

The code is standalone at `experiments/calibration/r2_l1_locality_kernels.py`; it is not imported by or connected to the production runner. The original v22, R1, and R2-0 source files remain immutable.

## 2. Notation, assumptions, and distance

An event at stream position (t) has a normalized patch descriptor (h\in\mathbb R^D) and an official source-group identifier (q_t). Anchor (j) has normalized descriptor (\xi_j\), source group (q_j), insertion position (\tau_j\), signed correction strength (\beta_j\), positive confidence (\kappa_j\), legacy Gaussian width (r_j>0), and prospective effective support radius (R_j>0). These two radii have separate fields in the implementation and are never equated implicitly.

The legal set is

\[
\mathcal E_t=\{j:\tau_j<t,\ q_j\ne q_t\}.
\]

The reference validates every anchor's provenance before scoring. A current or future anchor is rejected; it is not silently filtered. Source groups must be lowercase 64-character SHA-256 identifiers. Missing, malformed, or mismatched provenance fails closed.

For L2-normalized vectors the cosine distance is

\[
d(h,\xi)=1-\operatorname{clip}(h^\top\xi,-1,1)\in[0,2].
\]

The implementation rejects zero, nonfinite, dimension-mismatched, or non-unit feature vectors. A separate stable `l2_normalize` utility is available for callers that explicitly want normalization. Distances are computed in float64 after validation, independent of whether valid source arrays were float32 or float64.

The compact-support profile is

\[
\phi(u)=\begin{cases}(1-u^2)^2,&0\le u<1,\\0,&u\ge1.\end{cases}
\]

It is nonnegative, at most one, continuous, and has value and first derivative zero at (u=1). It is a custom nonnegative weighting function. This specification does **not** claim that (\phi(d/R)) is a positive-definite Mercer kernel in arbitrary dimensions.

## 3. Frozen nonlocal references

### 3.1 Original C1 normalized Gaussian RBF

\[
w_j(h)=\kappa_j\exp\!\left(-\frac{d(h,\xi_j)^2}{2r_j^2}\right),\qquad
\Delta_{\mathrm{old}}(h)=
\frac{\sum_{j\in\mathcal E_t}w_j(h)\beta_j}
{10^{-6}+\sum_{j\in\mathcal E_t}w_j(h)}.
\]

The single-anchor case is (\Delta=\beta w/(w+10^{-6})), not (\beta\exp(-d^2/(2r^2))). Thus normalization can cancel almost all Gaussian attenuation while (w\) remains larger than the absolute (10^{-6}) denominator term.

### 3.2 Original ungated Signed kNN

For eligible anchors, retrieve the (k_{\rm eff}=\min(k,|\mathcal E_t|)) smallest cosine distances, breaking exact ties by original anchor index. With signs (s_j\in\{-1,+1\}\),

\[
\Delta_{\mathrm{kNN}}(h)=a\,\frac1{k_{\rm eff}}
\sum_{j\in\mathcal N_{k_{\rm eff}}(h)}s_j.
\]

Distance determines which neighbors are selected but not how much they contribute. There is no distance cutoff; a nearest neighbor at the maximum cosine distance can still produce magnitude (a). For fewer than `k` legal anchors, using the mean of the available `k_eff` anchors matches the frozen R1 fixture convention. The R1 engineering fixture uses `k=5`, amplitude `a=0.20`, and FIFO capacity 256; none of these gives it a distance cutoff.

## 4. Candidate A: C1-Gated

Use a separate effective radius (R_j>0):

\[
u_j(h)=d(h,\xi_j)/R_j,\quad
g(h)=\begin{cases}\max_{j\in\mathcal E_t}\phi(u_j(h)),&\mathcal E_t\ne\varnothing,\\0,&\mathcal E_t=\varnothing,\end{cases}
\]

\[
\Delta_{\mathrm{gated}}(h)=g(h)\Delta_{\mathrm{old}}(h).
\]

### Guarantees for a fixed eligible Anchor set

- **Finite support:** (g(h)=0) when (d(h,\xi_j)\ge R_j) for every eligible (j). Therefore the output is exactly zero outside the union of effective support regions.
- **Bound:** if (|\beta_j|\le B), then
  \[
  |\Delta_{\mathrm{old}}|\le B\frac{\sum_jw_j}{10^{-6}+\sum_jw_j}\le B,
  \]
  and (0\le g\le1), so (|\Delta_{\mathrm{gated}}|\le B).
- **Boundary and continuity:** each (\phi(d/R_j)) is continuous and tends to zero at its support boundary. The maximum of finitely many continuous functions is continuous. The old mixture has a strictly positive denominator and continuous weights, so their product is continuous for fixed anchors and parameters. An overlapping anchor may keep (g>0) at another anchor's boundary; at the outer boundary of the union, all gates vanish.
- **Multiple signs:** positive and negative (\beta_j) values still mix globally through (\Delta_{\mathrm{old}}). They can reinforce or cancel. The gate controls where the total output may be nonzero, not which anchors influence its value within that union.

### Limitation: support gate does not localize mixture influence

An anchor whose own (R_j) does not cover (h) still contributes to (\Delta_{\mathrm{old}}(h)) whenever its Gaussian weight is nonzero. A different nearby anchor can make (g(h)>0), exposing that global mixture at (h). C1-Gated is therefore a minimal output-support control, not strict anchor-local mixing.

## 5. Candidate B: C1-Compact

For fixed (\lambda>0), (\kappa_j>0), and bounded corrections (|\beta_j|\le B), define

\[
v_j(h)=\kappa_j\phi\!\left(\frac{d(h,\xi_j)}{R_j}\right),\qquad
\Delta_{\mathrm{compact}}(h)=
\frac{\sum_{j\in\mathcal E_t}v_j(h)\beta_j}
{\lambda+\sum_{j\in\mathcal E_t}v_j(h)}.
\]

### Guarantees

1. **Exact zero outside the union:** if no eligible support contains (h), all (v_j=0), so the numerator is zero and the denominator is (\lambda>0). Empty eligible memory also returns exact zero.
2. **Continuous zero at boundaries:** each (v_j) continuously tends to zero as (d\uparrow R_j); the numerator tends to zero and the denominator tends to (\lambda). The ratio is continuous because its denominator is everywhere at least (\lambda>0).
3. **Correction bound:**
   \[
   |\Delta_{\mathrm{compact}}|
   \le B\frac{\sum_jv_j}{\lambda+\sum_jv_j}\le B.
   \]
4. **Image-max bound:** let the original dense patch map be (x\in\mathbb R^P), and apply one bounded correction (\delta_i\) to each patch, with (|\delta_i|\le B). Then
   \[
   |\max_i(x_i+\delta_i)-\max_i x_i|
   \le\|\delta\|_\infty\le B.
   \]
   This follows from the 1-Lipschitz property of the maximum under the sup norm; it does not assume the maximizing patch is unchanged.
5. **Conflicting signs:** simultaneous positive and negative anchors can cancel in the numerator. Unequal (\kappa_j\), radii, distances, or magnitudes change the balance; cancellation is possible, not guaranteed.
6. **Local influence:** anchors outside their own effective support have (v_j=0) and do not affect this query. This removes C1-Gated's global Gaussian mixing from the candidate formula.

These are algebraic properties for fixed legal Anchor state, not claims of better anomaly detection, pixel-space locality, statistical calibration, FPR control, or robustness to mislabeled feedback. Anchor-state changes over time are a separate causal process; continuity here is with the eligible state fixed.

## 6. Candidate C: distance-gated Signed kNN

Choose, but do not currently tune, a method-level support radius (R>0), regularizer (\lambda>0), neighbor count (k\ge1), and amplitude (a>0). First restrict to legal anchors with (d(h,\xi_j)<R); exact-boundary anchors are rejected. Among the local set, select at most (k) nearest anchors, ties by original index. Give each selected neighbor weight (q_j=\phi(d(h,\xi_j)/R)), and define

\[
\Delta_{\mathrm{knn-local}}(h)=
a\frac{\sum_{j\in\mathcal N_k^R(h)}q_js_j}
{\lambda+\sum_{j\in\mathcal N_k^R(h)}q_j}.
\]

If there are no legal local anchors, return exactly zero. If (k\) exceeds the local count, use all available local anchors. Because (q_j\ge0),

\[
|\Delta_{\mathrm{knn-local}}|
\le a\frac{\sum_jq_j}{\lambda+\sum_jq_j}\le a.
\]

The strictly positive denominator prevents division by zero; inputs and intermediate weights are checked for finiteness. At (d=R), the compact weight is zero and the neighbor is excluded, so its contribution approaches zero from inside. However, when the (k\)-th-neighbor identity changes at an interior distance tie, the selected signs may change discontinuously. Stable tie-breaking makes the value deterministic but does not make the function continuous there. This retains kNN retrieval and does not impose a hand-picked reduced neighbor count; the same (k) can be used for both legacy and gated references.

## 7. Numerical contract and reference implementation

The module rejects invalid radii, nonpositive regularizers, nonpositive/noninteger (k), nonpositive amplitude/confidence, nonfinite values, zero vectors, invalid signs, bad dimensions, and malformed source provenance. Distances are clipped only at the cosine similarity boundary. Compact weights are computed only for (d<R), avoiding overflow in (d/R) for very small radii. Legacy Gaussian attenuation is set to zero once (d/r>40), where its float64 exponential is already below useful precision; this avoids squaring an overflowing ratio. Weighted ratios scale weights and the positive regularizer before summation, preserving finite bounded results for very small/large finite magnitudes.

The implementation is mathematical CPU reference code. Its float64 reductions are not promised to be bit-identical to historical float32 PyTorch execution. It does not reproduce codebook admission/merge/eviction, image preprocessing, a model, or production score maps.

## 8. Synthetic falsification results

The focused suite contains **44 deterministic CPU tests** (`44 passed in 0.25s` in the final run). The reference counterexamples use only two-dimensional unit vectors:

- At cosine distance `d=0.8` with `r=0.2`, the single-anchor Gaussian weight is `exp(-8) = 0.0003354626`, but normalization by `w + 1e-6` leaves `w / (w + 1e-6) = 0.9970279` of `beta`. Thus the old normalized RBF still returns about 99.7% of its anchor strength at four Gaussian widths.
- At the same distance, with effective support (R=0.3), C1-Gated and C1-Compact return exactly zero. At maximum cosine distance 2, ungated Signed kNN still returns its full signed amplitude; the local version returns zero when its declared support is smaller than 2.
- Boundary tests cover zero distance, interior support, (R(1-10^{-3})), (R(1-10^{-6})), exactly (R), just outside, and distance 2. The compact candidates approach zero continuously at their support boundary.
- Opposite equal-weight anchors cancel; unequal confidence produces the weighted signed result. A test also confirms C1-Gated can be altered by a far anchor whose own compact support does not include the query, while C1-Compact is unchanged.
- A synthetic 30-category geometry example gives 26 large categories with 1,000 group-level query distances at 0.20 each and four small categories with 10 distances at 0.80 each. Pooled median and pooled 95th percentile are both 0.20; the category-equal mean of category medians is 0.28. At a purely illustrative radius 0.30, per-category coverage ranges from 0 to 1, category-equal coverage is 26/30 (86.67%), while pooled descriptor coverage is 26,000/26,040 (99.85%). This is weighting arithmetic, not an inferential claim that patch distances are independent.
- In a separate synthetic four-selected-category example, a radius equal to the selected four's 95th percentile covers all four selected categories but only 4/30 categories after the other 26 have distance 0.60. No such radius is proposed for real data.
- A synthetic same-source exact self-match gives apparent radius-0.10 coverage when the source group is included and zero after that group is excluded.

The tests also cover float32/float64 agreement, empty memory, zero/nonfinite vectors, tiny positive radii/weights, many anchors with huge finite confidence, bound checks, provenance failures, current/future rejection, same-source exclusion, kNN tie ordering, no-local-neighbor behavior, and an interior kNN neighbor-set discontinuity.

## 9. All-class geometry dependency and limitations

Real radius and distance calibration requires the separately specified [all-class geometry protocol](R2_L1_ALL_CLASS_DISTANCE_GEOMETRY_PROTOCOL.md). Background representation geometry (G1) may inform descriptor-distance distributions but does not measure behavior of feedback-derived correction anchors. Online Correction geometry (G2) requires causal, source-disjoint correction anchors and authorized feedback evidence. Neither population substitutes for the other.

The 30-category synthetic test demonstrates only that pooled and four-category summaries can hide heterogeneity. It does not model the actual category universe or provide any Real-IAD distance, anchor coverage, or radius evidence. All (R_j), shared (R), (\lambda), multipliers, and cutoffs remain `UNSELECTED_PENDING_ALL_CLASS_GEOMETRY_AUDIT`.

## 10. Identity and provenance

Audit base commit: `93ef00a074f03a9cb408cc787e4ccc25a40972f3`, branch `calibration-stage0-contract-20261029`.

SHA-256 values of the frozen references recorded before this task:

| Frozen file | SHA-256 |
|---|---|
| `test_phase3_AIF_simplified_v22.py` | `bfc871d5a9a32b2f5bdb9d7aab5177e702dec2e1110a6632f20bd13c20fcba30` |
| `experiments/calibration/r1_matched_correction_pilot.py` | `3208e2c52277ad76351eb86968e362aa315effe9e3c734bfe2b6ff7bab0db448` |
| `experiments/calibration/r2_0_label_blind_mechanism_audit.py` | `e2c9d9c6d22dba6a2222b84698d712dc70727d3f45d698e017d558b807b9a727` |
| `docs/calibration/audit/R2_0_LOCAL_CORRECTION_MECHANISM_AUDIT.md` | `b0b39bdc1ff98755710535030d4db32b9d75c289681ccd0bf0c759163fe488e9` |

SHA-256 of the new isolated reference and test sources:

| New file | SHA-256 |
|---|---|
| `experiments/calibration/r2_l1_locality_kernels.py` | `44b7f57e5542aff647d64b3a62efcc130ddce8d8c6840d250508476aee39d64f` |
| `tests/calibration/test_r2_l1_locality_kernels.py` | `691130ca983aeed3d6a91bdbe3a9701f2cb31e67297df6cee372ef81979df1c5` |

Final CPU verification command, with CUDA hidden:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& 'D:\anaconda3\envs\torch\python.exe' -m py_compile experiments/calibration/r2_l1_locality_kernels.py tests/calibration/test_r2_l1_locality_kernels.py
& 'D:\anaconda3\envs\torch\python.exe' -m pytest -q tests/calibration/test_r2_l1_locality_kernels.py
```

Verification command (Torch environment, CUDA hidden):

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& 'D:\anaconda3\envs\torch\python.exe' -m py_compile experiments/calibration/r2_l1_locality_kernels.py tests/calibration/test_r2_l1_locality_kernels.py
& 'D:\anaconda3\envs\torch\python.exe' -m pytest -q tests/calibration/test_r2_l1_locality_kernels.py
```

No historical source or output was modified. The full 30-category scan and every real-data radius decision remain unauthorized and unperformed.
