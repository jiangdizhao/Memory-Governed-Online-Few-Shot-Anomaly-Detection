# Stage 0.75-B B8 mathematical mechanism and falsification report

**Primary implementation:** [`STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v2.md`](STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v2.md)
**Superseded candidate:** [`STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v1.md`](STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v1.md)
**Status:** synthetic mechanics findings only. No native Greedy or Real-IAD performance claim.

## Frozen mechanism

For the current protected C64 scores (c_{1,t},\ldots,c_{64,t}), the controller uses the right-continuous finite empirical CDF

\[
\widehat F^C_t(a)=64^{-1}\sum_{i=1}^{64}\mathbf 1[c_{i,t}\le a].
\]

This is a C64-relative rank coordinate. It is not a defect probability, a deployment normal CDF, or a physical-defect FNR estimate. C64 contributes rank, q90, and scale coordinates only; its scores are not pooled into the queried-normal risk denominator.

The executed decision remains `score >= raw_threshold`. For queried, current-state, source-eligible normal scores (N_t) and distinct queried defect scores (D_t), the registered descriptive quantities are

\[
\widehat{FPR}_t(\theta)=\frac{\sum_{n\in N_t}\mathbf1[n\ge\theta]}{|N_t|},\quad
\widehat{FNR}_t(\theta)=\frac{\sum_{d\in D_t}\mathbf1[d<\theta]}{|D_t|},
\]

and \hat{\Delta J} is the sum of the paired risk reductions relative to the actual B3 threshold. A missing denominator is undefined and forces B3 fallback. Each eligible group contributes once, unweighted and without temporal decay. Anchor rescoring changes score state, never independent label count.

The full arm maximizes the frozen B3-relative objective over the global finite raw-score candidate grid, with `lambda_base=lambda_move=0.05`, a C64-IQR scale floor of `1e-12`, and a `0.01` raw-score execution movement cap. The reliability term is

\[
U=\sqrt{\frac{\log(2K/0.05)}{2n_N}}+\sqrt{\frac{\log(2K/0.05)}{2n_D}},
\]

where (K) is the number of global raw candidates. It is a concentration-inspired **heuristic penalty**, not a valid confidence bound. Selective disclosure, temporal dependence, adaptive memory, candidate selection, and source clustering block automatic confidence-coverage interpretation.

## What is mathematically established

If every reference and evaluation score is transformed by the same strictly increasing scalar function (g), then

\[
\mathbf1[g(c_i)\le g(a)]=\mathbf1[c_i\le a]
\]

for every anchor and evaluation value, including exact ties. Therefore the finite CDF coordinate and ranks are preserved. This argument does not apply to non-strict transforms that merge distinct values, class-conditional transforms, heterogeneous anchor transforms, or memory changes that alter score rankings non-monotonically. The diagnostic tests passed for common translation, positive affine, and strictly increasing nonlinear transformations with ties.

The score decision is preserved under a common strictly increasing transform only if the raw threshold is transformed consistently as (g(\theta)). A finite CDF value does not identify a unique raw threshold: in the tested 64-rank grid, raw thresholds 4.1 and 4.9 have the same CDF coordinate while a score of 4.5 receives different binary decisions. The implementation therefore logs and executes a raw threshold and never silently treats coordinate equality as prediction equality.

No theorem establishes that C64 represents deployment normals, that queried labels are representative, or that current-memory rescoring preserves defect concepts. No guarantee transfers from ACI, conformal methods, or confidence sequences to this sparse-feedback dynamic-memory controller.

## Falsification findings

* **S2 tail change:** a sparse upper-tail modification left the C64 median unchanged (`0.0` shift) while higher-q90 moved from `57.0` to `57.5`. Median translation therefore misses a relevant tail change in this fixture.
* **S3 asymmetric classes:** normal-reference rank consistency did not stabilize the defect distribution; the full controller stayed at B3 under its evidence penalty. This is consistent with the mathematical boundary, not evidence that B3 is generally safe.
* **S7/S8:** concept drift and C64/stream-normal mismatch remained distinct. In the mismatch scenario, Full reported descriptive BA `0.6123`, FPR `0.4219`, and FNR `0.3535`; C64 was not used as a stream FPR denominator.
* **S14/control C:** two worlds had identical controller-visible histories through event 512, then different future defect labels. Their predictions, scores, memory and query histories were identical through that event. Normal-only information cannot identify which future defect threshold is preferable.
* **Control D:** with eight queried sources and six searched thresholds, in-sample delta-J was `+0.5000` while the independent synthetic future delta-J was `-0.2857`. This is a bounded candidate-selection optimism example, not a population estimate.
* **Full-arm activation:** across 48 scenario/seed trajectories, the full-arm threshold digest matched actual B3 in all rows. Across the 41 rows with both queried classes, the final heuristic (U) averaged `1.2222` (range `1.0361`–`1.9595`), while the best unpenalized empirical delta-J averaged `0.1989` and never exceeded the complete penalized objective. B8-Full therefore always selected B3 fallback under the frozen candidate v2 values.

## Interpretation

The percentile-coordinate invariance is valid under its narrow common strictly increasing transform assumption. The synthetic suite does not support an incremental B8-Full decision mechanism: the registered reliability penalty suppressed every deviation, and B3+Offset performed better descriptively. The heuristic may be too conservative, but reducing it after seeing these outcomes would be post-hoc method tuning and is not done here. A fresh version and separate authorization would be required for any revision.
