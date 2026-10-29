# Stage 0.75-B B8 mathematical mechanism and falsification report

**Primary method:** [`STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v3.md`](STAGE075B_IMPLEMENTATION_CONTRACT_CANDIDATE_v3.md)
**Preserved v2 report:** [`STAGE075B_B8_MATHEMATICAL_FALSIFICATION_REPORT_v2.md`](STAGE075B_B8_MATHEMATICAL_FALSIFICATION_REPORT_v2.md)
**Status:** synthetic mechanics and equation-fidelity evidence only. Native Greedy parity and Real-IAD performance were not tested.

## Version and fidelity record

V1 restricted candidate search to the reachable movement interval and was not primary evidence. V2 corrected reachability, but final source review found that B8-Full's objective still used raw-score/IQR distances and did not use the registered C64 percentile coordinate. Its run and audit package remain preserved and superseded. V3 is the primary matrix version: Full uses the current C64 percentile coordinate and transports the prior B8-relative percentile residual. The added equation regression independently reconstructs the objective and raw candidate.

## Frozen v3 mechanism

For sorted protected C64 scores, `F_C(a)=count(c_i<=a)/64`. Set `u3=F_C(theta_B3)`, `rho_prev=F_C,prev(theta_B8_prev)-F_C,prev(theta_B3_prev)`, and `u_trans=clip(u3+rho_prev,0,1)`. Full maximizes `DeltaJ(theta)-U-0.05*abs(F_C(theta)-u3)-0.05*abs(F_C(theta)-u_trans)` over the full finite raw-score candidate set. Exact ties choose distance to B3, then prior B8, then lower raw threshold. The selected global target is projected once into the `0.01` raw movement interval.

Predictions remain `score >= raw_threshold`. Current-version queried normals and distinct queried defects define `FPRhat=sum(n>=theta)/|N|` and `FNRhat=sum(d<theta)/|D|`; a missing class makes the objective undefined and requests B3. Evidence is unweighted, distinct-group, no-decay. C64 supplies rank/q90/IQR coordinates and does not fill the queried-normal denominator. `U` is a concentration-inspired heuristic, not a confidence bound.

## Mathematical scope and findings

A common strictly increasing scalar transform preserves finite ranks because `g(c_i)<=g(a)` iff `c_i<=a`. This does not cover non-strict transforms with merged values, class-specific or heterogeneous transforms, non-monotone memory effects, or concept shift. A finite CDF coordinate does not identify a unique raw decision boundary; equal ranks may yield different binary decisions for an intermediate score. The implementation therefore retains and logs raw thresholds.

The direct v3 objective regression passed. The complete Full raw threshold digest equaled B3 across all 48 trajectories; BA was defined in 42 and exactly equal in all 42, with S0/S5 remaining `null`. At 41 final two-class snapshots, mean `U` was 1.2222 (range 1.0361–1.9595); the best final penalized objective ranged -1.4828–-0.5385. S2 changed q90 with zero median shift; S8 exposed C64 mismatch; S14/control C showed indistinguishable prefixes through event 512 despite different future defect labels. Control D showed training delta-J `+0.5000` versus fixed synthetic future `-0.2857` for eight anchors and six candidates.

No theorem establishes C64 representativeness, queried-label representativeness, online risk control, or transfer of ACI/conformal guarantees. The frozen synthetic evidence does not support an incremental Full action; no post-result tuning was done.
