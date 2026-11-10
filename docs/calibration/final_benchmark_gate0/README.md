# Final Benchmark Gate 0 — Executive Summary

**Status:** `GATE0_R2_MINIMAL_PATCH_COMPLETE_RESEARCHER_REVIEW_PENDING`
**Revision date:** 2026-11-11
**Checkout:** `calibration-stage0-contract-20261029`, R2 baseline HEAD `f202d4055d0219dfa377e9612d5bb570c3afd96e`

This R2 revision incorporates three targeted corrections to the R1 protocol candidate. None is final researcher approval; this document does not freeze Gate 0 or authorize Gate 1.

The six-document Gate 0 set is:

1. `GATE0_SCIENTIFIC_CONTRACT_DRAFT.md`
2. `GATE0_DATA_AND_INFORMATION_ACCESS.md`
3. `GATE0_METHOD_AND_SOURCE_AUDIT.md`
4. `GATE0_METRICS_AND_RESOURCE_CONTRACT.md`
5. `GATE0_DECISIONS_AND_BLOCKERS.md`
6. `README.md` (this summary)

**Pending Gate 0 decisions:** query cap and `Margin_rho` details, including whether the proposed 0.003 realized-rate tolerance is approved; Real-IAD Group-atomic feedback; selection/group-key policy, especially VisA's group identity exception; exposure terminology; the primary statistical/practical decision rule and a separate Signed kNN non-inferiority margin `delta_K`; Static-36+32 construction and VisA eligibility; and whether to require a credible source-faithful external online comparison or limit complete-system competitive claims. N-shadow's scientific definition is simplified here; implementation fidelity remains a Gate 1 source-review requirement. Exact identity manifests are deferred to a separately authorized Gate 1 metadata operation and do not prevent Gate 0 design review.

## Current recommendation

- Real-IAD: **full-category benchmark reassessment on a historically exposed dataset**.
- Real-IAD `H_Q`: the **primary prespecified benchmark contrast** on that historically exposed dataset, not independent confirmation.
- VisA: **cross-dataset external validation**, with independent-confirmation status conditional on exposure and identity/provenance certification.
- Query budgets: equal caps mean equal maximum allowances; realized query counts/rates can differ. The 0.003 candidate rate-matching rule is an additional description, not a condition for the equal-cap BER comparison.
- Support/C64 construction: freeze the deterministic metadata-only role-selection rule in the Gate 0 decision, then generate actual image/group identity hashes only after separate Gate 1 authorization.
- Static-36+32: required static baseline for all 30 Real-IAD categories at P0; conditionally required for all 12 VisA categories at P0 if identity/group eligibility and provenance pass. Exact identity split, reference construction, byte cap and threshold remain pending researcher review.
- Feedback: recommend Group-atomic feedback for five-view Real-IAD Source Groups; this changes historical R7/D2/B/C event timing and must not be described as historical parity.
- N-shadow: passive, non-querying matched-feedback mechanism control that reads F-B's Greedy state and adds only negative C1 anchors after verified-normal feedback.

For the Full C1 versus Signed kNN contrast, a nonsignificant difference is not equivalence or non-inferiority. A formal Signed kNN non-inferiority claim requires an independently justified, approved positive `delta_K`; it is not automatically 0.01. Static-36+32 is a same-total-normal-information allocation comparator, not a mechanism ablation and not historical A4's dynamic D arm.

The strict-four-total study remains separate and on HOLD: existing evidence is four deployed Supports plus 64 additional C64 normals, while the tracked C0 decision requires its own data-access comparison, model freeze and authorization. No identity manifests were generated. No data, labels, masks, protected outcomes, GPU or experiments were accessed. No historical source, test, result, checkpoint or artifact was modified; no commit or push was made.

Gate 1 remains unauthorized. The independent metadata operation must be separately approved after Gate 0 freezes dataset releases, role algorithms, group/timing rules, seeds, order construction and manifest schema. Gate 2 resource ceilings remain measurement-dependent.
