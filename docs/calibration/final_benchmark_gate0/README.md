# Final Benchmark Gate 0 — Executive Summary

**Status:** `GATE0_SCIENTIFIC_DESIGN_FROZEN_RESEARCHER_APPROVED`
**Revision date:** 2026-11-11
**Checkout:** `calibration-stage0-contract-20261029`, approved R2 base commit `2511d4f5b61efe869994b23afb398040eb49d270`

The researcher approved the six Gate 0 scientific decisions on 2026-11-11 following independent ChatGPT review of the R2 protocol. The scientific design is frozen at this level; this is not supervisor sign-off, data certification, implementation freeze, or Gate 1 authorization.

The six Gate 0 protocol documents remain:

1. `GATE0_SCIENTIFIC_CONTRACT_DRAFT.md`
2. `GATE0_DATA_AND_INFORMATION_ACCESS.md`
3. `GATE0_METHOD_AND_SOURCE_AUDIT.md`
4. `GATE0_METRICS_AND_RESOURCE_CONTRACT.md`
5. `GATE0_DECISIONS_AND_BLOCKERS.md`
6. `README.md` (this summary)

The separate [researcher approval and freeze record](GATE0_RESEARCHER_APPROVAL_AND_FREEZE.md) documents the six approvals.

**Approved design:** all six P0 protocol decisions recorded in the approval document. **Deferred dependencies:** actual identity/provenance manifests and VisA group verification require separate Gate 1 authorization; source-fidelity and implementation review remain Gate 1 work; resource limits remain Gate 2 measurements. `delta_K` is not approved and is required before any formal Signed kNN non-inferiority claim. The 0.01 BER value is not an approved hard success criterion. N-shadow remains a passive non-querying, normal-only matched-feedback control; its implementation fidelity remains subject to Gate 1 review.

## Researcher-approved scientific design

- Real-IAD: **full-category benchmark reassessment on a historically exposed dataset**.
- Real-IAD `H_Q`: the **primary prespecified benchmark contrast** on that historically exposed dataset, not independent confirmation.
- VisA: **cross-dataset external validation**, with independent-confirmation status conditional on exposure and identity/provenance certification.
- Query budgets: Real-IAD 3%, VisA 10%, category-level hard caps, approved `Margin_rho` and simplified BN+Band; equal caps mean equal maximum allowances, not equal realized spending. The approved 0.003 criterion governs only the additional approximately rate-matched description. No forced queries or top-ups.
- Support/C64 construction: use the approved deterministic metadata-only role-selection rule; generate actual image/group identity hashes only after separate Gate 1 authorization.
- Static-36+32: required for Real-IAD all 30 categories at P0; conditional for VisA all 12 categories at P0 if eligibility/provenance passes. Approved target: deterministic 36/32 identity allocation, single K-Center 8,192-prototype FP16 fixed bank, and C32 q90 static threshold, subject to Gate 1 source-fidelity checks.
- Feedback: approved Group-atomic feedback for five-view Real-IAD Source Groups; all five views are scored and sealed before purchased feedback changes model state. It differs from historical R7/D2/B/C event timing and is not historical parity.
- N-shadow: passive, non-querying matched-feedback mechanism control that reads F-B's Greedy state and adds only negative C1 anchors after verified-normal feedback.

The primary prespecified Real-IAD contrast is `H_Q` on a historically exposed dataset; it is not independent confirmation. VisA is cross-dataset validation, with any independent-confirmation claim conditional on provenance. A nonsignificant Full C1 versus Signed kNN difference is not equivalence or non-inferiority. A formal Signed kNN non-inferiority claim requires a separately approved positive `delta_K`; it is not automatically 0.01. Static-36+32 is a same-total-normal-information allocation comparator, not a mechanism ablation and not historical A4's dynamic D arm.

The strict-four-total study remains separate and on HOLD: existing evidence is four deployed Supports plus 64 additional C64 normals, while the tracked C0 decision requires its own data-access comparison, model freeze and authorization. No identity manifests were generated. No data, labels, masks, protected outcomes, GPU or experiments were accessed. No historical source, test, result, checkpoint or artifact was modified; no commit or push was made.

Gate 1 remains unauthorized. Actual data identities and VisA group provenance are not certified; any metadata operation requires separate authorization. Implementation/source-fidelity review and execution authorization are also deferred. Gate 2 resource ceilings remain measurement-dependent. Supervisor sign-off is not asserted.
