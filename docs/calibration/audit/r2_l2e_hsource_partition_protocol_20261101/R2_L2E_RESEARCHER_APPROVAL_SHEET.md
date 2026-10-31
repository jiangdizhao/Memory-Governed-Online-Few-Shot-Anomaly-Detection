# R2-L2E — Researcher Approval Sheet

**Decision requested:** approve or revise the proposed H-SOURCE role policy. This sheet does not authorize dataset access, image/feature extraction, Support/C64 selection, G1, or G2.

**Policy:** `r2_l2e.h_source_partition_policy.v1`

**Policy SHA-256:** `73afab2b6f306043094bd5e3c1643a8c004c04ba2237be6ba43b6eca25642719`

**Public inventory:** 30 categories from the pinned Tencent config. **Role memberships:** none assigned or certified.

## Recommended defaults

| Decision | Proposed default |
|---|---|
| Generalization target | H-SOURCE; complete source groups are held out within category. No unseen-category claim. H-CLASS would require a new version and separate authorization. |
| A | All authorized TRAIN-normal groups in all 30 categories. A roles are parameter-design only. |
| B | Development TEST groups. `audiojack`, `regulator`, `rolled_strip_base`, and `switch` remain exposed development populations. For other ordinary categories, assign prior-exposed groups to B, then `floor(3N/5)` fresh groups by deterministic rank. |
| C | Sealed TEST groups. The four exposed development categories are proposed `NOT_APPLICABLE`. Mint/PCB are proposed C-only after method freeze and explicit protected-category access. Other ordinary categories receive the remainder of the fresh pool. |
| Assignment unit / seed | One `official_sample_key`, all five camera views together; existing digest/rank algorithms; role seed `20261101`. Remove any Test group that overlaps A. Reject all remaining cross-role/category overlap. |
| Minimums | A: 68 groups for four Supports + 64 additional C64, 100 for full G1-A geometry. B/C: 200 groups per category/role and at least 100 normal plus 100 defect source groups by sealed evaluator. G2: at least 20 queried groups per outcome class after the frozen tape. These are feasibility floors, not power guarantees. |
| Temporal protocol | Q-A-TS-v1, 3% of image events, 16 bins, query seed 37, at most one queried view per group. Order-stress seed `20261102`; stress variants preserve the event multiset and keep labels offline. |
| Support/C64 | Select four deployed Supports plus **64 additional** C64 images from A, one image per distinct source group. Share A/B/C membership across Support Seeds; seed changes only the subset inside A. Not strict four-shot-total. |

The 60:40 split is conditional on the verified fresh TEST group count. Without A-overlap or exposure exclusions, 498 fresh groups are necessary to leave 200 in C. That count cannot guarantee the required 100 groups per outcome; the evaluator checks outcome adequacy after assignment without letting labels affect the split. If any floor fails, report insufficient and do not change seed, ratio, or category boundaries post-outcome.

The six proposed N/A roles are B for Mint/PCB and C for the four exposed categories. Each requires a written reason, researcher approval, and independent review under the existing empty-role contract. Missing receipts are not empty roles.

## Independent gates

- A/B/C memberships require a new verified combined TRAIN/TEST source manifest, field-scoped TEST identity authorization, a verified prior-exposure registry, researcher approval bound to the policy SHA above, and independent role/disjointness attestation.
- The existing L2C signature and TRAIN-census scope verify, but the historical timing of its trust-root pin is unverified. An independent source-fingerprint baseline is not established. Neither fact is repaired by this proposal.
- New TEST label access requires a separate authorization. C outcomes remain sealed to an independent evaluator after method and analysis freeze. Mint/PCB TEST access needs explicit protected-category scope.
- Image decode/features for G1-A, native category-specific C0 Top-32/M0 receipts for G1-B, actual Support/C64 selection, and causal queried-anchor G2 each need their own applicable authorization.

## Researcher decision

Select one:

- [ ] **APPROVE** the exact policy SHA above, including the H-SOURCE scope, role dispositions, 60:40 rule, minima, and separate seeds.
- [ ] **REVISE**; issue a new versioned policy and SHA before any new record is opened.

Researcher: ____________________  Date: __________  Decision ID: `R2-L2E-FREEZE:<policy SHA-256>`

Independent reviewer/trust-root pin and all data-access authorizations remain separate gates. This signature alone does not authorize an operation.
