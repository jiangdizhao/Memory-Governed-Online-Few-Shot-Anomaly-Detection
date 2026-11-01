# R2-G1 Researcher Approval Sheet

**This is a decision form, not an authorization. No box is checked.** The present task authorizes preparation only; it does not permit local data access, image decoding, DINO loading, feature extraction, Support/C64 selection, G1 execution, G2, TEST metadata, labels, or masks.

## A. Protocol decisions

Proposed protocol: `r2_g1.preexec.v1`.

- [ ] Approve the H-SOURCE role policy SHA-256 `73afab2b6f306043094bd5e3c1643a8c004c04ba2237be6ba43b6eca25642719` from R2-L2E. This approves a proposal only; it does not certify any membership.
- [ ] Approve camera ID 1 for G1-FULL-A; G1-PROTOCOL-MATCHED must use the existing D2 seeded view selector.
- [ ] Approve bounded group rank rule `r2-g1-bounded-pilot-v1`, sample seed 0, 100 geometry groups/category. This seed is independent of the L2E role seed and Support seed.
- [ ] Approve the proposed first G1-A category `switch`; consider `audiojack` only after review of the first report.
- [ ] Approve the matched G1-B planning set of four Supports plus 64 additional C64 groups (68 distinct groups/category); the 64 C64 images are both query and reference geometry rows with source-group leave-out; use history seed 190021, Support Seed 0, and view seed offset 891 for the initial pilot. C64 is not strict four-shot-total.

## B. Separate execution authorizations

Each approved operation must be covered by a new signed scope with an independent trust-root and source-fingerprint pin, exact execution commit, category list, unique run ID, and exact output limits.

- [ ] **A identity-only preflight:** permit count/group-digest projection for the named TRAIN-normal A population. No TEST fields, image bytes, labels, or masks.
- [ ] **G1-A one-category pilot:** permit image decode, DINO feature extraction, and 49-point descriptor projection for up to 100 certified A groups in `switch` under `realid-offical-sample-key-v2`; use the same sample for query/reference with source-group exclusion and compute G1-A distances. This does not depend on B/C membership certification.
- [ ] **G1-A two-category extension:** separately permit the same work for `switch` and `audiojack`, up to 100 geometry groups/category, after review of the one-category result.
- [ ] **Support/C64 selection:** separately permit selection of four deployed Supports and 64 additional C64 groups from certified A for G1-PROTOCOL-MATCHED; use Support Seed 0 initially and name the category explicitly.
- [ ] **G1-B switch pilot:** separately permit category-specific fixed-bank/M0 native C0 scoring and actual Top-32 receipt generation for the 64-image C64 geometry cohort in `switch`, seed 0.
- [ ] **G1-B audiojack extension:** separately permit a distinct `audiojack` fixed bank, M0 certificate, and native C0 Top-32 receipts. No switch receipt may be reused.
- [ ] **G1 distance summaries:** only after the relevant A/cache and, for G1-B, Support/M0/Top-32 receipts have independent verification.
- [ ] **B/C membership certification:** separate future decision only; requires TEST identity/exposure authorization and independent reviewer attestation.
- [ ] **G2:** not included. Requires separate future protocol and authorization.

## C. Proposed bounded pilot envelope

Applies only if separately authorized:

- Dataset root: `G:\dataset\realiad`; metadata root: `G:\dataset\realiad\jsons\realiad_jsons`.
- Allowed records: TRAIN-normal A groups in the specifically approved category only.
- G1-FULL-A view: camera ID 1; G1-PROTOCOL-MATCHED view: existing D2 seeded selector (history seed 190021, Support Seed 0, view offset 891). No score/label-based view selection.
- Unique output namespace example: `outputs/calibration/r2_g1_pilot/<run_id>`; reject existing paths, never resume another run implicitly.
- Maximum output bytes: **1,073,741,824** (1 GiB), including projected features, receipts, manifests, logs, and temp writes.
- Minimum free space before launch: **10 GiB**.
- Runtime: a hard, child-process deadline no greater than 120 minutes with verifiable Windows process-tree containment. A G1-specific command and adapter must be reviewed before approval; the R1 runner authorization does not cover G1.
- No source images, full feature maps, raw sample keys, raw paths, labels, masks, private feedback, or correction anchors in output.

## D. Evidence required before signing an execution scope

1. Researcher approval of the G1 view/sample rule and, when needed, the L2E H-SOURCE policy.
2. New signed access manifest specifically covering the requested TRAIN-normal identities/images/features/operation. The historical L2C grant permits only the TRAIN metadata census.
3. A independently verified availability/identity receipts for every requested category.
4. Independently pinned trust-directory and source-fingerprint digests for this new scope; historical trust-root pin timing remains unverified.
5. Reviewed G1 adapter, exact source/weight fingerprints, bounded feature-cache lifecycle, output-budget checks, and command.
6. For G1-B, independently verified category/seed Support bank, M0 state/certificate, and native C0 Top-32 receipts.
7. Independent reviewer attestation where source or role membership certification requires it.

Researcher: ______________________________
Decision date: ___________________________
Approved protocol version / policy SHA: __________________________________
Signed authorization ID (if any): ________________________________________
Exact category and run scope: _____________________________________________
Independent reviewer / attestation ID: ___________________________________

A signature on this sheet alone is not a dataset-access authorization. No credentials, passwords, or private signing keys are requested.
