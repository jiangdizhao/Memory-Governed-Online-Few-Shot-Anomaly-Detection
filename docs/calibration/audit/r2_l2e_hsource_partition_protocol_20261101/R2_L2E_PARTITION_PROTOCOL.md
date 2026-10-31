# R2-L2E — H-SOURCE Partition Protocol Proposal

**Status:** `PROPOSAL_READY_FOR_RESEARCHER_APPROVAL`

**Not certified:** A/B/C memberships, Support/C64 identities, G1-A/G1-B descriptors, G2 anchors, and any execution authorization remain pending. This report contains no Real-IAD group identities, Test counts, image features, outcomes, or local data scan.

## 1. Scope and evidence boundary

This protocol freezes a deterministic recommendation for assigning complete Real-IAD source groups to parameter-design role A, development role B, and sealed confirmation role C. It does not create any role membership or authorize metadata access, feature extraction, Support/C64 selection, G1, or G2.

The work was performed on branch `calibration-stage0-contract-20261029`, starting at HEAD `900dc6bffd0120b28ac008b3d0437f38a9b4518e`. The tracked tree was clean at start. Existing untracked audit/output artifacts were preserved and not read or staged for this task.

The public catalog is pinned to Tencent `98c28ffbe95b380d17a3777d77219152368c19e`, `experiments/RealIAD-full/config.yaml`, SHA-256 `9c17d4ba78210ad65a288fedcdd6ea8a65d0926da05a960d3ed8dd1c3210552f`, size 5,574 bytes. The public inventory contains the 30 category identifiers listed by the existing L2A/L2B contracts. A public category list is not local-data authorization.

Prior L2C provenance remains as recorded by [R2-L2D-R](../r2_l2d_provenance_recovery_20261101/R2_L2D_R_PROVENANCE_RECOVERY_REPORT.md): authorization SHA-256 `969fee786585fddfae88d1b3ecf19f2cd36a91e66ee614b6c0d9e9f57eee2587` verifies with TRAIN-census scope. The Stage B census report SHA-256 is `8aa27fb725002abc26e2a9d257ca065b4f86e789db1552357a7849212cf7fcce`; it reports 36,465 images, 7,293 groups, five views per group, and a minimum 167 TRAIN-normal groups/category. The researcher-provided trust-directory digest `b5e29d2d4efaa934054d36b6d7aa56ff122270c159ae6d7f4a7405a0a64c93c5` matches the current file, but contemporaneous evidence that this hash was independently pinned before Stage B is unavailable. An independent source-fingerprint baseline is also not established. The matching hashes inside a single Stage B output pair establish internal consistency only.

The existing TRAIN census is a candidate inventory, not certified A membership. Its minimum 167 groups/category is numerically above the 68-group support/C64 floor and the proposed 100-group full-geometry floor; this does not establish authorization, normality receipts, disjointness, or descriptor availability. The existing L2C authorization does not grant Test metadata or label access.

The [Stage 0.5 exposure ledger](../../STAGE05_DATA_EXPOSURE_LEDGER.md) records full Test-label access for `audiojack`, `regulator`, `rolled_strip_base`, and `switch`. Those outcomes are exposed exploratory information. They cannot be represented as untouched C outcomes. The ledger records no Mint/PCB Test outcome access; this proposal keeps those categories protected and C-only. No outcome values or local Stage 1/2 artifacts were opened for this task.

## 2. Recommended H-SOURCE role populations

The independence unit is one complete `official_sample_key` group, hashed with the existing `sha256:official-sample-group-nul-prefix-utf8.v1` algorithm. All five camera views belonging to a source group remain together. A group, its views, or a duplicate alias cannot be split across roles. Pairwise disjointness is global across categories and A/B/C.

| Role | Recommended population | Use and current status |
|---|---|---|
| A | All authorized TRAIN-normal source groups in all 30 categories | Parameter-design reference. Candidate census exists; no A role receipt or image/feature access is certified. |
| B | Development TEST groups under the dispositions below | Development and feasibility only, under a new scoped authorization. Existing four-class results remain explicitly `EXPOSED_DEVELOPMENT`. |
| C | New, sealed TEST groups under the dispositions below | Confirmation only after protocol, methods, source code, and analysis are frozen. An independent sealed evaluator controls outcomes. |

### Category dispositions

| Categories | A | B | C |
|---|---|---|---|
| `audiojack`, `regulator`, `rolled_strip_base`, `switch` | All authorized TRAIN-normal groups | Previously exposed development population; exact B identities require an authorized receipt or separately scoped identity read | `NOT_APPLICABLE` proposed because outcomes were already accessed; not an untouched confirmation set |
| `mint`, `pcb` | Authorized TRAIN-normal groups only | `NOT_APPLICABLE` proposed; keep protected | All eligible TEST groups in sealed C after a method freeze and explicit protected-category authorization |
| `bottle_cap`, `button_battery`, `end_cap`, `eraser`, `fire_hood`, `mounts`, `phone_battery`, `plastic_nut`, `plastic_plug`, `porcelain_doll`, `sim_card_set`, `tape`, `terminalblock`, `toothbrush`, `toy`, `toy_brick`, `transistor1`, `u_block`, `usb`, `usb_adaptor`, `vcpill`, `wooden_beads`, `woodstick`, `zipper` | All authorized TRAIN-normal groups | 60% of fresh, eligible TEST groups, plus any previously exposed groups | Remaining 40% of fresh, eligible TEST groups, sealed |

The six proposed `NOT_APPLICABLE` slots are B for Mint/PCB and C for the four exposed development categories. None is an empty-role certificate. Each requires researcher approval of its reason and the existing L2B frozen empty-role policy plus independent review. Missing identity receipts remain pending; they never imply an empty role.

This is **H-SOURCE**, not H-CLASS. All 30 category identities inform the A design pool. A successful C result would support only within-category, source-group-held-out evidence for the categories with an eligible C population. It would not support unseen-category generalization. If that claim is later required, freeze a separate H-CLASS policy with explicit held-out categories before accessing their data.

## 3. Deterministic assignment and role-manifest contract

The machine-readable, membership-free policy is [R2_L2E_HSOURCE_PARTITION_POLICY.json](R2_L2E_HSOURCE_PARTITION_POLICY.json). Its canonical SHA-256 is `73afab2b6f306043094bd5e3c1643a8c004c04ba2237be6ba43b6eca25642719`. The fixed H-SOURCE role-partition seed is `20261101`. It is distinct from any Support Seed, stream-order seed, and query-tape seed.

For each ordinary category, first construct a verified combined source manifest that binds the exact TRAIN and TEST source fingerprints, official catalog receipt, parser/identity-contract hashes, authorizations, and candidate-universe digest. For the 24 categories outside the four historically exposed categories and protected Mint/PCB, require a verified, independently reviewed prior-exposure registry for that same source snapshot. Then:

1. Keep all A-candidate TRAIN-normal groups in A. Exclude any TEST group whose digest appears in A; record the number excluded. A has priority on source overlap.
2. Assign every previously exposed TEST group to B. Remove those groups from the fresh split pool; never place them in C. The four Stage 0.5 development categories are treated as exposed at category level, so all of their eligible groups are B and their whole C role is proposed N/A. For the other categories, a verified group-level exposure registry is required before splitting.
3. Let `N` be the number of remaining fresh TEST groups. Rank each digest by ascending

   `SHA256("sha256:r2-l2d-hsource-group-rank.v1\0" + policy_sha256 + "\0" + category_id + "\0" + 20261101 + "\0" + group_sha256)`

   with the group digest as the deterministic tie-break. This reuses the existing L2D ranking algorithm without changing L2D.
4. Assign the first `floor(3*N/5)` ranked groups to B and the remainder to C. Thus `n_B=floor(3N/5)` and `n_C=N-n_B`; any integer remainder goes to C. The 60:40 ratio applies only to fresh eligible groups. Previously exposed groups increase B and are excluded from C.
5. For the four exposed development categories, route all eligible TEST groups to exposed B and propose C as `NOT_APPLICABLE`. For Mint/PCB, propose B as `NOT_APPLICABLE` and C as all eligible, unexposed TEST groups. Any prior exposure in Mint/PCB blocks this policy and requires a new researcher-approved version.
6. Reject duplicate/malformed source-group identities, mismatched source snapshots, missing exposure evidence, unknown categories, and any cross-category or cross-role overlap. Do not repair a shortage by changing the seed, ratio, or category boundary after outcomes are inspected.

The proposer in `experiments/calibration/r2_l2e_partition_protocol.py` accepts only a category ID, source-manifest digest, group digests, view counts, and explicit authorization/exposure gate assertions. It accepts no labels, paths, masks, features, scores, or images. The Boolean gate inputs are not credentials: a future caller may set them only after the existing L2B authorization/signature and field-scope validators succeed. The code always returns a `PROPOSED` or `PENDING` state; it cannot certify roles.

### Required provenance in future receipts

Reuse the L2B `RoleIdentityReceipt`, `ReviewedIdentityManifest`, `ResearcherApproval`, `IndependentReviewerAttestation`, and `ReviewerTrustDirectory` schemas and validation. A combined category source manifest must bind both TRAIN and TEST metadata fingerprints so A/B/C receipts share one source snapshot, as required by L2D. Public receipts contain only category, role, group digests, view counts, source-manifest SHA, policy SHA, authorization ID, and proposal/receipt hashes. No raw sample key, image path, label, mask, feature, or private key is included.

The researcher’s existing L2B H-SOURCE approval verifier must first validate the signature, approved proposal, and independently pinned trust directory. Additionally, the signed `decision_id` must equal `R2-L2E-FREEZE:<L2E policy SHA-256>`; `researcher_approval_binding_errors()` checks this semantic link and the complete 30-category/A-B-C/protected-category coverage. The reviewer trust directory must independently pin the researcher/reviewer keys, source fingerprints, the L2E policy hash, and the reviewed empty-role policy. A current-file digest supplied after the fact is not evidence that the trust root was pinned before execution.

Before any future Test identity work, a custodian needs a new signed scope naming the exact roots, categories, sections, metadata fields, operation, output boundary, and authorization ID. Parsing the existing official sample key can encounter `OK/NG` condition tokens. The custodian may read the allowlisted identity string only to compute the frozen group digest; the partition operator receives digests and five-view counts, not the token or labels. The current L2C TRAIN authorization does not cover this operation. B outcome labels require separate B evaluation scope. C outcomes require a later, separate authorization to a sealed evaluator after method freeze; only queried labels may be disclosed to the frozen online controller, after prediction.

## 4. Group-count, support, C64, and temporal feasibility rules

### Minimums and failure handling

| Gate | Proposed minimum | If insufficient |
|---|---|---|
| A, protocol-matched support | 68 distinct normal TRAIN source groups/category: four deployed Supports plus 64 additional C64 groups | G1 protocol-matched and its C64 selection are infeasible for that category. |
| A, full G1-A geometry | 100 distinct normal source groups/category, before the source-disjoint query/reference exclusions | Report `INSUFFICIENT_GROUPS_FOR_FULL_GEOMETRY`; do not substitute a B/C group. |
| B or C, category population | 200 distinct source groups/role/category, and a separate sealed count check of at least 100 normal and 100 defect groups for per-class rates | Mark the affected category/role infeasible. The total-group threshold is necessary, not sufficient; class counts are checked by the authorized evaluator only after assignment and must not affect assignment. |
| G2 feedback trajectory | At least 20 queried source groups of each outcome class after the frozen label-blind query tape is applied | Mark G2 anchor-to-future evidence insufficient. Do not change the query tape after seeing labels. |

With the 60:40 formula and no A-overlap exclusions, at least 498 fresh groups are necessary to leave 200 groups in C (`ceil(0.4*N) >= 200`). This is only a necessary group-count threshold. It does not ensure either role has 100 groups of each outcome or enough queried anchors. For G2, a 3% query budget needs at least 1,334 image events to make 40 queries arithmetically possible; with five views/group this is at least 267 complete groups. Neither length guarantees 20 queried groups per outcome because query selection is label blind. Actual B/C Test counts and outcome composition remain unknown and uninspected.

The 100-per-outcome floor is a planning screen, not a power guarantee: for a worst-case Bernoulli rate, 100 independent groups yield an approximate 95% normal half-width of 9.8 percentage points. Report exact group-cluster uncertainty (and its method/seed) and mark any wider interval as imprecise. Do not count five camera views or patch descriptors as five or many independent source groups. Macro, pooled-with-named-denominator, worst-category, and lower-tail summaries remain distinct.

### Support and C64 inside A

For every separately approved Support Seed, select four distinct normal deployment Supports and 64 **additional** normal C64 images from A. Each image comes from a different official source group; all 68 groups are mutually disjoint within that seed and each image's camera view is selected by a separately frozen rule. Support and C64 selection never consume B or C groups. Role membership is shared across Support Seeds; a Support Seed changes the selected subset inside A, not A/B/C membership. C64 remains read-only and never enters online memory. This is four Supports **plus additional C64**, not strict four-shot-total. At least 68 eligible A groups is necessary, not sufficient evidence of a valid bank.

Use the existing L2C `r2_l2c.support_c64_selection.v1` receipt and L2D `r2_l2d.support_c64_fixed_bank_binding.v1` contract. G1-B additionally needs the category-specific fixed-support and M0 scorer-state receipt. The historical structured switch/seed-0 bank digest remains switch/seed-0 only; it cannot identify another category/seed. Missing or unverified native receipts remain `NOT_YET_IDENTIFIABLE`.

## 5. G1-A, G1-B, and G2 are separate measurements

### G1-A — predefined normal spatial geometry

Reuse the existing L2A G1 implementation and frozen R1 representation contract. For each 28x28 patch map, take exactly 49 cell centers at row/column coordinates `{2,6,10,14,18,22,26}`. This is a fixed spatial sample, not score- or label-based Top-K. The cache/descriptor receipt must bind the encoder/weight fingerprint, preprocessing, hook layers, float dtype, normalization, category, A source manifest, role receipt, image/group digests, and sampling policy.

Keep `compare_source_disjoint()` mathematics unchanged: exact blockwise cosine distance `1 - clip(float64_dot, -1, 1)`; exclude same `official_sample_key` and same image before top-k; exact nearest and fifth-nearest values; deterministic ties by distance then original reference-row index; and strict `distance < radius`. Report nearest and fifth-nearest distributions, descriptor/image/group denominators, category-equal macro and explicitly weighted pooled summaries, worst category and lower tail, and source-group bootstrap intervals. The radius grid is descriptive and remains unselected; no threshold or radius is chosen by this task.

### G1-B — native Top-32 geometry

Use the frozen native C0 Top-32 selection indices from the correct category's pre-score M0 state. Bind the receipt to that category's own Fixed Bank, support seed, scorer profile, representation, baseline memory certificate, and source image. Never select Top-32 by locally sorting or reranking descriptors. A switch/seed-0 receipt cannot transfer to another category or seed. Require the independent source receipt, fixed-support receipt, and M0 certificate; otherwise return `NOT_YET_IDENTIFIABLE` and do not compute G1-B descriptors.

### G2 — causal queried-anchor-to-future study

G2 requires a separately authorized causal stream and legal C1/C2 queried-feedback anchors with event-time and pre-score memory-state provenance. A candidate anchor is available only after the event has been scored, predicted, queried under the frozen tape, and its label disclosed. Compare only against strictly future native patches; exclude same-source groups and current/future anchors. Record anchor sign, age, event index, memory version, query/update time, source-group digest, and scorer/feature identity under protected receipts. No private feedback or anchor ledgers were read here. G1 background normals do not establish G2 anchor density, sign balance, age, transport, or correction locality.

## 6. Temporal-stream contract

Use Q-A-TS-v1 only under its separately approved profile: 3% of image events, 16 frozen time bins, query seed 37, and at most one queried view per `official_sample_key`. The query decision is precommitted and label blind. Keep the query seed separate from the role seed `20261101` and the order-stress seed `20261102`.

For B-only stress studies, preserve exactly the same source-group/image event multiset in every order. Freeze the five-camera within-group expansion order. Rank groups by `SHA256("sha256:r2-l2e-temporal-order.v1\0" + 20261102 + "\0" + group_sha256)`, with group digest as tie-break. Register these order builders before reading labels: `RANDOM` (ascending rank), `NORMAL_FIRST`, `DEFECT_FIRST`, and `BURSTY_50_GROUP` (alternate 50-group class blocks starting with normal; if a class is exhausted, append the remaining class; the last block may be shorter). A separate evaluator may use B labels offline to build the latter three order vectors; no label, future-class flag, or hidden class histogram enters the controller. C receives only the single frozen primary order unless a future signed confirmation protocol explicitly permits stress variants.

Use the same group-level query mask across orders only if it satisfies Q-A-TS bin/budget constraints under every order. If it does not, freeze a separately named `order-plus-query-profile` comparison and do not attribute its result to order alone. After assignment, an evaluator checks queried normal/defect group counts. Failure to reach the G2 minimum is reported as insufficient; no outcome-informed query selection, reordering, or seed search is allowed. A temporal-stress tape created from labels is never passed to the online scorer/controller.

## 7. Gate and access matrix

| Operation | Evidence/authorization needed before operation | What is not implied |
|---|---|---|
| A role identity receipt | Researcher-approved L2E policy; normal-only TRAIN source-manifest linkage; existing L2B receipt schema; independent role review and pinned trust root | An existing verified Stage B TRAIN receipt may be reused if its provenance/source-root gates pass. The L2C census alone does not certify membership or authorize image decode/features. |
| A feature extraction for G1-A | Separate signed permission for exact TRAIN-normal categories/root, image decode, frozen DINO source/weights and preprocessing, feature output path/budget, and source-cache fingerprint | This L2E proposal authorizes no DINO load or descriptor extraction. |
| TEST identity-only B/C split | New signed identity-field authorization; explicit `official_sample_key` parsing/hash scope; verified combined TRAIN/TEST fingerprint; group-level prior-exposure registry for the 24 ordinary categories; researcher approval of exact L2E hash | Public category inventory and current TRAIN authorization do not authorize TEST metadata. |
| B outcome evaluation | Separate B label/evaluation scope after role assignment; evaluator-only labels before metrics; protocol freeze per study | B access does not authorize C or masks. |
| C confirmation | Researcher method/protocol freeze; independently reviewed C identity receipt and seal; separate C outcome authorization to an independent evaluator; protected categories explicitly named | Researcher-facing early C counts, outcomes, masks, or performance-driven changes are not permitted. |
| Support/C64 selection | Certified A receipt, frozen selection seed/rule, authorized normal identity pool, 4+64 disjoint receipt and independent review | A role proposal does not select Supports or C64. |
| G1-A / G1-B | Separate execution authorization after relevant A/source/cache receipts; G1-B also needs native category M0/Top-32/fixed-bank receipts | This protocol neither scans descriptors nor selects radii. |
| G2 | Separate execution authorization, legal B/C query scope, frozen stream/tape, causal anchor provenance, and native future-patch receipts | G1 distance statistics are not G2 evidence. |

No TEST metadata, class-condition distributions, labels, masks, features, Mint/PCB TEST identities, or private feedback were accessed to write this proposal. No Census was rerun.

## 8. Leakage, independence, and remaining blockers

The independent unit is a source group, not an image view or patch. Source IDs with known previous outcome exposure are excluded from C. The four fully exposed Stage 0.5 development categories are C `NOT_APPLICABLE`, pending the formal reviewed empty-role policy. Mint/PCB remain protected and C-only. The full C source-group manifest and outcome authorization are controlled by an independent reviewer/evaluator; developers receive no group-to-label table. The C run uses the frozen method; queried labels enter its controller only after prediction under the precommitted tape. C results are released only after the sealed run and analysis are complete.

The following gates are independent; none implies another:

1. **L2E engineering/protocol:** synthetic proposer and documentation are prepared; researcher approval is pending.
2. **L2C historical authorization provenance:** signature and TRAIN-census scope verify; historical trust-pin timing remains unverified; independent source-fingerprint baseline remains unestablished.
3. **A/B/C identity certification:** pending authorized source identities, verified exposure registry, source-manifest linkage, six explicit N/A approvals, global disjointness, and independent attestations.
4. **Support/C64 identity:** pending; no selection or category/seed bank binding was made.
5. **G1/G2 execution:** not authorized; no features, distances, native Top-32 run, query-anchor stream, or study was produced.

Before any new local record is opened, obtain researcher approval bound to this policy SHA and independently pin the trust root and source-fingerprint baseline. Then obtain field-scoped identity authorization. The independent reviewer must attest role identities and disjointness, including all N/A decisions. B labels, C sealed labels, protected Mint/PCB access, image/feature extraction, Support/C64 selection, G1, and G2 each require their own applicable authorization. If source counts, per-class group counts, query counts, or exposure exclusions miss the frozen minima, report `INSUFFICIENT`; do not repair by looking for a favorable seed, moving groups after outcomes, changing the 60:40 ratio, or merging categories.

## 9. CPU-only implementation provenance

The focused tests use synthetic SHA-256 group digests and synthetic five-view counts only. They exercise deterministic ranking/quotas, A-overlap removal, exposure routing, exposed/protected dispositions, missing and insufficient populations, access gates, view grouping, global overlap, policy/receipt tampering, and researcher approval binding. They do not import DINO weights, read dataset metadata, or run G1/G2.

Test result and final source/policy hashes are recorded in the completion report after the final focused regression run. This proposal is not a scientific result and does not claim role certification or evaluation readiness.
