# R2-L2B All-Class Data Role Certification and G1 Reference Pool Preparation

**Audit date:** 2026-11-01
**Branch:** calibration-stage0-contract-20261029
**Starting HEAD:** b030db170322393c08c03ed2d8d1c0bcf34c6819
**Classification:** R2_L2B_PROTOCOL_READY_FOR_RESEARCHER_APPROVAL
**Role certification:** R2_L2B_ROLE_CERTIFIED not reached
**Local metadata access:** PENDING_AUTHORIZATION
**G1 execution:** NOT AUTHORIZED
**Real-data distance statistics:** NOT RUN

## Scope and result

This work prepares a versioned H-SOURCE role proposal, a default-deny local
access template, and a CPU-only identity certification contract. It did not
read any local dataset JSON, image, feature cache, label, mask, private
feedback, Mint/PCB material, or source-group identity. It did not load DINO,
extract features, run G1, choose a radius, or evaluate outcomes.

The previously verified category-only receipt at
R2_L2A_R_OFFICIAL_PUBLIC_CATALOG_RECEIPT.json names the pinned Tencent
Real-IAD config at commit 98c28ffbe95b380d17a3777d77219152368c19e9 with
SHA-256 9c17d4ba78210ad65a288fedcdd6ea8a65d0926da05a960d3ed8dd1c3210552f.
It contains the same 30 public train/test category identifiers. This public
inventory is a source-provenance input only; it does not authorize local
metadata or establish a partition.

H-SOURCE is recommended because the protocol can retain all category identities
while holding out source groups. It does not support a claim of generalization
to unseen category identities. H-CLASS remains an alternative only after an
exact category holdout set is frozen in a new policy version. No exact group
assignment or H-CLASS set has been selected.

## Implementation and contract

New source and test files:

- experiments/calibration/r2_l2b_role_certification.py
- tests/calibration/test_r2_l2b_role_certification.py

New records:

- R2_L2B_PROPOSED_30_CLASS_ROLE_POLICY.md: readable proposal and unapproved
  researcher decision record.
- R2_L2B_PROPOSED_30_CLASS_ROLE_POLICY.json: canonical, digest-bound 90-row
  category by A/B/C proposal.
- R2_L2B_DATA_ACCESS_APPROVAL_TEMPLATE.json: default-false authorization
  template.

The 90 proposed assignments use A for parameter-design reference, B for
development evaluation, and C for sealed confirmation. Every assignment is
source-group based under identity contract realid-offical-sample-key-v2 and
grouping unit official_sample_key. The existing Real-IAD filename/path parser
is reused for future authorized inputs; it was exercised only with synthetic
path strings. Camera views that share one official_sample_key receive one
domain-separated group digest and cannot be split across roles. Receipts store
group digests and view counts; they do not store raw sample keys, image paths,
labels, or masks.

The proposal marks Mint and PCB PENDING_AUTHORIZATION_PROTECTED. Any other
protected confirmation groups must also remain sealed until the researcher
names them. The proposal has no role counts or availability claims.

### Independent review and fail-closed status

Role certification compares a producer receipt with a separately supplied
identity manifest and a signed reviewer attestation. Reviewer and researcher
approvals use Ed25519 signatures. The trust directory binds reviewer public
keys, identity-source digests, reviewed empty-role policy digests, A availability
receipt digests, and approved proposal digests. Its own SHA-256 must be pinned
out of band. No production trust directory or researcher signature was
provided here, so no category role is certified. Without the Ed25519 dependency,
signature verification fails closed.

Content digests are not signatures. A producer cannot certify its own receipt,
substitute an untrusted reviewer's name, use a digest-only self-attestation, or
certify C from an unreviewed digest. A/B/C source-group overlap raises a
contract error. An absent role receipt remains PENDING_MISSING_RECEIPT and
cannot be treated as an empty role. An explicitly empty role requires a frozen
NOT_APPLICABLE policy row, written rationale, and a reviewer-trusted policy
digest. No such role is proposed in this version.

A reference eligibility requires a separate normal-only A identity manifest,
an independently pinned A availability receipt with positive counts, and an
exact match between group and view counts. A role digest alone does not show
that its images are authorized normals.

## Access scope

R2_L2B_DATA_ACCESS_APPROVAL_TEMPLATE.json has:

- no approved local root or category;
- no physical JSON byte permission;
- no train or test parsed section or field;
- no permission to read train normal/OK status or official image identity
  metadata;
- no permission to read the OK/NG/defect tokens embedded in official sample
  identities;
- no Mint/PCB or other protected category permission;
- no C identity certificate supplied;
- no feature extraction, outcome-label, or mask access;
- no disclosure fields and a zero-byte output limit.

The prospective paths in the template are explicitly marked as not authorized.
The access contract distinguishes physical file bytes from parsed fields. Train
and test metadata may be co-located in the same JSON, so permission to parse one
section must separately acknowledge exposure to the other section's bytes.
Train normal/OK status is label-bearing and requires its own explicit decision.
The official_sample_key itself includes OK or NG and may include an NG defect
code; accessing those condition-bearing identity tokens requires another
explicit decision. Test outcome labels and masks stay separately denied.

Any future read must first pass the explicit scope guard for the approved root,
category, section, and fields. This task did not call that guard on any local
dataset path or open a local data file.

## G1 regimes and outstanding evidence

G1-FULL-A and G1-PROTOCOL-MATCHED are represented separately:

| Regime | Evidence | Current state |
|---|---|---|
| G1-FULL-A | All independently authorized normal A source groups across the public catalog | Not authorized |
| G1-PROTOCOL-MATCHED | Frozen protocol evidence only | Not authorized |

Both keep four deployed normal supports separate from 64 additional C64
calibration normals. C64 is additional to the four and is not strict
four-shot-total. Neither regime changes or merges into C0 online memory. No
radius is selected.

G1-B still needs category-specific native Top-32 selection receipts, fixed
support identities, and the correct category-specific M0 scorer state. The
historical switch seed-0 fixed-support receipt cannot be reused for other
categories. No native score, descriptor, feature extraction, or G1-B receipt
was created.

All 30 readiness rows currently report public catalog membership only, local
inventory PENDING_AUTHORIZATION, missing A/B/C source-group receipts, no source
group counts, no A reference availability, and no geometry. The readiness
builder does not convert public inventory into local access.

## Synthetic verification

The new CPU suite verifies:

- complete 30-category by A/B/C proposal coverage, digest integrity, protected
  defaults, and pending approval;
- a signed policy decision still does not grant local access;
- public catalog presence does not change local authorization;
- default-deny scope, physical bytes versus parsed train/test fields, mixed
  train/test file acknowledgment, train normal/OK status, protected classes,
  test labels, and official identity field permissions;
- official multi-view identity is stable across camera views using synthetic
  strings only, with category and malformed-key rejection;
- raw identifiers and paths are absent from role receipts;
- missing roles are not empty, explicit empty roles need an independently
  reviewed frozen policy, and producer self-attestation is rejected;
- trusted reviewer signatures, forged reviewer names, untrusted roots,
  category/source mismatches, and C identity review;
- normal-only A evidence and independently pinned availability/count matching;
- A/B, A/C, and B/C overlap rejection;
- distinct G1-FULL-A and G1-PROTOCOL-MATCHED contracts, C64 accounting, no C0
  memory merge, and no radius selection.

These are synthetic contract tests. They do not establish that any real
category has enough authorized examples or that any actual A/B/C partition is
disjoint.

## Researcher decisions required before local metadata access

1. Approve the exact H-SOURCE proposal or request a new H-CLASS version with
   its exact held-out category set.
2. Approve all category-role assignments, or supply independently reviewed
   NOT_APPLICABLE rationales; say whether Mint, PCB, and any other protected
   classes are included in the policy.
3. Name the local root and exact category allowlist. Protected-class policy
   approval and protected data access are separate decisions.
4. Decide whether physical JSON file bytes may be read and acknowledge that
   train/test sections may share those bytes.
5. List exact parsed fields for train and test, including whether train
   normal/OK status and condition-bearing official sample-key tokens are
   allowed.
6. Provide the independently provisioned reviewer public keys, pinned trust
   directory SHA, identity-source provenance, and signed identity manifests.
7. Provide pairwise disjointness evidence, normal-only A availability receipts,
   and a separate C identity certificate.
8. Select G1-FULL-A, G1-PROTOCOL-MATCHED, or both as separate future evidence
   regimes; decide separately on any DINO feature extraction.
9. Set an output location, byte limit, and disclosure allowlist. Keep outcome
   labels and masks denied unless separately authorized.

## Status matrix

| Gate | Result |
|---|---|
| Protocol specification and synthetic contract | R2_L2B_PROTOCOL_READY_FOR_RESEARCHER_APPROVAL |
| Public 30-category inventory | Verified by the prior pinned public receipt |
| Local data inventory | PENDING_AUTHORIZATION |
| A/B/C role certification | R2_L2B_ROLE_CERTIFIED not reached |
| G1 reference pool authorization | NOT AUTHORIZED |
| G1-B category-native receipts | PENDING |
| Real-data feature or distance statistics | NOT RUN |
| GPU, DINO, labels, masks, private feedback, Mint/PCB | NOT ACCESSED |
| R2-1, G2, outcomes | NOT RUN |

R2_L2B_BLOCKED applies to performing local role certification or G1 now. The
metadata-only protocol is ready for explicit researcher decisions; actual role
certification remains pending.

## Source control and provenance

The requested branch and starting HEAD matched the task. The work began with no
tracked or staged modifications. Pre-existing untracked audit and output trees
were left untouched. Only the new R2-L2B source, test, and audit records are in
scope for the local commit. No push is authorized.

Compilation and focused suites used D:\anaconda3\envs\torch\python.exe:

| Command | Result |
|---|---|
| python.exe -m compileall -q experiments/calibration tests/calibration | PASS |
| python.exe -m pytest -q tests/calibration/test_r2_l1_locality_kernels.py | 44 passed |
| python.exe -m pytest -q tests/calibration/test_r2_l2a_allclass_geometry.py | 38 passed |
| python.exe -m pytest -q tests/calibration/test_r2_l2b_role_certification.py | 32 passed |
| Combined R2-L1, R2-L2A, and R2-L2B suites | 114 passed |
| Proposal JSON and default-deny template validation | PASS; 90 role rows |

The tests create only synthetic identity strings, hashes, keys, and receipts.
They do not enumerate or open dataset paths.

SHA-256 provenance:

| File | SHA-256 |
|---|---|
| experiments/calibration/r2_l2b_role_certification.py | 0f7bd313b5d3774454868d0dd833d28040340ecdb0178dcdece1b0eb94f7759c |
| tests/calibration/test_r2_l2b_role_certification.py | 40fcea783ccfcd4502490af837d04dda287876454f60126e21e788c39b5dd08e |
| experiments/calibration/r2_l2a_allclass_inventory.py | 6ce759ebeeefb069a1a834fad6a343436e9e828e0bbd45003153dc44d345bc24 |
| experiments/calibration/r2_l2a_g1_geometry.py | 3fd3ce36a1317c7c408cdbdc829d87d238aefaab81e453ecf56c36a43aa2201f |
| experiments/risk_safe/preflight.py | 2070ba770ac36af06f688b0450c711d67b7bca526e3939178d559216535c25ba |
| docs/calibration/audit/R2_L2A_R_OFFICIAL_PUBLIC_CATALOG_RECEIPT.json | 77f30230e3752a5921ea500695b7ad066103d07d870ac823b65ac5af310340bc |
| docs/calibration/audit/R2_L2B_PROPOSED_30_CLASS_ROLE_POLICY.md | 5a8e0002884b58680e628cf91c3507ab55485a1f6ec1378b5183d911db69569c |
| docs/calibration/audit/R2_L2B_PROPOSED_30_CLASS_ROLE_POLICY.json | a490475f003a14bb24d83f771554df07812ae2c8900117620ecf9be24d3397b8 |
| docs/calibration/audit/R2_L2B_DATA_ACCESS_APPROVAL_TEMPLATE.json | aef3038b75d2ce5a6387a597173435c1058c7efab40f72cb5b4b3a1ae782dcca |
No local dataset path was enumerated or accessed.
