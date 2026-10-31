# R2-L2B Proposed 30-Category Role Policy and Decision Record

**Decision state:** PROPOSED, NOT APPROVED
**Recommendation:** H-SOURCE
**Proposal SHA-256:** 531a62e78519702f13dfc1fdb579e8476c29d3ff2993219605516d0ca7433324

This is a policy proposal only. The verified public category catalog does not
authorize local dataset access, establish any A/B/C assignment, or certify a
role. No local category JSON, image, label, mask, feature cache, sample key, or
source group was read for this record.

## Proposed holdout and role design

Recommend H-SOURCE: within each of the 30 public categories, assign complete
official_sample_key source groups to A, B, or C. Keep every camera view for one
official sample in the same role. Require pairwise source-group disjointness
across A, B, and C, including across category boundaries.

H-SOURCE retains all 30 category identities in parameter design. It therefore
supports all-category source-group-disjoint evaluation; it does not establish
generalization to unseen category identities. H-CLASS would hold out selected
categories from parameter design, but no categories are selected here. A
future H-CLASS proposal must freeze its exact category set and reissue the
versioned policy before any data access.

The proposed role purposes are:

| Role | Purpose | Current state |
|---|---|---|
| A | Parameter-design reference | PROPOSED; no source groups assigned |
| B | Development evaluation | PROPOSED; no source groups assigned |
| C | Sealed confirmation | PROPOSED; independent C identity review required |

All 30 category by A/B/C pairs are explicitly represented in
R2_L2B_PROPOSED_30_CLASS_ROLE_POLICY.json. The proposal contains 90 role rows,
all marked PROPOSED except the protected category rows below. No counts,
availability, source-group identities, or category-specific splits are
invented.

| Category | A | B | C | Treatment |
|---|---|---|---|---|
| audiojack | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| bottle_cap | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| button_battery | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| end_cap | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| eraser | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| fire_hood | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| mint | PENDING_AUTHORIZATION_PROTECTED | PENDING_AUTHORIZATION_PROTECTED | PENDING_AUTHORIZATION_PROTECTED | Protected and sealed |
| mounts | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| pcb | PENDING_AUTHORIZATION_PROTECTED | PENDING_AUTHORIZATION_PROTECTED | PENDING_AUTHORIZATION_PROTECTED | Protected and sealed |
| phone_battery | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| plastic_nut | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| plastic_plug | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| porcelain_doll | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| regulator | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| rolled_strip_base | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| sim_card_set | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| switch | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| tape | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| terminalblock | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| toothbrush | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| toy | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| toy_brick | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| transistor1 | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| u_block | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| usb | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| usb_adaptor | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| vcpill | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| wooden_beads | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| woodstick | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |
| zipper | PROPOSED | PROPOSED | PROPOSED | Researcher approval required |

Mint and PCB remain protected by default. The documentation reviewed for this
proposal also says that any other protected confirmation groups must remain
sealed. The researcher must identify any such additional groups before
approving the policy; this proposal does not infer that the public catalog is
the complete list of locally protected data.

## Role and identity contract

The identity is the established Real-IAD official_sample_key under contract
realid-offical-sample-key-v2 and grouping unit official_sample_key. The spelling
of that historical version identifier is preserved. Its parser validates the
category, session, sample directory, camera view, timestamp, and path layout.
The official key includes OK or NG and may include an NG defect code; reading
these condition-bearing tokens is a separate explicit permission.
All camera views sharing a key are grouped before role assignment. The
R2-L2B receipt retains a domain-separated hash and a view count, not the raw
sample key or local image path.

Missing role evidence stays PENDING_MISSING_RECEIPT. It never certifies an empty
role. An explicitly empty role requires a versioned frozen policy row with
NOT_APPLICABLE, a written rationale, and independent reviewer approval bound
to the exact policy digest. A producer-supplied digest alone is insufficient.
Any A/B/C overlap fails closed.

C requires a separate identity manifest, independent reviewer attestation, and
explicit confirmation that the C identity certificate was supplied. A source
group digest or public catalog entry cannot certify C.

## G1 reference evidence regimes

Keep the regimes separate:

| Regime | Reference population | Support accounting | Current state |
|---|---|---|---|
| G1-FULL-A | All independently authorized normal source groups assigned to A, across the public catalog | Descriptive broad geometry; deployed four supports and 64 additional C64 calibration normals remain separately identified | Not authorized |
| G1-PROTOCOL-MATCHED | Evidence admitted by the frozen few-shot protocol | Four deployed normal supports plus 64 additional C64 calibration normals | Not authorized |

C64 is additional to the deployed four supports. Neither regime is strict
four-shot-total, neither is merged into C0 online memory, and no radius is
selected by this proposal. G1-FULL-A and G1-PROTOCOL-MATCHED are separate
evidence regimes and must be reported separately.

For G1-B, each category also needs its own native Top-32 selection receipt,
fixed-support identity, and correct M0 scorer state. The historical switch,
seed-0 support digest is not transferable to another category. No DINO
features, Top-32 selections, scores, or native receipts were generated.

## Researcher decision record

No decision is recorded as approved. Before local metadata access, the
researcher must explicitly decide:

1. Approve this exact H-SOURCE proposal, or request a versioned H-CLASS policy
   with its exact held-out categories.
2. Approve the 30-category by A/B/C policy, including each required role and
   any NOT_APPLICABLE row with a reviewed reason. State whether Mint, PCB, or
   any other protected categories are included in the policy; this is separate
   from data-access approval.
3. Name the exact local root and category allowlist. List any protected
   categories separately; Mint and PCB remain sealed unless named explicitly.
4. Decide whether physical JSON file bytes may be read. The train and test
   sections may share a file, so approving only one parsed section does not
   avoid exposure to the other section's bytes.
5. List the exact permitted train and test fields. Decide separately whether
   train normal/OK status may be read; it is label-bearing. Test outcome labels
   and masks remain denied unless separately authorized.
6. Approve the official image-identity metadata fields and provide the
   independently provisioned identity-reviewer trust root and source
   provenance.
7. Name the independent reviewer and provide signed A/B/C identity manifests,
   pairwise source-group disjointness evidence, normal-only A reference
   availability evidence, and a separate C identity certificate.
8. Choose G1-FULL-A, G1-PROTOCOL-MATCHED, or both as separate regimes. State
   whether future DINO feature extraction is allowed in a distinct
   authorization.
9. Set the output location, output-byte limit, and exact fields permitted for
   disclosure.

The companion R2_L2B_DATA_ACCESS_APPROVAL_TEMPLATE.json starts with every
permission false, no approved root, no categories, no parsed sections, and a
zero output limit. The proposed H-SOURCE policy digest is a content identity,
not an approval receipt.
