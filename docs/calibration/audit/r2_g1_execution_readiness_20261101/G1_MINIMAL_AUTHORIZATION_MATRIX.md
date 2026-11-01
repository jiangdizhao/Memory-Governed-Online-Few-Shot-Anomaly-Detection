# G1 Minimal Authorization Matrix

This is a proposal for a later researcher-signed authorization. **No row is authorized by this document.** The historical L2C grant is limited to TRAIN metadata census fields and does not permit image decoding, feature extraction, Support/C64 selection, or geometry.

| Operation | Existing evidence | Additional authorization / receipt required | G1-A needs it? | G1-B needs it? |
|---|---|---|---:|---:|
| Public category names | Pinned 30-category public catalog receipt | None for names alone; catalog is not data permission | Yes, identity only | Yes, identity only |
| TRAIN-normal candidate group counts | L2C Stage B census; aggregate only | A availability receipt bound to the exact source manifest; independent verification of source/trust fingerprints | Yes | Yes |
| A membership and normal-only identity projection | H-SOURCE proposal SHA above; no membership receipt | Researcher-approved A rule; exact identity contract realid-offical-sample-key-v2 and group digest algorithm; field-scoped TRAIN identity authorization if new identity reads are required; count/group receipt; independent reviewer attestation | Yes | Yes |
| Decode authorized TRAIN images | No authorization | New signed image-access scope naming dataset snapshot, categories, TRAIN-normal only, camera-view rule, source hash, and output policy | Yes | Yes |
| DINO feature extraction / G1-A descriptors | Engineering profile exists; no G1 feature receipt | Explicit G1-A feature authorization, bounded run identity, adapter/source hashes, unique output directory and 1 GiB output cap for pilot | Yes | Yes |
| Four deployed Support selection | No new selection is authorized; switch seed-0 bank digest is historical only | Separate Support-selection authorization and category/seed-bound identities (initial proposed Support Seed 0); actual bank digest receipt and independent review | No | Yes |
| Additional 64 C64 selection for G1-PROTOCOL-MATCHED | Historical C64 protocol exists but does not authorize reuse here | Separate C64-selection authorization under history seed 190021 and view seed offset 891; 64 distinct additional normal groups outside Support; these C64 images form the matched geometry cohort | No | Yes for G1-PROTOCOL-MATCHED |
| Native M0 scorer state and C0 Top-32 | G1 validators exist; no eligible G1 A-image Top-32/M0 receipts are verified | Explicit native C0 scoring/Top-32 authorization; category-specific Support receipt; empty-Greedy M0-state digest/certificate; scorer/source fingerprints; external receipt verification | No | Yes |
| G1 distance computation and summaries | Exact synthetic geometry implementation and tests exist | Explicit per-profile execution authorization after all A/cache/Top-32 gates; separate category list, group limit, run ID, source commit, time/disk bounds | Yes | Yes |
| B/C membership certification | L2E policy proposal only; no A/B/C memberships | Researcher approval bound to L2E policy SHA; new scoped TEST identity/exposure authorization; verified combined TRAIN/TEST source manifest and exposure registry; independent role/disjointness attestation; approve each proposed N/A disposition | No for A-only geometry | No for A-only geometry |
| G2/query-label or mask work | Not part of G1; not authorized | Separate G2 authorization, frozen stream/tape and independent evaluator scope | No | No |

## Independent approvals

1. **Researcher protocol approval:** approve/revise the L2E H-SOURCE policy SHA and the G1-specific deterministic view/sample rules. Approval alone grants no data access.
2. **A-only image/feature access:** separately name category, A-group maximum, TRAIN-normal scope, image decoding, R1 feature extraction, G1-A geometry, exact output directory, 1 GiB maximum, and 10 GiB free-space floor. This can be approved without B/C membership certification.
3. **Support/C64 selection:** separate approval for actual four Support identities and 64 additional C64 identities. The Supports build the fixed bank; C64 groups are the matched query/reference geometry cohort.
4. **G1-B native scoring:** separate approval for category-specific M0 scoring and actual C0 Top-32 extraction. A prior R1 stream receipt is not a substitute for an A-cohort M0 receipt.
5. **B/C partition:** requires a distinct policy approval, TEST identity/exposure authorization, and independent reviewer certification. It is not a prerequisite to an A-only G1-A pilot.
6. **G2:** requires its own future approval; none of these approvals grants it.

No approval is implied by the checklist. A signed authorization must be bound to a unique non-overwriting run ID and the exact execution commit. Never include private keys or passwords in receipts.
