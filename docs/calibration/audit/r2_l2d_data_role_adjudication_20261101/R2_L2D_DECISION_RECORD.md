# R2-L2D Decision Record

- **Decision:** `R2_L2D_PROTOCOL_READY_CERTIFICATION_PENDING`
- **Provenance gate:** `BLOCKED_INPUT`
- **G1/G2:** `NOT_AUTHORIZED`

## Gate outcomes

| Gate | Result | Scope |
|---|---|---|
| `ARTIFACT_AUDIT_PASS` | PASS, scoped | Existing L2C pair has internally consistent schemas, 30-category aggregates, parser fingerprints, group/view counts, privacy flags, role-pending statuses, and size below 64 MiB. This does not authenticate the run. |
| Provenance authentication | `BLOCKED_INPUT` | Signed authorization payload and externally pinned trust-root fingerprint are absent. No independent prior source-fingerprint baseline exists; raw JSON was not reopened. |
| Local data inventory authorization | `PENDING / NOT INDEPENDENTLY VERIFIED` | The census reports an authorization ID/hash, but the signature and trust root cannot be verified; the tracked L2C report conflicts with the local outputs. |
| `IDENTITY_CONTRACT_PASS` | PASS, synthetic contract | New group-level proposal, overlap, role-receipt composition, Support/C64 selection, and bank-binding validators pass synthetic CPU tests. This is not a real role identity certification. |
| `ROLE_PROPOSAL_READY` | READY, procedure only | Deterministic quota/seed proposal procedure is specified and tested. No actual identities, quotas, or membership were proposed or assigned. |
| `INDEPENDENT_ROLE_CERTIFIED` | NO | A/B/C receipts, signed independent reviews, pinned trust-root material, and explicit role authorization are unavailable. |
| Actual Support/C64 selection | NOT RUN | No membership was approved or generated. |
| G1-A / G1-B / G2 | NOT RUN / NOT AUTHORIZED | No feature extraction, native bank verification, geometry, descriptor selection, radius work, or evaluation occurred. |

## Evidence and risks

The public L2C completion report has SHA-256 `8aa27fb725002abc26e2a9d257ca065b4f86e789db1552357a7849212cf7fcce`. The private receipt pair's digest and all private group identifiers are excluded from this report and Git. The existing report and receipt pair internally match on all 30 source JSON fingerprints. That does not establish those fingerprints against any earlier baseline or prove the prior run's authorization.

The committed L2C report still records that no census ran, in conflict with the present untracked output directory. The signed authorization and trust-root record needed to adjudicate this conflict are missing. The census report records that Test-section bytes were encountered but not parsed. No Test outcome was accessed by this task.

The R2-L2D code and test SHA-256 values, exact commands, final commit SHA, changed-file list, and regression result are recorded in the final execution response. The commit is a local WIP because provenance and certification remain blocked. No push is authorized.

## Minimum additional evidence before G1-A

1. The original signed L2C authorization payload and a trust-directory fingerprint pinned independently of that payload, matching the recorded authorization ID and digest.
2. A prior authenticated source-fingerprint baseline or separately authorized procedure to verify source metadata. This task did not reopen source JSON and did not authorize a new scan.
3. Researcher-approved H-SOURCE policy, category-level A/B/C quotas, source population, seed, and stage scope.
4. Category-specific A role receipts, normal-only evidence manifests, independent reviewer manifests/attestations, and a pinned reviewer trust directory.
5. Verified complete group-level A/B/C disjointness, with B/C source-split authorization. Any official TEST metadata requires a distinct signed access authorization; labels/masks require separate approval.

## Minimum additional evidence before G1-B

In addition to the G1-A gates, G1-B requires category-specific native receipts: correct structured Fixed Bank digest, category/seed Fixed Bank identity receipt, category-specific M0 scorer state, valid A-derived Top-32 selection receipt, an approved H-SOURCE policy, a separate signed category/seed/source-bound Support/C64 selection authorization, and approved 4 Support + additional C64 membership with independent selection and bank-binding review. The historical switch/seed-0 digest applies only to switch/seed0.

The decision is therefore to stop at protocol and synthetic identity-contract readiness. No HOLD is removed, no local group membership is certified, and G1/G2 must not start from this audit.
