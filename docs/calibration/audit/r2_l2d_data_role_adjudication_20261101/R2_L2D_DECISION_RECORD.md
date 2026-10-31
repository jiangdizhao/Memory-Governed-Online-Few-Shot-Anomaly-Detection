# R2-L2D Decision Record

- **Decision:** `R2_L2D_PROTOCOL_READY_CERTIFICATION_PENDING`
- **Stage A:** `R2_L2C_LOCAL_ACCESS_NOT_AUTHORIZED` at that historical attempt; report preserved.
- **Stage B:** Census outputs pass internal consistency; signed authorization ID, canonical digest, Ed25519 signature, researcher key, and TRAIN-census scope are verified. Historical pre-execution trust-pin timing remains unverified.
- **G1/G2:** `NOT_AUTHORIZED`

## Reconciled event record

The Stage A report records an earlier authorization gate failure before census execution or local data I/O. The later Stage B census output pair is present under `outputs/calibration/r2_l2c_hsource_all30_train_census_20261101_v1/`. These are sequential historical events and are not inherently contradictory. Stage A and Stage B outputs remain unchanged. The recovered signed authorization has an ID and digest matching the Stage B report, and its signature and scope verify. No contemporaneous record establishes when the trust-directory digest was pinned relative to Stage B execution.

The Stage B pair is internally consistent under `audit_l2c_artifact_pair`: 30 categories, 7,293 reported source groups, 36,465 normal images, five views per group, and 30/30 public/private source-fingerprint matches. The audit also verifies candidate group-hash uniqueness across the supplied candidate pool. These are receipt-internal checks only. No independent source-fingerprint baseline exists, and the source JSON was not reopened. Do not treat the recorded `cross_role_overlap_count: 0` as a role-disjointness result: all 90 role slots remain pending and no A/B/C membership receipts exist.

See [R2-L2D-R provenance recovery report](../r2_l2d_provenance_recovery_20261101/R2_L2D_R_PROVENANCE_RECOVERY_REPORT.md) for exact hashes, signature/scope verification, trust-pin timing limitation, role-policy proposal, and remaining gates.

## Five independent gates

| Gate | Current result | What this result covers |
|---|---|---|
| L2D engineering and protocol verification | `PASS_SCOPED_CPU` | Existing L2B/L2C/L2D synthetic contract suites pass. Stage B pair passes internal artifact consistency. No data access, feature extraction, geometry, or execution is authorized by this pass. |
| L2C Stage B authorization signature and scope | `VERIFIED` | The recovered signed manifest has a valid canonical digest and Ed25519 signature under the trusted researcher public key; the exact TRAIN-census scope passes the original validator and its ID/digest match Stage B. |
| Historical independent trust-pin timing | `UNVERIFIED` | The researcher-provided pin matches the trust directory now; no contemporaneous evidence shows it was independently pinned before Stage B execution. |
| A/B/C role-membership certification | `PENDING` | H-SOURCE is proposed only. No role groups are assigned; no role receipts, normal-only A certification, pairwise role-overlap evidence, or independent C certificate are available. |
| Support/C64 identity certification | `PENDING_NOT_SELECTED` | Candidate-pool capacity is reported; no category/seed-bound Support or C64 membership was selected or certified. C64 remains additional to the deployed four supports. |
| G1-A/G1-B/G2 execution authorization | `NOT_AUTHORIZED` | No geometry, native bank verification, descriptors, or evaluation may start from these records. |

An independent source-fingerprint baseline is `NOT_ESTABLISHED`. Same-run agreement between the public census report and private receipt file demonstrates internal consistency; it does not authenticate the source JSON fingerprints against an independent prior record. This evidence gap does not invalidate the demonstrated internal agreement or the verified authorization signature/scope.

## Role-policy disposition

The existing H-SOURCE policy remains `PROPOSED, NOT APPROVED` (proposal SHA-256 `531a62e78519702f13dfc1fdb579e8476c29d3ff2993219605516d0ca7433324`). A concrete deterministic identity-only assignment rule is specified in the linked recovery report, but it has not been run and does not assign protected groups. It requires a researcher-approved versioned policy containing the role populations, exact per-category A/B/C quotas, integer seed, and protected-category disposition. Mint, PCB, and any additional protected confirmation groups remain unassigned until explicitly authorized.

## Exact remaining authorization requirements

1. **Historical trust-pin timing:** the supplied trust-directory digest matches the trust file and validates the signature now. A contemporaneous external pin/manifest is still required to claim that this root was independently pinned before Stage B execution.
2. **Establish source provenance:** provide an independently pinned pre-run source-fingerprint baseline, or separately authorize a new metadata verification procedure. Do not infer a baseline from same-run matching files.
3. **Approve the role policy:** the researcher must approve H-SOURCE (or issue a versioned replacement) and freeze role populations, quotas, seed, category scope, protected-group handling, and source-split permissions.
4. **Authorize any required new metadata access:** B/C identities requiring TEST metadata need a distinct signed access authorization. Label or mask access requires separate approval. Reopening source JSON to establish fingerprints also needs separate authorization if it is outside the recovered manifest's exact scope.
5. **Certify identities independently:** supply category/role identity manifests, normal-only A availability evidence, pairwise A/B/C disjointness, independent reviewer attestations, and a separate C identity certificate bound to the independently pinned reviewer trust directory.
6. **Select Support/C64 and authorize execution separately:** approve actual category/seed/source-bound four-support plus additional C64 membership, independent selection review and bank binding. G1-A, G1-B, and G2 each require explicit execution authorization at their respective scopes.

No HOLD is removed. Do not infer authorization for G1/G2 from L2D synthetic readiness, L2C output existence, a public category catalog, or an internally consistent receipt pair.
