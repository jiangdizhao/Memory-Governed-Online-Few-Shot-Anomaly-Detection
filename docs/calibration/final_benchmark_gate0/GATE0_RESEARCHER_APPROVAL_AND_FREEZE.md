# Gate 0 Researcher Approval and Scientific Design Freeze

**Status:** `GATE0_SCIENTIFIC_DESIGN_FROZEN_RESEARCHER_APPROVED`
**Approval date:** 2026-11-11
**Branch:** `calibration-stage0-contract-20261029`
**Approved R2 base commit:** `2511d4f5b61efe869994b23afb398040eb49d270`

## Approval record and scope

The researcher explicitly approves the six scientific design decisions recorded in the Gate 0 R2 protocol, following the independent ChatGPT review reported by the researcher. This freezes the scientific design for subsequent separately authorized planning. It does not assert separate supervisor sign-off, certify data identities or provenance, freeze or validate an implementation, authorize Gate 1, or authorize experimental execution.

The freeze covers the six decisions below and the passive N-shadow design. Existing R2 source and data-access boundaries remain in force. No code, data, outcomes, masks, models, or experimental results are approved for access by this record.

## Six approved scientific decisions

1. **Query protocol and expenditure interpretation.** Use the registered simplified BN+Band and `Margin_rho` policies, target rate `rho=0.03` for Real-IAD and `rho=0.10` for VisA, and per-category cap `B_c=floor(rho*T_c)`. Retain the R2 `Margin_rho` cold-start, past-only utility history, deterministic identity-hash, and suppression rules. Equal caps mean equal maximum label allowance, not equal realized expenditure. The absolute realized-rate matching tolerance is `0.003`, checked per category and at the category-equal dataset macro within each paired condition; it permits only an additional approximately rate-matched description. Report actual query counts, rates, cap use, suppression, and obtained label composition. No forced queries, top-ups, or policy-specific budget compensation are permitted.

2. **Real-IAD feedback timing.** Use Group-atomic feedback: process the five ordered views of each official Real-IAD Source Group from one pre-group state; seal every view's score, prediction, and query proposal before disclosing the group's one purchased label; then apply legal updates for subsequent groups. Incomplete or invalid groups fail closed. This timing is a new protocol and does not claim historical event-level parity.

3. **Data roles and conditions.** Use deterministic metadata-only Support/C64 selection: four deployed verified-normal Supports plus 64 separate verified-normal C64 images. Preserve P0 `(Support Seed 0, Stream Seed 0)`, P1 `(Support Seed 0, Stream Seed 11)`, and P2 `(Support Seed 1, Stream Seed 0)`. No feature/performance search may select identities. Actual identities, group disjointness, release/annotation provenance, and VisA group provenance require separately authorized Gate 1 verification. Do not assume unsupported VisA physical groups; fail closed or limit the affected condition when a valid group unit cannot be certified.

4. **Exposure and claims.** Describe Real-IAD as a historically exposed full-category benchmark reassessment, not independent confirmation. Describe VisA as cross-dataset external validation; any independent-confirmation claim depends on later exposure/provenance evidence. Keep protected outcomes and masks sealed under their existing authorization boundaries.

5. **Statistical analysis.** H_Q, `BER(F-M)-BER(F-B)`, is the primary prespecified Real-IAD benchmark contrast. Keep the paired category-level bootstrap, category-equal aggregation, registered supporting analyses, and explicit FPR/FNR reporting specified in R2. Retain negative and mixed results. The proposed 0.01 BER value is not an approved hard success criterion. A nonsignificant H_C difference does not establish equivalence or non-inferiority. No formal Signed kNN non-inferiority claim is permitted without a separately justified and approved positive `delta_K`; no `delta_K` is approved by this record.

6. **Comparators and fidelity limits.** Static-36+32 is required for Real-IAD all 30 categories at P0 and conditionally required for VisA all 12 categories at P0 if its identity/group eligibility and provenance pass. The approved identity allocation uses the same P0 68 normal identities as the principal systems: four Supports plus 32 deterministically selected P0 C64 identities as the 36 reference identities, and the remaining 32 P0 C64 identities as C32 calibration. Rank the P0 C64 identities by SHA-256 of UTF-8 `Gate0-static36-c64-v1\0<dataset_release_sha256>\0<category>\0<support_seed_decimal>\0<official_group_key>\0<official_image_key>`, with ties broken by normalized group-key and image-key UTF-8 bytes; assign the first 32 to reference and the remainder to C32. Freeze this baseline-specific assignment from metadata before scoring in a separate checksummed schedule; it does not alter the principal manifest's Support/C64/TEST roles. Do not search other allocations. The approved implementation target is one K-Center procedure producing a fixed bank of 8,192 FP16 prototypes, with a static threshold from C32 q90. Gate 1 source-fidelity checks remain required; this is an approved target, not a claim that the implementation exists or passed parity. Static-36+32 is a same-total-normal-information system comparator, not a mechanism ablation or historical A4's dynamic D arm. External online comparisons require source fidelity; otherwise limit complete-system competitive claims.

**N-shadow remains approved as a passive, non-querying normal-only correction control.** It inherits F-B's legally executed query opportunities and disclosures, reads F-B's corresponding Greedy state, maintains its own initially empty normal-only C1 correction state, adds only negative anchors on verified-normal feedback, never adds defect anchors, and cannot influence F-B. Its implementation and source fidelity remain Gate 1 requirements.

## Deferred dependencies and limits

- **Gate 1 is not authorized.** Actual image/group identity manifests, source and annotation hashes, role/group disjointness, release provenance, and VisA group identity/provenance have not been certified. Any metadata/data operation requires separate authorization. Do not open TEST labels, masks, or protected outcomes as part of this freeze.
- **Implementation is not frozen, authorized, or validated.** Source review and parity checks must verify the approved algorithms, five-view barrier, label order, query accounting, C64 protection, N-shadow isolation, and Static-36+32 K-Center/8,192-prototype FP16/C32 q90 target before any future execution authorization.
- **Gate 2 resource measurements remain pending.** No GPU, RAM, disk, latency, or runtime measurement is represented here.
- `delta_K` remains unapproved. Without a separate justified approval, report H_C descriptively and make no formal Signed kNN non-inferiority claim.
- The 0.01 absolute Macro BER proposal remains unapproved as a hard success criterion. Real-IAD remains historically exposed; VisA independent-confirmation status remains conditional on provenance/exposure evidence. External competitive claims remain bounded by source fidelity.
- Strict-four-total remains on HOLD. No new allocation search, A4 rerun, or unregistered parameter search is approved.
- The researcher approved this scientific design; **separate supervisor sign-off is not asserted**.

## Final status

`GATE0_SCIENTIFIC_DESIGN_FROZEN_RESEARCHER_APPROVED`

This status denotes a researcher-approved scientific design only. It does not authorize Gate 1, implementation work, data access, or a scientific run.
