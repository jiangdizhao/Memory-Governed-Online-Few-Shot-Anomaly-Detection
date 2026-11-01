# P0-R1 Dataset Audit Decision

**Audit classification:** bounded metadata and historical-document review; no anomaly-detection experiment, G1/G2, or dataset-content inference was run.

## Real-IAD historical exposure

Real-IAD has documented full-30-category historical result exposure, four-category Stage 0.5/0.75 and Stage 1-A outcome exposure, Audiojack/Probe-A hindsight exposure, and additional switch-only R1 queried-feedback/engineering exposure. R2-0 was label-blind; R2-L2C Stage B was TRAIN metadata-only. Exact Source Group intersections remain `UNKNOWN` where the tracked summaries provide only digests or aggregate identities. The 30-category public catalog and TRAIN census do not certify A/B/C roles. Existing G1/G2 and independent-role-certification statuses are unchanged.

## MVTec AD 2 verdicts

| Question | Decision |
|---|---|
| Is local structure consistent with the official release? | Yes, at the inspected metadata level. Eight category directories and all expected splits are present; per-category counts match the official paper and total 8,004 images. `STRUCTURE_COMPLETE`. |
| Are any components detectably missing? | No missing expected split/category/mask folders or files were detected. `wallplugs` is the local folder spelling for the paper's “Wall Plugs”; official code-utility mapping was not verified. |
| Is file-level content integrity verified? | No. `CONTENT_INTEGRITY_UNVERIFIED`; no image decode or official byte/checksum comparison was done. |
| Can the local copy's official distribution origin be verified? | No. Official release information is documented, but local origin is `SOURCE_PROVENANCE_SELF_REPORTED` / `SOURCE_NOT_VERIFIED`: no archive or source manifest was present in the inspected scope. |
| Is the license identified? | Yes: CC BY-NC-SA 4.0. Academic non-commercial use is compatible in principle, with attribution/share-alike obligations. The researcher must confirm all intended use is non-commercial and compliant: `LICENSE_COMPLIANCE_REQUIRES_USER_CONFIRMATION`. |
| Is frozen external inference possible? | Yes, after freezing an MVTec-specific but non-outcome-tuned preprocessing and source-role protocol: `SUITABLE_FOR_FROZEN_EXTERNAL_INFERENCE`. No model was run here. |
| Is public-test evaluation possible? | Yes, using the public labels/masks under a frozen protocol: `SUITABLE_FOR_PUBLIC_TEST_EVALUATION`. This audit inspected only aggregate counts and did not view examples or mask pixels. |
| Can public test support limited online feedback? | Potentially, under a separately preregistered protocol: `POTENTIALLY_SUITABLE_FOR_LIMITED_ONLINE_FEEDBACK`. At 3%, each class yields only 2–5 total queries by count. |
| Can the previous 20-normal + 20-defect G2 anchor target be met at 3%? | No. Even pooling all 1,084 public images yields at most 33 queries by ceiling, before source-group constraints: `FULL_G2_ANCHOR_TARGET_INFEASIBLE_AT_3_PERCENT` / `INSUFFICIENT_FOR_FULL_3_PERCENT_ONLINE_PROTOCOL`. |
| What about private/mixed test? | Frozen inference and official server evaluation are possible; local labels and query-feedback simulation are not. Private and private-mixed correspond to the same scenes and cannot be treated as independent groups. |

Official reference: [MVTec AD 2 release and license](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2), [official paper](https://doi.org/10.1007/s11263-026-02743-0), and [benchmark server](https://benchmark.mvtec.com/).

## Remaining requirements before external confirmation

1. Preserve an acquisition receipt or original archive and compare its provenance with the official source; obtain a trusted official checksum if MVTec provides one.
2. Confirm non-commercial use and attribution/share-alike compliance for the intended research, code, and publication outputs.
3. Verify the `wallplugs` directory alias against official utilities or documentation before writing a dataset adapter; do not rename the existing data during this audit.
4. Freeze support/C64 Source Group identity rules, grayscale mapping, aspect-ratio/edge handling, score-map geometry and metrics before any public-test outcome analysis.
5. Decide whether the planned external study is frozen inference, a smaller online-feedback design, or official private-set submission. The full G2 queried-anchor target needs a different budget/dataset/protocol; it cannot be claimed from this public split at 3%.
6. Obtain the separate researcher/reviewer approvals required for any role certification, support selection, G1/G2 or execution. This audit grants none.

The MVTec audit does not block separately authorized Real-IAD Local Correction work. No MVTec inference or G1/G2 execution should be inferred from these reports.
