# MVTec AD 2 Source and License Review

**Local root reviewed:** `G:\dataset\mvtec_ad_2`
**Researcher confirmation:** newly downloaded and never used for experiments before this audit. No historical MVTec experiment search was performed.

## Authoritative release record

MVTec's [official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2) identifies eight scenarios, normal-only training and validation, public test annotations, private test ground truth, and the CC BY-NC-SA 4.0 license. The [official MVTec AD 2 paper](https://doi.org/10.1007/s11263-026-02743-0) reports 8,004 images across eight categories and describes the public/private test design. The linked [official evaluation server](https://benchmark.mvtec.com/) is the documented route for private-set evaluation.

The official page requires users to acknowledge non-commercial use for downloads. It states commercial use is not permitted and directs uncertain users to MVTec. Its code-utilities link was not successfully retrieved in this audit, so that package's directory aliases and any additional source manifest were not inspected.

## Local acquisition evidence

| Check | Finding |
|---|---|
| Dataset directory | Present at the researcher-specified path. |
| Expected category directories | Eight present; local `wallplugs` corresponds by name to the paper's “Wall Plugs” scenario. The exact canonical code token was not independently verified against official utilities. |
| Local documentation | `readme.txt` (964 bytes) and `license.txt` (21,288 bytes) are present. The README identifies MVTec AD 2 and CC BY-NC-SA 4.0; the license file contains the license text. |
| Original archive / source manifest | No matching archive was found in the dataset root or its immediate parent. No source manifest or official per-file checksum list was found in the inspected local documentation. No wider filesystem search was made. |
| Download provenance | `SOURCE_PROVENANCE_SELF_REPORTED`; local archive origin and byte-for-byte release identity are `SOURCE_NOT_VERIFIED`. Folder structure and file counts alone do not prove the distribution source. |
| Hashing | No image, mask, archive or full-dataset content hashes were computed. |

The local license is consistent with the official page. The official release and license are documented (`OFFICIAL_SOURCE_DOCUMENTED`; `LICENSE_IDENTIFIED`). Academic research use is compatible in principle when attribution and share-alike terms are followed and all use remains non-commercial. This is not a declaration of legal compliance: the researcher must confirm that intended publication, code/data redistribution, collaborations, and any downstream use meet the license (`LICENSE_COMPLIANCE_REQUIRES_USER_CONFIRMATION`).

## Verdict

- `OFFICIAL_SOURCE_DOCUMENTED`
- `SOURCE_PROVENANCE_SELF_REPORTED`
- `SOURCE_NOT_VERIFIED`
- `LICENSE_IDENTIFIED`
- `LICENSE_COMPLIANCE_REQUIRES_USER_CONFIRMATION`
- Research access: `NONCOMMERCIAL_ACADEMIC_USE_COMPATIBLE_IN_PRINCIPLE`, subject to the confirmation above.

No data was downloaded, moved, rewritten, or submitted to the benchmark. No browser history, account, or credential was inspected.
