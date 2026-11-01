# MVTec AD 2 Structural Inventory

**Inventory mode:** directory names, file names and file-size metadata only. No image headers, image pixels or mask pixels were opened.

## Per-category inventory

The published counts below are from Table 4 of the [official MVTec-hosted paper PDF](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/MVTecAD2_arXiv_preprint_2503.21622v1.pdf). Local counts matched that table. `Public normal/defect` counts come from the public split's `good`/`bad` directory names; they are aggregate metadata, not performance results.

| Local category | Train | Validation | Public total (normal/defect) | Private | Private mixed | Public masks | Total images* | Image bytes across image splits | Mask bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| can | 412 | 46 | 162 (72/90) | 321 | 321 | 90 | 1,262 | 2,856,068,929 | 217,224 |
| fabric | 387 | 43 | 156 (66/90) | 314 | 314 | 90 | 1,214 | 10,834,249,351 | 497,916 |
| fruit_jelly | 263 | 37 | 80 (20/60) | 255 | 255 | 60 | 890 | 1,296,596,536 | 314,335 |
| rice | 313 | 35 | 132 (42/90) | 277 | 277 | 90 | 1,034 | 6,760,802,321 | 536,836 |
| sheet_metal | 137 | 19 | 114 (24/90) | 142 | 142 | 90 | 554 | 1,651,380,153 | 714,924 |
| vial | 291 | 41 | 140 (35/105) | 276 | 276 | 105 | 1,024 | 811,573,891 | 457,205 |
| wallplugs (Wall Plugs) | 293 | 33 | 150 (60/90) | 232 | 232 | 90 | 940 | 2,209,933,337 | 512,261 |
| walnuts | 432 | 48 | 150 (60/90) | 228 | 228 | 90 | 1,086 | 6,319,045,668 | 777,318 |
| **Total** | **2,528** | **302** | **1,084 (379/705)** | **2,045** | **2,045** | **705** | **8,004** | **32,739,650,186** | **4,028,019** |

\*Total images count the private and private-mixed files as separate released image files. The paper notes that private-mixed includes the same scenes as private under seen and unseen lighting, so those two sets are not independent physical-scene groups.

Per-split byte subtotals were not retained in the bounded command output and were not recollected. The table therefore reports exact per-category image bytes across image splits and mask bytes separately; it does not infer split-specific byte totals.

## Structure and file checks

- All eight expected category folders were present. Each had `train/good`, `validation/good`, `test_public/good`, `test_public/bad`, `test_public/ground_truth/bad`, `test_private`, and `test_private_mixed` at the expected shallow depth.
- No expected directory was missing or unexpectedly empty. No unexpected child directories appeared at the inspected leaves.
- 8,004 image files and 705 mask files were counted. All 8,709 leaf files had `.png` extensions; zero-byte files: 0; unexpected extensions: 0.
- Public bad-image stems and mask stems paired 705/705 after removing the documented filename suffix `_mask`; unmatched images: 0; unmatched masks: 0; duplicate stems: 0. This was filename-only pairing.
- No reparse-point directories or files were observed in the enumerated tree.
- Local category spelling `wallplugs` represents the paper's “Wall Plugs”; the official code-utilities package was not retrieved, so the canonical software folder token is not independently confirmed.

## Integrity status

`STRUCTURE_VERIFIED_CONTENT_INTEGRITY_UNVERIFIED`
Dataset structure verdict: `STRUCTURE_COMPLETE`
Content/byte-integrity verdict: `CONTENT_INTEGRITY_UNVERIFIED`

The observed aggregate and category counts match the official paper. A name/size inventory does not establish that each PNG decodes or that file bytes match MVTec's release. No authoritative local checksum manifest was available in the inspected files. Deep decode and hash verification are deferred; they would require a separate bounded authorization and must not include private test pixels or full-content hashes under this task.
