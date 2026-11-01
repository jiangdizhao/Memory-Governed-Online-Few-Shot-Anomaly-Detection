# P0-R1 Audit Scope and Resource Record

## Repository and historical-document scope

- Repository: `E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection`
- Branch / starting HEAD: `calibration-stage0-contract-20261029` / `3d2b17c844e7e783f400c989bf968cd54889e4ab`
- Reviewed Git-tracked protocols, summaries and audit reports for Stage 0/0.5/0.75, Stage 1-A, R1/R2 and the original v22/PR evidence.
- Did not read any untracked output or private artifact. Did not access `G:\dataset\realiad`, Real-IAD category JSON, images, masks, labels or feature caches.

## MVTec filesystem scope

- Dataset root: `G:\dataset\mvtec_ad_2`.
- Inspected only its root, the eight expected category directories, and the documented `train`, `validation`, `test_public`, `test_private`, `test_private_mixed`, and public-mask leaf directories. Maximum inspected depth was the expected category/split/mask layout. The immediate parent was checked only for a directly present MVTec-named archive; no recursive parent search was made.
- Distinct entries in the final tree inventory: 8,808 (97 directories, 8,711 files including the two root documentation files). The 8,709 leaf files are 8,004 images plus 705 masks.
- Metadata enumeration was repeated in bounded passes to correct mask-stem matching and inspect reparse-point attributes. The full-listing commands returned at least 34,846 child-entry observations across those passes, plus earlier shallow directory listings. Thus the distinct filesystem tree was below 20,000 entries, but a cumulative-per-enumeration interpretation of the 20,000 limit would be exceeded. This audit-scope deviation is disclosed; no further dataset enumeration was performed.
- No image or mask content was opened. Image headers opened: 0. Private-test headers/pixels opened: 0. Mask pixels read: 0. Image decoding: 0. Image/mask/archive hashes: 0. No archive traversal, download, extraction or inference was run.
- Public bad-image and mask filenames were compared by stem only; no individual filename inventory is included in this report. Private filenames were counted but not emitted or interpreted.
- Local documentation: two unique text files (`readme.txt`, `license.txt`), three reads total, 23,216 bytes read. This is far below the 8 MiB text ceiling. No model or feature-cache files were accessed.
- File metadata checks found 0 zero-byte files, 0 unexpected extensions, 0 missing expected directories, 0 unexpected leaf subdirectories, and 0 reparse-point files/directories. All 8,709 leaf files were `.png`.

## Checks deliberately skipped or limited

- No PNG decode validation or byte comparison to the official release; content integrity remains unverified.
- No split-specific byte subtotals were retained from the bounded command output; exact per-category image bytes and separate mask-byte totals are recorded in `MVTEC_AD2_STRUCTURAL_INVENTORY.md`.
- No local acquisition archive/source manifest or authoritative official checksum was available in the inspected scope. The search did not extend outside the dataset root and immediate parent.
- No TRAIN/VALIDATION image header was opened because official documentation already establishes the grayscale/resolution issue and the task did not require a visual check.
- No MVTec historical experiment search was conducted, per the researcher's confirmation that it had never been used before this audit.
- No model inference, DINO feature extraction, AUROC/AUPRC, hyperparameter selection, private-test content inspection, or benchmark submission was run.

## Procedure note

The repeated metadata listings did not read file contents, but they were avoidable. The counts are based on metadata and agree across the inspected passes; the repeated traversal and missing per-split byte output remain explicit limitations rather than being presented as a single-pass resource audit.
