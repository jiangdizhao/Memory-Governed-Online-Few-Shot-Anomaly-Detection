# Stage 0.5 data-exposure ledger

This ledger records access under the explicit Stage 0.5 authorization. The machine-readable ledger, source hashes, class JSON fingerprints, tape hashes, support/C64 identity digests, and model provenance are in `audit/stage05_exploratory_v1/`. The development test outcomes below are now exposed exploratory information and cannot be described as unseen for subsequent method development.

| Information | Authorized scope | Access status | Purpose / record |
|---|---|---|---|
| Development training-normal labels | Four audited classes only | Accessed | Historical seed-190021 C64 and support seeds 0-4 selection. Identity digests are recorded; no test outcomes informed selection. |
| Development test labels for query composition | `audiojack`, `regulator`, `switch`, `rolled_strip_base` only | Accessed after all query tapes and support selections were frozen and hashed | Full-stream composition, query normal/defect counts, cameras, groups, bins, first-400 counts, defect arrival, gap, and query-window sensitivity. |
| Development test images | Same four classes only | Accessed by native scorer | Frozen M0 scoring after support/C64 freeze. No label or mask input to scoring. |
| Development test labels for score evaluation | Same four classes only | Accessed after all 20 score tapes were frozen and checksummed | AUROC/AUPRC, theta0 metrics, retrospective static threshold oracle, and temporal window metrics. The oracle was not exposed to any controller. |
| Segmentation masks | None | NOT ACCESSED | Outside scope. |
| Mint/PCB and other confirmation outcomes | None | NOT ACCESSED | Remain sealed. |
| Stage 1/2 streams or controller outcomes | None | NOT RUN | Stage 1/2 remain on HOLD. |
| Controller source or parameters | None | NOT MODIFIED | No algorithm selection or tuning. |

## Access timing and provenance

- Test-label access started: `2026-10-29T04:17:14Z` after the query tape freeze and support/C64 selection were finalized.
- Score evaluation labels were joined only after the score-tape manifest and all 20 class-seed score files were finalized.
- Official JSON SHA-256 values by class, query archive hash, score manifest hash, support/C64 identity digests, and per-phase source hashes are in `source_and_protocol_provenance.json` and `data_exposure_ledger.json`.
- The frozen query archive is identified by SHA-256 `db61283c1d2b368687604d8e29b668e200a229bf9cb0d4cbaa41970026e7e031`. The score-tape manifest SHA-256 is `0e09148778bf1ca1ff93cf2db5b966c6a4af825b7e88241db4f65bc64a17a18c`.
- The aggregate query-window output SHA-256 is `aefcfdc66579588a32f1188209cf9547ed1d41a2b9192619ceea00ad22aa4eae`. Static score files are linked by checksums in the local run manifest and audit package.
- No per-image labels, full score tapes, images, masks, weights, or feature caches are published or committed. Local raw score tapes remain under the isolated Stage 0.5 output directory.

Any future method or design decision informed by these exposed outcomes must disclose this loss of blindness. This ledger does not authorize Stage 1/2 or confirmation-class access.
