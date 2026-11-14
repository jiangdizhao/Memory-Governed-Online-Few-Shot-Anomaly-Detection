# VisA 1cls P0 - 12-category F-B external validation

## Executive summary

The frozen F-B configuration completed Stage A and post-seal Stage B for all 12 registered VisA 1cls P0 categories: 2,162 TEST images total. Category-equal macro AUROC was **0.946476**, AP **0.948709**, BER **0.125069**, and balanced accuracy **0.874931**. This is a single external short-stream compatibility study; it does not establish VisA-wide mechanism superiority beyond the prespecified F-B validation objective.

## Experimental protocol

- VisA 1cls, all 12 registered categories, P0 only; Support Seed 0 and Stream Seed 0.
- Four deployed TRAIN-normal Supports and 64 separate TRAIN-normal C64 images per category; no Support/C64/TEST overlap.
- Frozen DINOv2 ViT-L/14 Reg4 and the existing native four-arm runner/controls were used unchanged. F-B was the preselected primary configuration.
- The query target is 10% with a per-category hard cap. No forced queries or top-ups were used.
- VisA has no registered physical-object key, so query units are image-level; exact duplicate-content clustering remained inactive because the verified local census contained no exact duplicate TEST images.
- Stage A used sealed-before-feedback, query-only label disclosure. Official image-level outcomes were projected only after each valid category seal in separate Stage B evaluation.
- The study is external short-stream compatibility/performance validation, not a new internal BEA/Gaussian mechanism comparison.

## Primary F-B results

| Category | N | Normal | Anomaly | AUROC | AP | BER | FPR | FNR | Balanced accuracy | Queries | Query rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| candle | 200 | 100 | 100 | 0.942800 | 0.948358 | 0.170000 | 0.040000 | 0.300000 | 0.830000 | 20 | 10.0000% |
| capsules | 160 | 60 | 100 | 0.977500 | 0.981962 | 0.071667 | 0.133333 | 0.010000 | 0.928333 | 15 | 9.3750% |
| cashew | 150 | 50 | 100 | 0.939000 | 0.969796 | 0.175000 | 0.080000 | 0.270000 | 0.825000 | 15 | 10.0000% |
| chewinggum | 150 | 50 | 100 | 0.987200 | 0.994276 | 0.030000 | 0.000000 | 0.060000 | 0.970000 | 13 | 8.6667% |
| fryum | 150 | 50 | 100 | 0.984200 | 0.991138 | 0.060000 | 0.100000 | 0.020000 | 0.940000 | 15 | 10.0000% |
| macaroni1 | 200 | 100 | 100 | 0.961100 | 0.961384 | 0.140000 | 0.020000 | 0.260000 | 0.860000 | 20 | 10.0000% |
| macaroni2 | 200 | 100 | 100 | 0.923300 | 0.922931 | 0.170000 | 0.180000 | 0.160000 | 0.830000 | 20 | 10.0000% |
| pcb1 | 200 | 100 | 100 | 0.893700 | 0.875485 | 0.175000 | 0.110000 | 0.240000 | 0.825000 | 20 | 10.0000% |
| pcb2 | 200 | 100 | 100 | 0.903000 | 0.869096 | 0.140000 | 0.100000 | 0.180000 | 0.860000 | 20 | 10.0000% |
| pcb3 | 201 | 101 | 100 | 0.909307 | 0.927685 | 0.139653 | 0.069307 | 0.210000 | 0.860347 | 20 | 9.9502% |
| pcb4 | 201 | 101 | 100 | 0.949802 | 0.949510 | 0.159505 | 0.099010 | 0.220000 | 0.840495 | 20 | 9.9502% |
| pipe_fryum | 150 | 50 | 100 | 0.986800 | 0.992888 | 0.070000 | 0.140000 | 0.000000 | 0.930000 | 15 | 10.0000% |
| **Category-equal Macro** | **2162** | **962** | **1200** | **0.946476** | **0.948709** | **0.125069** | **0.089304** | **0.160833** | **0.874931** | **213 total** | **9.8285% mean** |

F-B executed **213** queries across the 12 categories. The category-equal mean realized query rate was **9.8285%**; pooled across all images it was **9.8520%**. Ten categories exhausted their hard cap. Capsules used 15/16 queries and chewinggum 13/15; the remaining categories used their full cap.

### Interpretation

F-B AUROC ranged from **0.893700** (pcb1) to **0.987200** (chewinggum), with BER from **0.030000** (chewinggum) to **0.175000** (cashew and pcb1). The result varies materially across categories. FPR and FNR trade off across categories: for example, macaroni2 has FPR 0.180000 and FNR 0.160000, while pipe_fryum has FPR 0.140000 and FNR 0.000000. Short streams also make the absolute query budget sparse (13 to 20 F-B queries per category), and TEST anomaly prevalence varies from 49.75% to 66.67%. These observations describe this registered P0 run; they do not identify future learning value for unqueried outcomes or establish internal module superiority.

The registered Static-4+64 and Static-36+32, F-M, K-B, K-M and N-shadow trajectories were retained for every category in the secondary machine-readable aggregate. They are not used here to claim BEA or Signed Correction superiority. Any descriptive static-versus-online difference is confounded with the registered online feedback trajectory and is not a separate causal claim.

## Limitations

- The data identity is a researcher-approved local snapshot, not a publisher-authenticated release.
- Image-level query units do not establish physical-object independence.
- Only P0 is tested; VisA seed stability is not independently evaluated.
- One short-stream external configuration cannot establish internal module superiority. It also does not remove separate limitations of the Real-IAD benchmark.
- The execution resource telemetry did not produce peak allocated/reserved GPU or peak RSS fields; point-in-time resource snapshots are recorded in each Stage A parity receipt and are not peak measurements.
- Static-36+32 was executed as registered; its source receipt retains the adapter status `CUDA_NUMERICAL_PARITY_PENDING_GATE2`, so this report does not claim a separate independent CPU-versus-CUDA K-Center parity study.

## Provenance and artifacts

- Source commit: `ddbd59efe8897877b03c1b2a2ff19222fa5c5997` (branch `calibration-stagea-consolidated-20261114`).
- Dataset snapshot SHA-256: `15535268a41c5c163b9906874fa148478f9da0bf9394659f448d320de778a7f2`; split CSV plus license artifact-set SHA-256: `5612ff8ec0b9ca5c1545b9f08aaba71b89f3b36bcaf8dc2e9fee82589949d029`.
- Official `1cls.csv` SHA-256: `a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995`; local dataset release SHA-256: `913e32622b9567d28d4b974c1e3444d1c3333f27522a8b1faa02a2a7440b97b5`.
- Registration index: `outputs/calibration/final_benchmark/visa_1cls/registration_01/registration_index.json`.
- All per-category seals, journals, authorization bindings and Stage B file hashes are in `outputs/calibration/final_benchmark/visa_1cls/macro_p0_12cat_01/integrity_checks.json`.
- Complete Stage B results for all seven methods are in `outputs/calibration/final_benchmark/visa_1cls/macro_p0_12cat_01/visa_p0_macro_metrics.json`.
- F-B category metrics and category-equal macro (including query and class counts) are in `outputs/calibration/final_benchmark/visa_1cls/macro_p0_12cat_01/visa_fb_12cat_summary.json`.
- Research instruction digest for the remaining-category execution: `58d801eef8da033a48086d608eeee3824140f3d4a9f584f4a40329e1c6cde888`. No independent reviewer, supervisor or publisher-authentication approval is claimed.
- TEST labels were not used during Stage A except after executed queries; remaining TEST outcomes were projected only after valid seals in Stage B. Pixel masks were never opened.
