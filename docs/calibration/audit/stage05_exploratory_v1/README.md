# Stage 0.5 exploratory audit package

This package contains aggregate development-set statistics from the four authorized classes. It is exploratory reconnaissance, not Stage 1 validation. Development test outcomes are exposed and must not be described as unseen in future method development. Mint/PCB were untouched; no controller was changed; Stage 1/2 remain on HOLD.

## Query feedback composition

`query_seed_aggregate_summary.csv` and `query_seed_sensitivity.json` contain seed-37 query composition, all 200 fixed schedule summaries, camera/bin/group-view summaries, first-400 composition, activation quantiles, and sliding 256-image query-feedback windows. The 200 schedules are a registered finite sensitivity set, not a probability distribution over all possible tapes.

| Class | Rate | Seed-37 queried normals / defects | Seed-37 queried defect fraction | Finite-set F(>=8) | Finite-set F(>=14) |
|---|---:|---:|---:|---:|---:|
| audiojack | 3% | 74 / 40 | 0.3509 | 1.000 | 1.000 |
| audiojack | 5% | 124 / 67 | 0.3508 | 1.000 | 1.000 |
| audiojack | 7% | 171 / 96 | 0.3596 | 1.000 | 1.000 |
| regulator | 3% | 81 / 35 | 0.3017 | 1.000 | 1.000 |
| regulator | 5% | 130 / 63 | 0.3264 | 1.000 | 1.000 |
| regulator | 7% | 178 / 92 | 0.3407 | 1.000 | 1.000 |
| rolled_strip_base | 3% | 33 / 81 | 0.7105 | 1.000 | 1.000 |
| rolled_strip_base | 5% | 58 / 132 | 0.6947 | 1.000 | 1.000 |
| rolled_strip_base | 7% | 83 / 183 | 0.6880 | 1.000 | 1.000 |
| switch | 3% | 54 / 61 | 0.5304 | 1.000 | 1.000 |
| switch | 5% | 90 / 101 | 0.5288 | 1.000 | 1.000 |
| switch | 7% | 132 / 135 | 0.5056 | 1.000 | 1.000 |

`query_window_sensitivity.csv.gz` reports every stride-one 256-image window position with seed-37 queried-defect count and min/P05/P25/median/P75/P95/max/mean/SD across the fixed seeds. It contains aggregate counts only.

## Static native scorer

`static_score_seed_summary.csv` records all class x support-seed AUROC, AUPRC, theta0 confusion metrics, the in-sample retrospective static-threshold oracle, headroom, and worst-window summaries. `static_score_distribution_seed_summary.csv` reports normal/defect score quantiles, overlap intervals, and score tie counts. `static_score_windows.csv.gz` holds aggregate confusion results for every static 256-image score window.

The oracle is an optimistic in-sample diagnostic that was unavailable to any controller. The `FROZEN_STATIC_NEAR_CEILING` flag uses the registered exploratory screen (Frozen BA >= 0.95 and oracle headroom <= 0.01). It is not a statement about industrial risk or whether dynamic-memory calibration is needed.

## Gates and interpretation

`gate_matrix.json` summarizes the audit status. `data_exposure_ledger.json` records authorized label exposure. `source_and_protocol_provenance.json` links the frozen inputs, score tapes, source/configuration, model, runtime, and support identities. `checksums.sha256` covers every package file except itself.

The 64 C64 images are additional protected train-normal anchors alongside four deployed supports; this is not a four-shot-total setup. The static scorer uses M0 only, with dynamic Greedy memory empty and disabled. This audit does not test adaptive memory, online threshold calibration, or controller value under changing memory.
