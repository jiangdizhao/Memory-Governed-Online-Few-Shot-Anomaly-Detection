# Experiment A: Calibration Scale, Initialization and State Dependence

Run `20261106_a1_full01` completed A1 only. This is a four-category development study, not independent confirmation.

## Protocol and integrity

Branch `calibration-stage0-contract-20261029` at parent `1c8dad859bba886438152491644dff0adbfe4336`. The frozen native DINOv2/VisionAD path was used; the fixed bank realized 8,192 float16 prototypes per layer. Test images were scored before Stage B; the separate Stage B process read image labels only after four category-specific Stage A seals. No masks or confirmation categories were accessed.

The Stage05 static score-tape manifest contains stale hashes for `stage05.py` and `stage05_runner.py`; those tapes were used only for full-vector numerical parity against fresh native scores, never as primary results. Support/C64 fixed-bank structured digests were independently verified.

## Data roles

| Category | Train groups | Eligible calibration groups | H groups | Cmax-V images | Test events |
|---|---:|---:|---:|---:|---:|
| audiojack | 247 | 207 | 32 | 1035 | 3815 |
| regulator | 249 | 209 | 32 | 1045 | 3855 |
| switch | 246 | 206 | 32 | 1030 | 3820 |
| rolled_strip_base | 249 | 209 | 32 | 1045 | 3805 |

## A1 results

Full threshold results, threshold draw stability, normal-only H exceedance, prefixes, and 512-event windows are in the protected local run output. Threshold arms shared the same frozen score vector in each support condition. Exact per-condition AUROC equality was verified.

The report does not select a calibration size by test BER. The decision candidate below uses H-tail behavior, draw stability, and resource use; test metrics remain evaluation and safety outcomes.

## Component status

- A1: `COMPLETE_VALID` for P0/P1/P2 across the four authorized development categories.
- A2: not executed; shared-state B3 rescoring has not been completed.
- A3a: not executed; calibration-coupled trajectories have not been completed.
- A3b: `HOLD_BASELINE_FIDELITY`; a deterministic equal-byte expanded coreset constructor is not specified sufficiently to implement without adding an unregistered method.
- No Stage 1/2 program, Experiment B/C, confirmation outcome, or mask was accessed.

## Reproduction

`D:\anaconda3\envs\torch\python.exe -m experiments.calibration.a_calibration_scale --run --run-id 20261106_a1_full01`

A1 run output: `E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection\outputs\calibration\a_calibration_scale_20261106_a1_full01`. Audit: `E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection\outputs\calibration\a_calibration_scale_20261106_a1_full01\integrity_audit.json`.
