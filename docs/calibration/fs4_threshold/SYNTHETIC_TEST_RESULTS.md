# Synthetic implementation verification

This is an engineering verification record, not a Real-IAD experiment or scientific performance report. All examples and outcomes are synthetic. No dataset, model weights, Real-IAD score, TEST label, or mask was accessed for this task.

| Check | Result | Scope |
| --- | --- | --- |
| `D:\anaconda3\envs\torch\python.exe -m pytest -q tests/calibration/test_fs4_threshold.py` | PASS: 22 passed, 0 failed, 0 skipped | CPU synthetic tests; four Pillow/torchvision deprecation warnings |
| `D:\anaconda3\envs\torch\python.exe -m experiments.calibration.fs4_threshold_synthetic_smoke` | PASS | Orchestration-double and native CPU synthetic streams |
| `python -m ruff check --select F,E9` on the eight new modules and test file | PASS | Undefined names and selected syntax errors; used the same torch environment |
| `python -m compileall -q` on the eight new modules and test file | PASS | Python compilation; used the same torch environment |
| `python -m experiments.calibration.fs4_threshold_audit` | PASS | Branch/start ancestor, exact changes allowlist, protected tracked source hashes, existing untracked pathname inventory, and runtime identifier scan |
| Native GPU, full real Support-bank build and Real-IAD execution | SKIP | Not authorized or verified in this assignment |
| Durable production ledger, checkpoint/resume and real dataset identity binding | SKIP | Specified for a future authorized campaign, not implemented or validated here |

Both smoke streams contain 70 synthetic events in 14 five-view groups. Each executes 14 sealed queries, discloses 12 synthetic normals and two synthetic defects, and checks all 70 synthetic outcomes only after the event and query seals. The orchestration-double run performs 36 current-state rescores. The native CPU run exercises native Greedy and C1 and performs 48 rescores; its final banks contain 96 Greedy slots and 14 correction slots. It uses a deliberately tiny synthetic static prototype bank; it is not a full native four-Support augmentation build. The builder's four-Support restrictions are separately exercised with a builder double. No scientific accuracy or risk metric is inferred from these checks.

The selection-bias diagnostic uses 80 seeded Bernoulli samples from a synthetic 1,000-point population. Its reference 90th percentile is approximately 0.89990; the unweighted selected-sample percentile has mean 0.95448 and standard deviation 0.00319, while the weighted percentile has mean 0.90013 and standard deviation 0.00799. Clipped-weight ESS ranges approximately 116.13–136.36 (median 124.89). This fixed simulation illustrates selection and estimator variability; it does not establish unbiasedness, coverage, or superiority for an adaptive deployment.

The audit preserves the starting inventory of 7,030 pre-existing untracked paths with SHA-256 `5d036161ab992ad68c8f98e7b03e91199fc7fa7b4d170603936c57064488dd18`. It compares tracked historical files and specifically checks protected source bytes after line-ending normalization. It does not open or hash the contents of existing untracked outputs, so their byte-level preservation was not independently established by that inventory check.
