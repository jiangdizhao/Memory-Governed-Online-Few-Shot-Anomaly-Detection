# FS4 implementation review handoff

The requested implementation is limited to four-Support initialization, stochastic group-slot BEA, L and E, native source provenance, and synthetic engineering verification. It does not establish scientific performance. Independent review should inspect the actual source, its test evidence and the limitations in the accompanying contract before any later experiment is considered.

Review entrypoints:

- `experiments/calibration/fs4_stochastic_bea.py`: lagged MAD, B/N/D equations, candidate-slot and query-action randomization, online price, and sealed receipts.
- `experiments/calibration/fs4_threshold_core.py` and `fs4_threshold_controllers.py`: normal-only evidence validation, weighted CDF/ESS and versioned L/E updates, diagnostics and capacity behavior.
- `experiments/calibration/fs4_greedy_provenance.py` and `fs4_source_excluded_scorer.py`: exact native write mirror, replacement union, full-feature cache, source-filtered clone, Top-32 reselection and C1 filtering.
- `experiments/calibration/fs4_threshold_integration.py`: four-Support builder, native feedback, shared independent query reference, coupled instance isolation, and ordered seal/disclosure callbacks.
- `experiments/calibration/fs4_threshold_synthetic_smoke.py`, `fs4_threshold_audit.py` and `tests/calibration/test_fs4_threshold.py`: CPU-only fixture, exact allowlist and tests.

Future scientific execution remains blocked pending independent review and a separate authorization. GPU native parity, full real Support-bank construction, identity binding to a dataset manifest, a durable ledger and checkpoint/resume are not verified by these tests. The statistical and source-exclusion limits are recorded in the companion reports. The untracked historical pathname inventory was compared with its recorded starting digest; untracked historical contents were not opened for a byte-for-byte comparison.
