# FS4 R1 engineering test matrix

| Gate | Result and evidence |
| --- | --- |
| Prior FS4 tests and new CPU tests | PASS: 55 passed, 0 failed, 0 skipped; four Pillow/torchvision deprecation warnings. Command: `D:\anaconda3\envs\torch\python.exe -m pytest -q tests/calibration/test_fs4_threshold.py tests/calibration/test_fs4_hardening_r1.py`. |
| Fatal Greedy/C1 provenance, nonfinite score, cache corruption, live mutation | PASS: prior native tests plus `test_native_c1_provenance_and_mutation_before_exception_fail_fatally`, fatal-injection parameterization, and missing/nonfinite cache tests. |
| Retry, duplicate token, G-only/GC-only changes and ESS hold | PASS: retry test and the prior version/ESS tests. |
| High/low query rates, drift, interleaving, ties, probabilities and no forced queries | PASS: ten parameterized stress scenarios and prior slot/exploration tests; full 400-group records are in the query stress JSON. |
| Shared reference isolation, coupled query-only mode, frozen C1 outcome categories | PASS: prior shared test, coupled disagreement/query test, and four native TN/FP/FN/TP outcome cases. |
| Prediction-before-label order, unsealed/duplicate feedback, corrupt ledger | PASS: durable order, synthetic executor, malformed-record and forged-utility tests. |
| Required source fingerprints, registered event-stream seal, multi-category separation | PASS: missing-source and stream-mismatch negative tests; two independently initialized category policies complete a global Stage A seal. |
| Native Greedy/C1 restore, E cache and receipt rebinding | PASS: native boundary checkpoint test and checksum rejection. |
| Crashes at prediction, query, feedback, C1, Greedy, E refresh, after update | PASS: replay equals the uninterrupted native CPU reference; no recorded feedback is fetched twice. Ambiguous intent without a recorded label fails closed. |
| Crash during actual E rescoring | PASS: failure after five of twelve native rescoring calls, atomic threshold hold, then replay matches the uninterrupted post-state and ledger. |
| Synthetic CUDA | PASS: 4 passed, 0 failed, 0 skipped; CPU/GPU G and GC tolerance `atol=1e-5`, `rtol=1e-4`, direct source exclusion, Top-32 reselection, native C1 update, actual Greedy replacement, 8,192-prototype score and E resource probe. |
| Four-Support TRAIN-only native construction | See `HARDENING_R1_NATIVE_GPU_PARITY.md` and its source-versioned manifests. |
| Compilation, selected Ruff checks, protected-source and pathname audits | PASS; 27 protected tracked source files and 7,030 pre-existing untracked pathnames checked. |
| Scientific campaign, TEST labels/images, masks and detection metrics | SKIP: not authorized or run. A broad historical-document search did surface a prior calibration summary; see the handoff for the scope limitation. |
| Production feature/model/Support binding | BLOCKED: the generic durable executor accepts caller-provided features and does not prove their association with the registered dataset and model. |

Each test passes only when its indicated integrity condition holds. A successful synthetic engineering check is not a scientific performance or FPR guarantee.

During development the first proposed mid-rescore fixture did not reach the rescore path: unequal query weights correctly made ESS fall below 12. The fixture was corrected to use distinct synthetic feature directions and to assert twelve equal-probability exploitation receipts; the ESS rule and all acceptance assertions remained unchanged. The final suite counts above are from the corrected source revision.
