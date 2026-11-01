# R4-FPSA isolated implementation

**Status:** synthetic CPU implementation complete; no performance claim or real-data result. This is an implementation review checkpoint only.

**Source parent:** `76f2f490cd35573a3c13e9bfcc072349d9f15f9e` on `calibration-stage0-contract-20261029`.

## Native lock-in and frozen FPSA rule

The v22 `SignedCorrectionCodebook.update()` keeps a full-bank candidate set to the incoming sign whenever any same-sign anchor exists. Before either sign is present, an incoming sign may replace an anchor of the opposite sign; after the first anchor of that sign is retained, subsequent updates of that sign compete only within its own set. This can lock the live sign allocation despite later queried feedback for the underrepresented sign.

FPSA counts distinct queried Source Group SHA-256 identities, using only feedback registered after the score seal:

```text
N_pos = distinct queried anomalous groups
N_neg = distinct queried normal groups
K_pos = floor(K * N_pos / (N_pos + N_neg) + 0.5)
K_neg = K - K_pos
```

With no feedback there is no quota intervention. Normal-only history targets `(0, K)`; anomaly-only history targets `(K, 0)`. With both labels observed, `K_pos` is clamped to `[1, K-1]`. Targets never inspect TEST prevalence or cause immediate eviction. They are intended capacities, not guaranteed occupancies.

At a full bank, after the native beta floor and merge/append paths, FPSA borrows replacement candidates from the opposite sign only when the incoming sign is below its current target and the opposite sign is above its target. It selects the weakest candidate using the original retention formula and applies the original replacement-acceptance comparison. Rejection leaves the bank unchanged and increments `quota_borrow_rejected`; an accepted cross-sign replacement increments `quota_cross_sign_replacements`. No quota bonus or forced admission is used.

## Native behavior preserved

`FPSASignedCorrectionCodebook` subclasses the frozen native class and reuses its `apply()` unchanged. Its copied `update()` keeps native feature normalization, beta clipping and floor, same-sign merge, append path, retention calculation, replacement acceptance, outcome counters, total capacity, and `last_step` updates. The only algorithmic difference is the marked quota-aware candidate-selection branch at full capacity. Native `inserted/merged/replaced/skipped` statistics and the native counter names remain available.

The native C1-Full writer calls each codebook through the `update(q, beta, *, radius, kappa, outcome, step)` interface. FPSA retains that exact signature, the `apply`, `size`, and `summary` interfaces, and the native slot metadata fields. Existing R1 source exclusion maps slot `last_step` to Source Group provenance; the synthetic integration test exercises that mapping and confirms current/same-source anchors remain excluded after a cross-sign replacement.

## Feedback API and diagnostics

Call `observe_verified_feedback(label, source_group_sha256, step, score_sealed=True)` once per queried group, after sealing scores and before its correction update. The method accepts only integer labels 0/1, a lowercase 64-hex SHA-256 identity, and strictly increasing nonnegative steps; it rejects unsealed feedback and duplicate group identities before changing state. For a multi-layer bank, the future caller must register each query once with each relevant layer Codebook, not once per patch descriptor.

`diagnostics()` exposes queried normal/anomalous group counts, current targets and occupancy, quota opportunity/success/rejection counters, and native operation counts. `last_update_diagnostics` provides per-update outcome counts while `update()` keeps the native return shape expected by C1-Full.

## Verification

Commands, using the existing `D:\anaconda3\envs\torch\python.exe` environment:

```powershell
& $PY -m py_compile experiments/calibration/r4_fpsa_codebook.py tests/calibration/test_r4_fpsa_codebook.py
& $PY -m pytest tests/calibration/test_r4_fpsa_codebook.py -q
```

Result: **10 passed** in 7.25 seconds. Compilation passed. The tests use only small deterministic CPU tensors and synthetic SHA-256 group identities. They cover native update-state/statistics/readout equivalence without quota pressure, normal-only parity, no-feedback and single-class/mixed quota semantics, half-up rounding and clamping, delayed anomalous feedback and native lock-in, weakest-candidate selection under the original retention formula, successful and rejected borrowing, capacity and sign occupancy, transactional feedback validation, R1 `last_step` provenance/LSO behavior, and the native C1 method signatures.

No DINO model was instantiated, no image or dataset file was read, no TEST labels were accessed, and no GPU experiment or score evaluation was run. The four existing torchvision deprecation warnings are from importing the frozen native source and do not affect the CPU tests.

## Subsequent audiojack integration

Only after independent source review and separate experiment authorization, construct an FPSA Codebook with the same `K=256`, feature dimension, device, and `CorrectionConfig` as its native counterpart. Install it into the corresponding `corr_cb[class][view][layer]` slot after native bank initialization. After the event's score outputs are sealed, register a queried label and group digest once with each layer Codebook, then let the existing C1-Full writer perform its unmodified native gate, beta construction, and batched `update()` calls. Keep query labels unavailable to FPSA unless they have passed the existing query-delivery gate.

The eventual comparison must retain the native SC/GC arms and report the feedback counts, targets, actual occupancies, borrow opportunities, successful cross-sign replacements, rejections, and native update counts. No outcome or balanced-accuracy benefit is implied by the synthetic occupancy test.

## Remaining limitations and risks

- Full `PatchVMB.update_corrections_from_topk()` execution was not run; compatibility is established by source inspection, exact method-signature/subclass tests, and direct synthetic Codebook calls. This is not end-to-end C1-Full or GPU parity evidence.
- The native Codebook stores `last_step`, not Source Group IDs. FPSA deliberately preserves that contract and relies on the existing external R1/R3 provenance map and source-exclusion filtering; integration must keep that map synchronized.
- The FPSA feedback identity set and counters are in-memory state. A future resumable runner must persist and verify them alongside native Codebook state, or disallow resuming an FPSA stream.
- Real audiojack occupancy, sign lock-in frequency under the full trajectory, balanced accuracy, and any improvement remain unmeasured. The next step is researcher push and independent source review, followed by a separately authorized experiment.
