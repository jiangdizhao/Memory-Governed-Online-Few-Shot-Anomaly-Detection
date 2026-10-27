# D1-A-R2 — Direct GitHub review and local validation

**Status:** Code-only corrective review, committed on `risk-safe-admission-sprint-20261027`; **local execution pending**. This is not D1-B nor a CDARG implementation. The previous D1-A-R1 outputs stay immutable as evidence of the R1 source snapshot.

## 1. Verified historical state (R1)

- R1 was uploaded in commit `82539dde463413f5c997f4738c2c6b811727c978`.
- Real-IAD official sample key is the primary unit, not the `S####` diagnostic directory. Six development classes each have 525 primary stream samples / 2,625 images, with 79 queries at the V0.2 reference budget.
- The previous D1-A-R1 experiment artifacts remain under `outputs/risk_safe_sprint_20261027/d1a_preflight/`. They must never be silently overwritten.
- R1 code reported 52 successful CPU-only tests; the new R2 tests and behavior **have not yet been executed on the user's Windows environment**.

## 2. R2 corrections

### 2.1 Label-bearing file paths must not reach a pre-query controller

Real-IAD filenames and official sample keys include `OK`, `NG` and defect-code tokens, so merely omitting an `is_good` field is **not sufficient** to prevent a policy from reading the true label. Q-A randomization itself remains unchanged, but the synthetic shared-tape replay now calls `predict(step, PublicObservation)`, which exposes only `position`, an opaque `observation_handle`, `camera_id` and `phase` (no path, sample key, timestamp or label). Trusted loaders may map opaque handles to image data outside the controller; model state updates happen only after an authorized query.

The historical `StreamImage`, dataset IDs and Q-A tape manifest retain full filenames for *trusted audit, lookup and reproducibility*, **not** to be passed directly as pre-query Q-B/Q-C controller arguments. The future D1-B/C code must enforce the same access boundary at runtime and test it with a malicious/probing fake controller.

This does **not** assert statistical non-informativeness of visual content, camera IDs or capture order; it strictly prevents direct label leakage through names and metadata.

### 2.2 Provenance after commit/push

R1 serialized the run's original Git HEAD into `source_manifest.json`. When the result-generating code and previously untracked outputs were subsequently committed, the new HEAD no longer matched the recorded one, despite identical bytes.

The R2 preflight preserves the **recorded original run's Git metadata** when reusing an existing source manifest only if Git proves a fast-forward ancestry relationship. All source hashes, dataset fingerprints, configuration hashes, serialized output digests and all other fields must still match exactly; any change still raises `OutputMismatch`. The old manifest is *not rewritten*.

The R2 Python sources differ from the R1 sources. **Do not rerun R2 against the old R1 canonical output directory.** This should fail closed by design. Generate and verify a **new**, versioned output directory.

## 3. Local validation after syncing this branch

From PowerShell in `E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection`:

```powershell
git status --short
git fetch origin
git switch risk-safe-admission-sprint-20261027
git pull --ff-only
git rev-parse HEAD

& "D:\anaconda3\envs\torch\python.exe" -m py_compile experiments/risk_safe/preflight.py experiments/risk_safe/query_protocol.py tests/test_risk_safe_d1a.py
& "D:\anaconda3\envs\torch\python.exe" -m unittest discover -s tests -p "test_risk_safe_d1a.py" -v

& "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.preflight `
  --dataset-root "G:/dataset/realiad" `
  --json-dir "G:/dataset/realiad/jsons/realiad_jsons" `
  --out-dir "outputs/risk_safe_sprint_20261027/d1a_preflight_r2"

& "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.preflight `
  --dataset-root "G:/dataset/realiad" `
  --json-dir "G:/dataset/realiad/jsons/realiad_jsons" `
  --out-dir "outputs/risk_safe_sprint_20261027/d1a_preflight_r2" `
  --verify-only

git status --short
```

**Expected:** CPU tests pass; six-class metadata audit passes; first R2 run creates four JSONs in `d1a_preflight_r2`, and `--verify-only` validates all four. If any command fails, preserve the traceback and do not rerun destructively.

After review of outputs, commit *only* the new `d1a_preflight_r2` directory:

```powershell
git add -- outputs/risk_safe_sprint_20261027/d1a_preflight_r2
git diff --cached --stat
git diff --cached --check
git commit -m "data(risk-safe): archive validated D1-A-R2 metadata preflight"
git push origin risk-safe-admission-sprint-20261027
```

Commit and push **only if the tests and verify-only pass**. The provenance fix is designed to permit future verification after this result commit, provided the source/data/configuration remain unchanged.

## 4. Acceptance gates before D1-B

1. Tests show controller prediction cannot access filename or official sample key.
2. Tests demonstrate legacy R1 output bytes cannot be silently reused after R2 source changes.
3. Source manifest verification survives a provenance-only fast-forward commit, while changed source/dataset/configuration fails.
4. Six categories retain the official-sample grouping, 525 groups and 2,625 images per stream, object/key partition nonoverlap, deterministic nested Q-A queries.
5. No legacy v22 / V0.2 code, historical outputs or GPU experiments are changed.
6. D1-B query-budget grid, Q-B rule and research hyperparameters remain **unfrozen and unexecuted**, pending independent review.
