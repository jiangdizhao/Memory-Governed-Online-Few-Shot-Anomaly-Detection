# FS4 R2 validation matrix

Source under test: `f39f31b53856ffddfbcce5f7733a0795d1ffd7a7`. Tests use synthetic data, models, keys and fixtures unless explicitly noted. Tests of the production plumbing bypass the fail-closed hold only inside the test fixture; they do not prove a trusted producer boundary.

| Gate | Result | Evidence and limitations |
| --- | --- | --- |
| Trusted encoding | **FAIL** | `test_production_rejects_preconstruction_encoder_substitution` demonstrates a same-process encoder substitution accepted after the synthetic fixture bypasses the hold. Production context is disabled by default. |
| Support binding | **PASS, synthetic**; **SKIP, real R2** | Registered Support identity and bank mismatch tests pass using synthetic inputs. No real Support images or checkpoint were read in R2; prior R1 preflight is historical evidence, not an R2 production test. |
| Ledger provenance | **PASS, synthetic**; **BLOCKED, production** | Registered image, feature/CLS, runtime and tamper tests exercise plumbing; the unresolved producer boundary prevents a production provenance claim. |
| Stage B authorization and completion | **PASS, synthetic**; **SKIP, provisioned production** | Signature, scope, missing-result and tampering tests use ephemeral keys; no real trust root, separate provisioned Stage B account, grant or evaluation was used. |
| Access controls | **PASS, Windows synthetic**; **SKIP, two-account integration** | Windows Authz denies the untrusted World SID and tests reject a weakened ACL. A real separate Stage B principal was not provisioned or tested. Full local administrators and code inside a privileged process remain outside this operational boundary. |
| Checkpoint recovery | **PASS, synthetic**; **BLOCKED, production** | Mismatched dataset/encoder and replay checks use synthetic fixtures; the real producer boundary and production recovery remain unvalidated. |
| CPU tests | **FAIL** | 80 passed, including the prior 55; one failed: `test_production_rejects_preconstruction_encoder_substitution`. Four dependency deprecation warnings. |
| CUDA verification | **PASS** | Six bounded synthetic CUDA tests passed, including all four prior tests; four dependency deprecation warnings. |
| Compilation and repository audit | **PASS** | `compileall` passed; 27 protected source files and 7,030 unrelated untracked paths match the starting inventory; R2 identifier scan passed. |
| Scientific run and Stage B evaluation | **SKIP** | Not authorized by this task; neither was performed. Future execution requires separate scientific authorization and provisioned principals and signing authorities. |

Commands, using `D:\anaconda3\envs\torch\python.exe`:

```text
python -m pytest -q tests/calibration/test_fs4_threshold.py tests/calibration/test_fs4_hardening_r1.py tests/calibration/test_fs4_hardening_r2.py
python -m pytest -q -s tests/calibration/test_fs4_hardening_cuda.py tests/calibration/test_fs4_hardening_r2_cuda.py
python -m compileall -q experiments/calibration/fs4_r2_audit.py experiments/calibration/fs4_r2_security.py experiments/calibration/fs4_r2_trust.py experiments/calibration/fs4_r2_stage_b.py experiments/calibration/fs4_r2_context.py experiments/calibration/fs4_durable_ledger.py experiments/calibration/fs4_durable_runtime.py experiments/calibration/fs4_checkpoint.py experiments/calibration/fs4_threshold_integration.py tests/calibration/test_fs4_hardening_r2.py tests/calibration/test_fs4_hardening_r2_cuda.py
python -m experiments.calibration.fs4_r2_audit
```
