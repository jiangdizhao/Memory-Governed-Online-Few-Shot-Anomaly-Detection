# FS4 R4 label-blind research engineering review

Source freeze: `45f518a7673cd26474f17cd5cc481d3bbda645eb`. Starting revision: `0d058dddff9279fd03cb1e791792b1db7334c99c`. This task used synthetic images and labels only. No Real-IAD TEST image, outcome, mask, scientific run or real Stage B evaluation was accessed or performed.

## Label-access and image audit

`fs4_r4_preparation.py` checks an ordered original registration, verifies each original image's registered content and official Source Group/camera identity, and creates two separate, non-overwriting stores. The private mapping contains original paths, original groups, labels, camera views and image fingerprints. The public registration contains random 256-bit opaque group and event IDs, opaque image-resource names, camera and stream positions, dimensions, and content/resource/pixel digests. The mapping commitment includes a private random salt; it is not a direct path hash. Five registered camera views share an opaque group ID, all positions remain ordered, and the original Support groups are disjoint from the TEST groups.

The preparer decodes to RGB, writes a fresh metadata-free lossless PNG, and verifies exact pixel and geometry equality. `fs4_r4_blinded.py` validates the complete public registration and preparation receipt and reads only those opaque resources. The Stage A worker is not passed the private mapping root, original TEST paths or label store. Its configuration, manifest, journal and checkpoint are scanned in the synthetic integration test for original TEST filenames; an attempted mapping dump through the broker interface is rejected. Queried labels are written to the durable feedback record only after the registered action.

The separate broker validates the private/public mapping before the worker starts. For each request it independently replays the durable hash chain and query policy, and requires the matching prediction, executed query, query receipt, group, position and `LABEL_INTENT`. It creates a one-use disclosure marker before returning a label. A duplicate, fabricated receipt, wrong group, future event or request without an executed query fails closed. Checkpoint recovery replays already recorded feedback without a second broker release; an intent without durable feedback remains ambiguous and cannot be replayed safely.

The separate CPU Stage B evaluator first validates the global Stage A and category seals. Only afterward does its mapping-verification subprocess open the private mapping and derive labels, check original and opaque mapping consistency, and compare all legally queried feedback. It recomputes metrics from sealed scores and thresholds and writes a protected outcome projection, aggregate result, Stage B seal and completion seal. The R4 completion verifier rechecks the projection checksum as well as the result and seals.

These are operational controls for a trusted local researcher. The worker and broker use the same Windows account; this is not OS-account isolation or a security guarantee against a malicious local operator, arbitrary code execution in the privileged process, or a full local administrator. The earlier high-security `ProductionContext` remains blocked and its adversarial regression is unchanged.

## Four-Support fidelity

R4 uses the existing native bootstrap, not renamed Support files or a reconstructed approximation. The public registration retains the four original Support records and source root, so the same absolute Support paths enter P0's augmentation and prototype-subsampling seed calculations. The native builder creates the fixed bank, Support CLS coverage and cold-start reference; the context checks its actual fixed-bank digest against the immutable registered preflight and rejects nonempty initial Greedy or C1 memory. R4 does not alter L/E, Greedy, C1, stochastic BEA or their parameters.

Synthetic tests check the exact Support paths and seed inputs, acceptance of the matching synthetic bank, and rejection of a wrong fixed bank. No real R4 DINO or Support-bank rebuild was authorized or run; the historical R1 preflight remains historical evidence, and a future real bootstrap must pass the same digest check.

## Verification

| Gate | Result | Evidence and limit |
| --- | --- | --- |
| Blinded registration and image-only access | **PASS synthetic** | Opaque IDs, group/view/order binding, metadata removal, pixel equality, tampered/swapped resources, repeated IDs and changed registrations were exercised. |
| Query-only broker | **PASS synthetic** | A genuine durable pending query was independently replayed; fabricated receipts, wrong group, future/unqueried requests, duplicate disclosure and a mapping-dump request were rejected. |
| Spawn and recovery | **PASS synthetic** | The spawned worker scored and replayed its registered stream, without a second broker release; Stage A artifacts were checked for original TEST names. |
| Support and bounded model hashing | **PASS synthetic** | Original Support paths and seed inputs were preserved; wrong fixed bank was rejected; full-model hash count did not increase during events. Real R4 Support reconstruction was **SKIP**. |
| Stage B | **PASS synthetic** | Before-seal evaluation and projection failed; post-seal independent CPU evaluation succeeded; altered mapping and projection were detected. Real evaluation was **SKIP**. |
| Research CPU tests | **PASS** | 20 passed: 14 existing research tests and six R4 tests. Four dependency deprecation warnings. |
| Synthetic defect-path supplement | **PASS** | One additional synthetic test used five `NG` views with defect-like path and image metadata, verified opaque public rows and exact metadata-free pixels, and checked the private mapping retained the correct synthetic labels. Four dependency deprecation warnings. |
| Synthetic CUDA | **PASS** | Six passed, including the four original CUDA tests. Four dependency deprecation warnings. |
| Prior CPU suite | **FAIL overall** | 80 passed and the preserved high-security adversarial encoder-substitution regression failed. The high-security production context remains blocked and unsupported. Four dependency deprecation warnings. |
| Compilation and repository audit | **PASS** | Compilation succeeded; 27 protected historical files matched the starting revision; 7,030 unrelated untracked paths retained inventory SHA-256 `5d036161ab992ad68c8f98e7b03e91199fc7fa7b4d170603936c57064488dd18`; R4 runtime identifier scan passed. |
| Real-data staging and scientific execution | **SKIP** | Not authorized and not performed. |

Commands, using `D:\anaconda3\envs\torch\python.exe`:

```text
python -m pytest -q tests/calibration/test_fs4_research_execution.py tests/calibration/test_fs4_r4.py
python -m pytest -q -s tests/calibration/test_fs4_hardening_cuda.py tests/calibration/test_fs4_hardening_r2_cuda.py
python -m pytest -q tests/calibration/test_fs4_threshold.py tests/calibration/test_fs4_hardening_r1.py tests/calibration/test_fs4_hardening_r2.py
python -m compileall -q experiments/calibration/fs4_r2_context.py experiments/calibration/fs4_research_ledger.py experiments/calibration/fs4_r4_preparation.py experiments/calibration/fs4_r4_blinded.py experiments/calibration/fs4_r4_broker.py experiments/calibration/fs4_r4_worker.py experiments/calibration/fs4_r4_mapping_verify.py experiments/calibration/fs4_r4_evaluator.py tests/calibration/test_fs4_r4.py
python -m pytest -q C:\Users\rico\AppData\Local\Temp\fs4-r4-synthetic-ng-test.py
```

The supplemental synthetic test script is a local temporary validation artifact, not a committed source file. Its SHA-256 is `4f25c7e0b446d5d09e52ca63cd5f14d6738d41c47b2a7ef35ceb6e029d3cd0b4`; it imports the committed R4 fixture and creates all input images under pytest's temporary directory.

The local source commit is marked `WIP / STAGE0_BLOCKED` because the historical high-security critical regression remains unresolved. That label does not claim an adversarial production pass or authorize execution; the R4 research classification below is for independent review under the stated local trust model.

## Future authorization gates

Independent review is required before any real-data staging. A separate explicit authorization must then cover creation and verification of the private mapping and blinded resources from the intended Real-IAD registration, protection and retention of those artifacts, and the TRAIN-only four-Support bootstrap/parity check. A later scientific campaign needs its own frozen manifest, source/model/Support and stream identities, query-only feedback arrangement, and explicit execution authorization. Real Stage B outcome access and evaluation require the registered global Stage A seal and separate authorization. Nothing in this engineering task authorizes any of those actions.

Classification: `FS4_LABEL_BLIND_RESEARCH_READY_FOR_INDEPENDENT_REVIEW`.
