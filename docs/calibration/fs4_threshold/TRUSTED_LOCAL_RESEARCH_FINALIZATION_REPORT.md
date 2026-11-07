# Trusted local research execution: blocked engineering handoff

Source freeze: `8ca8b574fb307cc2239bd27705e64facfe99862e`. Branch: `calibration-stage0-contract-20261029`. Scope is engineering and synthetic tests only; no Real-IAD scientific experiment, TEST outcome access, or real Stage B evaluation was performed.

## Blocking causal boundary

The inherited execution registration stores each event's `relative_path` in the manifest loaded by the Stage A process. The native `_read_registered_image` calls `parse_image_name`, whose directory and filename checks explicitly extract the `OK` or `NG` condition. Thus a Stage A encoder/policy process can access outcome-bearing path information for unqueried events. This violates the requested information boundary even though the current policy functions are passed opaque event and group IDs. It is not resolved by trusting the local operator or by spawning a process.

The research context and worker therefore fail closed through `require_label_blind_dataset_adapter()`. The worker checks the hold before opening its launch configuration or manifest; the context checks it again before initialization. Research execution is not ready for independent scientific use. The synthetic adapter explicitly overrides this hold only within the spawned test child, using fabricated images, model, preflight and feedback; passing those tests does not validate the missing data firewall.

Completing this work requires an independently verified, label-blind image adapter and registration: the Stage A process must receive only opaque registered identity, image content, geometry and a verifiable Source Group binding, without receiving outcome-bearing paths, directory names, defect codes or a mapping that recovers them. Its causal boundary, event mapping, content and group verification, recovery, and legal query-only feedback must then be tested end to end. The proposed process separation remains an operational research boundary, not a security claim against an operator with arbitrary code execution or a full local administrator.

## Implemented engineering components

The separately named research worker uses Windows-compatible `multiprocessing.spawn` and accepts position-only event commands. It initializes the registered native model, four-Support bank, memory and controller state inside the fresh process. Full model and source checks occur at startup, restoration and final sealing; per-event checks cover identity, geometry, full features and CLS, event order and runtime state. A separate private research ledger binds provenance to durable event, query and transition records. Checkpoint recovery re-encodes and compares registered events.

An independent CPU evaluator launches a read-only Stage A ledger validator before it opens an authorized outcome projection. It checks the global and category seals, registered identities, event/query state and disclosed-feedback consistency, recomputes metrics from sealed scores and thresholds, and writes checksummed result and completion records to a separate private store. These paths have synthetic evidence only. No real evaluation authorization, outcome projection or scientific result was used.

The existing high-security `ProductionContext` remains blocked. Its preserved `test_production_rejects_preconstruction_encoder_substitution` still fails because arbitrary same-process code can substitute the encoder before context construction. The local research mode expressly does not claim protection against that adversary.

## Validation

| Gate | Result | Evidence and limit |
| --- | --- | --- |
| Research Stage A label blindness | **BLOCKED** | Registered event paths disclose OK/NG. Worker and context fail closed before event processing; worker holds before reading configuration. |
| Spawned worker, parent monkeypatch, provenance and model hash schedule | **PASS synthetic** | Synthetic-only adapter bypasses the causal hold. Parent encoder monkeypatch did not propagate; the full-model hash count did not increase for consecutive events. |
| Model, Support, source, dataset, event, feature and query identities | **PASS synthetic** | Registered-identity and actual synthetic model/bank mismatch tests; not a real DINO or Real-IAD validation. |
| Checkpoint and resume | **PASS synthetic** | Worker checkpoint/resume and changed-image replay rejection; scientific recovery remains **BLOCKED** by the label-blindness hold. |
| Stage B causality and metrics | **PASS synthetic** | Incomplete Stage A rejected before labels; a fresh CPU evaluator recomputed synthetic metrics and detected modified results. Scientific evaluation remains unauthorized and **SKIP**. |
| Prior CPU and research tests | **FAIL overall** | 94 passed, including 14 research tests and all 55 original CPU tests; one preserved high-security adversarial regression failed. Four dependency deprecation warnings. |
| CUDA | **PASS synthetic** | Six tests passed, including the four original synthetic CUDA tests. Four dependency deprecation warnings. |
| Compilation, protected source and output inventory | **PASS** | Compilation passed; 27 protected source files match the starting revision; 7,030 unrelated untracked paths retain SHA-256 inventory `5d036161ab992ad68c8f98e7b03e91199fc7fa7b4d170603936c57064488dd18`. |
| Adversarial production | **UNSUPPORTED / FAIL** | Production context stays blocked; the prior same-process substitution regression remains failing and unchanged. |
| Scientific execution and real Stage B | **SKIP** | Not authorized and not performed. |

Commands, using `D:\anaconda3\envs\torch\python.exe`:

```text
python -m pytest -q tests/calibration/test_fs4_threshold.py tests/calibration/test_fs4_hardening_r1.py tests/calibration/test_fs4_hardening_r2.py tests/calibration/test_fs4_research_execution.py
python -m pytest -q -s tests/calibration/test_fs4_hardening_cuda.py tests/calibration/test_fs4_hardening_r2_cuda.py
python -m compileall -q experiments/calibration/fs4_r2_context.py experiments/calibration/fs4_research_context.py experiments/calibration/fs4_research_ledger.py experiments/calibration/fs4_research_worker.py experiments/calibration/fs4_research_evaluator.py experiments/calibration/fs4_research_stage_a_verify.py tests/calibration/test_fs4_research_execution.py
```

The repository audit compared each of the protected files with starting revision `d8866933e567e9e88bf6f6f61fb6626e153f2e09`, hashed the sorted unrelated untracked path inventory, and checked research runtime identifiers. No protected source or pre-existing output was modified. The local branch was not pushed.

Classification: `RESEARCH_EXECUTION_BLOCKED`.
