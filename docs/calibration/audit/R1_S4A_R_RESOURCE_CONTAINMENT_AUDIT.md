# R1-S4A-R Hard Runtime and Storage Containment Audit

**Status: `CPU_RESOURCE_GUARD_BLOCKED`**

**R1-S4B: BLOCKED pending a hard filesystem quota and the remaining native parity checks.**
This is a CPU engineering audit, not experimental evidence.

## Scope and baseline

The task began on branch `calibration-stage0-contract-20261029` at expected parent `451e736b18ef11c2b9621fb592cb245c03ccb2c0`. The scoped implementation and test commit is `bce241d49fef1add265f60fe34132c60d0c9b18a`; this report is committed separately.

No R1 pilot, DINOv2 model, GPU, image decoder, Real-IAD image, label, mask, or scientific metric was accessed. The existing failed R1 output remains untouched. The 82 untracked paths present at the start were preserved and not staged. Protected files were unchanged:

| Protected file | SHA-256 |
|---|---|
| `test_phase3_AIF_simplified_v22.py` | `bfc871d5a9a32b2f5bdb9d7aab5177e702dec2e1110a6632f20bd13c20fcba30` |
| `vmb_visionad_new.py` | `bd6545bf5ff63acce01792e5d42aa964c88e73fb59670febeb55f2d1139a409a` |

## Exact source changes

### `experiments/calibration/r1_runtime_supervisor.py`

- Adds a Windows Job Object supervisor with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`.
- Creates the child suspended, assigns it to the Job Object, and resumes it only after assignment. This closes the startup window in which a child could create an uncontained process. The implementation does not set Job Object breakaway flags or pass `CREATE_BREAKAWAY_FROM_JOB`. Windows assigns ordinary `CreateProcess` descendants to their parent's job by default; the Job Object can terminate all associated processes. ([Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects), [Process Creation Flags](https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags))
- Starts the fixed 120-minute deadline before child creation, so the deadline covers model initialization and event processing. The real R1 entry point always uses this full deadline.
- On timeout, terminates the Job Object and checks both the root process signal and the Job Object active-process count. If termination cannot be confirmed, closing the job applies the kill-on-close guard, the supervisor records `BLOCKING_TERMINATION_FAILURE` when the output path remains writable under the contract, and raises `SupervisorBlockingFailure`.
- If the root process exits while ordinary child processes remain, the supervisor gives them one second to exit, then terminates the job and reports failure. It never retries or resumes an incomplete run.
- Keeps the runner's branch, HEAD, source revision, profile, and authorization checks. The real pilot requires both the explicit `--authorize-real-execution` switch and an authorization ID. The child checks a supervisor authorization token and verifies that it is in a Job Object. The supervisor source is now included in the run's source fingerprints.
- Refuses an existing output namespace. Supervisor status is written through the shared budget in the reserved failure bytes.

Windows Job Objects cover ordinary process descendants created with `CreateProcess`. Microsoft documents an exception for processes started through WMI `Win32_Process.Create`; the R1 pilot does not use that API. Brokered or service-launched processes remain outside this supervisor's verified guarantee.

### `experiments/calibration/r1_matched_correction_pilot.py`

- Adds the centralized `OutputBudget` writer with a 64 KiB reserve for termination/failure metadata. It checks projected file bytes before each write, then audits the resulting directory.
- Routes event NPZ shards, public score and transition JSONL, private queried-feedback JSONL, correction-anchor JSONL, attempt history, the run manifest, and supervisor status through the accountant. Manifest temporary files are allowlisted and accounted while being written.
- Rejects writes outside the unique R1 namespace, non-allowlisted files, symlinks/reparse points, hard-linked files, reused output paths, unaccounted subdirectories, and concurrent writers. A process-local lock and a named Windows mutex serialize supported writers.
- Records the byte limit, reserved bytes, enforcement method, and the requirement for an OS filesystem quota in the run manifest. Failure-manifest replacement and supervisor status can use the reserved bytes.
- Adds `_require_supervised_authorization` at the real pilot entry point. Direct invocation fails before model setup.

The accountant measures logical file lengths and serializes cooperating application writers. It cannot prevent an unrelated same-user process from writing to the directory, and its post-write audit cannot undo bytes already written by such a process. It also does not measure filesystem allocation metadata. Thus this accountant is auditable and fail-closed for the runner's known writers, but it is **not a hard storage bound**.

### `tests/test_r1_s4a_resource_containment.py`

- Adds synthetic Windows Job Object child tests for normal exit, a sleeping/hung process timeout, descendant termination with partial output retained, unexpected nonzero exit, and termination that cannot be confirmed.
- Adds synthetic budget tests for pre-write rejection, the 64 KiB reserve behavior using small test limits, NPZ shard refusal before file creation, all four JSONL classes, invalid/reused paths, unaccounted files, and thread/process writer contention.
- Exercises the actual `experiments.v02_memory.MemoryArm("greedy_coreset", ...)` adapter against a small synthetic bank. Two repeated event positions with identical C0 score/threshold/prediction/candidates and differing injected C1/C2 values produce the same memory digests and transitions. Queried defects and unqueried events do not call the normal writer.

No Correction algorithm, C2 amplitude, query budget, dataset protocol, original scoring mathematics, protected source, or historical output was changed.

## Runtime termination tests

The CPU tests ran on Windows with `CUDA_VISIBLE_DEVICES=''`. All synthetic children used the Job Object supervisor; no real pilot command was invoked.

| Case | Result |
|---|---|
| Child confirms that it runs in the Job Object and exits normally | PASS |
| Sleeping/hung child reaches deadline and is terminated | PASS |
| Child plus spawned descendant are terminated; partial manifest remains | PASS |
| Child exits nonzero; failure status is recorded; no restart occurs | PASS |
| Injected termination API failure yields blocking status and exception | PASS |
| Explicit authorization is required before the R1 launch path | PASS |

## Storage-limit tests

Synthetic byte budgets verify that the projected write is rejected before I/O, the final test directory remains under its synthetic ceiling, and failure/status metadata can use the reserved region. The test refuses an oversized event NPZ before creating its shard. Public scores, transitions, private feedback, and correction anchors are each passed through the common JSONL accountant. Concurrent supported writers, unknown files, invalid names, reused namespaces, and the wrong output namespace fail closed.

**Hard quota result: NOT CONFIGURED / NOT VERIFIED.** On Windows Server 2016–2025, File Server Resource Manager can apply a hard quota to a folder, and an auto-apply quota can cover existing and future child folders. The next readiness step is a dedicated R1 output parent with an auto-apply **hard** 2 GiB quota for each run directory; verify that the quota is enforcing (not soft) and test refusal on a disposable scratch directory before using it. ([FSRM quota management](https://learn.microsoft.com/en-us/windows-server/storage/fsrm/quota-management), [Create a Quota](https://learn.microsoft.com/en-us/windows-server/storage/fsrm/create-quota), [New-FsrmQuota hard-limit behavior](https://learn.microsoft.com/fil-ph/powershell/module/fileserverresourcemanager/new-fsrmquota?view=windowsserver2019-ps))

Until an enforcing per-run quota is installed and verified, the hard storage requirement is unresolved and GPU readiness remains **BLOCKED**.

## Common Greedy Writer contract

The two CPU contract tests passed using the production `common_c0_greedy_writer_action` and actual `v02_memory.MemoryArm.update` implementation with a synthetic codebook/bank and synthetic features. The same C0 inputs and candidate arrays resulted in identical common-memory states when C1/C2 values differed. Repeated event updates were deterministic. Unqueried events and queried defects performed no normal-memory write. Existing R1-S4A tests also continue to cover the causal ordering and source-bound identity checks.

This does not establish GPU or full native-score parity. S4B would still need to verify the native scorer's C0 Top-32 selection and the real support bank's dynamic MemoryArm update against the intended native trajectory using authorized GPU inputs. No such check was run here.

## Regression, source SHA, and test provenance

Command run from the repository root:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
@'
import py_compile, sys, runpy
for path in (
    r'experiments/calibration/r1_matched_correction_pilot.py',
    r'experiments/calibration/r1_runtime_supervisor.py',
    r'tests/test_r1_s4a_resource_containment.py',
):
    py_compile.compile(path, doraise=True)
sys.path.append(r'D:\anaconda3\Lib\site-packages')
sys.argv = ['pytest',
            'tests/test_r1_s4_cpu_protocol.py',
            'tests/calibration/test_r1_matched_correction_pilot.py',
            'tests/calibration/test_stage1a_p12_label_provider.py',
            'tests/test_r1_s4a_resource_containment.py',
            '-q', '-p', 'no:cacheprovider']
runpy.run_module('pytest', run_name='__main__')
'@ | & 'D:\anaconda3\envs\torch\python.exe' -
```

| Check | Result |
|---|---|
| Python compilation of pilot, supervisor, and new tests | PASS |
| New runtime/storage/writer tests | PASS, 13 tests |
| Existing R1-S4A causal protocol tests | PASS, included below |
| Existing R1 synthetic pilot and queried-label provider tests | PASS, included below |
| Combined focused CPU regression | **58 passed, 0 failed** |
| Warnings | 4 existing torchvision/Pillow interpolation deprecations |

SHA-256 fingerprints at the implementation commit:

| File | SHA-256 |
|---|---|
| `experiments/calibration/r1_matched_correction_pilot.py` | `78d310d596afdacfe33bfa521b69258397bf568b736c3f81bb428336538671fa` |
| `experiments/calibration/r1_runtime_supervisor.py` | `a819e08bbb0c8c225842a918ceff7a150935a11012bb1ed050a86cc9122e605e` |
| `tests/test_r1_s4a_resource_containment.py` | `75e3a0360618ac8df7088226ff1063d250ceb58a171196bfb5c5c11acc049569` |

The pilot's parent-file SHA-256 at the expected starting HEAD was `ddfbab56c75c008c7b416b4c655418150add9ea8645fa85bb23a7223b216ff3c`. Branch and starting HEAD matched the task expectation. The focused tests confirmed that incorrect revision/profile identity is rejected. Protected-file fingerprints above match the unchanged repository files.

## Remaining risks and disposition

- The application accountant is not an operating-system quota; a non-cooperating writer can bypass it. **This is the controlling blocker.**
- Job Objects cover ordinary process descendants; brokered WMI/service launches are outside the checked tree. The existing pilot does not use those launch paths.
- Real R1 model initialization, GPU hangs, native scorer behavior, actual feature sizes, disk allocation, and support-bank write parity were not exercised.
- The reserved failure area is bounded at 64 KiB. Current failure records are small and exception text is bounded by the runner; an unexpectedly oversized status record fails closed rather than exceeding the configured limit.

**`CPU_RESOURCE_GUARD_BLOCKED`** — runtime containment and synthetic storage accounting tests pass; the hard storage quota and full native GPU parity remain unresolved. This is not an R1 engineering pass, does not authorize R1-S4B, and includes no scientific evaluation.

Implementation commit: `bce241d49fef1add265f60fe34132c60d0c9b18a` (parent `451e736b18ef11c2b9621fb592cb245c03ccb2c0`). No push was performed.
