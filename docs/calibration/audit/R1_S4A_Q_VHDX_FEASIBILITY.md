# R1-S4A-Q Windows 11 Pro VHDX Feasibility Audit

**Recommendation: `BLOCKED` for R1-S4B readiness.**
Windows 11 has native VHDX attach support without enabling Hyper-V, and the host volume has ample free space. The current R1 runner is not wired to an external volume, the current token cannot create or mount a VHDX, and an exact 2 GiB VHDX cannot offer 2 GiB of file capacity after filesystem overhead. No VHDX was created, attached, formatted, detached, or deleted.

## Scope and provenance

The read-only audit began on branch `calibration-stage0-contract-20261029` at expected HEAD `2a067f018971f9f4171f67ce8d41f3a5033af192`. The tracked worktree had no modifications. Existing untracked research outputs were left untouched.

Reviewed the current R1-S4A-R report, `experiments/calibration/r1_runtime_supervisor.py`, and `experiments/calibration/r1_matched_correction_pilot.py` including `OutputBudget`, all of its write wrappers, output path construction, and the private-feedback, manifest, event-shard, transition, correction-anchor, and supervisor-status call sites. The required calibration contracts were also reviewed. This task changes no code or experiment output; only this audit document is to be committed.

## Read-only Windows checks

| Check | Observed |
|---|---|
| OS identity | `Win32_OperatingSystem` reported Microsoft Windows 11 Pro, version `10.0.26300`, build `26300`, 64-bit. The CurrentVersion registry reported `DisplayVersion=26H2`; its legacy `ProductName` still says Windows 10 Pro. `Get-ComputerInfo` likewise returned legacy Windows 10 Pro / 2009 labels. |
| VHD storage commands | `Mount-DiskImage`, `Get-DiskImage`, `Get-Disk`, `Get-Partition`, and `Get-Volume` are available from the Windows `Storage` module. `New-VHD`, `Mount-VHD`, `Get-VHD`, and the Hyper-V module are absent from this session. |
| Hyper-V feature query | Read-only DISM query returned error 740 because the current shell is not elevated. Feature state was not verified. Hyper-V is not required for the documented Disk Management VHD workflow. |
| Current token | Windows principal check: not in Administrators and not in Backup Operators; token is not elevated. No elevation was requested. |
| Repository volume | `E:` is NTFS, 515,762,024,448 bytes total with 299,801,182,208 bytes available (about 279.2 GiB). This is ample for a fixed VHDX of roughly 2 GiB plus measured overhead at the time checked. |
| VHDX operation performed | None. Checks were metadata queries only; no image or volume was created or mounted. |

Microsoft lists Windows 11 support for creating, attaching, and detaching VHD/VHDX through Disk Management, and recommends VHDX. Its documented procedure requires Backup Operators or Administrators. `Mount-DiskImage` also requires administrator privileges. The current token has neither role, so those operations were unavailable and intentionally not attempted. The Hyper-V PowerShell module is not present in this session, but is unnecessary for the built-in Disk Management path. ([Manage VHDs](https://learn.microsoft.com/en-us/windows-server/storage/disk-management/manage-virtual-hard-disks), [Mount-DiskImage](https://learn.microsoft.com/en-us/powershell/module/storage/mount-diskimage), [New-VHD](https://learn.microsoft.com/en-us/powershell/module/hyper-v/new-vhd))

## Capacity and compatibility

A fixed VHDX reserves host storage at creation and remains approximately the size of its virtual capacity. Its attached filesystem has less usable space than the virtual disk size because partition and filesystem structures consume capacity. ([VHDX format overview](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-vhdx/83f6b700-6216-40f0-aa99-9fcb421206e2))

The current `MAX_OUTPUT_BYTES` is `2 * 1024**3` bytes **including** the 64 KiB `OutputBudget` failure reserve. Normal writes stop at `MAX_OUTPUT_BYTES - 64 KiB`; failure/status metadata may use the reserve. Therefore:

- An exactly 2 GiB virtual disk does **not** guarantee 2 GiB of file-byte capacity. The filesystem may fill before the application limit, and the runner could then fail while writing its final manifest or supervisor status.
- To preserve a 2 GiB application file-byte ceiling and its failure reserve, the VHDX virtual capacity must exceed 2 GiB. After formatting, verify `SizeRemaining >= 2 GiB + M`, where `M` is a registered safety margin. The exact VHDX size must be chosen from the measured formatted volume, not from an assumed fixed NTFS overhead. A provisional 64 MiB margin can be evaluated on a disposable volume but is not yet validated.
- If the VHDX capacity itself must be exactly 2 GiB, compute an effective application cap as `min(2 GiB, formatted SizeRemaining - M)` and keep the 64 KiB failure reserve inside that cap. This is a smaller output allowance and may stop R1 early.

These constraints make “exactly 2 GiB total VHDX capacity” and “retain 2 GiB of application file capacity” incompatible. The requested size semantics must be settled before S4B. A fixed VHDX does provide a hard capacity boundary for writes **to that virtual volume**, but it is not by itself a process filesystem sandbox.

## Current output-path and escape audit

The current code does not support an external mounted output volume:

- `_output_namespace()` hard-codes `ROOT / "outputs/calibration"`; `run_pilot()` does not accept an output path.
- `supervise_r1_pilot()` independently constructs that same repository output path and passes it to the child. The child CLI has no `--output-dir` argument.
- `OutputBudget` only accepts a direct child of its `allowed_parent` with the R1 namespace prefix. It checks for reparse points at the output root and within the run tree, but does not bind the path to an expected volume GUID or VHDX image. Parent mount configuration and the actual volume backing are not validated.
- `_budget_for_path()` returns `None` for a writer path outside an active budget. `_append_jsonl`, `_write_once`, and `_atomic_json` then use their unbudgeted fallback path. Thus an accidental or newly introduced out-of-root writer is not currently guaranteed to fail closed.
- The known run artifacts are routed under the current `output` path: event NPZ shards, score/transition JSONL, private feedback JSONL, correction-anchor JSONL, run manifest, failure manifest, and supervisor status. Their existing path consistency is useful, but path consistency is not a volume boundary. `OutputBudget` alone cannot prevent other code or a library from writing elsewhere.
- The supervisor's Job Object controls process lifetime, not filesystem access. The child retains the current user's access to the repository and other volumes. Python bytecode, temporary files, Git's optional index writes, and model/library caches must be redirected, made read-only, disabled, or otherwise contained. A drive letter or volume-mounted output path alone cannot stop the child from writing to `E:` or another accessible location.

### Minimum future implementation requirements

1. Add an explicit `output_parent`/`output_dir` input to `_output_namespace`, `run_pilot` and its CLI, `supervise_r1_pilot`, and the supervisor CLI. Keep `ROOT` for source imports, data references, and Git identity. Keep the existing profile IDs, scoring, query policy, and algorithm settings unchanged. The output parent must be the root of the separately mounted volume; create exactly one new `r1_digest_contract_v1_{mode}_{run_id}` directory beneath it.
2. Preserve exact branch, HEAD/source-commit, profile, authorization-ID, and Job Object checks. A runner source change requires a newly authorized exact commit. Record the output-binding contract version, run ID, expected VHDX path, VHDX/disk identity, volume GUID, filesystem, formatted capacity, free bytes, and effective byte limit in a runtime-storage manifest section separate from the algorithm profile digest.
3. Before child creation and again before model initialization, verify the expected VHDX is attached and maps to the exact output volume. Require NTFS, the expected volume GUID/disk and partition identity, enough usable space for the effective limit plus margin, and a unique new namespace. Fail before creating output if a query is unavailable or values disagree. Do not allow automatic mount, resume, or fallback to the repository output tree.
4. Validate the output path using handle-resolved final paths and volume IDs, not only lexical `Path` checks. Reject a mount point used as the run directory itself, unexpected volume/mount aliases, reparse points in every ancestor and descendant, nested mount points, and any opened output handle whose volume GUID differs from the authorized volume. Recheck for path substitution at write time to reduce symlink/junction race risk.
5. Remove the unbudgeted writer fallback during a pilot. Make every output writer receive the active `OutputBudget` explicitly or fail if a write path is outside the one registered run root. Keep private feedback and both failure/supervisor manifests on the same volume. Route or disable temporary/cache/bytecode writes and prevent optional Git index writes; if arbitrary library writes must be ruled out, use a restricted child token/ACL or an OS sandbox that grants write access only to the authorized volume. Job Objects alone do not provide that restriction.
6. Keep a global run-ID uniqueness check across the historical repository output namespace and the external volume namespace, in addition to rejecting an existing destination. Bind the chosen run ID and exact volume identity to the explicit authorization. Do not resume a failed run on a remounted or replacement volume.
7. Preserve the 120-minute Job Object deadline and write reserve. Handle `ERROR_DISK_FULL`/`ERROR_HANDLE_DISK_FULL` without overwriting/truncating completed shards; write NPZ shards through a temporary file plus atomic rename or retain and classify an incomplete shard. The volume sizing margin must leave enough actual filesystem space for failure status and cleanup metadata after normal output reaches its budget.

## CPU-only test design

These are design requirements for the follow-up implementation; none were added or run in this audit. Use an injectable volume-identity/free-space provider and small `tmp_path` fixtures. Do not create a real VHDX for unit tests.

| Test | Required assertion |
|---|---|
| Correct volume identity | Matching VHDX, disk/partition, volume GUID, NTFS and run root passes; no DINO/model import occurs during the check. |
| Wrong/detached/replaced volume | Wrong GUID, VHDX path, disk ID, filesystem, drive mapping, or unmounted image rejects before output directory creation and before child launch. |
| Insufficient free space | A mocked `SizeRemaining` below effective app limit plus safety margin blocks before child launch; existing/historical paths remain unchanged. |
| Reparse/mount escape | Symlink, junction, parent reparse point, nested mount, and a write handle resolving to another volume all fail without creating an outside file. |
| No writer fallback | While a run budget is active, attempt every wrapper with an outside-volume path; each fails closed and creates no file there. Include private-feedback and supervisor-status paths. |
| Full-volume write | Inject Windows disk-full errors during NPZ, JSONL, temporary manifest, and final-status writes. Keep prior complete files and the partial run; report `INCOMPLETE`/blocking status where the reserved physical space permits; never resume automatically or exceed the test volume cap. |
| Partial-output retention | Fail after at least one synthetic event shard; verify prior shard and failure metadata remain readable and checksum state marks the run incomplete. |
| Cleanup contract | Mock detach and reattach: detach makes the volume unavailable but preserves the VHDX and its files; reattach reads the same volume identity. Verify cleanup never deletes an output-bearing container implicitly. |
| Regression | Re-run the existing 58-test R1-S4A-R/S4A CPU regression unchanged, then add the tests above. |

Actual full-volume behavior, NTFS allocation, and VHDX attach/detach still require a separately authorized disposable-volume integration check. This audit did not run the prior regression suite; the previous report records 58 passed at implementation commit `bce241d49fef1add265f60fe34132c60d0c9b18a`.

## Reversible setup and removal procedure (future authorized operator only)

1. Confirm the authorized output-volume size semantics and effective budget first. Select a host path outside historical results, such as a dedicated disposable directory on the repository's E: host volume; recheck free space and choose a currently unused drive letter. Current checks showed no `R:` drive, but that must be checked again at setup time.
2. With an authorized administrator/Backup Operators operator, use Windows Disk Management **Create VHD**, choose VHDX and **Fixed size**, and use a capacity that passes the formatted `SizeRemaining` requirement above. Initialize GPT, create one NTFS partition, and assign the chosen drive letter. Do not enable Hyper-V, use a dynamic/differencing disk, or nest another VHDX.
3. Before a run, verify the VHDX path-to-disk-to-partition-to-volume mapping, volume GUID, NTFS type, formatted capacity and free space. Pass that exact identity to the source-bound supervisor. Run only with a newly authorized commit, run ID, and output namespace.
4. On completion or failure, wait for the Job Object to report zero active processes; inspect the manifest/status and retain the VHDX as the durable output. Use Disk Management **Detach VHD** or `Dismount-DiskImage` to detach. Detaching leaves the VHDX and its data intact and can be reversed by attaching it again. ([Manage VHDs](https://learn.microsoft.com/en-us/windows-server/storage/disk-management/manage-virtual-hard-disks))
5. Do not delete the VHDX as routine cleanup. Deleting it deletes the contained output; only remove a truly disposable test container after its contents are explicitly discarded or separately preserved and verified. No setup, attach, format, detach, or deletion step was executed here.

## Security limitations and final disposition

- A VHDX constrains writes that reach its filesystem; it does not deny the child process access to other writable host paths. The current runner has no output-volume pinning and has unbudgeted fallback paths. Strict “cannot escape” is therefore **not currently established**.
- Job Object termination and VHDX capacity are complementary controls, not a filesystem sandbox. Reparse-point rejection also needs handle-based ancestry/volume checks to close path substitution and mount escape cases.
- The VHDX is not encryption. The E: BitLocker/encryption state was not queried; detached image confidentiality and access control remain the host filesystem's responsibility.
- A fixed 2 GiB VHDX is incompatible with retaining 2 GiB of output file capacity after formatting. Either make the VHDX larger based on measured formatted capacity, or reduce the runtime effective output cap for an exact 2 GiB virtual disk.
- Current privileges do not permit the documented VHDX create/attach path. No feature was enabled, no elevation was requested, no dependency was installed, no GPU/model was loaded, and no dataset label was read.

**`BLOCKED`** — Windows 11 Pro supports the VHDX mechanism, but S4B must wait for a separately authorized volume setup, a resolved capacity definition, source-bound volume identity checks, removal of writer fallbacks, and CPU-only tests of escape/full-volume behavior. This report does not authorize administrator operations or GPU execution.
