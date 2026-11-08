# FS4 joint calibration campaign — live execution status and B/C prioritization (2026-11-08)

> **Point-in-time research operations record, not a final experiment report.** The figures below come from the latest researcher-supplied Codex status. The FS4 worker continues independently, so counts may have advanced. This document does not authorize any source, specification, or experiment change.

## Identity and scope

- Repository branch: `calibration-stage0-contract-20261029`.
- **Pushed and active FS4 source revision:** `c184e7130c943746a9911e92e987c4749b67318f`.
- Frozen scientific specification SHA-256: `1c96077bfdaa58f9ff93a67a9349a4cc6bc8b63578ae8915ff23762b50aee896`.
- Campaign ID: `fs4joint_20261107_s0q37`.
- Windows experiment base (not for Git tracking): `E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection\FS4_Joint_Experiment\fs4joint_20261107_s0q37`.
- Current active scientific campaign root: `FS4_Joint_Experiment/fs4joint_20261107_s0q37/campaign/attempt_02`.
- Population: 15,295 Real-IAD TEST camera views in 3,059 five-view Source Groups across audiojack (3,815), regulator (3,855), switch (3,820), and rolled_strip_base (3,805); all are **previously exposed development categories**, not untouched confirmation data.
- FS4 scientific scope: shared trajectory S comparing historical L and current-state source-excluded E thresholds, plus independent L-coupled CL and E-coupled CE trajectories. The S/CL/CE trajectory names are **not** shorthand for research-program Experiments A/B/C.

## Preparation, data integrity, and execution history

1. **Real-data STAGING: PASS** at the active source revision. A sealed registration contains 15,295 events. All private per-image labels were reconciled against the official category JSON `anomaly_class` relative to `meta.normal_class`.
2. The filename/session-versus-image-label reconciliation found **3,455 NG-session camera views with JSON-normal image labels**, and **zero OK-session / JSON-defect reverse mismatches**. Image labels must not be propagated from a defective capture session to its five views. The narrow R4 preparation/broker consistency error was corrected before execution; official JSON image-level truth was retained.
3. **Native four-Support parity: PASS** for all four categories, using original registered TRAIN-normal supports. Greedy and C1 start empty; no C64 calibration samples are added.
4. **Resource preflight: PASS under an explicit documented operational exception.** The strict report recorded an RTX 3090 Ti CUDA total-capacity reading 13,107,200 bytes (12.5 MiB) below nominal 24 GiB; the derived receipt accepted **only** that discrepancy. Host RAM, free disk, blinded resource completeness and synthetic native benchmark conditions passed. The original measurements were preserved. No frozen scientific method was changed to obtain this operational acceptance.
5. Earlier staging/worker attempts and their failure evidence remain preserved under separate non-overwriting attempt paths. An early worker/ledger model-identity binding error required the narrowly scoped source fix committed as `c184e713`. The current staging and scientific campaign use that revision.
6. During Stage A, multiple worker shutdown exceptions required verified checkpoint/ledger recovery. The recovered event boundaries were checked to avoid duplicate label disclosure; previous failure records were retained. These interruptions increased wall-clock cost. Their presence does **not** mean Stage A is complete.

## Last reported live progress (not final results)

| Gate or trajectory | Last researcher-supplied status |
| --- | --- |
| STAGING | PASS, 15,295 / 15,295 images |
| Four-Support parity | PASS, 4 / 4 categories |
| Resource preflight | PASS with documented nominal CUDA-capacity exception |
| Stage A / S | **5,818 / 15,295 completed image events (~38.0%)**, still running |
| S / audiojack | 3,815 / 3,815; category seal reported |
| S / regulator | At least 2,003 / 3,855 completed after audiojack |
| Stage A / CL | 0 / 15,295; not started |
| Stage A / CE | 0 / 15,295; not started |
| Stage A / combined event count | **5,818 / 45,885 (~12.7%)**; this is an event-count fraction, not a runtime forecast |
| Stage B | Not started; **no** completed H1/H2 BER estimates |
| Last verified checkpoint mentioned by Codex | `regulator-event-05000`; committed state and tensor hashes matched |
| Checkpoint cleanup | **Not performed**; recovery checkpoints are retained |

**Progress counting:** count `EVENT_COMPLETE` markers, not raw `events.jsonl` lines, because the append-only journal contains multiple record kinds per image. No Stage A/Stage B performance claim may be inferred from partial progress.

## Pause feasibility and explicit operational decision

**Codex assessment:** `A_PAUSE_NOT_SAFE_UNDER_CURRENT_PROTOCOL`.

**Decision:** **DO NOT PAUSE A. KEEP A RUNNING.** No pause, suspension, termination, or discretionary restart was performed in the reported pause-feasibility check. The experiment remains in progress.

Reasons established from the active executor:

- No supported intentional pause-at-event or pause-at-category operation exists.
- `--resume` is a conditional **recovery** mechanism, not a guaranteed planned-pause mechanism. Recovery requires permitted incomplete/unclean-process evidence, valid ledger/checkpoint state, and no unresolved `LABEL_INTENT`.
- The frozen timer counts wall-clock gaps between processes. Time spent paused therefore consumes the registered **7-day per-trajectory** and **21-day campaign** limits.
- At **2026-11-08 00:20 UTC**, the supplied timing-ledger reading showed approximately **6 days, 11 hours, 54 minutes, 50 seconds** remaining for both active S and the campaign. This is a dated snapshot, not the present remaining allowance.
- A `RECOVERY_COMPLETION` receipt is created only after fully completed Stage A; no separate safe-pause receipt exists.

Do not deliberately kill the worker or modify timing, recovery semantics, frozen scientific source, query policy or checkpoints merely to free the GPU. Retain all currently valid A registrations, private mapping, sealed journal entries and recovery evidence. Do not perform the checkpoint cleanup procedure before verified Stage B completion and explicit researcher confirmation.

## Research priority: B and C ahead of expanding A

The **research program's** key upcoming integrated experiments are:

- **Experiment B — Signed Correction mechanism attribution:** existing R9 and R10 development runs already provide sealed full-stream, retrospective evidence, including Full C1, beta-only and Signed Exemplar/knn comparisons. See [R9](../audit/r9_correction_mechanism/REPORT.md) and [R10](../audit/r10_signed_knn_baseline/REPORT.md). Repeating these identical development conditions is **not** a substitute for the still-needed integrated scientific contrast or independent confirmation.
- **Experiment C — BEA policy attribution:** historical E6/D2 B1 closed-loop artifacts exist for the same four development categories, including reported Stage A seals and Stage B files. This inventory was reported by Codex; this record does **not** newly validate or re-run private feedback, metrics or seals. Any genuinely new C conditions need a distinct registered identity and a frozen protocol, not a silent replay under the old D2 run identity.

**Priority decision:** use existing R9/R10/D2 evidence to prepare B and C without rerunning known historical experiments; retain the currently running A, and do not enlarge A into a new calibration-scale sweep. B/C preparation can proceed without GPU contention, but new GPU execution requires an explicitly agreed, scientific-condition-specific protocol. A and its Stage B cannot be called complete until all S/CL/CE trajectories and the final verification finish.

## Next status update requirements

When new durable evidence is available, update only the corresponding **observed** counts, category/global seals, execution failures/recoveries, resource consumption, and final verified outcomes. Keep timestamps with snapshots. Do not invent BER/AUROC/AP, query counts, Stage B outcomes, or GPU-time estimates. Store large outputs, feedback, dataset data and checkpoints **outside Git tracking**, under the existing experiment base.

This record documents a **rejected pause proposal and continued execution**, not a paused experiment or an independently confirmed research result.
