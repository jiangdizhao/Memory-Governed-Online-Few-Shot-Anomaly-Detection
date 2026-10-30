# Stage 1-A P1.2 - Bounded Real Native Replay and Dense Anchor Evidence

**Decision: `P12_BLOCKED`**

**Branch:** `calibration-stage0-contract-20261029`

**HEAD/base:** `a674b7633d1049af3cc34156314a812a0319b864`

**Verified Q-A-TS aggregate binding:** `dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183`

## Decision and scope

The P1.2 run stopped at the researcher authorization gate. Neither required approved record was found: `STAGE1A_P1_BOUNDED_NATIVE_REPLAY_GO` or `STAGE1A_P1_QUERIED_FEEDBACK_ACCESS_GO`. Both exact templates are prepared in this audit directory and remain `PENDING_RESEARCHER_DECISION`. No real image, queried label, Native Scorer, model weight, or GPU was accessed or initialized. No P1.2 trajectory, anchor measurement, query-response digest, or private checkpoint was created.

The stored metadata-only Q-A-TS-v1 tapes were verified against the requested aggregate identity. Their query counts are 114 audiojack, 116 regulator, 115 switch, and 114 rolled_strip_base (459 unique image positions). Verification used only frozen metadata and stored tape artifacts.

## Separate approvals required

1. **Bounded Native/data authorization** ? Real-IAD development queried-image access only; frozen Native scoring and private checkpoint/evidence generation; maxima of 459 unique feature encodings, 2,295 static-map score calls, and 75,406 query-anchor Native score calls.
2. **Queried-feedback access authorization** ? only the 459 fixed Q-A-TS positions across the four verified tapes; maximum 459 unique queried-label reads. No unqueried labels, masks, Mint/PCB, or confirmation outcomes.

Both records must bind every per-class tape identity and the aggregate digest `dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183`. The templates leave `decision` pending and `approved_by`, `approved_at_utc`, and `approval_reference` blank. Do not change the class set, query tape, seed, source identities, or quotas while completing them. Independent C64 numerical rescoring has no authorization here; it requires a separate decision and compute quota if needed. These P1.2 records do not grant Stage 1-A performance-comparison permission.

## Gates

- Metadata tape/source preflight: PASS.
- Exact Native memory parity: NOT RUN; blocked by both missing approvals. Phase A (audiojack, seed 0), phases B/C, and resume phase D have not started.
- Dense anchor evidence: NOT RUN; independent and separately mandatory. Memory parity has not been established.
- Checkpoint/resume: NOT RUN; no real private state exists.
- Native scorer identity: recorded from approved P0/P1 manifests and current code fingerprints, but no scorer/model was initialized.
- Historical G2: remains BLOCKED; independent package-integrity certificate remains PASS with `RESEARCHER_REVIEW_REQUIRED` and its existing scope limitations.

No BA, AUROC, FPR, FNR, controller comparison, or Stage 1-A performance evaluation was run. No historical artifact or protocol was modified.

## Preflight search note

An initial overly broad repository text search for approval language returned a truncated portion of unrelated, pre-existing audit output. No evaluation was run, and that output was not used in this P1.2 decision. This incident is disclosed for provenance; the approval blocker is independently established by the P1/P1.1 audit records and the absence of either required approved record.

## Request to researcher

Please provide both completed, separately approved JSON records in the prepared templates, including the researcher decision reference, approver identity, and UTC approval time. Once both records are present and validate against this exact tape binding, P1.2 can resume with Phase A only; later phases remain conditional on their exact gates.
