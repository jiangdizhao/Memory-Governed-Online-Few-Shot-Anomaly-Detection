# Stage 1-A P1.2 low-memory recovery — final reconciliation

**Decision: `P12_EXACT_PARITY_AND_EVIDENCE_PASS`**
**Acceptance: `RESEARCHER_REVIEW_REQUIRED`**
**Execution branch/HEAD:** `calibration-stage0-contract-20261029` / `9a60222b70f8e9353b2937c5c128d41d967b72a2`
**Tape binding:** `dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183`

The class-sharded replay completed all 20 frozen class/seed trajectories with exact event-level memory, writer/source-map, C64 q90 and transport parity. The 16 previously persisted evidence files passed independent hash/C64/LSO verification and replay-time state binding. Four rolled-strip evidence files were recomputed through Native all-layer LSO, then independently verified. The query tape contained 459 unique image positions and 2,295 query events across seeds. Only queried positions were read; no Mint/PCB, masks, or unqueried labels were accessed.

The first final reconciliation attempt stopped on a Windows path separator mismatch in its prior-evidence linkage comparison. This append-only finalizer normalized manifest path separators and verified all 20 existing receipts and evidence hashes. It did not replay events, encode images, access labels, or run Native inference. The original blocked interim report and gate remain preserved alongside this final reconciliation record.

Compute counts, repeated attempt reads/encodings, cache reuse, support-feature encoding, memory samples, per-trajectory digests, and evidence paths appear in the accompanying JSON files. The original Stage 0.75-A G2 remains BLOCKED. No Stage 1-A performance comparison, Stage 1-B/2 run, confirmation analysis, or masks were accessed. Researcher review remains required.
