# Stage 1-A P1.2 bounded Native replay - partial reconstruction

**Decision: `P12_PARTIAL_RECONSTRUCTION`**
**Branch:** `calibration-stage0-contract-20261029`
**HEAD:** `7ffcb9c30d7822d420e0d9e9d16a88a46247af12`
**Q-A-TS-v1 binding:** `dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183`

## Outcome

The audiojack/seed-0 bounded smoke and resume continuation passed exact parity. Fifteen trajectories have durable completed status through `switch/seed4/Q-A-TS-v1-3pct-seed37`. The runner then reported `MemoryError`. Its last emitted trajectory result was rolled-strip-base/seed-0, 3,805/3,805 events, `PASS_EXACT_PARITY`; that trajectory did not receive a durable checkpoint or progress update. Four rolled-strip-base support seeds were not started.

The CLI recorded no allocation traceback or error message. The runner's pre-existing blocked-audit writer declined to write because the active output and audit directories already existed. This compact audit records the last durable manifest and the separately observed CLI result without treating it as a durable completion. The run's `INCOMPLETE` marker is preserved.

## Gate interpretation

The last durable compute counters are 345 queried-label reads, 345 feature encodings, 1725 static-map calls, and 60631 anchor-score calls. These counters precede the uncheckpointed trajectory and are **not final totals**. Final causal evidence reconciliation and compute-ceiling verification are therefore incomplete. Private image-derived traces, evidence, and the 7359652922 byte checkpoint remain local and are not committed.

Historical C64 artifact linkage passed, but independent C64 rescoring was not performed. The scoped current-source reconstruction was accepted for this execution; original historical source-byte parity was not recovered. Original G2 remains blocked. No Stage 1-A performance comparison, Stage 2, Mint/PCB confirmation outcome, or mask access occurred.
