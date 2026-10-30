# P1.2 scoped current-source reconstruction decision — 2026-10-30

**Scope:** P1.2 runtime source identity only. **Historical G2 remains BLOCKED.**
This decision does **not** modify Stage 0.5, Stage 0.75-A, the approved Stage 1 design, frozen manifests, query tape, or calibration controller code.

## Historical finding

The frozen Stage 0.5 score manifest and Stage 0.75-A protocol agree on the SHA-256 of all 11 recorded Native source files. The current checkout matches 9/11. Only these two exact pairs differ:

| File | Historical frozen SHA-256 | Current SHA-256 |
|---|---|---|
| `experiments/calibration/stage05.py` | `af1ef435addfe2f08cb2e40a1d141b617fb15c852ed469c8aa7062663ccf8626` | `4b8aae252f5f1e096b127906794064da269aa79f7cd5514f6f738cda585c017e` |
| `experiments/calibration/stage05_runner.py` | `1e40d9c5e48c252eef3d96686d2a315c4a87f1f325130a6c2961f7cc41cafaea` | `1d110e1ca813a064767ebeab597b609b3d863f8afd4c8085b06948ad88347e14` |

Both files were introduced in Git commit `fe692a855f283c37fc6919d3e739058eb3248757` and match that commit's current text. The expected frozen bytes are not present in this branch's file history. This establishes a missing historical source snapshot, **not** that the old and current implementation are functionally equivalent.

## Execution rule

P1.2 may proceed using the explicitly fingerprinted *current* sources, with a **scoped source-provenance exception** only if:

1. Both historical manifests agree on the exact frozen SHA-256 in the table.
2. Current source bytes match the exact current SHA-256 in the table.
3. Every other recorded Native source matches both historical manifests and the current filesystem.
4. Frozen model checkpoint, DINOv2 source tree, Native Scorer configuration, fixed support/C64 identities, query tape identities and historical artifact checksums retain their independent strict checks.
5. Existing per-event M0 scalar, pre/post memory hashes and versions, writer schema/source attribution, stored C64 transport, all-layer queried-normal LSO, and causal evidence gates remain mandatory and fail closed.

No historical source manifest is rewritten. The original three-way source binding retains `BLOCKED_SOURCE_LINKAGE_MISMATCH`; the **separate new-execution assessment** is `PASS_SCOPED_CURRENT_SOURCE_RECONSTRUCTION` only under these exact conditions. Never report this as `PASS_EXACT_HISTORICAL_SOURCE` or closure of G2. Any other or changed mismatch blocks.

## Interpretation

Numerical parity on real data, if later achieved, can support new-execution reproducibility and equivalence of the *measured* historical event/memory trajectory; it cannot establish identity of the unavailable historical executed-source bytes. Report the current execution HEAD, source hashes and resulting tests in the new immutable P1.2 audit; do not change prior blocked artifacts. New code is limited to source-identity gate logic and CPU unit tests. No GPU or Real-IAD evaluation was performed by this change.
