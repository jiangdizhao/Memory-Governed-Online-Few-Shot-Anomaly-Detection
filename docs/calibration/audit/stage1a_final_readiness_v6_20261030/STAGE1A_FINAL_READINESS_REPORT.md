# Stage 1-A Final Readiness Report

**Decision: `STAGE1A_READINESS_PARTIAL`**

This package freezes a focused, controller-only Stage 1-A execution specification. It does not execute the algorithm comparison, report Real-IAD outcomes, or evaluate performance.

## Frozen scope

- Classes: audiojack, regulator, switch, rolled_strip_base.
- Support seeds: 0–4.
- Query tape: Q-A-TS-v1, 3%, seed 37; binding `dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183`.
- No Q-B, 5%, or 7% runs. Protocol v3 and the focused addendum candidate are unchanged; this is `FOCUSED_STAGE1A_READINESS_SPEC_v1_2026-10-30`.

## Readiness evidence

The adapter consumes precomputed scores and recorded memory transitions. It calls each prediction before queried feedback, supplies feedback only to registered consumers, applies the recorded writer/memory transition, delivers the corresponding evidence, then updates controller state for the next event. It never invokes Native Scorer, DINOv2, or Greedy writer. Deterministic synthetic CPU tests passed. A separate bounded production-input smoke processed audiojack/seed0 through its first taped query at t=35 (36 events, one query), produced no metrics, and stopped; it did not replay the stream.

The execution manifest links all 20 P1.2 receipts and evidence hashes, and Stage 0.75-A score/event/C64 provenance by class. The query tape verifier reads only frozen metadata and the label-blind tape archive. Offline performance outputs, full-stream labels, masks, and confirmation outcomes were not opened.

## Method registry and metrics

Executable methods: B0–B5b, B7, and Stage075BOffset; B5a/B5b source-native methods and their Frozen-Theta hybrids remain separate. B6 is retained as applicability-only and B8 as an existing historical negative control. Stage075BOffset preserves its own raw-median-offset behavior and holds its own threshold if either class-side evidence is unavailable; it is not an additive B3 offset.

Metrics are specified for a later private post-prediction evaluator. Undefined denominators remain null; source-native SPOT/dSPOT startup-unavailable predictions do not receive ordinary full-stream BA; common horizon begins at zero-based t=1050; worst-window reporting uses defined full nonoverlapping 256-image windows. B5a/B5b/B6 applicability reporting retains actual and class-conditional coverage, makes no post-outcome cutoff, and leaves inappropriate metrics undefined; fallback decisions are separately attributed.

## Source caveats and limits

The historical Stage 0.75-A G2 gate remains BLOCKED. P1.2 uses scoped current-source reconstruction; exact historical executed source bytes and G2 query-response digests were not recovered. This package preserves that caveat and does not close G2. The readiness result is an engineering readiness decision only, not a scientific validation or risk guarantee.

## Remaining readiness blocker

The P1.2 `complete.json` and two preserved final checksum manifests refer to report hashes that do not match the current preserved report bytes. The independently enumerated 20 P1.2 receipt records match their shard receipt fields, and each evidence file matches its receipt SHA-256. Historical files were left unchanged. Before Stage 1-A execution, an append-only P1.2 integrity reconciliation or independent researcher-approved closure should bind the current report bytes without overwriting the prior audit.

| Preserved reference | Expected SHA-256 | Current actual SHA-256 |
|---|---|---|
| `complete.json` final report pointer | 295e8011652b2d0c7f67ffa2357cfd0df95de7c308693def4a5448fde824e0b8 | 1ed4b889b6c9f7372fdea746f8cf0c4f05aec3431892458f6638b67930f0b48a |
| `checksums_final.json` → `STAGE1A_P12_LOW_MEMORY_RECOVERY_REPORT_FINAL.md` | 295e8011652b2d0c7f67ffa2357cfd0df95de7c308693def4a5448fde824e0b8 | 1ed4b889b6c9f7372fdea746f8cf0c4f05aec3431892458f6638b67930f0b48a |
| `checksums_final_v2.json` → `STAGE1A_P12_LOW_MEMORY_RECOVERY_REPORT.md` | 209eaaee4576079a72a04394ee522eb8f4044972dad88b56583fe82e4fa82d65 | aef3d4aa2e0f7a26de074705ce0d7eca0b1782f14730ed53d223c4d9b51fd12d |

## Gate decision

**`STAGE1A_READINESS_PARTIAL`** — the replay adapter, synthetic controller/metric checks, bounded input smoke, registry, query tape, and per-trajectory input links pass. Do not begin the formal comparison until the P1.2 report checksum discrepancy is closed in a new append-only integrity record. No formal performance evaluation was started here.
