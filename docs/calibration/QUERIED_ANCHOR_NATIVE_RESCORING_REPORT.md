# Native Query-Anchor Rescoring Report

**Status:** source-faithful native measurement; independent research review required.

Protocol candidate SHA-256: `959ee7cfaaa258596d34f2b3a095921f9c58c9d0f923698943f586b0de093ff3`.

The unchanged v22/VisionAD `score_encoded` path scored the original frozen
snapshot memories through the Native Scorer. The frozen four supports remain
separate from the additional protected C64 reference. No Greedy writes occurred.

## Scope and integrity

- Source trajectories verified: 20.
- Snapshot states expected: 120.
- Stage 0.75-A stale `INCOMPLETE` marker preserved: True.
- C64 parity uses all 64 protected references at every restored snapshot.
- Fixed support digest and snapshot memory/source-map identities are checked exactly.
- C64 q90 parity tolerance: abs(delta) <= 1e-6 + 1e-6 * max(abs(a), abs(b)).
- DINO source and weights were loaded only from the verified local cache.

- Actual-B3 package linkage status: BLOCKED; listed source files and the replay/run-manifest link were independently checked. Linkage findings: ['complete manifest checksums.json SHA-256 mismatch', 'complete manifest file count 5 differs from current checksum entries 4'].
- Prior incomplete implementation attempt preserved at `outputs/calibration/queried_anchor_identifiability_v1_20261030` with status `BLOCKED_WIP`; no future result artifact was produced.
## Rescoring totals

Newly encoded queried images: 386.
Protected C64 images encoded once per category: 256.
Current-memory normal LSO score calls: 2525.
Primary normal LSO anchor scores: 2450; repeated query-anchor parity scores: 100.
Current-memory defect score calls: 2645.
Normal full-memory sensitivity score calls: 2450.
Native score calls total, including C64 parity/repeats: 15445.
Queried feature-cache reuse count: 4709.
Protected C64 feature-cache reuse count: 7424.
Elapsed native measurement seconds: 811.418.
Peak CUDA allocated bytes: 2433839616.

Normal anchors use all-layer source-aware LSO. Defect anchors use unchanged
snapshot memory and are rejected if any prototype provenance indicates prior
normal admission. Arrival-time scores are used only in the staleness sensitivity.

The original Stage 0.75-A trace stores query positions and writer outcomes but
does not persist the disclosed label values. This run reconciled query labels
against the unchanged official metadata SHA, the frozen query tape, and every
seed's writer response; no separate historical label-response digest exists.

## Gate matrix

See the machine-readable gate matrix in the audit package. Blocked cells are
not replaced by reconstructed or approximate states.
