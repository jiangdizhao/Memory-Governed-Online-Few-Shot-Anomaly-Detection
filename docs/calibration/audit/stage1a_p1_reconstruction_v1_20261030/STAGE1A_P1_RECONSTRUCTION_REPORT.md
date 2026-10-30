# Stage 1-A P1 native replay and evidence reconstruction

**Status:** `P1_IMPLEMENTATION_PREPARED; DATA_REPLAY_BLOCKED_PENDING_TWO_APPROVALS`

**Decision:** Stop before GPU/image execution and queried-label access. The repository contains no explicit bounded P1 scorer/data authorization and no separate queried-feedback access decision. This package records the query-only replay design and tests synthetic runner primitives only. It does not mark original memory parity PASS and does not authorize Stage 1-A performance evaluation.

**Repository:** `calibration-stage0-contract-20261029`, base `fd29a31d49613ed87530ad71c71fa1bfa483f4c8`. The approved Stage 1 v3 protocol and focused addendum candidate remain unchanged.

## 1. Query-only reconstruction feasibility

The original memory path is structurally reconstructible by visiting all event metadata while regenerating model inputs only at the 459 precommitted query positions across the four class tapes. The shared Greedy writer is threshold-independent and can change memory only after queried verified-normal feedback and an accepted write. Nonqueried events, queried defects, and rejected/no-content-change writes do not add memory prototypes. Therefore a full-stream DINOv2 rerun is not justified by current evidence.

Reuse the 76,475 static and 76,475 dynamic scalar score values, the fixed Q-A-TS-v1 3% seed-37 tape, the unchanged native writer/reference configuration and source, and all 76,160 stored post-write C64 scores. For each queried event, re-encode that image once and share the immutable feature bundle across support seeds only when model/checkpoint and preprocessing fingerprints match exactly. Regenerate the static M0 score map for that image under each exact static support-bank identity; compare its scalar to the saved static score tape. Static maps are absent from the saved output. Exact-key caching may share a map across seeds only when image, static memory, scorer config, and scorer source fingerprints are identical. The maximum without such reuse is 2,295 queried-event map calls (459 positions × 5 seeds). The map is computed before queried-label disclosure, preserving the historical causal order. Dynamic event scores are reused, not recomputed.

The feature bundle for every queried image is retained privately for later anchor evidence. A queried normal's writer call uses its regenerated M0 map and full feature bundle; a queried defect is not written but its feature bundle is needed as a defect anchor. C64 maps/features are not regenerated while their stored memory-addressed scores and transport digests match.

Any before/after memory version/hash, writer action, source-attribution, legacy-version, M0 scalar, C64 q90, or transport mismatch stops that trajectory at that event. The runner has no repair or approximate path. If an authorized query-only replay fails exact parity, report the exact field and event first. A full-stream backbone replay is not an automatic response; it would require separate evidence that query-only feature reconstruction cannot explain the mismatch and a further bounded approval.

No real query-only replay was started. Thus structural reconstructibility is **supported, not verified**. Neither original G2 nor its historical missing source/label provenance has been repaired.

## 2. Minimum additional work and accounting

Actual P1 computation in this task: zero image reads, zero DINOv2 encodings, zero Native Scorer calls, zero queried-label reads, zero GPU calls. P1 maxima are 459 unique query feature encodings, 459 unique queried-label reads cached across support seeds, at most 2,295 static map calls before identical static-bank reuse, and a metadata-derived upper bound of 75,406 version-addressed query-anchor score calls. The combined physical scorer bound is 77,701 calls; DINO feature calls are counted separately. Event score calls and new C64 calls remain zero if the saved tapes pass exact identity checks.

The prior six-snapshot audit recorded 15,445 score calls, 811.418 seconds, and 2,433,839,616 peak CUDA bytes. Its measurement artifact contains labels and was not opened or reused. These measurements are not extrapolated to dense P1 replay. P1 runtime, CPU RAM peak, GPU peak, feature-cache size, and output storage are `UNKNOWN`.

The tested cache key binds image bytes, encoder/model identity and preprocessing. Anchor score keys bind memory version/content SHA, per-layer source-map SHA, anchor/group identities, scorer config/source SHA, and exclusion semantics. C64 q90/transport entries are reused only under exact memory identity. A normal anchor is scored with every row from its source group removed in every layer; the adapter must provide an all-layer zero-residual proof. A defect is scored against the current bank and is never written. One official source group may produce one query anchor only. New anchor arrival without a memory change reuses all existing exact-key scores and scores only that new anchor. A memory/source-map change changes the key and requires fresh scores for all eligible anchors in the evidence snapshot.

The cache primitives and exact event replay are callback-driven so they do not load a dataset or initialize a model on import. A production adapter to the frozen Native Scorer and `SharedNativeGreedyWriter` remains a P1 integration gate; no real artifact loader was invoked before authorization.

## 3. Label firewall and authorization requirements

Historical G2 remains `BLOCKED`; the independent certificate remains package-integrity `PASS` with researcher review required. This package does not recover the exact historical executed source bytes or historical query-response digest. The previous label-bearing snapshot payload remains unopened.

Real P1 requires two separately recorded decisions:

1. A bounded Native Scorer/data authorization for exactly the four development classes and precommitted query schedule, with maximum feature/map/anchor call counts and explicit exclusion of confirmation classes and masks.
2. A queried-feedback access decision for only those Q-A-TS-v1 positions. It must exclude unqueried labels, Mint/PCB, masks, and confirmation outcomes. A new query-response digest may then be emitted as a digest only; raw labels remain private.

The runner requires both records before it can invoke lazy event, image, or label providers. The helper `make_p1_authorization_record` creates a `PENDING_RESEARCHER_DECISION` template only; it does not approve access. No applicable approval record was found in the calibration docs, so execution stopped here.

## 4. Controller timing and evidence availability

Source review confirms the prequential engine scores and predicts first, discloses a label only when the fixed tape queries, calls feedback/score updates, applies the independent shared writer, refreshes evidence on query or memory change, and finalizes thresholds for the next event. No method comparison was run.

| Method | Legal update point | P1 input availability |
|---|---|---|
| B0 | Predict from current score and fixed theta; no feedback update. | Score and theta0 tapes reusable. |
| B1 | After prediction, append only queried verified-normal stale arrival score; adapt after 16. | Needs separately approved queried labels; no image scoring. |
| B2 | After prediction, feed queried labels to the adapted v22 sidecar; its unlabeled callback is a no-op and its source normal-only fallback remains disabled. | C64 initialization is reusable; feedback needs separate label decision. No full-loop parity claim. |
| B3 | After shared writer/C64 evidence refresh, use current protected C64 q90 for the next event. | Stored C64 transport can support reconstruction if exact memory hashes match. |
| B4 | After prediction, update matching queried-label error state; then use current-version normal/defect evidence for the next prediction. | Needs labels and dense current-memory anchor scores. ACI guarantee transfer is not claimed. |
| B5a | Source-only score update after prediction at each event; 1,000-observation startup remains unavailable. | CPU replay from dynamic score tape; preserve missing startup predictions. |
| B5b | Source-only residual update after prediction; 50 local points plus 1,000 residual warm-up observations remain unavailable. | CPU replay from dynamic score tape; preserve missing startup predictions. |
| B7 | Consume only versioned queried-normal/defect evidence after writer and evidence refresh; HOLD under its unchanged v0.1 rules. | Needs labels plus dense current-memory anchors; CP values remain descriptive. |
| Stage075BOffset | `observe_feedback` is a no-op; on evidence revision it takes the midpoint of queried normal/defect score medians only when both sides exist, then clips to 0.01. | Needs dense queried-anchor scores. One-sided or missing-side evidence sets desired theta to its own theta, so it holds. It does not use C64 and is not a continuously updated B3-plus-additive-offset controller. Its historical `B3+Offset` name describes the synthetic candidate, not a mathematical identity with B3. |

The Stage075BOffset behavior above is verified directly from its implementation and with a deterministic synthetic one-sided/two-sided unit test. The earlier P0 method card overstated C64 as an input for this row; this P1 report records the inspected source behavior without editing the prior P0 artifact or changing the candidate protocol.

## 5. Test evidence and gate status

The existing-Torch-environment P1 test command was:

```text
D:\anaconda3\envs\torch\python.exe tests/calibration/test_stage1a_p1_reconstruction.py
```

**14 passed.** These are synthetic CPU-only tests covering authorization fail-closed behavior, query-only event replay, one-shot versus checkpoint/resume identity, memory/writer/C64 mutation rejection, queried-only label access and deduplication, feature/static-map cache identity, anchor key mutation, score reuse on same-memory new-anchor arrival, memory-change invalidation, source fingerprint mutation, compute-cap enforcement, SharedNativeGreedyWriter/C64 lookup adapters, and Stage075BOffset's one-sided HOLD behavior. Python compilation also passed for the new runner and tests. These tests do not establish native artifact parity, GPU correctness, or scientific performance.

Gate detail is in `p1_gate_matrix.json`. Real P1 replay is blocked on both approvals and production data/scorer adapter execution. Original G2 is still blocked. Stage 1-A performance evaluation remains out of scope even if P1 later passes; it requires the separate Stage 1 execution go and protocol approval. No Stage 1/2 experiment, unqueried label, mask, Mint/PCB image, or confirmation outcome was accessed in this task.
