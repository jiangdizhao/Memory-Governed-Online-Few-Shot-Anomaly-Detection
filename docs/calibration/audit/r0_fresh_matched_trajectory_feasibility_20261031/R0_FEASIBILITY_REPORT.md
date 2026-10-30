# R0 — Fresh Matched-Trajectory Feasibility and Low-Cost Correction Replay Design

**Date:** 2026-10-31

**Repository / branch:** `E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection` / `calibration-stage0-contract-20261029`

**Starting commit:** `4047e7062c33617f02ba5bae719b7ab95c6dc60a`

**Decision:** `R0_FEASIBLE_BUT_RESOURCE_INTENSIVE`

**Execution status:** design only; no implementation, test, experiment, image, label, mask, GPU, or scorer execution.

## 1. Executive decision

A fresh matched-trajectory correction experiment is technically feasible with an isolated replay adapter and a two-tier evidence tape. The current native scoring API provides an image score and summary map statistics, but not all dense per-layer/per-view patch maps required for exact replay. A new adapter can call the unchanged native functions to expose those maps and preserve C0 scoring semantics. This adapter does not exist yet, and C0 parity has not been run.

The earliest safe split is after native scoring has computed the dense pre-correction base maps, native top-32 patch IDs, and C0 image score for the event's exact fixed-support and Greedy-memory state. Correction variants then consume the same stored maps and selected feature rows. A full feature bundle is still required transiently during the common pass for native scoring and Greedy writes. Persisting full stream features is possible but costly; the compact tape is enough only while corrections retain the native top-32 selection.

Existing receipts are not a replacement for a fresh tape: the P1 cache was not built or serialized, P12 stores only partial queried-event evidence, and its all-event state/feedback provenance was not certified. Historical correction state is not a portable Greedy-state artifact. The new experiment must be labeled as a Greedy-adapted correction unless it reproduces and pins an exact historical writer path.

Recommend Strategy 3: a preselected one-class, one-seed, 512-event engineering pilot, followed by selective scaling only after independent review and separate researcher authorization. The current master plan still says no further experiment is authorized. This R0 report does not authorize R1. No outcomes were recomputed or evaluated for R0; historical aggregate results embedded in planning documents were not reanalyzed.

## 2. Repository and scope record

The requested branch and starting SHA matched. The tracked worktree was clean. The target report directory did not exist before this task. Twenty-nine existing untracked first-level `outputs/calibration` directories were identified and preserved. No branch switch, worktree, clone, cleanup, or source edit occurred.

The only new file is this report. No tests were run because this was a read-only design task. The existing FT-C0 report and small committed/readiness manifests were read as documentation and metadata only. No image bytes, raw official labels, label-provider responses, masks, confirmation classes, private checkpoints, or prediction payloads were opened. No DINO inference, GPU work, scorer call, or experiment was run.

## 3. Native scoring pipeline and source map

| Stage | Current source and behavior | Shape / dtype ascertainable from source | Dependence and cache consequence |
|---|---|---|---|
| Image identity and preprocessing | `test_phase3_AIF_simplified_v22.py:5913` loads RGB. `vmb_visionad_new.py:318–335` resizes the RGB image to square 448×448 with PIL BICUBIC, center-crops to 392×392, converts to float32, and normalizes by mean/std. `:337–447` loads DINOv2 using `torch.hub`. | Input tensor `[B,3,392,392]` float32, cast to float16 on CUDA when `use_fp16=True`. | Independent of support seed and memory if source bytes, view, transform, and encoder identity match. Cacheable by exact content and transform identity. |
| DINO patch features | `vmb_visionad_new.py:DINOv2MultiLayerBackbone.encode_batch` (around `:423`) hooks four transformer blocks, strips CLS and register tokens, casts patch tokens to float32, normalizes each patch, and returns four patch-feature tensors plus a CLS vector. `test_phase3_AIF_simplified_v22.py:5919–5946` wraps this in `encode_pil` and `fuse_patch_features`. | Architecture/profile expectation: DINOv2 ViT-L/14, 392/14 = 28×28 = 784 patches, width 1024 at each of four hooks. Concatenation gives one `[784,4096]` float32 patch matrix, then L2 normalization. These dimensions are inferred from the named architecture and source profile; no model was loaded, so runtime shapes remain to be recorded. | Feature values do not depend on support seed or dynamic memory. They do depend on image content, view, preprocessing, checkpoint/model weights, source revision, hook layers, fusion, and dtype. The global CLS output is not used by this scorer. |
| Fixed and dynamic normal distance | `test_phase3_AIF_simplified_v22.py:3827` `_dN_min_fixed_dyn` takes the minimum of fixed-normal and Greedy STM distances, with optional LTM/buffer branches disabled in the flat Greedy profile. `:3918` `score_single_view` normalizes features and computes these distances. The locked native profile uses `knn=5`. | Per fused layer, query `[N=784,D=4096]`; output dense `dN [784]`. The actual native distance implementation and fixed-bank occupancy determine compute/memory. | Depends on the support-seed-specific fixed bank and the event's current Greedy state. It cannot be shared across seeds or memory versions unless their exact bank digests match. It is not affected by current-event labels, which are disclosed later. |
| Patch selection and base map | `score_single_view` selects `min(32,N)` largest `dN` values per layer and records their IDs. With `use_defect_bank=False` and corrections disabled, the locked Greedy path has no defect boost or correction term, so the map is the native `dN` map. | Top-32 IDs and values plus dense float32 map `[784]` per fused layer/view. Current `feature_fuse="concat"` makes one feature layer; the top-32 selection is therefore over one 4096-D fused layer. | Selection depends on support seed and current Greedy memory. Candidate IDs and maps must be recorded separately for every trajectory event. |
| Layer/view/image aggregation | `:3999` `fuse_layers`; `:4033–4159` `score_sample_multiview` aligns view maps, fuses views, and calls `aggregate_map` (`:4013`). `experiments/a0_a1_strict4_matched.py:58–71` `score_bank` passes the native configuration. | Profile is one identity view, one concatenated feature layer, layer mean, view mean, and image max over the fused patch map. Thus layer/view fusion are degenerate one-item means; image output is a scalar. `image_topk=64` does not affect max aggregation. | The map and scalar depend on the same fixed and dynamic state as `dN`. Every correction arm must reuse the same base maps and preserve this aggregation order. |
| Native score interface | `score_bank` returns the image scalar and `A_map_stats`; `score_sample_multiview` returns a scalar, summaries/debug statistics, and the feature bundle. Neither is an all-event dense patch-map tape. `:4164+` `score_sample_correction_debug_maps` exposes fused `A_pre/A_post/A_corr` arrays for diagnostics, but not the full per-layer/per-view base maps needed for a general replay tape. `experiments/risk_safe/native_bridge.py:123–171` wraps the scalar path and sets scalar parity tolerance to `atol=1e-5, rtol=1e-4`. | Scalar float; summary metadata; diagnostic fused map arrays where explicitly requested. | Existing bridge parity is useful for scalar checking but does not by itself supply reusable dense maps or patch features. |
| Greedy writer | `experiments/v02_memory.py:108–158` `MemoryArm.update` is called after queried verified-normal feedback. It calls `select_informative_normal_write` with score, theta, sigma, and `pred_defect`; then consumes candidate indices and full features. Greedy replacement uses `greedy_slot` at `:21+`. | Full per-event feature matrices are required transiently for normalization and candidate admission. Dynamic prototypes use native bank dtype (often float16) and have a 256-per-layer cap in the locked flat profile. | Writer inputs include score/threshold-derived eligibility and candidate selection, so a fixed query tape alone does not guarantee a fixed memory path. Writer changes also depend on query identity/label and prior memory. |

`experiments/calibration/profiles.py:20–90` locks `dinov2_vitl14_reg`, image size 392, resize 448, FP16 model input, concat fusion, top-32 patches, one `id` view, `knn=5`, image max aggregation, four deployed supports, and mutable capacity 256 per layer. The four deployed supports plus any separately registered protected C64 calibration normals are not strict four-shot-total. C64 is not part of the normal memory unless a separately approved profile says so.

## 4. Valid cache boundary and tape options

There are two distinct boundaries:

1. **Feature boundary:** immediately after DINO encoding and feature fusion. The feature matrix can be shared across support seeds only when its exact source image, view, transform, model/checkpoint, source code, layer, fusion, dtype, shape, and feature-byte digest match. The feature is independent of the support and Greedy memory.
2. **Correction replay boundary:** after the common trajectory has computed the dense pre-correction patch map(s), native top-32 IDs, and native C0 scalar against that event's exact fixed support plus Greedy state. These values are state-dependent and must be stored per trajectory/event. C1/C2 can then be replayed without repeating DINO encoding or nearest-neighbor scoring if they keep the native selected patch set.

| Tape option | Replay properties | Finding |
|---|---|---|
| A — full features for every event | Supports later reselection, new patch rules, and broad correction kernels; retain full maps for exact C0. | Most flexible, but about 183 GiB for 15,295 unique byte-identical stream images or 915 GiB if duplicated for all 76,475 trajectory events, before metadata. |
| B — selected top-32 features and their scores only | Can apply a fixed correction to those rows, but cannot always recompute image max after a negative correction: an unselected patch can become the maximum. It also cannot support changed patch selection. | Insufficient alone for exact image aggregation. It becomes exact for the locked selected-patch treatment if accompanied by the dense base map (or a rigorously sufficient reduction for the precise frozen aggregation); this report chooses the dense map. |
| C — two-tier tape | During the initial common pass, keep full features transiently for native scoring and the Greedy writer. Persist per-trajectory dense base maps, selected IDs, selected FP32 feature rows, and aggregation/state receipts for correction replay. | Recommended. Compact relative to full features and exact for C1/C2 constrained to native top-32 candidates. If a future correction changes selection or needs all patches, require Option A and rerun under separate authorization. |

The correction split preserves current VisionAD selection and aggregation because the source selects top patches from `dN`, applies correction only to those selected rows, and then runs unchanged layer/view map fusion and image aggregation. The tape must retain the full dense base map because correction can change which patch wins max aggregation. Any method that reselects after correction, applies correction outside the native top-32, or changes view/layer semantics is outside the compact tape contract.

## 5. Minimum immutable Evidence Tape

### 5.1 Manifest and event record

Use an append-only, versioned, manifest-locked tape with independently checksummed shards, resumable offsets, and non-overwrite behavior. Every smoke or partial run is marked `INCOMPLETE`. The tape contains no label values.

**Tape header:** schema/version; task and protocol ID; parent commit; source-tree, scorer, writer, profile, and preprocessing digests; run/trajectory IDs; class and support seed; fixed-support digest; C64 identity digest and role if used; Q-A-TS version/seed/fraction; runtime/device/determinism identity; expected event count; shard manifest and checksums.

**Per-event record:**

- event position, opaque image ID, source-image content SHA-256, official group digest, class, camera/view, source timestamp, and data role;
- encoder/checkpoint/source and preprocessing/view fingerprints; full-feature key and full-feature byte digest;
- fixed-bank digest and dynamic Greedy memory version, content digest, and source-map digest before scoring;
- each layer/view's dense pre-correction base map in its original float32 representation, map checksum, native selected top-32 IDs, selected feature rows in the original float32 representation, and feature-row checksum;
- native C0 scalar, map/fusion parameters, per-layer/view and fused-map digests, C0 score-state digest, and scorer config/source digests;
- fixed Query Tape decision, query ID or null, and a reference to the next causal event, but no label;
- common C0 writer inputs, writer action/candidate IDs, memory version/digest after update, and before/after transition digest;
- C64 state/version/checksum and common C0 threshold identity where C64 participates in the common C0 prediction or correction-write outcome.

The full feature matrix is required transiently during common scoring/writing. The compact persistent tape stores selected FP32 feature rows; it stores the full feature digest as provenance but cannot reconstruct full features if a future algorithm needs them. The score tape is per trajectory because the dynamic state, top-32 IDs, and dense maps vary with support seed and feedback history.

### 5.2 Separate label-disclosure and correction-anchor ledgers

Represent every query response in a separate restricted disclosure ledger, ordered after that event's sealed C0/C1/C2 predictions. It may contain query ID, source/group digest, reveal sequence, verified label payload, and response-chain digest. Do not put evaluator labels, unqueried labels, future labels, or query-response payloads in the online score tape.

Correction anchors are a separate, versioned sidecar derived only after a legally queried label is disclosed. Each anchor records its query/disclosure reference, source group, event position, common C0 prediction/outcome, pre-update Greedy memory version/digest, selected patch IDs and references to their stored feature rows, native base patch scores, sign/coefficient parameters, eligibility decision, and anchor-state before/after digests. Store feature bytes once in the tape; the sidecar holds references and checksums. The source/group and memory references allow leave-source-out checks. Defect anchors may update correction state but never the normal Greedy bank.

### 5.3 Cache keys and reject conditions

**Feature key:** SHA-256 over canonical fields containing source image content SHA-256; transformed-view content SHA-256 and transform ID; view name; RGB conversion; resize/crop/interpolation/normalization parameters; augmentation recipe and seed (when applicable); model name, checkpoint/weight SHA-256, DINO source revision/tree SHA-256, hook-layer list; input and output dtypes; feature-fusion mode; patch grid; actual tensor shapes; and normalization semantics. Keep role, class, official group, and support/C64/stream partition as separate provenance/access-control fields even when the same numerical feature bytes are reusable.

**Score-tape key:** trajectory/class/support seed and event position; feature key and feature-byte digest; exact fixed-support digest; C64 state digest if it affects this score or common C0 threshold; Greedy memory version/content/source-map digests; scorer config/source-tree digests; view/layer map identities; `knn`, top-k, defect/correction flags, map fusions, image aggregation, and runtime numerical identity. A score/map record is not reused across seeds or events unless every state-dependent key is identical.

Reject a cache entry or replay on any content/group/role mismatch; support/C64/query overlap; different view or augmentation; missing checkpoint/source fingerprint; preprocessing or feature-fusion change; unexpected shape/dtype/grid; missing or corrupt checksum; nonfinite tensor/map/score; memory version/digest or source-map mismatch; scorer/config mismatch; query label in the score tape; or incomplete/duplicate event position. An opaque same-image cache hit never waives source-group or split-exclusion checks.

Existing interfaces are less complete: P1 `FeatureKey` at `stage1a_p1_reconstruction.py:182` has image, encoder, and preprocessing hashes, but its `FeatureCache` at `:196` is in-memory and reports `feature_payload_serialized=false`. Its `StaticMapKey` at `:217` is for static memory and lacks dynamic Greedy version/content. P12 `feature_key` at `stage1a_p12_runner.py:1138` uses queried-event image SHA and encoder/preprocessing fingerprints; its writer rebuilds static-map evidence at queried positions. Neither is a complete all-event feature and state-dependent score tape.

## 6. Cross-seed feature reuse

The read-only `stage1a_execution_manifest.json` in `docs/calibration/audit/stage1a_final_readiness_v6_20261030/` lists 20 trajectories: four classes with five support seeds each. Within each class, the five rows share event count, query-position digest, stream identity digest, scorer-config digest, and source fingerprint. The source comparison runner also reads the class stream metadata by event position for each seed (`stage1a_formal_comparison.py:400–446`).

| Class | Seeds | Events per seed | Query-position digests within class | Stream identity digests within class |
|---|---:|---:|---:|---:|
| audiojack | 0–4 | 3,815 | 1 | 1 |
| regulator | 0–4 | 3,855 | 1 | 1 |
| rolled_strip_base | 0–4 | 3,805 | 1 | 1 |
| switch | 0–4 | 3,820 | 1 | 1 |
| **Total** | **20 trajectories** | **76,475 events** | — | — |

This verifies common stream metadata/order and query-position digests, not byte-identical image files. The existing query-event code hashes content only for queried images; R0 did not hash all stream images. The 15,295 unique stream-image estimate is therefore conditional, not confirmed. The native profile fixes model/preprocessing/view configuration, and the five per-class rows share source/config digests, but the exact checkpoint bytes must still be pinned by a fresh run before claiming feature identity.

If future execution hashes each source image and verifies exact feature keys, stream DINO features may be encoded once per unique class-stream image and shared read-only across the five support seeds. The five seed-specific fixed banks and evolving Greedy memories still require five separate native distance/map computations per event. Support features and stream features should use separate role namespaces: support references use deterministic augmentation and FP16 bank preparation (P0 `encode_reference_features`, `:136–166`), while stream queries use the identity view and FP32 fused features (`encode_query`, `:321–324`). Cross-role numerical reuse is allowed only if transformed pixel bytes and every feature-key field match, while overlap exclusion remains mandatory.

## 7. Shared Greedy memory and feedback causality

The common event order is:

1. Read the current stream event and compute frozen features.
2. Score C0/C1/C2 using the same pre-event Greedy memory `M_t`; C1/C2 may add their own correction states only after the shared native base maps are computed.
3. Seal all three predictions and their pre-event memory and correction-state digests.
4. Apply the fixed label-blind Query Tape; disclose only a queried response in the separate disclosure ledger.
5. Update C1/C2 correction state from this legally revealed response and the common C0-derived prediction/outcome, score, threshold, and base-map information. No current label is visible before the current prediction.
6. Run the normal Greedy writer once from frozen common C0 inputs and the queried verified-normal label, producing `M_(t+1)` for every arm. Unqueried and queried-defect events cannot enter normal memory.

The shared writer must freeze the C0 scalar, theta, sigma, C0 `pred_defect`, AMap/candidate IDs, write eligibility, keep count, source/group identity, event number, and the verified query response. `MemoryArm.update` uses score/threshold-derived `pred_defect`, `sigma`, candidate selection, and features; therefore feeding C1/C2 scores or predictions into the writer would change admission or candidate count even with identical query IDs. The common C0 reference prevents this feedback path. C64 threshold/evidence state, if used, is likewise common and separately keyed.

Required assertions at every event:

- identical pre- and post-event Greedy memory version/content/source-map digests for C0/C1/C2;
- identical query IDs, decisions, and reveal order;
- correction update timestamp strictly after prediction and query disclosure;
- writer input/action/selected-candidate digests identical to the common C0 reference;
- only queried verified normals enter Greedy memory; defect anchors remain correction-only;
- no official source group is used as its own correction anchor or as a support/C64/stream overlap;
- no support/C64/query-source overlap is silently accepted;
- C0 scores and maps match the native scorer with correction disabled.

This design requires an isolated new replay adapter. `NativeScoreAdapter` currently installs a Greedy state into a native clone but returns only the scalar. The current writer can already be called once after queried-normal feedback, but it does not implement C0/C1/C2 orchestration or produce this tape. Do not patch v22, Greedy, the B3 controller, the scorer, or the query controller.

## 8. C0 native-score parity design

Parity is feasible in principle because the unchanged scorer exposes `score_single_view`, `fuse_layers`, `aggregate_map`, view alignment, and the normal `score_sample_multiview`/`score_bank` path. It is not established until an authorized pilot tests it.

For each pilot event, take a frozen feature bundle and exact fixed/dynamic bank snapshot with defect and correction banks disabled. First run unchanged `score_bank` and retain the native scalar plus AMap top-32 IDs. Through the isolated adapter, call the unchanged per-view `score_single_view` once per view to get dense maps and per-layer top-32 IDs; reconstruct the unchanged layer fusion, view alignment/fusion, and image aggregation with the same config; compare the reconstructed C0 scalar to the native scalar. In the pilot, also compare the fused map to the native diagnostic `A_pre` map and verify event state digests before and after scoring are unchanged. The separate native and map-extraction passes may temporarily duplicate NN work in the parity pilot; after this adapter passes, the full common pass need only compute its stored dense maps once.

**Pass gate:** exact grid, layer/view ordering, and top-32 IDs; bitwise map equality where the same runtime operation permits it; exact scalar equality on the same deterministic runtime, or at most the existing native-bridge tolerance (`1e-5` absolute + `1e-4` relative) if a documented serialization/runtime effect explains the difference. Any unexplained map difference, selection mismatch, nonfinite output, or tolerance exceedance blocks the run. Empty correction state must reproduce C0. A tolerance result is an engineering parity finding, not empirical evidence.

The built-in `score_sample_correction_debug_maps` can assist with fused pre/post diagnostics but does not replace per-layer/view tape extraction. No parity test was run for R0.

## 9. Resource and storage estimate

All numerical estimates below are illustrative source-profile calculations, not runtime measurements. They assume one identity view, one concatenated feature layer, 784 patches, 4×1024 concatenated channels, and stored stream features/maps in float32. Actual hook shapes, checkpoint, bank occupancy, kernel implementation, and runtime must be recorded by a separately authorized pilot.

| Item | Calculation / scenario | Estimate and caveat |
|---|---|---|
| One full fused feature | `784 × 4096 × 4` bytes | 12,845,056 bytes = 12.25 MiB per image. |
| DINO stream encodings | `15,295` unique images if same-byte sharing is proved; otherwise up to `76,475` trajectory events | One-time encoding count is conditional. No duration can be inferred. Add support/C64 references separately: `N_ref × patches × feature_dim × dtype_bytes`, with each augmentation/view counted as a distinct transform. |
| Full stream feature cache | `15,295` or `76,475` times 12.25 MiB | About 196.5 GB / 183.0 GiB if shared; 982.3 GB / 914.9 GiB if fully duplicated. These are not needed for the recommended constrained replay tape. |
| Compact selected features | `76,475 × 32 × 4096 × 4` | 512 KiB/event; about 40.1 GB / 37.34 GiB across 20 trajectories. Top-32 selection depends on each Greedy state, so do not assume cross-seed deduplication. |
| Dense base maps | `76,475 × 784 × 4` | 239,825,600 bytes = 228.7 MiB for one layer/view. Add map copies only if the frozen profile has more layers or views. |
| Compact tape total | Selected features + one dense map/event | About 37.56 GiB before event metadata, checksums, anchor receipts, C64 receipts, and filesystem overhead. |
| Pilot tape | `512 × (512 KiB + 3,136 bytes)` | About 257.5 MiB before metadata for one trajectory, one layer/view. It can validate schema/parity without building 20 trajectories. |
| Greedy scoring | `O(events × views × layers × Npatches × (Nfixed + Ndyn) × D)` | Requires one state-specific nearest-neighbor pass per trajectory/event. At 76,475 events it is not avoided by sharing DINO features. Exact runtime is unknown and must not be inferred from P12 or queried-anchor rescoring. |
| Correction replay | Upper bound `32 × 256 × 4096 = 33.55M` multiply-accumulates/event for a dense top-32-to-256-anchor comparison | Across 76,475 events: about 2.57T MAC, or about 5.13 TFLOP when counting 2 FLOPs/MAC. Actual active anchor count is usually lower; C2 may be cheaper. No CPU/GPU time is measured. |
| Peak RAM/VRAM and disk | Model + fixed bank + mutable bank + current feature/map chunks; compact tape on disk | No reliable VRAM/time estimate from source alone. Distance chunking, bank occupancy, device, precision, and checkpoint are runtime variables. Full cache estimates exclude support/C64 data and metadata. |

A small pilot is viable without constructing the full 20-trajectory tape. The complete run remains resource-intensive due to state-specific Greedy scoring and roughly 37.6 GiB of compact selected features. Use resumable shards, content-addressed keys, checksums, manifest locking, and non-overwrite output names.

## 10. Minimal R1 engineering pilot proposal

**Proposal only; not authorized.** Preselect `audiojack`, support seed `0`, first 512 ordered stream positions (`0–511`). This is selected by fixed class/seed/order before any performance inspection. Use the existing fixed Q-A-TS-v1 tape, seed 37 and 3%, without regenerating or adjusting it. Use the four deployed supports and the registered additional C64 references if required by the common C0 threshold protocol; report the access budget as four supports plus additional C64, never strict four-shot-total.

The pilot should build exactly one valid Greedy trajectory and test engineering correctness only: feature and state keys, query order, source/group/timestamp checks, exact writer transitions, checksummed tape write/resume behavior, dense map extraction, C0 parity, and C1/C2 score-interface compatibility. Do not calculate AUROC, AUPRC, BA, FPR, FNR, or treatment effects.

Before any R1 implementation, freeze method identities. A reproducible legacy-style starting point is available in current source: the normalized signed RBF application is `Δ(h)=Σ_j w_j(h)β_j/(Σ_j w_j(h)+ε)`, with `w_j=κ_j exp(−d_j²/(2r_j²))` and cosine distance `d_j=1−cos(h,p_j)`. The current source's update uses label sign, common C0 outcome, score/theta, selected patch dN gates, outcome weights, clipping, and same-sign merge/retention. The existing FT-C0 audit records the exact current-source defaults and their ranges. If reused on Greedy maps, name it `C1-Greedy-adapted`; do not claim the old v22 correction state or historical path was transplanted faithfully. Pin whether the direct `strength_scale=1.0` path is used or the more context-dependent strength calculation; do not switch after observing results.

For a compact simple comparator, preregister a `C2-Greedy-signed-kNN` candidate before R1: use source-disjoint, legally queried correction anchors; for each native top-32 patch select up to the five nearest anchors by cosine distance and add the unweighted mean of their signs (`+1` defect, `−1` normal; unit scale) to that patch's base score. Use deterministic tie-breaking by anchor event order and a 256-anchor-per-class/view/layer cap; freeze the retention order before R1. This is a proposed adapted comparator, not a validated method; retain the same source-group, label-timing, and top-32 restrictions. If reviewers require a different fixed amplitude or neighbor rule, freeze that before R1 and do not optimize it on pilot outcomes.

**Outputs, if later authorized:** locked profile/source digests; exact source/content/feature-key manifest; per-event dense base maps, top-32 rows/IDs and native scalars; query-event references (no label payload in score tape); C0 writer inputs/actions; pre/post memory and correction-state digests; parity results; anchor activation counts; checksums; disk/RAM/VRAM and timing receipts.

**Stop conditions:** stop on C0 map/scalar parity failure, different query IDs/reveal order, any Greedy digest or writer-action divergence, label-before-prediction, invalid source-group/timestamp ordering, support/C64/query overlap, defect admission to normal memory, duplicate anchor group, cache-key/checksum mismatch, incomplete resume state, or nonfinite feature/map/score. If suitable queried evidence has not produced both normal and defect correction anchors—or no eligible correction activates—within 512 events, return `INSUFFICIENT_ANCHOR_EVIDENCE`; do not call it an algorithm failure. No outcome evaluation or parameter edit is allowed in this pilot.

## 11. Implementation strategy comparison

| Strategy | Benefits | Costs and limitations | Assessment |
|---|---|---|---|
| 1 — recover historical P12 | Could clarify old source and feedback history if every private artifact were recovered. | Current P1 feature provenance says `NOT_BUILT_NO_AUTHORIZATION`; P12 evidence is partial, all-event parity and final response provenance are incomplete, and the legacy correction bank lacks Greedy source/group/state provenance. Recovery still would not create an all-event compact tape and cannot make the legacy correction state portable to Greedy. | Do not prioritize. It is a high recovery burden and does not answer the fresh matched-trajectory question. |
| 2 — fresh full 20-trajectory native pass with shared encoding | Directly produces a reusable fair state-matched tape for all four development classes and five seeds; one feature computation can serve five trajectories if byte/feature keys match. | Requires 76,475 state-specific Greedy score/map events; feature reuse is not yet content-certified; compact tape is about 37.6 GiB, while full features can reach 183–915 GiB. Needs the isolated adapter, pinned checkpoint, correction protocol, and separate compute/data authorization. | Correct eventual scale-up, but expensive as the first engineering step. |
| 3 — minimal pilot then selective scale | Exercises shapes, state/key contracts, C0 parity, causal order, writer equality, and serialization on at most 512 events before paying for full trajectories. Preserves reproducibility if it writes the same versioned tape schema and freezes all gates before scaling. | Pilot may end `INSUFFICIENT_ANCHOR_EVIDENCE`; its engineering PASS cannot support AUROC claims. It still needs separately authorized image/label/GPU access, exact method freeze, and independent review. | Recommended next, after review and explicit separate R1 authorization. |

The new infrastructure can support fair future comparisons beyond reproducing old v22 behavior: multiple correction kernels can consume one state-matched base-map/feature tape if they share eligible feedback, selected patch rows, and no-change memory/query contracts. Full-feature storage or new scoring is required if a future method changes patch selection, spatial support, or feature representation.

## 12. Remaining risks, required inputs, and final recommendation

Remaining gates before R1:

- pin and hash the exact DINO checkpoint/weights, DINO source tree, device/runtime, preprocessing and actual output shapes;
- hash every pilot stream image and official group without opening outcome labels, and prove support/C64/stream/query disjointness;
- obtain a complete fixed Query Tape and legally permitted response path, with a separate response-chain ledger;
- define the exact C1 writer/strength route and freeze C2 neighbor/amplitude semantics before any pilot outcome can be observed;
- implement and independently review the isolated map/tape adapter; test C0 parity, resume integrity, source exclusion, and every event's shared memory/writer/query digest;
- obtain separate researcher authorization for the named class/seed/event count and image/label/GPU operations. The current master plan does not grant this authorization.

**Final R0 status: `R0_FEASIBLE_BUT_RESOURCE_INTENSIVE`.** The architecture is feasible, the recommended compact tape can avoid repeated DINO and nearest-neighbor scoring across correction variants, and the pilot is small enough to limit initial risk. Exact C0 parity is feasible in principle but unverified. Fivefold encoding reduction is conditional on per-image content and feature-key equality; no dynamic map or Greedy scoring reduction follows from it. The full 20-trajectory tape is substantial in storage and state-specific scoring cost.

**Recommendation: HOLD R1 for independent protocol/method review and separate explicit researcher authorization.** If authorized, begin with the fixed one-class/one-seed/512-event engineering pilot; scale only after all engineering gates pass. No experiment was implemented or executed here, and no claim about correction benefit or scientific success is made.

## Sources inspected

- `docs/calibration/REJECTION_RECOVERY_FAST_TRACK_EXECUTION_PLAN_20261031.md` — correction-task scope and approval gates.
- `docs/calibration/audit/ft_c0_preflight_20261031/FT_C0_PREFLIGHT_REPORT.md` — historical artifact gap, signed-correction source semantics, and current-input limitations.
- `docs/calibration/B3_CALIBRATION_PAPER_RECOVERY_MASTER_PLAN_AND_PROGRESS.md` — decision vocabulary, development split, causal ordering, inherited constraints, and current authorization status.
- `docs/calibration/audit/stage1a_final_readiness_v6_20261030/stage1a_execution_manifest.json` — small trajectory/query/source digest metadata only.
- `docs/calibration/audit/stage1a_p1_reconstruction_v1_20261030/feature_cache_provenance.json` — cache provenance status only.
- `outputs/calibration/stage1a_formal_controller_comparison_v4_20261030/run_manifest.json` — run/event-count and query-tape metadata only; no prediction payloads opened.
- `vmb_visionad_new.py`; `test_phase3_AIF_simplified_v22.py`; `experiments/v02_memory.py`; `experiments/risk_safe/native_bridge.py`; `experiments/a0_a1_strict4_matched.py`; `experiments/p0_protocol_audit.py`; `experiments/calibration/profiles.py`; `experiments/calibration/engine.py`; `experiments/calibration/stage1a_p1_reconstruction.py`; `experiments/calibration/stage1a_p12_runner.py`; `experiments/calibration/stage1a_formal_comparison.py` — only relevant functions/metadata were inspected.
