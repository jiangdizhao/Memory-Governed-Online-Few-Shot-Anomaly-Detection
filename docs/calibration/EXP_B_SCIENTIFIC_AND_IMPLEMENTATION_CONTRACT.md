# Experiment B — Signed Defect Feedback Causal Attribution

**Status:** implementation contract; awaiting independent source review. No full Real-IAD execution was performed by this implementation task.

## 1. Scoped authorization and repository record

On 2026-11-08, the researcher explicitly authorized this Experiment B implementation despite the pre-existing untracked worktree inventory. This scope permits new Experiment B code in `experiments/calibration/`, new tests in `tests/calibration/`, this contract, focused synthetic CPU verification, and a local commit containing only those new files. It does not authorize a full Real-IAD Stage A/B run, protected confirmation outcomes or masks, edits to historical R7/R9/R10/v22/FS4 sources, or unrelated files.

Before implementation, the path-only worktree audit recorded branch `calibration-stage0-contract-20261029`, HEAD `36ff59684dae0c3f5d746cca93aac4076a307676`, zero tracked modifications, and 7,030 pre-existing untracked paths. The SHA-256 of the sorted relative path inventory was `b57116cc14bb9ee94859eba1d4cafd3cecc4ec1687f9b57bccc2aa896109394a`. The inventory had 231 paths under `docs` and 6,799 under `outputs`. The five implementation targets did not exist before creation. Existing untracked outputs were left in place and excluded from staging.

The complete-run CLI additionally requires a separate review-complete researcher authorization JSON bound to the source commit, this contract hash, four registered categories, run ID, and Stage A/B scope. This task did not create that execution authorization and did not call the full-run CLI. Each category runs in a fresh Windows Job Object child with a one-hour deadline; the CPU-only Stage B runs in a separate bounded Job Object child. The parent verifies process-tree exit receipts before it accepts category seals or begins Stage B. No automatic restart or resume is supported.

## 2. Frozen scope

Experiment B uses the R9/R10 P0 development stream: Real-IAD categories `audiojack`, `regulator`, `switch`, and `rolled_strip_base`; Support Seed 0; Shuffle Seed 0; 15,295 image events; and the exact R9/R10 P0 Q-A-TS queried image identity set (459 queries total). The four fixed Support images remain separate from the 64 additional, disjoint TRAIN-normal C64 commissioning images. This is **four Support plus C64**, not strict four-shot-total.

The frozen scoring path is `dinov2_vitl14_reg`, resize 448, center crop 392, concatenated blocks 5/11/17/23 (4096 dimensions), native spatial maximum, and native C1 Top-32 Patch selection. The native fixed bank has K=25,000 nominal capacity and 8,192 realized prototypes. The one shared Greedy memory and native C1 memories use K=1,024. TEST images are encoded once in the integrated loop and share their descriptors across all seven principal arms.

No confirmation category, mask, or unqueried TEST outcome is available to Stage A. Query decisions come from the frozen R9/R10 tape. Every queried outcome is read through `IndexedQueriedLabelReader` only after the current score row has passed `ScoreDisclosureGate` fsync sealing.

## 3. Principal estimands and arms

For each registered score (J), the factorial contrasts are:

- Defect-side increment: `ND - N`.
- Defect-only effect: `D - G`.
- Normal-side effect: `N - G`.
- Metric-scale interaction: `ND - N - D + G`.

The interaction is an estimand on the chosen metric scale; it does not by itself establish an intrinsically nonlinear Correction operator. AUROC, AP, and balanced accuracy prefer larger values; BER, FPR, and FNR prefer smaller values.

| Arm | Score and state | Legal updates |
|---|---|---|
| G | Unmodified native R9/R10 Greedy-only score | The sole shared Greedy writer; no Correction state |
| N | Native R9 Full C1 readout with independent initially empty K1024 Codebook | Queried normals only, sign -1 |
| D | Native R9 Full C1 readout with independent initially empty K1024 Codebook | Queried defects only, sign +1 |
| ND | Unmodified native R9 Full C1 readout and update operator | Both verified signs |
| SignedKNN | Original R10 `SignedKNNFIFO`, capacity 1024, k=5, amplitude 0.02, FIFO/source exclusion and R10 dense-map correction | Every queried verified label with its registered sign |
| HITL_FP | Independent Abbas-inspired 9NN normal-memory editor | Queried false positives among verified normals; FP expansion only |
| HITL_FP_FN | Same external scorer and FP editor as HITL_FP | Same FP rule plus queried false negatives among verified defects; FN surgery |

The N and D trajectories are separate counterfactual Codebooks. Their different occupancy and replacement histories are part of their treatment effects. They do not share Correction state with ND. Every factorial arm scores against the same Greedy memory trajectory.

## 4. Causal event ordering

The integrated event loop follows:

1. Extract one TEST image's native DINO feature tensor.
2. Score G, N, D, ND, SignedKNN, HITL_FP, HITL_FP_FN and read-only diagnostics from pre-event state. Record each arm's state version, state digest, threshold, prediction, and the frozen query decision.
3. Seal and fsync the outcome-free score row through the existing R7 `ScoreDisclosureGate`.
4. If and only if the frozen query tape selects the event, reveal the official JSON label through the R7 indexed reader, bound to the score seal.
5. Update only eligible per-arm Correction and memory-editor states. Derive every N/D/ND Correction `score`, `theta`, and FP/FN/TP/TN outcome from the common pre-feedback G reference. Apply the single shared normal Greedy writer using the original G score, theta, sigma, prediction, Top-32 IDs and features. Defect labels never enter the normal Greedy bank.
6. Recompute changed-state C64 evidence, refresh B3 thresholds, and apply every change from event (t+1).
7. Commit the private queried-feedback receipt and finish the event.

Unqueried events perform no supervised update. A current event's disclosure cannot affect its sealed prediction. Stage A score JSONL files reject label/outcome keys; labels are written only to the private feedback ledger.

## 5. Native C1 readouts and attribution diagnostics

N, D, and ND use the existing `SignedCorrectionCodebook`, native `apply()`, `r1._c1_score_delta`, R9 dense-map reconstruction, and `r2_exp2a_c1_strength._capture_update`. N/D use the unchanged native C1 operator but independent empty K1024 states, accepting only the sign registered for that arm. G owns no Correction Codebook.

ND positive-only and negative-only readouts are diagnostics on the same pre-event ND state. The adapter first applies the R9 current/future and Source Group eligibility filter, then creates a read-only native Codebook clone containing only anchors of the selected sign. Consequently, omitted signs are excluded from both the numerator and normalization denominator. An empty eligible sign has zero correction. The live ND state is not modified.

The G global-offset diagnostic is the current-state C64 mean of `ND - G`, added to the current G score. It is versioned with the corresponding C64 evidence. This is a read-only, time-varying diagnostic and not an independently trained arm.

## 6. Abbas-inspired external method adaptation

The external method is labeled an **Abbas-inspired online protocol adaptation**, not a faithful reproduction. The implementation adapts the paper's PatchCore-style normal-memory editing idea to frozen normalized DINO descriptors, the project's single-group query tape, and its queried-feedback firewall. It does not claim author-code equivalence. The original full-label error-pool protocol is not used.

Each E bank starts as an independent copy of the actual 8,192-prototype native fixed Support bank. It is mutable, and original Support prototypes are not protected from removal or capacity subsampling. The E1/E2 starting features, IDs, threshold evidence, and state digest must match. C64 features are cached read-only and never inserted.

For each image, each patch receives the mean of its (k_E = \min(9, |M_E^{eligible}|)) nearest Euclidean distances in L2-normalized 4096D space; the image score is the maximum across all 784 patch values. Similarity is computed with `sqrt(max(0, 2 - 2 h·m))`. Exact nearest-neighbor calculation is chunked across patches and prototypes; stable sorting resolves ties by bank order. Source-group-ineligible prototypes are excluded before scoring. The arm's own maximum-response patch is retained for FN surgery.

**FP expansion** occurs only for a queried, verified normal image that the arm itself classified as an FP. The nearest-reference 1-NN distance of each patch is compared with that arm's current median 1-NN distance across all cached C64 patch descriptors. Strictly novel patches are inserted with their Source Group digest, event position, and unique event/patch ID. Empty insertions are no-ops.

**FN surgery** occurs only in E2 for a queried, verified defect that E2 itself classified as normal. It removes up to five nearest source-eligible mutable prototypes to E2's own anomaly patch, selected by stable distance order and removed by prototype ID. A removal that would leave fewer than nine total prototypes is a recorded no-op. No defect prototype is inserted.

After insertions exceed K=9,216, the complete merged bank is uniformly subsampled without replacement. A local NumPy generator is keyed by category, event position, registered seed, and pre-update state digest; the arm name and process-global RNG state are excluded. Support and online prototype retention/removal counts are recorded. Capacity is not presented as equal total memory bytes across methods.

E1/E2 thresholds use their own empirical higher-q90 among the same 64 C64 images and the existing B3 current-state refresh with numerical movement cap 0.01. The contract reports this cap in each score's own units; it does not imply equal physical movement in native and external score spaces. E1/E2 share later FP edits only while their own pre-feedback predictions and histories make the insertion conditions identical. `HITL_FP_FN - HITL_FP` is the complete online effect of enabling FN surgery, including any downstream change in predictions and FP opportunities.

## 7. Calibration, seals, metrics, and resources

Native arms and SignedKNN initialize from registered native C64 q90 evidence. On every corresponding state change, C64 is rescored from the cached native feature/base-map data and R7 B3 is refreshed with its version and state digest. External arms use their own C64 score scale and state-consistent q90. C64 is never mixed into the TEST feedback stream.

Each category Stage A seal binds source commit, source fingerprints, Support/C64/JSON receipts, event/query identities, score and private-feedback checksums, final state digests/versions, and resource/timing telemetry. The global Stage A seal is written only after all four category seals validate. Historical parity is checked before Stage B against R9 G/Full C1 and R10 G/SignedKNN scores, event identities, query identities, Support registration and pre-event Greedy digests using R10 tolerances. Historical rows are parity references only and do not replace the new score trajectories.

Stage B is a fresh CPU-only process. It checks all four category seals and score checksums before projecting full TEST labels. It verifies queried private feedback against official image-level JSON labels, then reports per-category and category-equal macro AUROC, AP, BA/BER, FPR/FNR, confusion counts, complete 512-event worst BER windows, query count/composition, first legal defect-query position, and descriptive before/after-first-defect-query metrics. The pre-feedback interval includes the first defect-query event; the post-feedback interval starts at the next event because the current event's prediction precedes disclosure and update. Undefined rates remain null when their class denominator is zero. It reports the factorial, SignedKNN, external, sign-readout, and global-offset comparisons. The implementation provides paired Source Group cluster-bootstrap intervals resampling whole five-view groups; views are not treated as independent, and no GO/NO-GO rule is encoded.

Resource seals report the fixed reference, shared Greedy arrays, each C1 allocation, R10 FIFO arrays, both external banks and metadata, cached C64 feature bytes, peak CUDA allocated/reserved memory, peak host RSS, encoder and score time, C64 rescore call/time totals, novelty recalibration calls/time, and FN surgery/removal counts. Extra external capacity is not described as byte-matched to C1.

## 8. Implementation and execution limits

Historical implementation dependencies are imported without editing their files:

- R9: deterministic order/query mapping, common Greedy writer action, native C1 scorer, C1 state digest, C64 current-state scorer, and traced native update.
- R10: `SignedKNNFIFO`, original C2 dense-map adapter, and sealed R9/R10 parity tolerances.
- R7/R3/P0: frozen Support/C64/TEST identities, native feature extraction, indexed queried-label reader, score disclosure gate, B3 refresh, official Stage B label projection, and model/runtime receipts.

The implementation does not rerun historical R7/R9/R10 development experiments. It does not run Stage A/B, open confirmation labels or masks, tune on TEST scores, or claim any arm improved performance. The exact full-run command is shown for a future, separately authorized run and was **not executed**:

```powershell
python -m experiments.calibration.exp_b_signed_feedback `
  --full-run `
  --run-id expb_20261108_reviewed_source_v1 `
  --authorization-file private/experiment_b/full_run_authorization.json `
  --experiment-seed 0
```

The authorization JSON must contain `researcher`, `independent_review_status: PASS`, and `reviewed_source_commit_sha`, and bind the approved `source_commit_sha`, contract `protocol_sha256`, run ID, all four categories, `authorization_status: AUTHORIZE_FULL_REAL_IAD_STAGE_A_B`, `allow_full_run: true`, `stage_b_after_all_stage_a_seals: true`, `confirmation_data_access: false`, and `test_masks_access: false`. Existing output leaves are non-overwriting; failed runs remain incomplete and are not resumed automatically. Category children use the repository's existing R3 offline/CUDA environment policy and R1 Windows Job Object supervisor; Stage B cannot be launched directly and is available only as the supervised CPU child of the full-run command.

## 9. Source-review corrections

The local correctness repair imports the existing R7 module inside `_refresh_threshold()` and continues to call the unchanged `_refresh_b3()` algorithm and registered movement cap. FN surgery L2-normalizes query descriptors before ranking them against the normalized external memory, matching the external scorer's distance semantics. The synthetic regressions exercise an actual B3 state/version refresh and a deliberately non-unit FN query whose normalized Euclidean nearest IDs differ from the unnormalized ranking.
