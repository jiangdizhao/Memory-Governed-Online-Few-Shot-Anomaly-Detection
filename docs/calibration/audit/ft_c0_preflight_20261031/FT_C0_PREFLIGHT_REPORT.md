# FT-01 Closure and FT-C0 Preflight

**Date:** 2026-10-31

**Preflight status:** BLOCKED_INPUT

**Existing-frozen-receipt causal classification:** NOT_CAUSALLY_IDENTIFIABLE

**Recommendation:** NO-GO for replay from the currently certified receipts; HOLD any future run for input recovery, independent protocol review, and separate researcher authorization.

## Executive conclusion

FT-01 should be treated as closed for planning. Native queried-anchor rescoring was already performed. Its development-only Next-256 results are unfavorable for the particular query-to-threshold direction rule tested: the selected action had negative mean balanced-accuracy change, low unique-best direction agreement, and a high harmful-activation rate. This is evidence against that rule, not evidence that every sparse-feedback calibration method is impossible.

The legacy Signed Local Correction mechanism is a separate score modifier. It does more than move a threshold: its signed, outcome-weighted correction is added to selected patch scores before layer, view, and image aggregation. The source is available, but the current receipts do not establish a faithful, no-GPU replay on the frozen Greedy path. The P1 feature-cache audit says no persistent feature cache was built; P12 stopped with partial, non-finalized evidence; the final queried-label count and 20-trajectory parity were not established; and the historical Stage 0.75-A executed source bytes and query-response digests were not recovered. The frozen Greedy arm also explicitly disables correction.

Therefore the existing-receipt C0/C1 comparison is **NOT_CAUSALLY_IDENTIFIABLE** from the artifacts now certified. The overall preflight is **BLOCKED_INPUT** because the missing or unverified inputs could, in principle, be supplied through a separately authorized and preregistered study. This does not establish universal impossibility.

Do not attribute the historical M2-to-M3 result to correction. Its AUROC moved from 0.8885 to 0.9005 while realized query rates changed from 0.0832 to 0.0511 and downstream states diverged. That comparison does not isolate a score-only correction effect.

## 1. Workspace and scope record

- Repository: E:/Memory-Governed-Online-Few-Shot-Anomaly-Detection
- Branch: calibration-stage0-contract-20261029
- Parent HEAD: 7ed99d199f1a8345440aa62564ecdf2340cdd6bd
- Tracked worktree: clean before this report.
- The report target directory did not exist before this task.
- Twenty-nine existing untracked outputs/calibration directories were identified and left untouched:

  - b3_state_lag_formal_v1_20261030
  - queried_anchor_identifiability_v1_20261030
  - queried_anchor_identifiability_v1_postmeasurement_gate_fix_tests_20261030
  - queried_anchor_identifiability_v1_repair1_20261030
  - queried_anchor_identifiability_v1_repair2_20261030
  - stage0
  - stage05_exploratory_20261029
  - stage075a_b3_headroom_v1_20261030
  - stage075a_b3_headroom_v1_20261030_verified2
  - stage075a_greedy_ranking
  - stage075a_greedy_ranking_repaired_20261029
  - stage075a_greedy_ranking_repaired2_20261029
  - stage075a_greedy_ranking_repaired3_20261029
  - stage075b_synthetic_20261029
  - stage075b_synthetic_v3_20261029
  - stage075b_synthetic_v3_blocked_20261029
  - stage075b_synthetic_v3_blocked_final_20261029
  - stage075b_synthetic_v3_final_20261029
  - stage1a_formal_controller_comparison_v1_20261030
  - stage1a_formal_controller_comparison_v2_20261030
  - stage1a_formal_controller_comparison_v3_20261030
  - stage1a_formal_controller_comparison_v4_20261030
  - stage1a_p12_low_memory_recovery_v1_20261030
  - stage1a_p12_low_memory_recovery_v2_20261030
  - stage1a_p12_low_memory_recovery_v3_20261030
  - stage1a_p12_native_replay_v1_standing_authorization_01_20261030
  - stage1a_p12_native_replay_v1_standing_authorization_02_20261030
  - stage1a_p12_native_replay_v1_standing_authorization_04_20261030
  - stage1a_p12_native_replay_v1_standing_authorization_05_20261030

This was a documentation-only audit. No images, dataset files, raw queried-label values, new official labels, masks, confirmation classes, label provider, GPU, or scorer were accessed. No experiment or test was run. Private checkpoints and prediction payloads were not opened.

## 2. FT-01 historical evidence reconciliation

The committed Queried-Anchor reports record:

| Evidence | Result | Interpretation |
| --- | ---: | --- |
| Snapshots with both normal and defect queried evidence | 90/120 | Query-anchor coverage, not representative population coverage. |
| Next-256 selected-action equal-class mean ΔBA | −0.00435525 | Retrospective development outcome; not a prospective online treatment effect. |
| Next-256 unique-best direction agreement | 0.3625 | Weak support for choosing the best threshold direction. |
| Next-256 harmful / false-activation rate | 0.585317 | Harmful actions occurred frequently among evaluated active actions. |
| Next-256 class means | audiojack −0.006556; regulator −0.007044; switch −0.001979; rolled_strip_base −0.000735 | All four category means are negative. |

The class values above are the **Next-256 selected-action means**. The decision-candidate report also includes all-remaining cohort means, which differ for regulator and rolled_strip_base; those cohorts must not be mixed.

Native Query-Anchor Rescoring had already run across 20 trajectories and 120 snapshots. The report records 2,450 primary normal leave-source-out scores, 2,645 current-memory defect scores, and 15,445 total native score calls. It used frozen snapshot memories and made no Greedy writes. This was rescoring of the old evidence states, not a new FT-01 experiment.

The Actual-B3 package link remains blocked. The audit records a checksums.json SHA-256 mismatch and a file-count mismatch: five files in the completion manifest versus four current checksum entries. The gate matrix marks G2 BLOCKED, requires researcher approval, and makes no scientific validation claim.

These results disfavor the **specific tested local query-to-threshold direction rule** and support closing FT-01 for planning. They do not falsify every sparse-feedback calibration method, establish an IID or population-risk result, or show that all feedback-based methods must fail. The snapshots and future windows overlap, and all four categories were development data.

**Evidence:** docs/calibration/QUERIED_ANCHOR_IDENTIFIABILITY_RESULTS.md; docs/calibration/QUERIED_ANCHOR_IDENTIFIABILITY_DECISION_CANDIDATE.md; docs/calibration/QUERIED_ANCHOR_B8_RESEARCH_IMPLICATIONS.md; docs/calibration/QUERIED_ANCHOR_NATIVE_RESCORING_REPORT.md; docs/calibration/audit/queried_anchor_identifiability_v1/provenance.json; docs/calibration/audit/queried_anchor_identifiability_v1/gate_matrix.json.

## 3. Legacy correction source and function map

The inspected current source is test_phase3_AIF_simplified_v22.py, SHA-256 BFC871D5A9A32B2F5BDB9D7AAB5177E702DEC2E1110A6632F20BD13C20FCBA30. This identifies current checkout bytes; it does not recover the exact bytes used by an older historical run.

| Mechanism | Source location | Finding |
| --- | --- | --- |
| Configuration | CorrectionConfig, lines 2536–2583 | Default capacity 256 per class/view/layer; correction weight 1; beta scale 0.5, clipped to [0.02, 0.20]; radius0 0.10, clamped to [0.05, 0.25]; kappa0 1; merge distance 0.05; sign-dependent dN store gates 0.10 and 0.12; keep_defect_bank defaults false. Current Greedy separately disables the defect bank. |
| Stored state and application | SignedCorrectionCodebook, lines 2599–2780; apply at about line 2688 | Stores normalized prototypes, beta, radius, kappa, counts, outcome counters, and last step. It does not store source image/group ID, feature-cache key, source-map ID, or memory-version digest. |
| Coefficient construction | VisionADMemory.update_corrections_from_topk, lines 4496–4590 | Uses queried label sign, predicted outcome, score, theta, optional strength_scale, and selected patch dN values. Base magnitude is clipped beta_scale × abs(score − theta); outcome weights are FP 1.30, FN 1.20, TN 0.60, TP 0.80. Patch multipliers are clipped to [0.75, 1.25]. |
| Anchor selection and retention | Same update function; SignedCorrectionCodebook.update | Retains selected per-layer patch features that pass sign-dependent dN gates; merges close same-sign anchors and may replace the weakest retained anchor when capacity is full. |
| Dynamic write strength | compute_correction_strength, about line 5786; online caller about lines 9760–9805 | A caller path scales writes using score-to-threshold distance, normal-score spread, novelty, ReasoningBank error, and threshold uncertainty; FP/FN may receive an additional mistake bonus. Other legacy call paths pass strength_scale=1.0. The exact historical path and config must be pinned before calling a comparison faithful. |
| Patch scoring | VisionADMemory.score_single_view, lines 3918–3990 | Selects top-k patches by normal-bank distance dN, then adds correction only to those selected patch scores. |
| Aggregation | score_sample_multiview, lines 4033–4160 | Default layer fusion is mean; views are aligned and fused; default image aggregation is top-k mean. The C0 and C1 path must preserve the exact frozen profile and map aggregation. |

The caller sets correction sign positive for a defect label and negative for a normal label. Thus a queried false negative can write positive defect evidence and a queried false positive can write negative normal evidence. Correctly classified queried TP/TN observations can also write, with lower default outcome weights. Coefficients therefore depend on label sign, outcome, image score relative to threshold, and—in one path—other uncertainty/novelty terms. This is not a fixed label-only residual.

The normalized RBF application is:

Δ_C(h) = Σ_j w_j(h) β_j / (Σ_j w_j(h) + ε)

with cosine distance d_j = 1 − cos(h,p_j) and w_j = κ_j exp(−d_j²/(2r_j²)). The correction is added to patch scores before the unchanged score-map aggregation.

Current Greedy memory source is experiments/v02_memory.py, SHA-256 3C4E247E3A41DC9ACC8147A35AA892CF3F3571ACA85BEB4169809BD70AD48AE0. MemoryArm explicitly sets corr_cfg.enabled=False and disables the defect bank. experiments/risk_safe/native_bridge.py, SHA-256 9575214EA8D14EBE32A537FA15AFA332446B41D66C1C8C85F890D2251EAEF8F7, installs a Greedy state into a native clone and scores through experiments/a0_a1_strict4_matched.py:score_bank. That source uses the native multiview scoring API and native score map.

The original correction bank is not a portable state blob for this Greedy trajectory. It was part of the legacy VisionADMemory correction codebook and its dN gates inspect that bank’s fixed/dynamic normal state and optional legacy stores. Greedy provides a different dynamic-memory policy and current memory history. The same DINO features do not establish equivalent source rows, memory state, or correction anchors. A new corrector conditioned on Greedy states would be an adapted candidate, not evidence that original v22 correction states transfer unchanged.

## 4. Frozen artifact availability and integrity

| Required input | Available evidence | Status for FT-C0 |
| --- | --- | --- |
| Frozen C0 event run | outputs/calibration/stage1a_formal_controller_comparison_v4_20261030/run_manifest.json records 20 trajectories, 76,475 events, query-tape binding dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183, and COMPLETE prediction phase. Its actual_head is 5546f037834b780eb71019a70655396499bb7800. outcome_label_vectors_saved is false. | Useful baseline receipt, but not a patch-feature replay cache. It predates the current parent HEAD. |
| Stage 1-A source and feedback lineage | The v4 preflight says exact historical executed source bytes were not recovered, historical G2 is not closed, and query-response digests were not recovered from historical G2. The P12 source-provenance manifest also marks historical source-byte parity UNRESOLVED; its recorded runner-entry hash is bac1698c4f31ca0ee987063ab56015b36c572f9da5d1f1aa3b33ebc85f21caf3. | Historical replay identity is unresolved. Current source hashes cannot substitute for the missing historical execution identity. |
| Full event patch features | docs/calibration/audit/stage1a_p1_reconstruction_v1_20261030/feature_cache_provenance.json says NOT_BUILT_NO_AUTHORIZATION, feature_cache_created=false, zero encodings, zero images read, and zero cache entries. P12 used feature bundles during execution, but no finalized all-event cache manifest is in the audit package. | Not certified for no-GPU correction scoring. Without a verified reusable feature bundle for every event, C1 requires new image encoding or a separately verified cache source. |
| Queried labels and response history | The P12 report records 345 durable queried-label reads before its last durable checkpoint; final queried-label count is explicitly unverified after runtime stop. The response history is not certified as complete in the current audit package. The old Stage 0.75-A trace did not persist disclosed label values or a separate historical label-response digest. | Insufficient to reconstruct a complete legitimate anchor sequence from certified receipts. No label values were read in this audit. |
| Greedy memory trajectory | Formal comparison receipts are checksummed and the source emits before/after memory versions. However, the P12 audit marks exact event parity for all 20 trajectories NOT_ESTABLISHED and durable private trace artifact count 0. Its private checkpoint is present but not committed or part of the audit package. | Partial provenance is not a complete, independently verified per-event memory replay. The private checkpoint was not opened. |
| Correction anchor ledger | No current Greedy correction bank was enabled. The legacy codebook does not retain image/group provenance or memory-version digests per anchor. | No source-aware correction history can be verified or restored from current receipts. |
| Native anchor measurements | docs/calibration/QUERIED_ANCHOR_NATIVE_RESCORING_REPORT.md records 386 newly encoded queried images and native scores against old frozen snapshot memories. The report explicitly says there were no Greedy writes. | Does not supply all-event C1 patch features or a correction bank generated along the current Greedy path. |
| Actual-B3 package | Queried-anchor provenance records checksum SHA mismatch and file-count mismatch, despite replay identity checks. | G2 remains BLOCKED; do not treat this as repaired or closed. |

The P12 run report records a runtime stop after partial work, with 15 trajectories durably complete, a subsequent trajectory lacking a durable progress update, and four trajectories not started. Its last durable counters were 345 feature encodings and 60,631 anchor-score calls; these are explicitly not final totals. The latest private checkpoint is 7,359,652,922 bytes and is not committed. The audit records historical source-byte parity unresolved and exact native event parity across all 20 trajectories not established. These files were not opened or used as a backfill.

Consequently, current feature-cache and memory-version receipts do **not** support a certified no-GPU C1 replay. The existence of a large private checkpoint or a cache-use count in another report does not certify an all-event patch-feature cache with the required encoder, preprocessing, memory, source, and score-map identities.

## 5. Causal identifiability assessment

For the **existing frozen C0 receipt**, classify the proposed replay as **NOT_CAUSALLY_IDENTIFIABLE**. The required valid Correction Anchor history cannot be reconstructed from the shared Greedy trajectory using the presently certified inputs: all-event patch features are not in a verified cache; the queried feedback trace is incomplete/unverified; full event memory parity is not established; exact historical execution bytes are absent; and no per-anchor source/version ledger exists. This finding is limited to this frozen receipt package. It is not a theorem that a future, newly authorized experiment cannot identify a correction effect.

The design could be made identifiable in a future study if it has complete immutable inputs and freezes every non-correction path:

1. Start each paired event from the same fixed normal bank, Greedy memory version/digest, and prior correction state.
2. Use the same image feature bundle, native score function, selected patch indices, score aggregation, stream order, and externally frozen Query Tape in C0 and C1.
3. Produce both scores and predictions before revealing any queried label. Unqueried labels never enter controller or corrector state.
4. Reveal only the verified queried label after the prediction. Update C1 correction state only then, using the preregistered outcome, score, threshold, and anchor rule.
5. Freeze Greedy memory updates and query decisions to the common C0 writer inputs and tape. Assert identical memory digests at every event. Corrections may affect future C1 scores but may not alter query selection, Greedy admission, or memory candidates.
6. Use the identical, frozen calibration algorithm and initialization in both arms. Do not tune thresholds on the metric evaluator’s labels. Seal both score streams before offline AUROC/AUPRC evaluation.

This is a score-modifier treatment with a causally updated corrector. If C1 is allowed to use its own previous corrected predictions to select later correction outcomes, that feedback is part of the preregistered C1 treatment; query and Greedy memory paths still must remain fixed. A baseline-outcome-frozen correction history is another possible, open-loop adapted algorithm and must be named separately.

### Required source and self-match identities

Before any future run, each anchor/event needs an immutable event ID, class, official sample-group ID, query decision/reveal time, verified queried label, feature reference and image-content SHA, encoder/checkpoint/DINO-source/preprocessing fingerprint, source-map SHA, native scorer/config SHA, Greedy memory version and digest, and selected patch IDs. The correction ledger must retain source IDs and the complete codebook state after each update.

For scoring an anchor against normal support or Greedy memory, exclude its complete official source group from every layer and verify zero source rows remain. Deduplicate queried anchors by official group; reject support/C64 overlap. Defect anchors must not enter the normal Greedy writer. The existing P1 anchor manifest specifies image, encoder, preprocessing, source-group and memory-content keys and all-layer leave-source-out rules; the legacy correction codebook itself does not record these identities.

### Historical versus adapted algorithm

The historical M2-to-M3 comparison is confounded by changed realized query rates (0.0832 versus 0.0511) and divergent writer/threshold states. It cannot identify correction’s independent contribution. The old Stage 0.75-A trace also lacks the exact disclosed label sequence and original executed source bytes. Enabling the signed corrector on the current Greedy trajectory would create a **new Greedy-conditioned correction candidate**. It must not be presented as exact historical v22 correction ON/OFF evidence or as a transferred v22 correction state.

## 6. Mathematical locality assessment

The normalized RBF expression does not provide a finite neighborhood in which correction is guaranteed to act and outside which it is guaranteed to vanish. When anchors are similarly distant and total weight remains large relative to ε, the numerator and denominator shrink together and the normalized value can remain close to a weighted average of anchor coefficients. As total weight eventually falls below ε, the correction tends toward zero, but the formula gives no useful distance threshold or bound for that transition. The Gaussian tails have global support, and the largest-radius anchors can dominate the mixture.

The implementation’s radius defaults to 0.10 and is clamped to 0.05–0.25 cosine-distance units. Correction is computed only on the current top-k dN patches, which limits where the patch operation is applied on an image; it does not prove that selected patches are semantically local to the queried source or that image-level changes remain local after map aggregation.

Required diagnostics for a future preregistered run:

- Residual magnitude and sign versus nearest-anchor cosine distance, reported by class, anchor sign, and memory version.
- Far-field residual quantiles and maximum absolute residual over prespecified distance bins, including the weight-sum-to-ε ratio.
- Per-patch change maps, fraction of patches changed, affected-patch neighborhood overlap, and spillover outside source/group neighborhoods.
- Image-score change and ranking change versus nearest-anchor distance; correction activation rate; score-distribution shifts and upper-tail changes.
- AUROC/AUPRC and BA/FPR/FNR stratified by anchor proximity, with class-by-seed reporting.
- Source self-match and group leakage checks; recompute anchor-local metrics under all-layer source-group exclusion.
- Off-neighborhood degradation checks and a preregistered definition of the anchor neighborhood.

Do not change the RBF, normalization, epsilon, or radius in a purported faithful test. Any compact-support kernel, unnormalized residual, or other locality fix is a different algorithm requiring its own preregistration and experiment.

## 7. Minimum future FT-C0 matched-tape specification

This is a proposal only; it does not authorize implementation or data access.

### Arms and fixed factors

- **C0:** frozen Native DINOv2/VisionAD scorer, immutable fixed normal bank, and the frozen verified Greedy normal-memory history with correction disabled.
- **C1:** identical fixed bank, feature bundles, scorer/profile, Greedy memory history, Query Tape, stream, and calibration rule, with only a causally updated signed correction patch modifier enabled.
- **Strong simple comparator:** a fixed label-conditioned nearest-prototype residual using the same legitimately queried normal and defect anchors, for example λ × [d(q, nearest normal anchor) − d(q, nearest defect anchor)]. Fix λ before evaluation; use source-group leave-one-out and no extra feedback or memory writes. This baseline tests whether a simple label-conditioned scorer explains the gain without the legacy outcome weighting, retention, or RBF codebook.
- **Negative controls:** correction disabled/zero beta; a preregistered anchor-sign permutation or label-shuffled anchor assignment preserving event order and query count; and, if compute allows, a random source-matched anchor control. Controls must be described before any evaluator labels are opened.

### Causal event and state order

For each event, first score from its frozen features and the common previous Greedy state. C0 emits the native score; C1 applies only the previous-state correction to the same native patch map and aggregates it through the same frozen map path. Emit predictions before disclosure. Apply the externally fixed query decision. If queried, reveal the verified label after prediction; update the C1 corrector using the preregistered outcome and coefficient rule for future events. Update the Greedy bank once from common C0 writer inputs and the verified-normal rule, regardless of correction score. Confirm C0 and C1 Greedy state digests, query positions, label history, and event order remain identical.

If a threshold or writer variable can change a query or memory update, it must be frozen to the common C0 path for this score-isolated experiment. The calibration algorithm and initial state must be identical; any score-dependent threshold state is reported as a downstream result and cannot feed back into query or Greedy memory. All correction updates occur after the current prediction and queried-label reveal.

### Scoring, metrics, and evaluation

Require exact C0 score and map parity with the frozen native scorer when correction is disabled. In C1, insert the correction at the same patch-score point as the legacy source and preserve top-k selection, layer fusion, view alignment/fusion, and image aggregation. Verify no correction changes C0’s selected Greedy writer candidates.

Primary metric: image-level AUROC. Additional ranking metric: AUPRC. Secondary metrics: BA, FPR, FNR, correction activation and magnitude, patch-level spillover, score-distribution change, compute and memory overhead, and category-by-support-seed stability. Report the four categories separately and paired seed results; support seeds are not independent categories. The offline evaluator may access only the specifically preregistered permitted development labels after score streams are sealed. No evaluator labels enter either arm. No threshold, downside tolerance, or effect-size cutoff may be chosen from the reported development results; preregister these before a separately authorized evaluation.

### Input gates and stop conditions

Required before protocol review: complete image-feature bundles with exact content/encoder/preprocessing keys, complete query responses from the frozen tape, all-event Greedy state and writer receipts, source-group IDs, score maps or sufficient patch features, frozen scorer and correction source/config hashes, complete artifact checksums, source-exclusion proofs, and an independent evaluator plan. Recovering missing private payloads is not authorized by this audit.

Stop with BLOCKED_INPUT if any of those inputs is missing, incomplete, or unverifiable. Stop if correction changes query positions, label disclosure order, Greedy memory digests, writer candidates, or C0 parity. Stop on any support/C64 overlap, source-group self-match, duplicated anchor group, missing class evidence, or unresolved source identity. No algorithm or parameter edits may be triggered by test outcomes.

## 8. Resource estimate and value for a rapid paper

The Stage 1-A v4 manifest records 76,475 event scores across 20 trajectories. As a rough feature-work bound, that is up to 76,475 event bundles if no cross-seed reuse is possible; if all five support seeds share each category stream and feature keys match exactly, it would be about 15,295 unique stream-image bundles. The latter is an assumption-based estimate, not a verified cache count. Current P1 provenance reports zero persistent feature-cache entries. Therefore a no-GPU estimate is not supportable from current records; absent a recovered, fully hashed feature cache, full event features would need to be generated under separate authorization.

With features available, C1 adds correction work on selected patches: approximately O(events × views × layers × selected patches × active correction anchors × feature dimension), with at most 256 anchors per class/view/layer under the legacy default. It also requires per-event corrector state and parity receipts. The P12 partial run’s 345 encodings and 60,631 anchor-score calls, and the earlier 811-second native rescore, are measured workloads from different tasks; neither is a valid FT-C0 runtime benchmark.

Planning estimate only: roughly 2–4 developer-days for an isolated adapter, artifact reader, deterministic C0 parity and state checks, correction ledger, and paired analysis once all inputs are verified. Input recovery or new feature encoding could dominate and cannot be estimated reliably from the available manifests.

Given the rapid-submission objective, do not spend that effort on the current incomplete receipt set. Local Correction has a falsifiable ranking hypothesis, but the present evidence does not support faithful historical attribution or no-GPU replay. A future study is worth considering only if a complete, provenance-rich feature/feedback/state package can be recovered or newly collected under separate authorization, and the simple nearest-prototype baseline is included from the start.

## 9. Decision

**Preflight status: BLOCKED_INPUT.**

**Existing frozen trajectory replay: NOT_CAUSALLY_IDENTIFIABLE from certified receipts.**

**Scientific recommendation: NO-GO for FT-C0 using the current receipts; HOLD any new run pending independent protocol review and explicit researcher authorization.**

FT-01 is closed for planning and must not be rerun. No conclusion here authorizes FT-C0, FT-Q0, FT-Q1, FT-02, FT-H0, new labels, GPU work, or confirmation-class access.

## 10. Read-only evidence and command record

Reviewed the six task-named calibration documents and the focused source functions listed above. Read small JSON provenance, gate, run, feature-cache, and checkpoint-verification manifests only. Inspected private checkpoint filenames and file sizes as metadata only; did not open their payloads or any large prediction files.

Read-only command families used: git branch/status/rev-parse and untracked path listing; rg searches on named source and audit paths; bounded PowerShell source/report ranges; JSON parsing of named small manifests; Get-FileHash on the six named source files; Test-Path for the new report directory. No cleanup, branch switch, worktree creation, source edit, dataset access, experiment, scorer, GPU, or test command was run.

Key source identity fingerprints for the current checkout:

- test_phase3_AIF_simplified_v22.py: BFC871D5A9A32B2F5BDB9D7AAB5177E702DEC2E1110A6632F20BD13C20FCBA30
- experiments/v02_memory.py: 3C4E247E3A41DC9ACC8147A35AA892CF3F3571ACA85BEB4169809BD70AD48AE0
- experiments/risk_safe/native_bridge.py: 9575214EA8D14EBE32A537FA15AFA332446B41D66C1C8C85F890D2251EAEF8F7
- experiments/a0_a1_strict4_matched.py: E916659681607BF75E097DD62379DCC363CCC3ADF6F991D4108592997E1B53BF
- experiments/calibration/stage1a_p12_runner.py: 8AF1AB63530FD50D205321870A54C786110596241CB54839E6AA3C03D32A0979
- experiments/calibration/stage1a_formal_comparison.py: A9391AB94E9A994198484CD3FB5EFC7F534B3515C870723C141C3417EA717C84

Historical reports, source manifests, and incomplete private runtime receipts must remain distinct. The current hashes above are not substitutes for missing historical execution hashes or unverified query-response and feature payloads.
