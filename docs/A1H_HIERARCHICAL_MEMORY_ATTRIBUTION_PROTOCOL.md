# A1-H: Causal Attribution of Hierarchical Normal Memory under Sparse Feedback

**Status:** Eight-arm memory-only code + automated GPU screening runner. Offline CI PASS is not GPU validation.  
**Created:** 2026-10-26  
**Branch:** \`a1h-hierarchical-memory-attribution\`  
**Baseline:** A0/A1 strict-four-shot native 3-class × 2-seed experiment (immutable).  
**Roadmap:** [Route A/B](POST_REJECTION_RESEARCH_ROADMAP_A_B_2026-10-26.md).

## 1. Questions and boundary

**RQ-H1:** Does physically consolidated STM→LTM memory retain more discriminative information than an equal-total-budget flat Reservoir / Greedy Coreset?

**RQ-H2:** When does maturation actually promote recurrent verified-normal motifs into LTM, and does **LTM retrieval**, not simply a bookkeeping sidecar, change scoring?

**RQ-H3:** Does the optional v22 utility-aware retention add benefit beyond ordinary v22 STM under the same hierarchy settings?

The experiment isolates verified-normal **memory**. It deliberately **does not** activate the AIF controller, online threshold controller, Signed Correction, defect bank, negative-correction bank, pseudo-normal writes or any oracle defect mask. Do NOT call it the full journal manuscript architecture. No new model is trained; DINOv2 frozen features and P0 fixed-reference support construction are reused.

## 2. Eight arms and exact interventions

| Arm | Mutable memory mechanism | Allowed retrieval |
|---|---|---|
| \`static\` | None | Fixed only |
| \`reservoir\` | Patch-level Algorithm R, total dynamic prototype budget K | Fixed + flat |
| \`online_greedy_coreset\` | The A1 online pair-gap greedy coverage heuristic, capacity K | Fixed + flat |
| \`v22_normal_stm\` | Original v22 \`FixedCapacityCodebook.update\` ordinary novelty gates, capacity K | Fixed + flat |
| \`H1_track\` | STM K/2, real v22 Maturation Buffer K/4, reserved LTM K/4; tracking only | Fixed + STM |
| \`H2_shadow_promotion\` | H1 + **actual physical LTM promotion**, no LTM retrieval | Fixed + STM |
| \`H3_ltm_retrieval\` | H2 + real LTM retrieval | Fixed + STM + LTM |
| \`H4_utility\` | H3 + optional utility-aware STM retention | Fixed + STM + LTM |

This is a controlled additive intervention chain. **H1/H2** must have identical STM and scored outputs (promotion cannot change online scoring when retirement and LTM retrieval are disabled); **H2/H3** must have identical physical STM and LTM contents at snapshots (only retrieval differs). The driver FAILS if these assertions break.

STM retirement and provisional Maturation Buffer retrieval are off throughout the primary matrix. The Buffer's contents are tracked and counted toward capacity, but not used in score retrieval. The H2 shadow group is **a physically stored LTM**, not a hypothetical eligibility counter.

### Pre-registered capacity budgets

- K=128: flat=128, hierarchy=64 STM +32 LTM +32 Buffer.
- K=256: flat=256, hierarchy=128 STM +64 LTM +64 Buffer.
- **Measured:** actual occupied prototypes, feature bytes, Maturation metadata bytes, per-method update time, promotion triggers / attempts / successes / new LTM prototypes.
- Original fixed support bank is exactly shared and excluded from the *mutable* budget. v22 candidate buffers are actually reconstructed at K_buffer, because otherwise P0's default buffer preallocates 4096 slots and silently violates physical capacity. Disabled Flat buffers are reset to one unused bookkeeping slot.
- Flat vs hierarchical comparison matches total *feature-prototype capacity*, **not total bytes or compute**, because metadata and sorting are heterogeneous. Reported bytes/time must be considered with accuracy; never call the comparison bitwise-equal memory cost.

## 3. Frozen input and feedback

1. Four distinct random normal TRAIN originals per class/support seed; no extra normal image encoded to select them. All initial augmentations derive from those four only.
2. Identical P0 DINOv2/VisionAD-substrate fixed support bank per arm, with fixed-bank SHA verified.
3. Common 3% deterministic **label-blind** query tape, same positions and labels across arms. All eight score before the queried label becomes available. Only queried *verified normal* items are fed into mutable normal memory.
4. Identical candidate embeddings for all arms: top 8 per fused layer selected from an immutable Static Bank score, independent of policy-specific scores. Distinct-event recurrence counts separate queried images, not different patches from one image.
5. A deterministic, label-blind split removes **96 entire test images** from the online stream. The frozen encoder computes their features once, caches on CPU; repeated probe checkpoints use identical images and frozen features. Probe scores/labels are evaluation-only.
6. Stream order defaults Native. Optional Normal-first and Defect-first are deliberately label-derived offline stress interventions, not on-policy sequence access. Real-IAD has **no validated independent A→B→A factory operating-mode labels** in this experiment, so no genuine operating-mode recurrence benchmark is claimed.
7. The old A0 LOSO θ is recorded ONLY as a **diagnostic** fixed operating point. Previous A0 found \`mean+2.5 sd\` in a 3-shot LOSO fold distribution was often inappropriately high for four-shot deployment. BA/FNR/FPR are diagnostic and NOT claims of valid field calibration. Primary metrics are **probe AULC, end-of-probe AUROC, prequential AUROC and memory retention traces**.

## 4. Retrieval confounding control

### Primary: unified retrieval

All methods share **global top-5 cosine-distance retrieval** over \`fixed ∪ STM ∪ (LTM if enabled)\` with one feature fusion / map aggregation. This replaces the heterogeneous-k native v22 \`min(d_F,d_S,d_L)\` as the primary attribution scorer. No optimization/checkpoint is allowed to choose a better method-specific k.

### Orthogonal probe diagnostics

At each checkpoint, for H3 and H4, evaluate the **same frozen probe under identical memory state** with LTM enabled vs disabled; report \`probe_mean_ltm_score_delta\` and \`probe_any_ltm_score_effect\`. At final checkpoint additionally evaluate native v22's original heterogeneous-k retrieval for diagnostic context (not used online). This separates *retained memory content* from *retrieval-specific* gain.

Actual nonempty LTM is not itself a performance claim. The summary explicitly records \`ltm_activation\`. If LTM remains empty or rescoring it changes nothing, mark it **unactivated or ineffective**, not as evidence for hierarchy superiority.

## 5. Main endpoints and analysis

- **Primary:** Area Under fixed-Probe AUROC Learning Curve, using trapezoidal integration weighted by actual stream steps. Report H4−Greedy Coreset, H3−H2, H4−H3 with paired differences over every class/seed/budget.
- **Secondary:** final/initial/minimal fixed-Probe AUROC, pooled prequential AUROC (temporally mixed), normal-score drift and diagnostic FPR/FNR at the stale LOSO θ.
- **Mechanism:** genuine promotion rate, cross-event evidence, candidate counts, H2/H3 state invariance, LTM scoring delta, capacity occupancy, update wall-time, metadata/bytes.
- **Strict decision gate:** A hierarchically enabled method cannot be declared to fail or succeed when LTM was never triggered. A win in only the native heterogeneous-k scorer is a retrieval-design result rather than memory-governance improvement.
- **No premature journal claim:** three classes × two support seeds is a screen; add 30-class confirmation with a pre-registered seed plan and published external coreset/memory controls after selecting defensible hypothesis. The implemented online Greedy Coreset remains an approximation, not ContCore or FOADS.

## 6. User workflow: one command

From repository root on the new branch, with the original Real-IAD files and \`torch\` conda environment:

\`\`\`powershell
conda activate torch
python experiments/run_a1h_hierarchical_memory.py --all --dataset-root "G:/dataset/realiad" --math-sdpa
\`\`\`

One call performs:
- A tiny **CPU-only executable v22 MaturationBuffer mechanism preflight** with synthetic repeated events. It checks that distinct-event recurrence and promotion eligibility work and that one image cannot fake recurrence. This is a functionality test, not reported dataset performance.
- First 64-stream-image GPU smoke (identical eight-arm/strict4 code paths).
- **Twelve** full Real-IAD conditions: audiojack / rolled_strip_base / switch × support seed 0 / 1 × total dynamic K=128 / 256 × Native.
- Every condition writes row-level outputs and is strictly validated (supports, LOSO folds, full query tape vs rows, memory budgets, data split, state invariants).
- Aggregates JSON/CSV and stops at first failed condition with a preserved \`*.run.log\`. No manual per-condition commands.

### Optional after initial interpretation

For stress only (adds Normal-first, Defect-first, 36 total conditions, and resumes existing Native results):

\`\`\`powershell
python experiments/run_a1h_hierarchical_memory.py --all --dataset-root "G:/dataset/realiad" --math-sdpa --orders native,normal_first,defect_first --no-smoke
\`\`\`

For a smaller first screen the same entrypoint supports \`--classes audiojack --seeds 0 --budgets 128\`. These change the resulting planned matrix, and should be recorded as exploratory, not silently pooled as full evidence.

If interrupted, re-run the **same** command. Completed valid per-condition directories are rechecked and skipped. Incomplete result directories or leftover failed preflight logs are intentionally not overwritten; inspect error logs and recover by archiving the affected incomplete directory **without changing completed evidence**.

### Files

\`\`\`text
outputs/a1h_hierarchical_memory/
  mechanism_test.run.log
  mechanism_test.passed
  audiojack_s0_K128_native_smoke/
  audiojack_s0_K128_native_full/
    summary.json
    prequential.csv
    probe_scores.csv
    probe_summary.csv
    memory_snapshots.csv
  ...
  aggregate.json
  aggregate.csv
\`\`\`

Offline lightweight CI cannot run Real-IAD/DINOv2; the actual CPU mechanism preflight and GPU full-condition code require the user's local environment. The repository's existing A0/A1 experiment files, R6 results and historical v22 script are **not modified** by this branch.

## 7. Next scientific step after outcomes

We first inspect if LTM actually formed and influenced scores, then evaluate hierarchy vs strong flat baselines at paired budgets. Only then decide between completing Route A, commissioning a separate true normal operating-mode A→B→A benchmark, or investigating Route B counterfactual risk-governed memory. Do not alter thresholds, budgets, or maturation rules midway through the matrix to obtain desired results.
