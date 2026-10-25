# A0/A1: Strict-Four-Shot, LOSO Calibration, Matched-Feedback Memory Baselines

**Status:** Implemented screening infrastructure; GPU metrics NOT YET PRODUCED / NEVER FABRICATE.  
**Parent:** Post-rejection [Routes A/B roadmap](POST_REJECTION_RESEARCH_ROADMAP_A_B_2026-10-26.md).  
**Branch:** \`a0-a1-strict4-memory-matched\`.  
**Scope:** A0 calibration repair + A1 *memory-only* causal screen; NOT full v22, NOT a publication-level external baseline comparison.

## 1. Research rationale

The reviewer questioned both support-pool access and self-nearest-neighbor calibration. R6 further showed that adding STM can improve AUROC but worsen fixed-threshold BA. A0/A1 therefore separates memory-adaptation effects from threshold tricks under an explicitly controlled training and evaluation budget.

Our target question is **whether a specific verified-normal retention mechanism improves score discrimination compared to FIFO, reservoir and online greedy coverage when support, queried labels, image encoder, patch selection, mutable capacity and numeric threshold are matched**.

## 2. Frozen primary protocol (no performance-tuned changes)

- **Dataset:** Real-IAD; default screening classes \`audiojack,rolled_strip_base,switch\`; support seeds \`0,1\`; Native stream. \`--with-stress\` optionally adds Normal-first/Defect-first for the same class/seed. Stream rearrangement uses offline test labels ONLY to construct diagnostic stress streams.
- **Four-shot:** exactly four distinct normal *training* source images, sampled with a seeded Python \`random.sample\` from metadata-indexed eligible normal paths. No nonselected normal images decoded, ranked or embedded for support selection. Allowed augmented views are derived only from these four. Model weights are frozen.
- **Fixed bank:** same P0/v22 scoring substrate DINOv2 (defaults inherited from P0), DINO multi-layer frozen features, source-image augmentations, fixed patch-bank limits and scoring options across arms. This is not claimed bitwise identical to the historical original v22 initializer: the P0 strict4 constructor is reused for internal matched comparisons.
- **LOSO threshold:** for each of the four support originals, build a separate bank from the other *three* originals including only their allowed augmentations. Score the completely excluded held-out original against that fold bank. Compute \`theta0=mean(LOSO scores)+2.5*sd(LOSO scores)\`. Numerical \`theta0\` is exactly shared and **frozen for all arms**. Store fold support IDs, fold hashes, four LOSO scores and diagnostic self-match scores. **Limitation:** LOSO fold banks contain 3 originals whereas deployment uses 4; this may bias the operating point. The small-sample normal-only calibration does NOT guarantee FPR control.
- **No hidden extra normal pool:** the whole training class can be indexed for paths, but only four selected images are decoded for support selection and calibration. No disjoint commissioning images are used in this primary regime. If independent commissioning data are later introduced, call it a separate labeled data-access regime.
- **Held-out probe:** deterministic label-blind selection of 32 test images *before stream replay*, entirely removed from the query/learning stream. Probe images are only scored at t=0, every 512 stream images, and at stream end. Their labels are used solely to compute offline evaluation metrics. Neither labels nor image features are allowed into support, feedback tape, memory, candidate selection or threshold update.
- **Test stream:** remaining images shuffled by fixed stream_seed=0, then optionally stress-reordered. Default max_samples=0 evaluates all available non-probe images. Smoke takes first 64 of the post-probe stream; all metrics should be labeled smoke/diagnostic.
- **Fixed feedback:** label-blind, deterministic seeded 3% query tape via existing \`e02_matched_feedback.make_query_tape\`, query seed=37. Exactly the same queried positions and labels for every arm in one condition. Each current image is scored by all arms **before** its own queried label is revealed. Only confirmed normal queries provide mutable normal-bank candidate patches.
- **Shared candidate extraction:** top-8 anomaly-ranked patch IDs *per feature layer* from a shared static fixed-bank score of that confirmed normal image. Every dynamic arm receives precisely these candidate embeddings. This isolates retention rather than confounding it with method-specific patch selection; unlike legacy v22 it is not a full architecture clone. No defect feature bank, Signed Correction, LTM promotion, maturation retrieval, pseudo-normal update or AIF query controller.
- **Identical maximum mutable capacity:** v22 P0 \`K_normal_dyn=1024\` per layer/view. Audit actual occupied prototypes (methods may differ in occupancy due to novelty gates). If comparing engineering memory costs, account for metadata and the online coreset's ephemeral O(K²) distance computation.
- **Decision endpoint:** exactly the same numeric fixed LOSO \`theta0\` for all methods and all time steps within a condition. Threshold updates are purposely disabled. Differences in BA, FPR and FNR arise from memory-score changes only; they must not be attributed to Sidecar.
- **Ranking endpoint:** per-arm prequential AUROC and fixed-probe snapshot AUROC. Pooled prequential AUROC may be temporally confounded when scoring memory changes; the fixed held-out probe repeatedly evaluates the *same* images at each checkpoint.
- **No test-label tuning:** labels only used for post-prediction queried feedback and offline evaluation; never choose hyperparameters from the result itself.

## 3. Memory arms (all score through the same PatchVMB and fixed bank)

| Name | Dynamic normal memory | Notes |
|---|---|---|
| \`static\` | Never writes | VisionAD-like fixed reference, same score/threshold substrate |
| \`fifo\` | Patch-level circular FIFO | Appends until full, then replaces oldest slot |
| \`reservoir\` | Algorithm R | Uniform reservoir over all admitted **candidate patch events**, seed-controlled per layer |
| \`online_greedy_coreset\` | Online min-redundancy swap | Append until full; when full compare candidate nearest distance against the smallest existing within-bank nearest-neighbor gap; replace most redundant prototype only when candidate improves coverage. **Heuristic**, not exact offline coreset or published ContCore reproduction |
| \`v22_utility_stm\` | v22 \`FixedCapacityCodebook.update\`, \`use_utility_retention=True\` | Original local utility/recency/repeated-normal-aware update primitive with novelty gates and metadata; **isolated component**, not original complete STM/LTM |

Important limits: FIFO/reservoir fill memory even with redundant examples; v22 method novelty-gates, so *same max capacity* does not mean *same occupancy or number of admitted patches*. Treat occupancy and update time as outcomes. The special \`v22_utility_stm\` arm tests only one candidate mechanism from v22, **not** the full memory-governed BEA paper's main claim.

## 4. One-command Windows/PowerShell procedure

From repository root (after synchronizing this branch), within \`torch\` conda environment:

### 4.1 Local preflight (no GPU data required)

    python -m py_compile experiments/a0_a1_memory_methods.py experiments/a0_a1_strict4_matched.py experiments/run_a0_a1_validation.py
    python experiments/run_a0_a1_validation.py --self-test
    python -m unittest discover -s tests -p "test_a0_a1_strict4_memory.py" -v

### 4.2 ONE command: smoke + ALL 6 screening experiments + automatic summary

    python experiments/run_a0_a1_validation.py --all --dataset-root "G:/dataset/realiad" --math-sdpa

Default builds a 64-image GPU smoke first, then runs (audiojack, rolled_strip_base, switch) × (support seeds 0,1) × (Native) automatically, stopping at first failed command with a logged traceback. No per-experiment commands are needed.

**Optional later stress matrix**, 18 total conditions; reuse validated Native outputs:

    python experiments/run_a0_a1_validation.py --all --dataset-root "G:/dataset/realiad" --math-sdpa --with-stress --no-smoke

Specify a smaller screening selection or larger campaign with:
\`--classes audiojack,switch --seeds 0,1\`. Actual default matrix is for initial *feasibility screening*, not a 30-class publication claim. Class order in \`--classes\` is execution order.

### 4.3 Resume and validation

The same \`--all\` command skips and fully **revalidates** already completed conditions. Existing incomplete or invalid directories block the run to protect evidence. Inspect partial outputs and the \`*.run.log\` sidecar before any manual cleanup; no silent overwrite.

After uploading outcomes by your normal Git workflow, the assistant reads the remote output directly.

Offline validation of already completed directory (no GPU runs):

    python experiments/run_a0_a1_validation.py --validate

## 5. Output tree and interpretation

\`\`\`text
outputs/a0_a1_strict4_memory/
  audiojack_s0_native_smoke/summary.json
  audiojack_s0_native_smoke/{prequential,probe_scores,probe_summary}.csv
  audiojack_s0_native_smoke.run.log
  audiojack_s0_native_full/...
  audiojack_s1_native_full/...
  rolled_strip_base_s0_native_full/...
  rolled_strip_base_s1_native_full/...
  switch_s0_native_full/...
  switch_s1_native_full/...
  aggregate.json
  aggregate.csv
\`\`\`

Every \`summary.json\` contains exact supports, test/probe hashes, four LOSO fold lineages, both LOSO and diagnostic self-match threshold, exact queried indexes, immutable fixed Bank profile and per-arm AUROC/BA/FPR/FNR, actual prototype count and writes. \`prequential.csv\` records all arms' same-step scores/predictions before feedback; probe score tables track memory-state discrimination separately.

The aggregator validates schema, class/seed/order, four supports and LOSO disjointness, probe/stream separation, query counts/tape, fixed threshold, per-layer capacities, no static memory writes, full CSV length/hash and declared query fraction.

## 6. Pre-registered analysis and gates

1. First inspect **static self-match diagnostic theta vs LOSO theta**. A large change warns against using original self-matched static classification as the claimed baseline.
2. Compare AUROC and *same-probe snapshot* metrics; distinguish normal/defect score changes. Do not conflate time-varying pooled AUROC with calibrated classification.
3. Compare v22 retention to FIFO, reservoir and online greedy **at the same label tape**; report occupancy differences explicitly.
4. Compare BA/FPR/FNR under shared frozen numerical theta; a lower FPR accompanied by a higher FNR is a trade-off, not necessarily a gain.
5. For any strong outcome, test additional classes/seeds and strengthen greedy coreset with established implementation. Do not make journal SOTA claims from the three-class screen.
6. If existing v22 utility retention does not beat strong simple methods, do not call its complexity a justified independent contribution. Consider Route B only after documenting this result.

**Remaining A1 requirements NOT YET IMPLEMENTED:** comprehensive full STM/LTM/maturation comparison; exact implementation of named external online AD methods; full four-arm static-memory × Balanced Rescue 2×2 under fair replay; statistically powered 30-class multi-seed testing; fair extra commissioning calibration; independent pixel-level metrics; runtime instrumentation of every GPU operation. These are subsequent gates, not claimed completed.

## 7. Known engineering constraints

- Requires the project's existing Python GPU environment (Torch, torchvision, SciPy, NumPy, scikit-learn, DINOv2). GitHub CI only checks offline protocol/tests, **not Real-IAD GPU execution**.
- Requires Real-IAD at \`G:/dataset/realiad\`, JSON index under \`G:/dataset/realiad/jsons/realiad_jsons\`, unless overridden.
- Default probe size is 32 and scoring occurs at multiple checkpoints. This is more expensive than the old one-pass R6 benchmark; log time/VRAM.
- \`online_greedy_coreset\` recomputes an O(K²) pairwise distance matrix when full; this may be CPU/GPU-memory or latency intensive. Report actual execution and do not hide computational differences.
- Standard P0/VisionAD models may download DINOv2 weights if not already cached; internet/cache prerequisites are inherited, no new trainer is added.
- This A0/A1 experiment intentionally *does not* add or alter v22/R6 query controllers, Sidecars, correction or prior archives.

**Status:** Code is available for user-run GPU validation; scientific conclusions depend on collected results.
