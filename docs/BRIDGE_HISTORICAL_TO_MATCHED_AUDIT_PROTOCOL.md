# Bridge Audit: historical normal-memory mechanism vs matched-budget controls

## Status and scientific boundary

**Branch:** `bridge-historical-matched-audit`. Independent of all previous A0/A1/A1-H experiment outputs. Adds only new code/tests/docs/CI.

The rejected paper's **Supplementary Table S1** reports fixed bank 25,000, per-image cap 5,000, four supports, augmentation four, concat features, identity view, STM=1024, LTM=512 with k=1, full maturation buffer and utility retention. **Table S16** compares STM=512/768 to LTM-no-buffer and full hierarchy; STM-only also turns off utility retention. **Table S2** specifies maturation event repetition minimum 2, score threshold 0.45, usefulness threshold 0.15 for LTM eligibility, event merge 0.03/max eight prototypes, candidate merge 0.06. The v22 constructor initializes maturation usefulness at 0.10, but the published value and the v22 CLI default are both 0.15. The Bridge audit explicitly sets **0.15** instead of relying on the constructor. The v22 CLI default maturation period is 32 events. The exact original full-run CLI logs remain necessary to assert exact historical equivalence.

**Essential limit:** Even the "historical" Bridge runs are NOT a faithful re-run of the paper's original end-to-end protocol. They use **strict four-shot random supports** rather than pre-deployment pool-assisted selection, **fixed 3% random human label tape** rather than endogenous BEA, **no signed correction**, **no adaptive threshold sidecar**, and **one fixed candidate patch tape selected by immutable Static** rather than method-specific patch policies. Thus, claims from old Table S16 cannot be compared to new scores as if everything else were held constant.

## Intervention matrix — SAME supports, image stream, queries, candidate patches

All eight arms share the original v22 **native** retrieval code: min of fixed k=5, STM k=5, LTM k=1, and *eligible* buffer k=1 with penalty. LTM and buffer switches are genuinely disabled for the respective arms.

| Arm | Capacity and mechanism | Causal interpretation |
| --- | --- | --- |
| `flat_stm` | STM=512 or 768; no retention utility, LTM or Buffer retrieval | Original STM-only comparison |
| `flat_utility` | Same STM; utility retention on, no LTM/Buffer retrieval | **Utility alone**, separates Table S16 conflation |
| `flat_equal_total` | One large STM of Kstm+512+4096 slots; utility on | Equal aggregate **prototype SLOT** max; report actual bytes separately |
| `track_utility` | STM=Kstm + 512 LTM reserved + 4096 Buffer tracking; no promotion or buffer retrieval | Maturation tracking with no score input |
| `shadow_utility` | + physical consolidation to LTM, no LTM/Buffer retrieval | Separate storage/promotion from retrieval; LTM contents may enter retention coverage |
| `ltm_utility` | same promotion + LTM retrieval, Buffer retrieval off | Pure LTM retrieval vs shadow: **state must match** |
| `full_utility` | LTM + eligible Buffer retrieval + Utility-aware STM | Full historical memory package under fixed supervision |
| `full_no_utility` | same hierarchy/feedback but utility retention off | Utility retention within hierarchy |

**Warning:** `full_utility` and `ltm_utility` may follow different STM paths because the v22 utility retention rule uses currently retrievable Buffer motifs as a coverage source. A full-system arm comparison is therefore **not** pure retrieval attribution. At each fixed-probe checkpoint, for `full_utility` ONLY, one frozen memory checkpoint is re-scored in five modes: full, LTM-off, Buffer-off, both-off, and global unified top-5 without Buffer, with no state updates in between. That produces a direct paired read-only retrieval counterfactual, including the question of whether original v22 k=1 LTM retrieval differs from A1-H global top-5.

`shadow_utility` and `ltm_utility` must have **identical full STM, LTM and Buffer prototypes** at every checkpoint; a mismatch fails execution.

## Fixed experimental contract

- Real-IAD classes: audiojack, rolled_strip_base, switch. Support seed=0,1.
- Four distinct randomly sampled **normal train originals** as sole deployment support data. Per-original support augmentation is derived from those images only.
- DINOv2 ViT-L/14 registers, concat 4-layer tokens, same frozen Fixed Bank; support SHA-256 and per-LOSO fold exclusion recorded.
- Identity query view; 96 probe images held out *before* streaming by label-blind seeded selection. Frozen probe features cached once; never added to memory, candidate selection or calibration.
- Native stream order and single prequential pass.
- Three percent of stream positions queried using same precommitted label-blind tape (seed 37). Only queried truly verified normal images write to normal memory. Defect labels may be observed but do not enter the tested normal-memory arms.
- Candidate patches: top 24 per fused layer, selected from an immutable Static anomaly map. Fixed for all arms at each queried normal event. This is a **new common candidate policy**, not original BEA's dynamic 24/16/8 gate. This difference must be acknowledged.
- Native v22 normal update operators, including actual MaturationBuffer and `consolidate_normal_dyn`. No gradient learning.
- Main `--evidence-mode neutral`: every queried verified normal supplies utility=0.5, boundary=0.5, outcome=TN. That isolates state organization from the previously detected invalid LOSO fixed threshold. **It intentionally disables FP-rescue utility.**
- Optional independent `--evidence-mode loso_margin`: restores v22 score/LOSO-based normal writing meta-values but is **not** a calibrated deployment protocol. Use a different output root and do not pool results.
- STM=512 initially; later STM=768 pressure sensitivity. LTM=512; Buffer=4096 (float32 tracker, higher actual bytes per prototype).
- Constant utility, consolidation and retrieval gates across all classes/seeds. In the current v22 version: `mature_min_events=2`, `mature_min_usefulness=0.15`, `mature_score_threshold=0.45`, promotion every 32 verified events, retention replacement margin .03, no retirement, native LTM k=1 and optional Buffer k=1.
- Historical `tau_U=0.15` agrees with the v22 CLI default (but not its constructor default), and is explicitly applied. No test-label-based hyperparameter tuning is permitted.
- Fixed LOSO theta recorded only for diagnostics. **Do not interpret this trial's BA/FPR/FNR as valid calibrated-deployment performance.** Focus on Probe AULC, fixed Probe AUROC, prequential AUROC, activation, memory coverage and counterfactual score effects.

## What is an independent result?

1. `flat_utility - flat_stm`: utility retention with exactly the same flat STM capacity.
2. `ltm_utility - shadow_utility`: native LTM retrieval with identical memory trajectory and identical STM/LTM/Buffer contents.
3. `full_utility - ltm_utility`: native Buffer-enabled *system* effect (not pure due to STM utility coupling); counterfactual read-only Buffer-off on the full checkpoint gives direct retrieval effect.
4. `full_utility - full_no_utility`: utility retention within equal-capacity hierarchy; normal memory representation changes.
5. `full_utility - flat_utility`: additional memory layers and capacity at *same STM* (NOT equal total resources).
6. `full_utility - flat_equal_total`: equal **maximum prototype slots** but very different architecture and memory bytes (Buffer uses FP32 vs flat FP16). Report allocated/occupied bytes and never describe it as byte-equal.

A bridge run only shows hierarchy effects if LTM promotion and/or eligible Buffer retrieval actually engage. Record number of nonempty LTM and Buffer and explicit counterfactual probe score changes. Check performance **after** state engagement in addition to whole-trajectory AULC.

## Local workflow

First sync branch, then run one command in your `torch` conda environment:

```powershell
git fetch origin
git switch bridge-historical-matched-audit
git pull --ff-only origin bridge-historical-matched-audit
git rev-parse HEAD

conda activate torch
python experiments/run_bridge_audit.py --all --dataset-root "G:/dataset/realiad" --math-sdpa
```

This invokes the genuine v22 CPU mechanism preflight and 64-stream-image GPU smoke, followed by **six full experiments**: three classes × two support seeds × historical STM=512 (eight methods per experiment). Fail-fast on incorrect memory state invariants, source mismatch, query tape, or overbudget prototype count. Existing complete conditions are validated and reused; interrupted GPU directories are never overwritten. Results:

```text
outputs/bridge_historical_matched/
  cpu_mechanism_preflight.run.log
  cpu_mechanism_preflight.passed
  audiojack_s0_STM512_native_smoke/
  audiojack_s0_STM512_native_full/
    summary.json
    prequential.csv
    probe_scores.csv
    probe_summary.csv
    memory_snapshots.csv
    retrieval_counterfactual.csv
  ...
  aggregate.json
  aggregate.csv
```

After first analysis, **optional** original pressure setting STM=768, preserving completed 512 runs:

```powershell
python experiments/run_bridge_audit.py --all --dataset-root "G:/dataset/realiad" --math-sdpa --stm-capacities 512,768 --no-smoke
```

Optional independent utility evidence sensitivity with a SEPARATE root:

```powershell
python experiments/run_bridge_audit.py --all --dataset-root "G:/dataset/realiad" --math-sdpa --evidence-mode loso_margin --out-root outputs/bridge_loso_margin
```

**Do not launch extra settings until the STM=512 neutral results are audited.** If the CPU preflight or smoke fails, save its full log; do not alter the scientific protocol in an attempt to make results look favorable.

Offline CI checks syntax, runner matrix, historical capacity arguments and counterfactual restoration. It does not run DINOv2 or Real-IAD. Actual CPU operator preflight and GPU execution must be verified locally.

## Scientific gate before the next manuscript

- Did historical k=1 LTM or mature-buffer retrieval change probe scores and rankings at constant supervision? How many images and when?
- Is improvement driven by **utility retention** alone (conflated in original S16)?
- Does any full-hierarchy benefit survive the equal-total-slot flat control, after reporting real memory bytes?
- Is the result stable over independent supports and operating modes? This screen is only 3 categories × 2 seeds; no external SOTA, 30-class, or A-B-A claims.

A negative answer does not invalidate original Table S16; it narrows the conditions under which its effect can be attributed. A positive answer does not retroactively prove original results were strict four-shot.
