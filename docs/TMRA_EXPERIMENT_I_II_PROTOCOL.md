# TMRA: Targeted Memory Retention & Attribution Audit

## Status and frozen research question

The branch **tmra-targeted-audits** adds experiments only. It retains the full original Bridge Audit and A1-H results and does not change v22, baseline code, or previous results.

* **Experiment I:** At exactly the same frozen memory checkpoint, which individual images/patch descriptors receive a lower normal-distance from LTM vs provisional Buffer, and how does that affect normal/defective images?
* **Experiment II:** Under disjoint original images, a shared label-blind 3% query tape, and a common physical mutable-tensor **byte ceiling**, can a hierarchy retain an earlier synthetic normal appearance mode A through exposure to changed mode B, and reduce labels needed to reacquire A?

All positive and negative results are reported; no post-hoc retuning on the probe/test labels.

## Experiment I — Source attribution on Bridge (historical-capacity operator)

Conditions: **Switch support seeds 0 and 1**, plus **Audiojack support seed 0** as a prespecified negative control. STM=512, LTM=512, Maturation Buffer=4096, real v22 native k=1 per-source LTM and maturity-gated Buffer retrieval. Four original normal support images, fixed 3% tape and common top-24 static candidate patches.

Each run replays exactly the Bridge trial using independent memory banks, then at checkpoints 0, 1024, 2048, 3072 and final, for each of 96 held-out probe images:

- Keep the whole learned memory fixed; change only retrieval flags.
- Record normal-distance image scores `S0` (Fixed+STM), `SL` (+LTM), `SB` (+eligible Buffer), `SLB` (both).
- Report per-image dual-player Shapley score-reduction contributions:
  - `phi_ltm = ((S0-SL)+(SB-SLB))/2`
  - `phi_buffer = ((S0-SB)+(SL-SLB))/2`.
- Report signed score reductions for **normal and defect** probes separately. A reduction on a defect is a possible false-negative risk, not a "rescue".
- If any source changes an image score, inspect a bounded set of high-effect patch indices (top 16 + highest-score patch), storing effective cosine distances to Fixed, STM, LTM, eligible Buffer; identify source with minimum native distance.
- Patch winners are geometric descriptors, **not** validated semantic/localization masks. This is no pixel AUROC/AUPRO claim.
- Recorded sample path/label columns are evaluation-only; none influence the memory updater or test split.

**Output:** `outputs/tmra/I_*/{summary.json,source_image_attribution.csv,source_patch_evidence.csv,retrieval_counterfactual.csv,probe_scores.csv,prequential.csv,...}`.

**Criteria:** independent marginal score effect; overlap/redundancy; normal-score rescue versus anomalous-score suppression; activation time. Aggregate AUROC is NOT enough to claim semantic memory.

## Experiment II — Controlled A-to-B-to-A recurrence

**Eight paired conditions:** (Switch, Audiojack) × support seeds {0,1} × orders {ABA, AAA}. Every ABA/AAA pair shares the *same original image paths, supports, labels and query positions*. Only appearance mode of Phase II differs.

- **Support:** exactly four distinct normal-train originals; static bank built with frozen ViT-L/14 DINOv2 register feature fusion. No train/test cross-access.
- **Raw source split:** label-blind shuffled original test images, 96 held-out A-Probe originals, 96 disjoint B-Probe originals, and **three disjoint sets of 900 stream originals**. Original images never repeat between phases/probes.
- **A mode:** exact original input image.
- **B mode:** predeclared clipped channelwise transform `RGB gains=(0.84,0.94,1.08), additive +8`; no geometry change, *synthetic covariate shift only*. Defect status is assumed preserved but not externally verified. Six Phase-II B images are additionally encoded in A-mode **for shift diagnostics only**, never added to a memory state; a negligible feature shift fails closed.
- **Query tape:** 3% of the **whole 2700-image stream**, label-blind seed 37. Only truly queried normals update the normal banks, **after scoring**. The same image is scored before potential update by all arms. Image defect labels are offline evaluation only unless queried.
- **Candidate patches:** 24 highest static-anomaly patches from the common immutable fixed scorer in the observed phase image, identical across arms per trial.
- **Memory arms:**
  - Static (no update; diagnostic)
  - Online Greedy Coreset, flat K=384
  - Flat v22 Utility STM, K=384, constant verified normal utility .5
  - STM(128) + LTM(64) + Maturation Buffer(64), original v22 native k=1 mature LTM/provisional Buffer retrieval and utility retention, `E_min=2`, promotion every **16** verified normal events; this compact stress-protocol promotion period differs intentionally from historical 32 and is fixed before evaluation.
- **Resource accounting:** every active dynamic PyTorch tensor allocated in STM/LTM/Buffer, including preallocated FP32 Buffer and metadata, is counted with storage deduplication and checked `<= 3,500,000 bytes` after every verified feedback. All models have **the same upper ceiling**, not equal actual usage. Frozen reference/model tensors and transient k-center computation are reported as outside this *mutable-state* measure. Exact used bytes appear in every memory snapshot. The simple-memory arms may occupy different byte amounts and have different compute costs.
- **Probe checkpoints:** before learning, after Phase I, after Phase II, immediately at Phase III start **before any A exposure**, then after every Phase III newly verified normal label, then Phase III end. A-Probe is the **same held-out set** at every time point; B-Probe is evaluated at selected checkpoints.
- **Forgetting:** `A_probe_AUROC(end I) - A_probe_AUROC(end II)`; additionally track normal A score drift. A-Probe label evaluation never updates memory.
- **Reacquisition:** first Phase III normal feedback count for which A-Probe AUROC returns to `end-I AUROC - 0.01`. If no recovery, result is **null/censored**, not made-up zero or a fixed maximum. Counts 0 when memory already meets the target before any Phase-III feedback.
- **Negative control:** AAA uses identical original paths, query positions, supports and stream staging. Compare difference-in-differences `(forget_ABA - forget_AAA)`; do not label the shift a real plant operating mode.

**Output:** `outputs/tmra/II_*/{summary.json,probes.csv,prequential.csv,memory_snapshots.csv,reacquisition.csv}`.

**Interpretation gate:** A credible long-memory advantage should require actual LTM promotion and demonstrated retention benefit against Greedy/Flat Utility under the common memory ceiling, reproducible across supports. No promotion is a valid negative outcome, NOT a reason to post-hoc adjust gates. If the static bank itself is strong enough to retain A, explain the ceiling effect.

## Execution on Windows (conda torch, RTX GPU, local Real-IAD)

From your existing repository root:

```powershell
git fetch origin
git switch tmra-targeted-audits
git pull --ff-only origin tmra-targeted-audits
git rev-parse HEAD
conda activate torch
```

First run Experiment I **only**:

```powershell
python experiments/run_tmra.py --experiment I --dataset-root "G:/dataset/realiad" --math-sdpa
```

It executes the actual v22 CPU mechanism/physical byte-ceiling preflight, a 64-sample GPU smoke and **3 full Experiment-I conditions**. Collect `aggregate_i.json`.

Then, if Experiment I completes, run Experiment II:

```powershell
python experiments/run_tmra.py --experiment II --dataset-root "G:/dataset/realiad" --math-sdpa
```

A separate 3×64-sample Phase GPU smoke precedes **eight full Experiment-II conditions**. Collect `aggregate_ii.json`. The runner verifies matched ABA/AAA manifests. A combined resumable command also exists: `--all`. Inspect outputs and reject failures before ever changing scientific settings.

Existing complete conditions with unchanged source hashes are validated/skipped. Incomplete results/logs are **never silently overwritten**; inspect and archive only the affected incomplete run before a rerun. Preserve every prior successful experiment's evidence.

Offline GitHub CI tests only contracts and syntax. The CPU test runs actual v22 operators locally. Real GPU runs are not declared passed until verified on the machine containing Real-IAD.

## Research claims excluded from these experiments

These experiments **do not** reproduce the earlier paper's BEA/Sidecar/Correction coupled system; do not prove external SOTA superiority, human-like world modeling, real factory mode robustness, true lifelong learning, or pixel segmentation quality. Experiment II measures controlled synthetic appearance-shift retention and recapture, under finite recurrence.

## Minimal analysis plan

For I: source contribution distribution normal vs anomalous, number of modified high-score patches/images, interactions between LTM and Buffer, their activation timeline; pair with old Bridge AULC.

For II: paired ABA vs AAA contrasts by class/seed, incremental A forgetting (including CIs reported as descriptive with tiny n), B-probe performance, A normal-score drift, actual LTM formation, same-ceiling but per-arm actual bytes, and censored reacquisition counts. If hierarchy fails to promote, treat that as an architectural sparse-feedback limitation, not an implementation exception.
