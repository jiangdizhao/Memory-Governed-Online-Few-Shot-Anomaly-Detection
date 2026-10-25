# E02-R1 / E02-R2 controlled follow-up experiments

**Authority:** unchanged original `test_phase3_AIF_simplified_v22.py` cold-start
(initialized through E01/E02). These are **exploratory** controlled experiments,
not reproduction of the full BEA/correction/LTM paper.

The original E02 Audiojack complete run (3815 images, seed 0, 114 queries)
reported Static AUROC 0.643439 vs STM-only 0.801397, but STM-only FNR
rose from 75.31% to 90.59%. Those observations motivated the two registered
follow-ups; do NOT choose hyperparameters after looking at probe or new
evaluation labels and then report the same data as confirmatory.

## Environment

Use the repo on your local Windows CUDA workstation. Offline GitHub CI
checks syntax and numpy/scikit-learn unit tests, **not** local DINOv2/Real-IAD.

~~~powershell
git fetch origin
git switch e02-r1-r2-probe-calibration
git rev-parse --short HEAD
git status --short
conda activate torch
python experiments/e02_r1_r2.py --self-test
~~~

Keep `--math-sdpa` consistent with previous E01/E02 runs.

## R1: strict held-out probes / frozen snapshot transfer

- Preselect probe images solely by paths/indices and `--probe-seed`,
  independently of labels, BEFORE online stream starts.
- Exclude probes completely from the streaming/query tape. No probe image
  may enter STM, training, or sidecar observations at any time.
- Run **one** canonical Fixed Bank initializer and two arms:
  Static vs Verified Normal STM-only. Keep theta0 fixed.
- Score the **same** probes at t=0, 25%, 50%, 75%, 100% of the
  remaining streaming images.
- Output paired AUROC/Balanced Accuracy/FPR/FNR, normal/defect score
  drift, number of observed verified-normal writes.
- Static probe scores are checked for drift across checkpoints.
- This is a held-out within-dataset split, not a new external benchmark.

### First GPU smoke: 100 class images, 16 held-out probes

~~~powershell
python experiments/e02_r1_r2.py --experiment r1 --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --legacy-seed 0 --stream-seed 0 --query-seed 37 --query-rate 0.03 --max-samples 100 --probe-seed 97 --probe-count 16 --math-sdpa --device cuda --out-dir "outputs/e02_r1_audiojack_smoke"
~~~

**No scientific conclusion** from 16 held-out probes.

### R1 full dataset, 256 strictly held-out probes

~~~powershell
python experiments/e02_r1_r2.py --experiment r1 --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --legacy-seed 0 --stream-seed 0 --query-seed 37 --query-rate 0.03 --max-samples 0 --probe-seed 97 --probe-count 256 --math-sdpa --device cuda --print-every 100 --out-dir "outputs/e02_r1_audiojack_full_seed0"
~~~

3815 total samples minus 256 probes = **3559 streaming images**. The same
query rate yields **107 label requests** in the training stream (not 114;
query budget is always measured against the *eligible* stream length).

R1 output:
- `e02_r1_probe_manifest.json` (actual probe paths, split and query hashes)
- `e02_r1_probe_report.json` (all checkpoint metrics and provenance)
- `e02_r1_probe_scores.csv` (paired score for every probe at each checkpoint)
- `canonical/canonical_v22_init.json`

**Pass/fail interpretation:** Positive gain requires AUROC on exactly the
same probes to rise from t=0 to final, with no probe feedback and no
score-only leakage; track defect FNR as carefully as AUROC.

## R2: legal threshold calibration on one identical STM path

**No held-out probe partition here**: entire same class stream and
identical query tape as prior E02 (query_seed=37, stream_seed=0).

- `static`: canonical unchanged Frozen Bank + theta0.
- `stm_frozen`: a single Verified Normal STM stream + frozen theta0.
- `stm_guarded`: SAME STM scores + original v22
  `GuardedDualAnchorThresholdController.observe_labeled`.
- `stm_normal_quantile`: SAME scores; after a minimum of 8 verified-normal
  query observations, update theta toward a rolling empirical q95 of
  the latest 64 verified-normal scores; alpha=0.25, maximum change of
  0.01 per update. All values are bounded relative to original theta0.
- `stm_mistake_feedback`: SAME scores; for queried verified false
  negatives reduce theta by 0.002; verified normal false positives
  increase theta by 0.002; otherwise do nothing; bounded near theta0.

Only the predeclared 3% query tape discloses labels for controllers and STM.
No unlabeled pseudo-normal writes, no retrospective ROC-based threshold
setting. In this experiment **all threshold policies reuse one identical
STM score path**, and STM write control uses the *fixed original theta0*;
therefore a difference in FPR/FNR is attributable to the controller
rather than a different stream of written STM prototypes.

### R2 100-image smoke

~~~powershell
python experiments/e02_r1_r2.py --experiment r2 --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --legacy-seed 0 --stream-seed 0 --query-seed 37 --query-rate 0.03 --max-samples 100 --math-sdpa --device cuda --out-dir "outputs/e02_r2_audiojack_smoke"
~~~

With only three queries, normal-quantile typically will NOT update because
the preset min_normals=8 gate is not reached. This is expected.

### R2 full 3815-image Audiojack run

~~~powershell
python experiments/e02_r1_r2.py --experiment r2 --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --legacy-seed 0 --stream-seed 0 --query-seed 37 --query-rate 0.03 --max-samples 0 --math-sdpa --device cuda --print-every 100 --out-dir "outputs/e02_r2_audiojack_full_seed0"
~~~

Full Audiojack R2 should draw **114 labels**, as original E02 did.
The original E02 and R2 will share supports, stream and query-tape
position sets. R2 deliberately uses one shared STM write trajectory and
may be numerically identical to E02 STM-only, but this should be checked.

R2 output:
- `e02_r2_calibration_report.json` (all controller AUROC, FPR, FNR,
   BA, threshold minima/maxima, queried-normal counts and STM state)
- `e02_r2_prequential_scores.csv` (pre-update scores, thresholds, preds)
- `e02_r2_query_tape.json` (exact label-blind positions)
- `canonical/canonical_v22_init.json`.

**Pass/fail interpretation:** All STM threshold methods MUST have
identical AUROC since they share the same score path. Any method that
reduces FNR but hugely increases FPR cannot be described as an unqualified
improvement. Compare balanced accuracy, FPR, FNR and the controller's
actual number of updates.

## Study limitations and reproducibility

These two experiments use fixed, exploratory hyperparameters. For a paper,
run multiple support seeds (0,1,2), pre-register final parameters before
cross-class testing, and compare simple FIFO/reservoir/coreset updating
under matched label budgets. Do not treat a favorable R2 threshold policy
chosen using Audiojack outcomes as independently validated on Audiojack.

If a GPU smoke test fails, send full traceback. Do NOT change v22 or
increase label budget to hide an experiment driver bug.
