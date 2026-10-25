# E02 — matched-label 2×2 STM / threshold experiment

Status: **implemented, offline CI tested**; requires user-local Real-IAD/CUDA run.
Core v22 and `vmb_visionad_new.py` remain completely unchanged.

## Scientific question

With *exactly the same* frozen DINOv2 / v22 support draw, Fixed Bank,
initial threshold, test stream, and externally fixed label/query positions,
does verified-normal STM provide a score-ranking gain independent of online
threshold calibration?

Four arms operate on a single shared image encoding at each time step:

| Arm | Verified-normal STM | Queried-label threshold |
|---|---|---|
| static | off | off |
| stm_only | on | off |
| theta_only | off | on |
| stm_theta | on | on |

The controller is the v22 `GuardedDualAnchorThresholdController`, using
the same original `init_from_support` and `observe_labeled` methods.
STM update is the v22 `normal_dyn_only` *update primitive*:
`layer_to_topk_from_amap` / `select_informative_normal_write` /
`PatchVMB.update_from_topk(which="normal_dyn")`.

Differences from the original full v22 policy are intentional: no
BEA query selection, no correction, no defect bank, no LTM/maturation,
no unverified pseudo-normal updates, and no unlabeled threshold observations.
For STM write informativeness, novelty is explicitly held at 0 instead of
using the changing coverage-memory estimator. Therefore **this is a causal
mechanism audit, not an exact reproduction of the previously submitted full
online v22 system**.

## Run on your Windows workstation (PowerShell)

Inside the repository (keep any existing uncommitted changes safe):

~~~powershell
git fetch origin
git switch e02-matched-feedback-factorial
git rev-parse --short HEAD
git status --short
conda activate torch
python experiments/e02_matched_feedback.py --self-test
~~~

Use the same math-SDPA kernel used for your P0/E01 comparisons.

### First smoke: Audiojack, 100 streamed test images

~~~powershell
python experiments/e02_matched_feedback.py --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --legacy-seed 0 --stream-seed 0 --query-seed 37 --query-rate 0.03 --max-samples 100 --math-sdpa --device cuda --out-dir "outputs/e02_audiojack_smoke"
~~~

This selects exactly **3 label-blind query positions** among the 100
images. Avoid drawing scientific conclusions from this small smoke test.

### Confirmatory within-category pilot: full Audiojack stream

~~~powershell
python experiments/e02_matched_feedback.py --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --legacy-seed 0 --stream-seed 0 --query-seed 37 --query-rate 0.03 --max-samples 0 --math-sdpa --device cuda --print-every 100 --out-dir "outputs/e02_audiojack_full_seed0"
~~~

`--max-samples 0` means the full class test stream (3815 images in
the previous Audiojack variant). For the full 3815-sample stream,
the query tape contains **114 randomly selected positions**.

Only after reviewing full Seed 0, vary the support seed (1, 2), keeping
the query seed, stream seed, feature configuration and query rate fixed.
Changing support seed changes the canonical v22 support images. Using the
same stream positions across seeds permits paired comparisons.

### Output files

- `canonical/canonical_v22_init.json`: genuine v22 cold-start audit
- `e02_query_tape.json`: 1-based query positions and SHA256 of tape and stream
- `e02_prequential_scores.csv`: every row contains the four arms'
  **pre-update** scores, theta and predictions; evaluation labels are
  stored only for retrospective metric computation.
- `e02_factorial_summary.json`: matched initial support paths, Fixed
  Bank fingerprint, query counts, per-arm AUROC/FPR/FNR/accuracy/
  balanced accuracy, final thresholds and verified STM update counts.

### Critical interpretation rules

1. Per time step: encode once → all four arm scores/predictions →
   record outcomes → if in tape reveal label → supervised state updates.
2. The tape is independent of ground truth and model score. Static gets
   zero labels for model updates; its evaluation labels are used only
   for computing metrics.
3. Exact initial Fixed Bank SHA256 and theta must match across arms;
   branches are deep copies of one canonical v22 initialized state.
4. **No test labels from non-queried images are used to adapt any state.**
5. `theta_only` scores should equal `static` scores step by step;
   updating only the threshold cannot alter AUROC, but can alter
   FPR/FNR/Balanced Accuracy.
6. AUROC computed across an evolving stream can be affected by time-dependent
   score distribution shifts. Subsequent research should test transfer to a
   held-out probe set that is never accessed for memory, queries, or calibration.
7. Query-rate 3% is an **experimental label-budget constraint**, not
   an output of BEA. Do not credit BEA with improvements seen here.
8. If online STM improves AUROC over static under fixed feedback, it is
   consistent with useful external-memory score adaptation. This alone
   does not prove superiority to FIFO/Reservoir/coreset or long-term
   continual learning. Those remain later P1 experiments.
9. This script shares the v22 initializer but intentionally implements a
   smaller, controlled stream and update driver; its scores should not
   be conflated with old PR 30-class full-system numbers.

## First acceptance gate

1. Offline self-test and GitHub CI pass.
2. GPU run has four arms and identical first-image scores and theta.
3. Query positions and query counts identical across all arms.
4. Static/STM-only theta remains unchanged; static/theta-only
   normal STM remains empty.
5. Fixed Bank fingerprint unchanged in every arm at end.
6. Report and CSV are saved successfully.

If the first 100-sample run raises an exception, provide the exact traceback,
not just an abbreviated summary. Do **not** modify v22 or increase the
query budget to work around an experiment driver defect.
