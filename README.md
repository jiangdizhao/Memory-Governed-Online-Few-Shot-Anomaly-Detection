# P0-R1: Reproducibility and protocol audit

This research branch preserves the ORIGINAL v22 and VisionAD-style baseline files
**without edits**. The P0 experiment is a frozen-score diagnostics tool,
**not a validated full online continual learning evaluation**.

## P0-R1: isolation of randomness

- --seeds specifies support-selection seeds. Strict4 randomly chooses four normal images.
- --augmentation-seed specifies image augmentation randomness per support path.
- --bank-seed specifies per-image/per-view/per-layer feature subsampling and final fixed-bank seeding.
- --stream-seed specifies only evaluation stream order.
- --commissioning-seed applies solely to extra calibration images.
- --verify-replay reconstructs the SAME bank and scores the SAME evaluation image twice, asserting identical fingerprints and scores within --replay-tolerance (1e-6 by default).

All configuration, backbone identity/resolution, per-layer **actual** fixed-bank
prototype counts/dimensions, fingerprints, threshold statistics, actual supports,
and source checksums are recorded to JSON. With identical supports and fixed
augmentation/bank seeds, the batch test asserts identical bank fingerprints
and evaluation scores even when the support-selection seed changes.

IMPORTANT: pool_assisted4 may deterministically choose the SAME four supports
for every seed. The summary reports the number of genuinely different support
sets per protocol. Repeating the same set is NOT support-draw replication.

## Run immediately: Windows PowerShell

If the repo is already cloned:

~~~powershell
git fetch origin
git switch p0-r1-reproducibility
git rev-parse --short HEAD
git status --short
conda activate torch
python experiments/p0_protocol_audit.py --self-test
~~~

If you do not have a local clone:

~~~powershell
git clone --branch p0-r1-reproducibility https://github.com/jiangdizhao/Memory-Governed-Online-Few-Shot-Anomaly-Detection.git
cd Memory-Governed-Online-Few-Shot-Anomaly-Detection
~~~

First GPU smoke test (Audiojack, support seed 0, 100 images):

~~~powershell
python experiments/p0_protocol_audit.py --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-policy strict4 --seeds 0 --augmentation-seed 0 --bank-seed 0 --stream-seed 0 --max-test 100 --verify-replay --device cuda --out-dir "outputs/p0_r1_audiojack_smoke"
~~~

Then (only after successful smoke) use the full Audiojack test set:

~~~powershell
python experiments/p0_protocol_audit.py --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-policy both --seeds 0,1,2 --augmentation-seed 0 --bank-seed 0 --stream-seed 0 --max-test 0 --verify-replay --device cuda --out-dir "outputs/p0_r1_audiojack_full"
~~~

For a dedicated augmentation-sensitivity comparison, hold --seeds fixed
and vary ONLY --augmentation-seed, using a different output folder.
Do not combine augmentation sensitivity with support-draw sensitivity.

## P0-R1 outputs

The case name encodes support, augmentation and bank seeds, e.g.
audiojack_strict4_support0_aug0_bank0.

- <case>.json: effective configuration, actual bank capacities/counts,
  real support identities, threshold inputs, score digest and replay check.
- <case>_scores.csv: unmodified static scores and evaluation-only labels.
- <case>_threshold_curve.csv: retrospective FPR/FNR/Balanced Accuracy vs threshold.
- <case>_roc.png: ROC curve with support-self/LOSO operating points.
- <case>_threshold_sweep.png: FPR/FNR tradeoff versus threshold.
- p0_batch_summary.json: batch comparisons, number of unique supports, and
  identical-support reproducibility checks.

If a class has only one type of test label, ROC diagnostics will reject it.

## Scientific caveats

1. In strict4, only the four selected normal TRAIN images contribute decoded
   pixels to support selection and bank construction. In pool_assisted4,
   selection looks at many additional train-normal images. The latter is NOT
   a strict four-shot protocol.
2. LOSO excludes the held-out original and all its augmented references,
   but builds a 3-original-image bank. Hence LOSO is a **diagnostic**, not an
   automatically unbiased deployed threshold estimator. It can increase FNR
   catastrophically and is NOT recommended as a new threshold without evidence.
3. The configured fixed-bank capacity (25,000 by default) is distinct from
   the initialization seed limit (currently 8,192). Inspect the ACTUAL bank
   counts in the output.
4. Frozen bank initialization uses original v22 primitives plus an independent
   deterministic support cache. It is not guaranteed to be numerically
   equivalent to the original historical v22 pipeline. Keep the submitted
   manuscript claims separate from P0 pilot results.
5. Test labels are used only for retrospective evaluation/visualization.
   **Do not pick a threshold on the displayed ROC/FPR/FNR curves and then
   claim that same test set is unbiased.**
6. A one-process bank replay does NOT verify cross-machine numerical identity.
7. The strict4 experiment is frozen-only, so it cannot show or refute dynamic
   memory learning. Matched static-vs-online P0c is subsequent work.

## Validation

GitHub Actions runs Python compile, a numerical self-test, and CPU-only unit
tests in tests/test_p0_protocol_audit.py. GPU Real-IAD results still have
to be generated locally. The original v22 and vmb_new sources are unchanged.
