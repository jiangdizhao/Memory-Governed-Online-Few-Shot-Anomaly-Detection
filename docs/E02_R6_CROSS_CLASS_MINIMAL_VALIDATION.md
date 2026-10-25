# R6 Extension — Defect-dominated + Near-balanced Minimal Validation

**Frozen parent**: R6 archive head 47f3b84a8a516f5bc9baaa452cd1228cad4365d4.
**Development branch**: e02-r6-extreme-balanced-validation.
**Driver**: experiments/e02_r6_cross_class_validation.py.

## Research protocol

The prior R6 prevalence profile was generated **without looking at accuracy**.
It selected:
- defect-dominated: **rolled_strip_base**, 1,280 normal + 2,525 defects;
  N=3,805, query budget K=114.
- near-balanced: **switch**, 1,826 normal + 1,994 defects;
  N=3,820, query budget K=115.

Two predetermined support seeds **0 and 1**.
Three predetermined orders for each seed: **native** (unmodified R6
stream), **normal_first**, **defect_first** (offline-only use of labels to
sort a copy of the same stream). Total: 2 classes × 2 seeds × 3
orders = **12 conditions**.

Run each native before its corresponding stress trials. Native is the
reference for initial fixed Bank, support files, theta0, original stream
fingerprint, full sample membership and identical global query budget
for each pair. Stress runs must pass the existing R6 membership and
reference checks. This is a query-policy + STM + Sidecar **joint
closed-loop response to input ordering**, NOT a matched-query-tape
causal test. Do not interpret pooled AUROC across different memory
states as an isolated model discrimination improvement.

All runs use canonical original E02/R5-D:
device=cuda, math_sdpa, local_saliency, dataset Real-IAD, K≈3%,
max_samples=0, legacy_seed=0, stream_seed=0, query_seed=37,
same R5 time-only pacing and four original threshold choices.
The second support seed modifies only the independent support sample
draw. The driver refuses arbitrary category/seed/order combinations.

The original v22, experiments/e02_r6_sidecar_stability.py, archived R6
results and prior R3–R5 files are NOT changed.

## Stage 0 — pull and offline validation (PowerShell, repository root)

    git fetch origin
    git switch e02-r6-extreme-balanced-validation
    git pull --ff-only origin e02-r6-extreme-balanced-validation
    git status --short
    git rev-parse HEAD
    conda activate torch
    python -m py_compile experiments/e02_r6_cross_class_validation.py tests/test_e02_r6_cross_class_validation.py
    python experiments/e02_r6_cross_class_validation.py --self-test
    python -m unittest discover -s tests -p "test_e02_r6_cross_class_validation.py" -v

## Stage 1 — six GPU runs, support-seed 0

One run per invocation. In PowerShell, initialize:

    $X = "experiments/e02_r6_cross_class_validation.py"
    python $X --plan

Then in this exact dependency order:

    python $X --run --class-name rolled_strip_base --support-seed 0 --order native
    python $X --run --class-name rolled_strip_base --support-seed 0 --order normal_first
    python $X --run --class-name rolled_strip_base --support-seed 0 --order defect_first

    python $X --run --class-name switch --support-seed 0 --order native
    python $X --run --class-name switch --support-seed 0 --order normal_first
    python $X --run --class-name switch --support-seed 0 --order defect_first

Check current progress:

    python $X --plan

Each command validates its full R6 output and refuses to overwrite a
preexisting target directory. If any run fails, STOP and send the
traceback; do NOT treat its partial canonical initialization folder as
accepted. Do not rerun into the same folder without first inspecting it.

Optional clearly-labeled interim report:

    python $X --summarize --allow-partial --summary-out "outputs/e02_r6_extreme_balanced_seed0_interim.json"

Do not present an interim file as all-seed robustness evidence.

## Stage 2 — independent support seed 1 (six GPU runs)

    python $X --run --class-name rolled_strip_base --support-seed 1 --order native
    python $X --run --class-name rolled_strip_base --support-seed 1 --order normal_first
    python $X --run --class-name rolled_strip_base --support-seed 1 --order defect_first

    python $X --run --class-name switch --support-seed 1 --order native
    python $X --run --class-name switch --support-seed 1 --order normal_first
    python $X --run --class-name switch --support-seed 1 --order defect_first

The final aggregator rejects identical support file lists across seeds
and verifies per-seed native/stress lineage. If you encounter accidental
same supports, do NOT quietly count them as two independent seeds.

## Stage 3 — strict audit and cross-category result aggregation

    python $X --plan
    python $X --summarize

The strict summary creates:

    outputs/e02_r6_extreme_balanced_validation_summary.json

It will FAIL if any of the 12 matrix conditions is missing or invalid.
It validates: original prevalence, overall stream length, exact 3% K,
class-verified query counts, native/stress bank fingerprint, support
files, theta0, original native stream SHA, reference and exact
image-membership checks, presence of class-conditional windows, and
separate support seed selections. It reports per-run AUROC, BA,
FPR, FNR, class-conditional queried-label rate, STM writes/prototypes,
forced-terminal query counts, query-free gap, Sidecar updates and
paired differences relative to identical-class/identical-seed native.
One-class-only window FPR/FNR are left undefined (null) in original R6.

The aggregator refuses to overwrite an existing report; use a
versioned new filename if doing an explicit new round.

**Two support seeds do NOT establish statistical confidence, and each
stress permutation is diagnostic, not representative of a real factory
drift process. These numbers cannot establish general SOTA or the
causal contribution of an individual Sidecar component.**

## Stage 4 — commit experimental output only after all checks

    git status --short
    git add outputs/e02_r6_rolled_strip_base_seed0_* outputs/e02_r6_switch_seed0_*
    git add outputs/e02_r6_rolled_strip_base_seed1_* outputs/e02_r6_switch_seed1_*
    git add outputs/e02_r6_extreme_balanced_validation_summary.json
    git diff --cached --stat
    git commit -m "experiment: R6 defect-dominant and balanced two-seed robustness audit"
    git push origin e02-r6-extreme-balanced-validation
    git ls-remote origin refs/heads/e02-r6-extreme-balanced-validation

Inspect staged files for dataset imagery, secrets, checkpoints and large
raw feature caches; none should be included.

## Why this extension

Audiojack and Regulator are both normal-majority categories. The
current extension is deliberately restricted to the **pre-registered
defect-majority** and **near-balanced** Real-IAD categories, which
completes the smallest meaningful across-prevalence contrast in R6.
It deliberately does not optimize query schedules, sidecar controls,
memory capacities or hyperparameters based on these experiments.
