# E02-R6 — Sidecar Stability & Extreme-Stream Audit

**Current experiment branch: e02-r6-sidecar-stability-audit.**
New audit-only entrypoint: [experiments/e02_r6_sidecar_stability.py](experiments/e02_r6_sidecar_stability.py).
Full parity requirements and local Real-IAD GPU runbook:
[docs/E02_R6_SIDECAR_STABILITY_PROTOCOL.md](docs/E02_R6_SIDECAR_STABILITY_PROTOCOL.md).

R6 wraps the original R5-D Balanced Sidecar with read-only event tracing
(candidate decisions, cooldown, missing-class evidence), plus offline-only
normal-first/defect-first/alternating-block stress permutations. Exact R5-D
stepwise parity is REQUIRED before interpreting the native Audiojack audit.
Other classes first build their own native R6 reference trajectory.

Original v22 and all R3/R4/R5 experiments/results remain unchanged. R6 is
**not a new threshold algorithm and not a full signed-correction/LTM experiment**.

~~~powershell
git fetch origin
git switch e02-r6-sidecar-stability-audit
conda activate torch
python experiments/e02_r6_sidecar_stability.py --self-test
python -m unittest discover -s tests -p "test_e02_r6_sidecar_stability.py" -v
~~~

---

# E02-R4 — A/B/C Query Governance Stability Audit

**Active branch: e02-r4-abc-query-governance.** New experimental entrypoint:
[experiments/e02_r4_abc_query_governance.py](experiments/e02_r4_abc_query_governance.py).
Full preregistered A/B/C comparison, local Real-IAD GPU commands and
replay acceptance rules:
[docs/E02_R4_ABC_QUERY_GOVERNANCE_PROTOCOL.md](docs/E02_R4_ABC_QUERY_GOVERNANCE_PROTOCOL.md).

A uses the original R3 bootstrap cost feedback unchanged. B excludes
Bootstrap queries from adaptive cost feedback while retaining them in the
actual queried-label budget. C adds score-only region quotas and paced
query spending to B. All variants use exactly the same full label budget
and run frozen/guarded/balanced/mistake threshold controls on matched
within-variant STM scores. **No full v22/correction claims are made.**

~~~powershell
git fetch origin
git switch e02-r4-abc-query-governance
conda activate torch
python experiments/e02_r4_abc_query_governance.py --self-test
~~~

**All earlier v22/P0/E01/E02/R1/R2/R3 files and existing outputs
remain unchanged.**

---

# E02-R3: Original Simple-AIF Bootstrap under matched label budgets

**New active research branch:** `e02-r3-bootstrap-sidecar-audit`.
Runs native v22 Bootstrap → Simple-AIF query functions and compares the
original Guarded Sidecar, Balanced Rescue, frozen theta, and a simple
mistake-feedback reference **under each fixed-feedback trajectory**.
Two alternative query schedules both have a hard final query cap.

- Source: [experiments/e02_r3_bootstrap_sidecar.py](experiments/e02_r3_bootstrap_sidecar.py)
- **Runbook:** [docs/E02_R3_BOOTSTRAP_SIDECAR_PROTOCOL.md](docs/E02_R3_BOOTSTRAP_SIDECAR_PROTOCOL.md)

The original v22 research script, E01/E02/R1/R2, and all existing
reports are unchanged. This is a controlled STM-only mechanism audit,
**not** an exact replay of the earlier `normal_dyn_plus_correction` historical
full-system run.

~~~powershell
git fetch origin
git switch e02-r3-bootstrap-sidecar-audit
conda activate torch
python experiments/e02_r3_bootstrap_sidecar.py --self-test
~~~

---

# E02-R1/R2 — independent probe generalization and legal threshold calibration

**Active follow-up experiments:** `experiments/e02_r1_r2.py`. Two experiments
reuse the unchanged canonical v22 initializer and E02 verified-normal STM:
R1 tests held-out probe generalization without probe feedback; R2 tests
predefined verified-label-only threshold controllers on a **shared STM
score trajectory**. Neither uses BEA or retrospective test-label tuning.

Local Windows PowerShell instructions, controls, outputs and interpretation:
[docs/E02_R1_R2_PROTOCOL.md](docs/E02_R1_R2_PROTOCOL.md).

~~~powershell
git fetch origin
git switch e02-r1-r2-probe-calibration
conda activate torch
python experiments/e02_r1_r2.py --self-test
~~~

**Original v22, vmb_new, P0, E01 and E02 files are unchanged on this branch.**

---

# E02 — Canonical v22 matched-feedback factorial experiment

**Active branch:** `e02-matched-feedback-factorial`.

The new experimental entry point is
[experiments/e02_matched_feedback.py](experiments/e02_matched_feedback.py).
The exact local GPU commands, the matched query budget, prequential protocol,
and output interpretation are in
[docs/E02_MATCHED_FEEDBACK_PROTOCOL.md](docs/E02_MATCHED_FEEDBACK_PROTOCOL.md).

This branch preserves the original v22, vmb_new, and all P0/E01 experiments.
E02 runs four arms (Static, STM-only, Theta-only, STM+Theta) from the SAME
original-v22 initialization, NOT a reimplementation or comparison against
the non-equivalent standalone P0 bank.

~~~powershell
git fetch origin
git switch e02-matched-feedback-factorial
conda activate torch
python experiments/e02_matched_feedback.py --self-test
~~~

---

# E01 — v22 initialization equivalence audit

**Active experiment on this branch:** original v22 initialization versus the
P0-R2 frozen-bank pilot. Run instructions, scope and output interpretation:
[docs/E01_V22_INITIALIZATION_AUDIT.md](docs/E01_V22_INITIALIZATION_AUDIT.md).

Run first: `python experiments/p0_v22_equivalence.py --self-test`.

**E01 does not change the original v22 or vmb_new source.** The E01 verdict
`not_equivalent` is an expected, informative result if the two different
initialization implementations disagree; it is not a model-crash indication.

---

# P0-R1: Reproducibility and protocol audit

This research branch preserves the ORIGINAL v22 and VisionAD-style baseline files
**without edits**. The P0 experiment is a frozen-score diagnostics tool,
**not a validated full online continual learning evaluation**.

## P0-R2: GPU numerical replay correction (2026-10-25)

The original P0-R1 replay checked two scores computed by **independent DINOv2
forward passes** with a very strict absolute tolerance of 1e-6. On CUDA
memory-efficient attention, those two passes can differ slightly even if the
fixed bank's bytewise fingerprint matches. Example observed locally:
0.5284950137138367 versus 0.5285062193870544 (delta 1.12e-5).

P0-R2 separates the checks:

1. Exact fixed-bank SHA256 fingerprint comparison (unchanged).
2. A **shared encoded feature tensor** is scored against both frozen banks;
   the strict --replay-tolerance=1e-6 applies here.
3. A second DINOv2 feature forward pass is evaluated and its numerical drift
   is recorded separately; default --inference-tolerance=1e-4.
4. When the same four supports occur under different support-selection seeds,
   bank fingerprints must match; scores are compared with the same GPU inference
   tolerance, and exact numeric hashes are logged for audit but no longer
   required to be bitwise identical.

Optional --math-sdpa disables CUDA flash/memory-efficient scaled-dot-product
attention for this P0 process, forcing the PyTorch math implementation.
This can reduce nondeterministic attention behavior but may be **slower and
consume additional GPU memory**. It also changes low-level numerical execution;
do not compare its scores to an accelerated-attention run as if identical.

**Do NOT simply set --replay-tolerance=1e-4**: that weakens the fixed-feature
bank-retrieval check. Use --inference-tolerance for actual re-encoding
variance instead.

### P0-R2 first run

~~~powershell
git fetch origin
git switch p0-r2-numerical-replay
git status --short
conda activate torch
python experiments/p0_protocol_audit.py --self-test
python experiments/p0_protocol_audit.py --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-policy strict4 --seeds 0 --augmentation-seed 0 --bank-seed 0 --stream-seed 0 --max-test 100 --verify-replay --math-sdpa --device cuda --out-dir "outputs/p0_r2_audiojack_smoke"
~~~

If --math-sdpa causes a CUDA out-of-memory issue, retry after removing
--math-sdpa, using a **new output directory**. Attach the case JSON
repeatability section and the threshold plots for review.

After the initial smoke passes, do a full Audiojack run with this SAME
--math-sdpa mode to avoid changing numerical execution mid-experiment.

---

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

## Historical P0-R1 commands (superseded by P0-R2 above)

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
