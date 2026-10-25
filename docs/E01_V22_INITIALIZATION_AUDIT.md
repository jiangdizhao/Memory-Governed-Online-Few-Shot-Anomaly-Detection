# E01 — original v22 vs P0 cold-start equivalence audit

**Scope:** Real-IAD, one class, single mode, strict four-shot, identity-only
view, no correction or set membership, no online state updates.

**Authority:** the unchanged `test_phase3_AIF_simplified_v22.py` is the reference.
The P0-R2 `experiments/p0_protocol_audit.py` is a *pilot* and is not assumed
equivalent to v22. A failed equivalence check is a valid scientific result.

## What E01 actually runs

1. Calls original `run_phase3_streaming_aif(max_samples=0)` with
   `--support-select legacy` semantics, `use_modes=False`, one identity
   view, fixed-bank max capacity 25,000 and per-reference patch cap 5,000.
2. Records the built-in original v22 `init_dump_json`, intercepts the
   native Fixed Bank initializer without modifying it, fingerprints actual
   prototype bytes and captures original augmentation view image hashes.
   No evaluation-stream label is accessed in this step.
3. Reconstructs the existing P0-R2 pilot bank using the **same exact four
   native v22 support image paths**. Records the independent/native P0
   support selection separately; this removes support *identity* as a
   confound in the bank content comparison.
4. Uses a **shared DINOv2 query encoding** for both systems to score the
   exact same held-out test image paths. No test labels or queries enter
   the bank or threshold initializer.
5. Writes `e01_equivalence_report.json` and
   `e01_paired_frozen_probe_scores.csv`.

This analysis is *diagnostic*. It does not derive a better test threshold,
evaluate online adaptation, or establish fidelity to the old 30-class runs.

## First local test: Windows PowerShell

In the existing repository folder (with no uncommitted changes to discard):

~~~powershell
git fetch origin
git switch e01-v22-initialization-audit
git rev-parse --short HEAD
git status --short
conda activate torch
python experiments/p0_v22_equivalence.py --self-test
~~~

Then the actual E01 small GPU audit (recommended 12 paired frozen probes):

~~~powershell
python experiments/p0_v22_equivalence.py --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --legacy-seed 0 --augmentation-seed 0 --bank-seed 0 --stream-seed 0 --probe-count 12 --math-sdpa --device cuda --out-dir "outputs/e01_audiojack_support0"
~~~

If your GPU runs out of memory with `--math-sdpa`, try without it
(using a different output directory), and report the difference in mode.
Do not vary attention mode within the same scientific comparison.

To test support seed 1, change `--support-seed 1` and
`--out-dir outputs/e01_audiojack_support1`, keeping all other flags.

**Do NOT pass `--max-test`** — that belongs to the prior P0 runner.
E01's `--probe-count` controls only how many held-out images are used for
frozen paired scoring. v22 itself processes **zero** online test samples.

## How to interpret

- `supports.native_order_identical`: native P0 support selection equals
  original v22's selection. When false, the P0 comparison STILL forces
  the original four v22 support files as inputs, so the bank comparison is
  meaningful.
- `augmentation.fixed_pass_fingerprints_equal`: both initialization paths
  produced the same pixels for their augmented references. Difference is
  expected if their randomization schemes differ.
- `memory.*.total_prototypes`: ACTUAL stored count, not configured maximum.
- `memory.*.sha256`: bytewise frozen-bank identity for supported single view.
- `threshold.*`: original v22 actual theta0 from its controller and P0
  simplified score mean + 2.5 SD. A difference is not automatically a bug.
- `paired_probe`: paired score gaps on the SAME query feature tensors,
  avoiding redundant encoder forward-pass noise.
- `verdict.status`: `not_equivalent` means at least one condition differs.
  It is **not** a failure of the E01 code or a model regression. The script
  intentionally records evidence rather than changing either model.

IMPORTANT: legacy v22 performs concept seeding and fixed-bank augmentation
passes sequentially using shared RNG state; P0-R2 independently derives
per-image/view random seeds and uses a simplified concept initialization.
Consequently exact bank hashes may differ even with the same image paths.
Don't infer novelty or online learning gains from these differences.

## E01 acceptance and next work

E01 engineering acceptance means:
- original v22 initialized successfully with zero online test samples;
- the same four support image paths were available to both banks;
- audit includes original v22 and P0 prototype shapes/hashes and actual
  initial scores and threshold;
- matched frozen probes yield finite values and a paired difference report;
- no pretrained weights, initial backbone or original scripts were modified.

It does NOT require the existing P0 bank to be equivalent. If it isn't,
future matched online experiments should use the **canonical v22 initializer**
on both branches, not artificially change v22 to resemble P0.

Only after E01 should E02/E03 use controlled query tapes and ensure score
before feedback before state update.
