# Memory-Governed Online Few-Shot Anomaly Detection

This branch holds a **non-destructive P0 protocol-audit pilot** based on the original
test_phase3_AIF_simplified_v22.py and vmb_visionad_new.py scripts. The original
Python scripts are imported without altering their source logic.

## P0 pilot: what is and is not tested

- P0a: compare genuinely random, strict-four-image support access with a
  pool-assisted four-image support selector (which accesses the normal train pool).
  Both only deploy four normal supports.
- P0b: compare the support **self-matching** initialization to **LOSO**
  initialization excluding the held-out image *and all its augmented references*.
  Optionally include a disjoint-normal commissioning condition (extra data).
- Static P0c audit: compute Accuracy, Balanced Accuracy, AUROC, FPR and FNR on
  exactly the **same frozen test scores** under different initialization thresholds.

**Not completed:** this is NOT the matched static-vs-full online P0c experiment.
The latter requires running the original stream under a common, audited support
bank and theta0 (plus fixed-feedback controls later). Do not use this pilot to
claim online-memory improvements.

### Environment

Install dependencies in the existing PyTorch/CUDA environment. For example:

```bash
pip install -r requirements-p0.txt
```

Ensure PyTorch, torchvision and your local CUDA driver versions are compatible.
DINOv2 weights are loaded by the original vmb_visionad_new.py implementation
and may require cached weights or network access on first use.

### Test immediately

```powershell
git fetch origin
git switch p0-protocol-audit
python experiments/p0_protocol_audit.py --self-test

python experiments/p0_protocol_audit.py `
  --dataset-root "G:/dataset/realiad" `
  --json-dir "G:/dataset/realiad/jsons/realiad_jsons" `
  --class-name audiojack `
  --support-policy strict4 `
  --seeds 0 `
  --max-test 100 `
  --device cuda `
  --out-dir "outputs/p0_audiojack_smoke"
```

PowerShell uses a backtick for command continuation; it must be the final
character of its line. Alternatively put the full command on one line.

### Follow-up comparative audit

After the smoke test passes:

```powershell
python experiments/p0_protocol_audit.py `
  --dataset-root "G:/dataset/realiad" `
  --json-dir "G:/dataset/realiad/jsons/realiad_jsons" `
  --class-name audiojack `
  --support-policy both `
  --seeds 0,1,2 `
  --max-test 0 `
  --device cuda `
  --out-dir "outputs/p0_audiojack_full"
```

An optional commissioning case adds extra normal training images and **must not
be called strict four-shot**:

```powershell
python experiments/p0_protocol_audit.py --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-policy strict4 --seeds 0 --commissioning-size 32 --out-dir "outputs/p0_commissioning"
```

### Outputs

- One JSON per (class, protocol, support_seed) recording support identities,
  data access and threshold initialization.
- One per-image CSV containing **the same static anomaly score** for all threshold
  comparisons.
- p0_batch_summary.json summarizing outcomes for every case.

### Scientific constraints and interpretation

1. The main Real-IAD score configuration used here is identity-only inference,
   multi-layer concat features, image-aggregation=max, 4 augmentations per support,
   Kfix=25000, and fixed_per_image=5000. If the canonical submitted configuration
   differs, configure the CLI explicitly.
2. This pilot constructs an independent, deterministic frozen bank through v22
   primitives with cached support augmentation. It is not a bitwise reproduction
   of the historical v22 seeded bank. Use it only for paired P0 audits until the
   original feature-bank construction is cross-checked.
3. LOSO uses **three references** per fold to score the held-out fourth support.
   Its threshold may be biased upward relative to a fully deployed four-reference
   bank. With four samples the threshold estimate has high variance.
4. Optional normal commissioning consumes further normal images but does not
   insert them into the fixed bank; report this as an extra-data protocol.
5. Offline evaluation labels are used **only to compute metrics**. They must not
   influence support selection, theta choice, or bank construction.
6. Run multi-seed and multi-class confirmatory experiments after the pilot,
   and report paired confidence intervals and classwise failure cases.
7. Do not modify the historical v22 script or use test labels for tuning.

Next stage: a controlled P0c **static vs online** experiment using exactly the
same audited initial support/memory/threshold in both runs.
