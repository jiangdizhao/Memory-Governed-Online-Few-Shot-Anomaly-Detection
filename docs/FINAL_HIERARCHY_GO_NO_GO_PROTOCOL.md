# FINAL Hierarchy Memory go/no-go test — immutable protocol

**Purpose:** Make a **one-time engineering research-direction choice**, NOT prove a method universally ineffective or reproduce the rejected paper's coupled BEA/threshold/signed-correction full system. The question is whether to keep improving the existing v22 hierarchical normal memory, or use strong online Greedy Coreset as substrate for **risk estimation and safe memory insertion (avoid increasing FNR)**.

**No subsequent cohort, parameter sweep, backup tests, or favorable-case selection.** After eight valid trials, we follow the automatic predefined verdict. No more "additional tests are required" before the architecture choice. Engineering defects can be fixed without changing data partition seeds, budgets, groups, outcome measures, or decision thresholds.

## Exactly two learning candidates

- **Greedy Coreset**: established A0/A1 online greedy replacement, 320 FP16 normal prototypes.
- **Legacy Hierarchy**: existing unmodified v22 update/consolidate/retrieve path: STM 128 FP16, LTM 64 FP16, Maturation Buffer 64 FP32, E_min=2, promotion every **32** verified normals, native LTM k=1, mature Buffer eligible for retrieval, neutral fixed usefulness .5.
- Both share **identical** four-support fixed VisionAD/DINOv2 bank, feature extraction, image/patch scoring, candidate patch selection, normal labels, frozen feature backbone. No signed correction, defect bank, BEA, or adaptive threshold. These components are intentionally excluded so that the two-arm comparison measures **memory architecture only**.
- Fixed maximum dynamic-state byte ceiling: 2,800,000 bytes. Their maximum reserved prototype-feature storage plus conservative metadata capacity estimate differs by less than 5%, checked at run start. Actual allocated bytes may vary because some algorithms do not fill their slots; this is a **reported outcome**. The overhead of the shared frozen DINO + fixed support reference and transient k-center computation are excluded from mutable-state capacity, but compute-update seconds are logged.
- This is **not** an exact historical full-system replication. Such a claim would again confound query, threshold, and memory. The original paper's positive effects remain historical evidence, not a shortcut to a fair attribution claim.

## EXACTLY eight GPU streams, no grid

Classes: `switch`, `audiojack`. Support seeds: `0`, `1`. Sequence: `ABA`, `AAA`. Thus 2×2×2 = **8 streams**, with both memory contestants run **within each stream**.

For each class and seed:
- Select exactly four normal *training images*, one from each of **four different real item identities**. Only these four images and derived rotations are read to build static support. LOSO threshold excludes the entire held-out support image and only serves as a **diagnostic operating point**, never to optimize memory.
- Parse Real-IAD *object identity* by stripping the `_C1..C5_<timestamp>` camera suffix from filename. If this convention cannot be verified from the filename, **abort** rather than silently merging/splitting groups. This grouping is verified against the available image naming pattern, not an independent manufacturer part ID.
- Divide test **physical-object groups** without overlap: 24 A-Probe groups, 24 B-Probe groups, 175 Phase-I groups, 175 Phase-II groups, 175 Phase-III groups. All camera views of a physical object stay in only one partition. Four support-object groups are excluded. If there are not enough independent units, fail *before GPU work* and do not adjust the number post-hoc.
- Stream all camera views of each selected object contiguously. Candidate patches are the common fixed-bank top 24 at an image queried as verified normal.
- Make a **precommitted label-blind 3% image query tape**, selected using independent physical-object identities. Query **at most one view per physical object**; only a queried truly normal view updates memory after prediction. Real-IAD JSON is authoritative for each view's label; filename `NG` is not a substitute.
- Stage 1: original RGB appearance mode A; stage 2: either predefined B photometric transformation (`RGB gains=[0.84,0.94,1.08], offset=8`) or identity AAA negative control; stage 3: original A. Same raw groups, supports and queries in paired ABA and AAA; only B images differ.
- A-Probe and B-Probe never enter any online learner, support bank, query controller, or candidate selector.
- Confirm that B actually changes the frozen features (predefined diagnostic threshold). No adaptive transformation tuning is allowed.

## Metrics: physical-object level first

Aggregate each object's anomaly score as the **maximum over its five camera views**, and object ground truth as positive if **any view has a verified anomaly**. Evaluate object-level AUROC on A-Probe at end of Phase I, II, III, and before Phase-III feedback. Evaluate object-level B-Probe AUROC after Phase II. Record image-level AUROC and LOSO fixed threshold FNR/FPR as *secondary descriptive metrics only*, because four-support LOSO is too fragile for a definitive deployed operating-point safety claim.

Measure number of Phase-III new normal human labels to reacquire `AUROC(end-I) − 0.01`; a miss is right-censored/null.

As a **read-only diagnostic** on one final Phase-II Hierarchy memory, compare full normal retrieval with both LTM and Buffer masked, plus separate LTM/Buffer masks. No alternative learning trajectories.

## Precommitted binary decision rule

For each of the **4** class×seed paired units, calculate

`shift advantage = [AUROC_H(ABA, A end-II) − AUROC_H(AAA, A end-II)] − [AUROC_G(ABA, A end-II) − AUROC_G(AAA, A end-II)]`.

A positive number means Hierarchy retained A better specifically during the B shift, after deducting normal AAA updates.

**Keep and improve Hierarchy ONLY IF ALL hold:**

1. Mean shift advantage at least **+0.015 AUROC**.
2. At least **3 of 4** paired units show an advantage of at least **+0.01**.
3. Hierarchy mean A-Probe end-II ABA AUROC is **no worse than Greedy by 0.005**.
4. Hierarchy mean B-Probe end-II ABA AUROC is **no worse than Greedy by 0.01**.
5. In at least **2 of 4** ABA runs, LTM contains an actual prototype and masking both extra banks on the **same checkpoint** reduces A-Probe object AUROC by **at least 0.005**. The report also separately records the independent LTM masking effect.

**Otherwise immediately select `USE_GREEDY_WITH_RISK_SAFE_ADMISSION`.** This is a conservative engineering investment gate, **not** a significance test. An inconclusive/negative result is not an invitation to test more settings. Running out of data or failing support/probe isolation is an **invalid experiment**, not negative evidence: fix the implementation/protocol bug without selecting a different science hypothesis.

### What happens next, without more memory-architecture experiments?

- If Hierarchy passes: retain and improve the existing v22 maturation/LTM governance; target promotion efficiency and FNR protection.
- If it fails: freeze old Hierarchy as historical comparison, adopt Greedy as online memory substrate; focus the next research on **predictive risk before inserting or replacing normal prototypes**. Guard unsafe novel-normal insertion, require conservative verification/supporting evidence and bounded forgetting, use calibrated defect-risk/FNR constraints. This is research development, not another architecture go/no-go test.

## Windows / RTX local commands

Run in repository root, after syncing **this specific branch**, in Conda `torch`:

```powershell
git fetch origin
git switch final-hierarchy-decision
git pull --ff-only origin final-hierarchy-decision
git rev-parse HEAD
conda activate torch
python experiments/run_final_hierarchy_decision.py --preflight --dataset-root "G:/dataset/realiad"
python experiments/run_final_hierarchy_decision.py --all --dataset-root "G:/dataset/realiad" --math-sdpa
```

The *metadata-only* preflight costs no DINO inference and catches unsupported filename identity patterns or insufficient disjoint groups before GPU. `--all` also rechecks metadata before executing exactly eight trials.

Output: `outputs/final_hierarchy_decision/{switch_s0_ABA,...,audiojack_s1_AAA}/summary.json`, per-image data CSV, and **`final_decision.json`** with the frozen automatic verdict. Full results are immutable and source SHA-verified. Re-run `--validate` to audit existing runs without GPU. If a trial crashes, preserve the traceback and affected partial evidence; do not retune parameters to force a favorable result.

**No further comparison is scheduled after the verdict.**
