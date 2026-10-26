# V0.2: Paired Cold-Start / Mature-History Memory Architecture Experiment

**Status:** Implemented for LOCAL GPU verification. CPU/CI checks do not establish Real-IAD results.  
**Branch:** `v02-paired-cold-mature-architecture` (independent from archived R6, TMRA, final-hierarchy-decision, and the rejected manuscript).  
**Decision target:** Which external normal-memory substrate should receive the next research investment: original-scale STM–Maturation–LTM, or a strong capacity-scalable online Greedy Coreset with future risk-safe admission?

## Scope and epistemic limitations

This is **not** a full recreation of v22's BEA/Sidecar/Signed Correction closed loop. The backbone, fixed support and all candidate patch selection are shared; online query positions are a precommitted 3% label-blind tape (one camera view at most per physical object); only verified normals are written after prediction. The static fixed-bank score drives an identical variable 8/16/24 candidate-patch budget and the v22 normal-write boundary/FP usefulness calculation with **novelty fixed at zero**. This preserves variable evidence and original v22 MaturationBuffer / LTM update and native separate-source retrieval, while deliberately removing controller-specific selection bias. Interpret a result as memory architecture evidence, not complete original-system superiority.

**No oracle probe labels or future test labels enter memory updates, query decisions, threshold or tuning.** The only label-aware sample selection is choosing balanced A/B **evaluation-only** probe objects. That stratification affects measured AUROC precision and must be disclosed.

## One independent experiment, four paired arms

| Arm | Additional verified history | Dynamic bank |
|---|---|---|
| greedy_cold | 0 | greedy nearest-neighbour-pair redundancy swap |
| hierarchy_cold | 0 | native v22 STM–Maturation–LTM |
| greedy_mature | 64 normal-labelled train objects | same greedy rule |
| hierarchy_mature | identical 64 labelled objects | native v22 hierarchy |

The immutable four original support images build a 25,000-feature capped fixed normal bank (5,000 per image), support augmentation 4, concat features, identity view. Frozen DINOv2/VisionAD-style scorer, same raw images, query tape, candidate patch ranking and scored-before-feedback semantics.

The **mature** setting is **history-assisted**, NOT a 4-image-total label budget. Exactly 64 additional verified-normal training-object images enter mature-only dynamic memory as 64 distinct events BEFORE stream time t=1. All four arms otherwise share support and online tape. History images are identical across the three support seeds of each category and do not overlap the four supports, A/B probes, or online test stream. Each extra label is charged and listed. Cold arms start with empty dynamic memory.

Proposed full v22-scale hierarchy budget per layer/view: STM=1024 FP16, LTM=512 FP16, Buffer=4096 FP32, min distinct verification events=2, promote every 32 normal updates, native mature Buffer k=1 and LTM k=1. Utility-aware STM retention is enabled. The implementation uses static-scoring evidence for outcome and threshold margin, not a full dynamically calibrated sidecar.

Greedy's prototype ceiling is **derived from the same maximum persistent dynamic feature+conservative metadata bytes** rather than clamped at its historical 1024. At full capacity the pre-registered nominal feature+metadata budget must match within 5%. Actual allocated tensor storage, updates, and total runtime are recorded separately; shared fixed bank, frozen encoder, and temporary greedy pairwise calculations are excluded from the mutable persistent limit. Do not interpret equal nominal capacity as equal occupancy or equal compute.

A0 greedy's min-redundancy swap is implemented using chunked distance queries to avoid a transient global O(K²) matrix; it is an online heuristic and not a faithful reproduction of published ContCore.

## Pre-registered data and experiment matrix

**Six named Real-IAD categories:** audiojack, bottle_cap, eraser, plastic_plug, switch, rolled_strip_base.  
**Three support seeds:** 0, 1, 2.  
**Two paired shifts:** ABA and AAA.  
**Total:** 6 × 3 × 2 = **36 streams**, each evaluating four memory states.

For each class, exactly the SAME object-disjoint test split and SAME label-blind query tape is used across support seeds and ABA/AAA. Only four normal support originals differ across seeds. Object identity follows the repository's validated Real-IAD filename suffix pattern (`_C1..C5_<14-digit timestamp>`); invalid paths fail closed.

Evaluation-only stratified A and B probes: each contains 24 normal + 24 anomalous **physical-object groups**, with multiple camera views never crossing data partitions. Online stream uses three groups of 175 further distinct physical objects. Raw stream objects and query positions are identical under ABA and AAA. A is original appearance. The ONLY B intervention is fixed photometric RGB gain [0.84, 0.94, 1.08] plus offset 8. This is a synthetic appearance change, NOT factory-verified concept drift. Every category must pass the metadata-only group-count and overlap preflight before any GPU experiment begins.

The 3% query tape is sampled from physical objects, at most one view per object, independent of labels. No unqueried anomaly may enter normal memory. LOSO support exclusion estimates an initial normal-only threshold; it is diagnostic only and is NOT claimed to guarantee deployed FPR/FNR. On-line comparisons primarily use object-level ROC-AUC from a fixed A-Probe and B-Probe. Physical-object score = maximum over available camera-view anomaly scores; object defect label = any abnormal annotated view.

## Primary/secondary endpoints and frozen research decision

Primary paired shift advantage for mature (and separately cold):
```
(H_ABA_A_endII - H_AAA_A_endII) - (G_ABA_A_endII - G_AAA_A_endII)
```
All terms are **object-level AUROC on the SAME A-Probe**. A positive difference alone does not prove higher absolute detection accuracy; additionally compare H-vs-G A and B end-II AUROC. Record cold and mature separately, not pooled.

Same-state read-only source attribution at A end-II masks LTM, Buffer, or both from the mature/cold hierarchy scorer. This does not generate a second learning trajectory. Record LTM prototypes formed, maturation occupancy, verification count and per-operation elapsed update seconds, plus retained dynamic tensor capacity. Phase-III reacquisition is the first number of additional normal labels needed to return to A end-I AUROC minus 0.01; unrecovered is null/censored.

**Pre-committed conservative engineering choice (NOT a significance test):** Retain the hierarchy if ALL:
1. Macro-over-classes mature retention advantage >= +0.015.
2. At least 4/6 classes have positive mean advantage (seeds averaged within class).
3. Mean mature absolute A AUROC gap (H-G) >= -0.01.
4. Mean mature absolute B AUROC gap (H-G) >= -0.01.
5. Mean cold absolute A AUROC gap (H-G) >= -0.02.
6. At least two distinct classes show a mature hierarchy LTM prototype **and** paired read-only no-extra effect >= +0.005 object AUROC in at least one support seed.

Otherwise `USE_GREEDY_WITH_RISK_SAFE_ADMISSION`. This is a **research investment decision**, not a claim of universal inferiority/superiority. Runtime cost and memory occupancy are reported alongside the decision, but there is no pre-authorised numerical update-latency gate; do not falsely represent it as one.

## One command / Windows PowerShell

From repository root (after initial clone), in a PowerShell where Conda is initialized:

```powershell
.\run_v02_experiment.ps1
```

It syncs the V0.2 branch, activates `torch`, compiles new code, runs CPU-only unit/self tests, preflights the dataset, runs all 36 cases in order, validates completed cases, and creates:

```
outputs/v02_paired_cold_mature/v02_final_decision.json
```

Direct Python equivalent (once already on the new branch):

```powershell
conda activate torch
python experiments/run_v02_architecture.py --all --dataset-root "G:/dataset/realiad" --math-sdpa
```

Restart/resume: rerun **exactly the same command**. Successful result directories are validated and reused. An incomplete/invalid condition fails closed; its evidence and `*.run.log` are preserved. Do not delete existing conditions automatically or retune parameters after seeing metrics. No manual per-class commands needed during successful execution. A GPU crash, invalid dataset protocol, or missing model weights **requires intervention**; automatic repair is not safe.

### Outcome provenance

Each case stores `summary.json`, `stream.csv`, `probe_scores.csv`, `memory_snapshots.csv`, `recovery.csv` and adjacent `*.run.log`. Source Python hashes, four support paths, history paths, all probe/stream physical-object IDs, query positions, observed normal/defect query counts, LOSO diagnostics, reserved/actual memory, and per-arm results are recorded. The runner validates row-level query positions, identities, exact source file SHA-256, and paired ABA/AAA input manifests before writing the final decision.

## Limitations requiring explicit manuscript disclosure

- Six intentionally selected diagnostic categories, not a representative random sample of 30.
- Three support seeds share evaluation objects; these are not 18 independent classes or separate stream/query seeds.
- The extra 64 normal training labels are a separate history-assisted supervision regime; cold and mature must not be mislabelled the same label budget.
- Shared frozen utility and static candidate selection differ from the original paper's fully coupled v22 BEA/Sidecar/correction.
- One B shift is only a synthetic appearance perturbation, not a measured A→B→A production mode transition.
- Under bounded 3% human feedback, LTM can fail to mature. This must be recorded as an actual mechanism limitation; do not manipulate stream length or feedback after results to force promotion.
- CI/offline tests do not verify Real-IAD GPU feature extraction. Scientific claims require validated row-level local GPU outputs.

**No modification to historical v22, rejected paper, R6, A1-H, Bridge, TMRA or old final-decision archives.**
