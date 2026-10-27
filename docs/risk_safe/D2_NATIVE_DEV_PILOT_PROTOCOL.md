# D2 — Paired Native v22 Memory Admission Pilot (engineering + developmental evidence)

**Status as of 2026-10-28:** code committed, GPU experiment **not yet run**. Immutable D1-A/B/C/D evidence and original v22/V0.2 are untouched. This document freezes the FIRST D2 **pilot** configuration and separates it from later full-budget and confirmatory analysis.

## Primary source and bibliography

Original project's "Risk-Safe Memory Admission for v22: A Deployment-Oriented Research Strategy.pdf" (Priority A, baseline table, fixed-probe metrics, risk-prediction-before-probe-damage test) and "Risk-Safe Memory Admission-7DAY-PLANS.docx" (Section 3.2, D2, Section 5). These require sidecar/correction OFF in mechanism mode, the SAME frozen Q-A tape, support, candidates, scorer, and byte capacity for every admission arm, and an independent evaluation-only defect probe.

Related references checked:
- HITL Memory Correction, Abbas et al. (2026), https://arxiv.org/abs/2608.17775 : original false-positive memory bank correction uses a novelty gate calibrated on **pool-normal** nearest-neighbor distances, and the paper evaluates reviewed samples/review pools. The "causal_novelty" implemented here uses only PAST queried verified-normal candidate scores, and cannot be misrepresented as a faithful authors' implementation.
- ContCore, Jung et al. (2026), https://arxiv.org/abs/2608.15277 and https://github.com/jungyg/ContCore : a taskwise greedy coreset expansion/consolidation method. Our flat Greedy is the **existing V0.2 baseline**, not "official ContCore". Native ContCore replication is outside this D2 pilot.
- FOADS / K-NG, Knowledge-Based Systems (2024), DOI 10.1016/j.knosys.2024.112168, https://arxiv.org/abs/2403.18201 : adapts normality from UNLABELED online data via neural gas and pseudo-anomaly filtering. The supervision assumption differs from verified-only Q-A; original FOADS is excluded from the same-supervision pairwise ranking and reserved for separate benchmarking.
- Real-IAD, Wang et al., CVPR (2024), https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Real-IAD_A_Real-World_Multi-View_Dataset_for_Benchmarking_Versatile_Industrial_Anomaly_CVPR_2024_paper.html : five cameras per capture sample. An NG group can contain image-level normal camera views; never propagate labels.

## Hypotheses and arms (pilot)

H1: normal-memory updates can lower held-out defect image scores despite human-confirmed normal input. H2: **past** verified defect anchor erosion contains predictive information about held-out defect score erosion. H3: CDARG should improve FNR-erosion/FPR-gain trade-off at comparable write rate; **H3 is NOT established by one smoke or one class.**

All arms get exactly the same Q-A query events and native image-level normal candidates (static pre-score and native 8/16/24 patch rankings). Static bank, DINOv2 and vision scorer are frozen; CDARG, novelty, Greedy and Reservoir use different causal dynamic admission only.

Method nomenclature:
- Static: no normal memory writes.
- Append: atomic event, rejects if the event cannot fit within identical dynamic prototype upper bound.
- Greedy: exact v02_memory.greedy_slot append/replacement, using D1-D verified native-normalized rows.
- Reservoir: seeded dynamic-patch replacement with the same per-layer cap, static fixed bank unchanged.
- Causal HITL-style Novelty: candidate mean nearest cosine distance to prior static+dynamic normal prototypes, with a causal median threshold calculated from only earlier verified-normal candidates. Cold first eligible normal admitted. Not authors' review-pool-calibrated algorithm.
- CDARG: original PDF's positive-part U_N hinge, R_D as normalized positive reduction of historical defect scores, hard H_D, and zero-defect-evidence means unavailable/reject. Uses the full native score of prior **entire verified camera views** as anchors in this phase. This is deliberately declared a "whole-image-native-anchor" variant, NOT the final 8–16 protected-patch proposal. The protected-patch variant and SAEG belong to D3, not a silent substitution here.

**Uncalibrated pilot coefficients:** epsilon_D=0.1 (normalized score units), lambda_D=1, tau_U=0, m_D=0.25*support-only LOSO sigma, sigma_N=LOSO sigma, sigma_floor=0.01, uniform anchor weights=1. Never treat these as selected/published or deployed hyperparameters. The purpose is to exercise actual memory commits, probe measurement and evidence logging. Do NOT use evaluation probe outcomes to tune these coefficients for subsequent confirmatory reporting.

**Important D1-D finding:** the LOSO theta0 was above every first B16 queried image score, including five image-level defects. This creates a potential floor in measured FNR and zero U_N. Report RAW held-out defect-score erosion and FPR gain separately; do not claim "no safety damage" merely because FNR is already at 1.0. The original strategy PDF includes an OUTER positive part on U_N; the D2 code now implements this explicitly, so do not compare D1-D diagnostic U_N numbers as if this were identical to pre-change code in every possible case.

## Post-pilot audit: immutable B8 evidence and latency correction (2026-10-28)

Commit `f1a7e6af4ac4121a02510aab7439535187298820` archived the first bounded eight-query pilot. Six arms made actual writes as intended (Static 0, Append/Greedy/Reservoir 48 patches each, causal novelty 24, CDARG 8). On the 20 held-out Probe A images (14 normal, six defect), **all arms had FNR=1 and FPR=0** at the fixed LOSO theta0, so this pilot did NOT establish differential threshold-classification safety. Terminal AUROC: Static .559524, Append/Greedy/Reservoir .654762, Novelty .619048, CDARG .583333. The three unconditional baselines had IDENTICAL dynamic-bank digests because no arm approached capacity: their apparent parity does not test replacement policy. Mean positive heldout-defect-score erosion: unconditional updates .001826, Novelty .000420, CDARG 0. These are descriptive tiny-probe numbers, not formal evidence that CDARG is safer at comparable adaptation or write count.

Two post-first-defect CDARG candidate events existed: at stream t=1085, observed normal utility U_N=0, anchor-estimated normalized risk R_D=.138958 (above pilot epsilon=.1) but independently evaluated mean positive defect-probe erosion was **zero**; at t=1159, R_D=0, mean heldout erosion=0 and one real write occurred. Thus the pilot risk-prediction Spearman status is `insufficient_evidence` (N=2), and the t=1085 rejection MAY reflect anchor coverage mismatch or over-conservatism. Do not represent this pair as validated risk prediction.

**Timing confound detected and corrected after this archived result:** in the first runner, per-arm `admission_seconds` accidentally included **offline heldout-defect-probe counterfactual scorer calls** for CDARG, but not for other arms. Therefore the archived pilot's CDARG-vs-baseline timing comparison is **INVALID**; raw times must not appear in a paper as online decision latency. The revised runner separates `admission_seconds` from `offline_probe_diagnostic_seconds`, includes both in per-event and summary CSVs, and includes a unit-test invariant for timing separation. GPU host-wall measurements remain exploratory, not synchronized CUDA kernel benchmarks. The first evidence files remain immutable and continue to identify their original generating commit.

**Next GO-only run (optional after CPU tests):** complete the same frozen audiojack B16 Q-A tape (`--max-query-events 16`) with a larger *still development-only* Probe-A subset (`--probe-groups-per-type 4`, 40 camera images). Choose a NEW out-dir `outputs/risk_safe_sprint_20261027/d2_audiojack_s0_q37_b16_full16_probe4`. The same strict four-shot, LOSO theta0, Q-A query seed, random stream and untuned CDARG settings remain fixed. Probe selection changes the evaluation data footprint and is for development *diagnosis*, not confirmatory reporting. These N4 probe groups are not a calibrated optimum.

## Pilot freeze: fixed Q-A, Random, no label distribution engineering

Stage D2 pilot #1: audiojack, support seed=0, query seed=37, budget B=16, max first **8** queried events, Probe-A first **2 normal +2 defect-containing official groups** (5 camera views per group; 20 images). All six arms use the same 20 images and same global checkpoint positions 0/875/1750/2625. Pilot #2 may process all 16 queries with N=2 probe groups per type if pilot #1 succeeds, but MUST use a new output folder. The original B16 total label composition is 11 verified normals and 5 verified defects; first eight queries have 6 normal, 2 defect.

**Critical:** a maximum of 8 queried events at B16 is a TRUNCATED replay, not a "full B16" condition. The manifest explicitly records full_budget_completed=false. Do not compare its final safety statistics to a complete B16 run.

All probes are selected BEFORE method outcomes from D1-A's evaluation-only Probe A cohort, by sorted sample IDs and OFFLINE group type. All five camera image labels are respected independently; their IDs and labels are held in the offline evaluator and NEVER passed to Admission, Query, novelty cutoff or threshold selection. Probe B is untouched, reserved for additional later evaluation. The pilot probe is small and correlated: not a publication-scale independent dataset or valid significance basis.

Fixed stream checkpoint positions and baseline score at time zero are identical across methods. Since Q-A is exogenous and unqueried images cause NO state updates in this mechanism-mode run, skipping feature extraction for unqueried images is acceptable for **memory/probe trajectories**, but it does NOT measure deployment inference cost/latency nor classify all stream images. Metrics are **discrete checkpoint averages**, not exact integrals over 2625 decisions.

## Files written if and only if run completes

A NEW immutable run directory contains:
- summary.json: model/config/LOSO, per-arm FNR-degradation over fixed checkpoints, FPR gain, AUROC/AUPRC, prototype counts, admission counts and high-level risk-prediction statistics.
- baseline_comparison_dev.csv: one paired method row per arm with fixed-checkpoint summary metrics; corrected runs have separated online admission seconds and offline probe diagnostic seconds.
- probe_snapshots.csv: per-arm per-checkpoint probe image metrics, raw score erosion and bank SHA.
- candidate_decisions.csv: all queried normal events per arm, native event-level candidate count, actual commit/reject, normal utility, historical defect risk, retrospective independent Probe damage (CDARG only), prototype deltas and latency.
- query_ledger.csv: acquired Q-A image-level labels (trusted experiment log), revealed after pre-query scores.
- source_manifest.json: producing Git SHA/branch, source-file hashes, Real-IAD class metadata/immutable D1-B query manifest hashes, all output SHA-256 and environment.

Do NOT overwrite D1-D output directories or move JSON/CSV to an unrelated branch. If any run or test fails, preserve traceback and contact repository maintainer for source correction, do not push invalid evidence.

## Local Windows PowerShell (single-line commands)

Run in E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection after clean git status and fast-forward pull.

Compile:

    & "D:\anaconda3\envs\torch\python.exe" -m py_compile experiments/risk_safe/d2_baselines.py experiments/risk_safe/d2_metrics.py experiments/risk_safe/d2_native_replay.py tests/test_risk_safe_d2.py

Cheap DATASET/TAPE preflight only (no DINOv2 or GPU); execute this before the real replay:

    & "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.d2_native_replay --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 16 --max-query-events 8 --probe-groups-per-type 2 --d1b-manifest-dir "outputs/risk_safe_sprint_20261027/d1b_protocol_v1" --preflight-only

CPU tests (including all D1 stages):

    & "D:\anaconda3\envs\torch\python.exe" -m unittest discover -s tests -p "test_risk_safe_d1*.py" -v

    & "D:\anaconda3\envs\torch\python.exe" -m unittest discover -s tests -p "test_risk_safe_d2.py" -v

GPU first bounded pilot (never automatic batch launch):

    & "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.d2_native_replay --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 16 --max-query-events 8 --probe-groups-per-type 2 --device cuda --math-sdpa --d1b-manifest-dir "outputs/risk_safe_sprint_20261027/d1b_protocol_v1" --out-dir "outputs/risk_safe_sprint_20261027/d2_audiojack_s0_q37_b16_pilot"

Read-only result/source/input SHA validation, no GPU compute:

    & "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.d2_native_replay --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 16 --max-query-events 8 --probe-groups-per-type 2 --device cuda --math-sdpa --d1b-manifest-dir "outputs/risk_safe_sprint_20261027/d1b_protocol_v1" --out-dir "outputs/risk_safe_sprint_20261027/d2_audiojack_s0_q37_b16_pilot" --verify-only

After success: git add ONLY the pilot output folder; git diff --cached --stat; git diff --cached --check; git commit; git push; git rev-parse HEAD.

## Acceptance and hard limitations

- CPU tests passed, frozen Q-A tape digest matched, four-shot support/probe/stream samples disjoint; no future/query/probe label to online arms; static support and baseline bank unchanged.
- All six arms use precisely the same pre-label candidate patch selection, but their dynamic memory differs due to admission, as intended.
- CDARG's gate receives only earlier verified normal/defect **whole-image** anchors; its independent defect-probe candidate damage is computed AFTER decision, recorded as offline diagnostic. Note risk selection bias from small anchor sets.
- At least some arms truly commit dynamic prototypes; verify bank SHA/version changes, admissions and patch counts. The Static baseline must remain unchanged.
- No zero-defect-anchor case can be called risk=0. Never imply acceptance is statistically risk-certified.
- Probe metrics represent only a tiny nonindependent pre-selected subset. No statistical power or confirmation-stage claims.
- If observed scores have no FNR dynamic range because theta0 is too high, state UNDERPOWERED/FNR-FLOOR and use continuous defect erosion to study memory effects. Do not tweak theta0 using the evaluation probe.
- SAEG, official ContCore, original FOADS, causal protected-patch CDARG variant, long-stream all-images scoring, threshold/Query closed-loop, sample-group bootstrap and cross-class confirmation remain future work.

**First GO/NO-GO decision only:** if the pilot executes true commits with intact causal constraints and immutable outputs, proceed to full B16 and more probe coverage. It does **not** decide whether CDARG is scientifically effective.
