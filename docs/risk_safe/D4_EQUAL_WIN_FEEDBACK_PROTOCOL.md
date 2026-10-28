# D4 — Three ablations: equal native wins, feedback credit, shuffled defect attribution

Status 2026-10-28: **GITHUB CODE COMMITTED; no claim of Windows GPU execution until you run it.**  This is a development-only POST-B16 checkpoint evaluation, not a full continuously updating prequential controller, nor a safety guarantee.

## Frozen sources / baseline

Rebuild original D3-C/D3 native full B16 Q-A tape, audiojack support seed0/query seed37, q90 training-normal-only registry, Greedy(104 prototypes) and causal Novelty(48 prototypes), NO sidecar or correction. Reproduce ALL original D3 480 scores to <=1e-6, along with 80 per-image Global original native score checks and full D2-A FPR/FNR/prototype-count checks. Original frozen Static support and dynamic bank content are immutable.

Only original 16 queried camera views provide feedback: 11 verified normal images and 5 verified defect images. At this POST-B16 checkpoint, all 16 labels are available BEFORE final Probe A evaluation. No labels, anomaly masks, or scores from either Probe A or Probe B may train/select a policy. Probe A includes only 40 previously established development images from 8 official sample groups. Probe B untouched.

**Three ablations in one GPU job**

A. Exact equal-win budget: detect original native unrestricted Global dynamic top5 winners (not merely eligible references); each policy chooses exactly K=min(ceil(0.02 * N_image_patches), number_of_global_winners) patch indices. Chosen patch has ALL original dynamic references available (same exact native mean-top5); unchosen patch uses frozen Static only. This is a fixed-PATCH-COUNT budget, not equal image score influence. No policy gets to alter Greedy/Novelty prototypes. Avoid label/score-aware oracle budget selection.

B. Feedback attribution: compute image and patch features for the 16 *already revealed* Q-A images ONLY. Native static-vs-global patch delta is treated as candidate influence. Normal feedback uses **leave-one-official-sample-out** dynamic bank when constructing its benefit evidence: a stored prototype derived from sample S is ineligible when scoring training-normal S. This prevents an image's own stored features from being credited with that image's improvement. Reward confirmed-normal FP patch corrections more heavily, pre-registered fixed factor, with no inference from Probe. Defect feedback uses native static high anomaly top32 patch *hypotheses* inside defect-labeled queried images, multiplied by static-to-global positive patch erosion. No pixel ground truth exists to the controller. Frozen DINO features + (u,v) + camera context yield an RBF-like nonparametric empirical patch credit. Fixed bandwidth=0.1, spatial bandwidth=0.25, risk penalty multiplier=1. Credit is source-only and prior-shrunk; no learned DINO, VAE, adaptive radius or Probe optimization.

C. Shuffled-defect attribution: among the same defect evidence rows, reproducibly shuffle feature/position/camera assignment relative to original risk weights, leaving the *number of observed defects*, source labels and risk-weight marginal distribution unchanged. This tests if the location/feature association of observed risk has incremental predictive value. Three fixed SHA256-derived seeds.

| Policy | Uses features/space | Revealed normal feedback | Revealed defect feedback | Seeds |
|---|---|---|---|---|
| random | native winner candidates, random winner selection | no | no | 0,1,2 |
| spatial | native top5 winning-prototype spatial proximity | no | no | 0 |
| normal_feedback | same frozen features and spatial context | yes, LOSO | no | 0 |
| normal_defect_feedback | same frozen features and spatial context | yes, LOSO | true B16 defect attribution | 0 |
| shuffled_defect_feedback | same as feedback model | yes, LOSO | shuffled B16 defect attribution | 0,1,2 |

Thus each memory arm has **9 scored conditions × 40 Probe A images**, yielding 720 score rows (two memory arms), and **18 summary rows**, plus source feedback diagnostic report. All strategies are guaranteed equal *native dynamic winning patch count* on every identical query image: selected patches use the complete native Global dynamic top5 and all others use Static. FPR/FNR, AUROC/AUPRC, continuous normal score gain vs Static and positive defect-image score erosion vs Static are reported. Winning patches count is NOT a guarantee of matched image-level effects or model utility.

**Scientific stopping rules**

Do not claim feedback is useful if defect evidence witness count=0, risk score constant, normal-only and normal+defect are identical, or true vs shuffled attribution does not change ranking. All such failures must be reported. Only B16 image-level feedback is available. Static-top32 suspected defect locations are *NOT real defect masks*. Current pilot does NOT perform unbiased calibrated risk uncertainty, online incremental credit updates, or a formal safety bound.

GO only if real normal+defect feedback improves the adaptation-erosion Pareto plane over same exact-win random/spatial/normal-only AND randomized defect attribution on Probe A, with credible risk witness coverage. Then FREEZE method and carry out incremental prequential replay plus independent Probe B and other classes/seeds. A single B16 seed/40 camera views is exploratory, no population-level significance or publication-worthy superiority by itself.

No original D1/D2/D3/D3-C evidence or old source code is overwritten. D4 has its own immutable atomic output directory. The manifest records exact original D2-A, D3, tape, calibration, dataset JSON, source and output hashes, true computing device. Read-only verification validates the exact historical generating Git revision (allowing only genuine Windows CRLF checkout normalization), input/output hashes and config.

## Windows PowerShell: all commands are ONE LINE EACH

Project workdir: E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection

1. `git status --short`
2. `git pull --ff-only origin risk-safe-admission-sprint-20261027`
3. `git rev-parse HEAD`
4. Syntax:
& "D:\anaconda3\envs\torch\python.exe" -m py_compile experiments/risk_safe/d4_retrieval_rights.py experiments/risk_safe/d4_equal_win_feedback_pilot.py tests/test_risk_safe_d4.py
5. New CPU tests:
& "D:\anaconda3\envs\torch\python.exe" -m unittest discover -s tests -p "test_risk_safe_d4.py" -v
6. D3/D3-C regressions:
& "D:\anaconda3\envs\torch\python.exe" -m unittest discover -s tests -p "test_risk_safe_d3*.py" -v
7. Metadata preflight (no CUDA):
& "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.d4_equal_win_feedback_pilot --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 16 --max-query-events 16 --probe-groups-per-type 4 --threshold-registry "outputs/risk_safe_sprint_20261027/d2a_audiojack_s0_train64_fixed_thresholds/threshold_registry.json" --fixed-quantile 0.90 --reference-d2a-summary "outputs/risk_safe_sprint_20261027/d2a_audiojack_s0_q37_b16_q90_probe4/summary.json" --reference-d3-dir "outputs/risk_safe_sprint_20261027/d3_audiojack_s0_q37_b16_q90_probe4_fixedk" --d1b-manifest-dir "outputs/risk_safe_sprint_20261027/d1b_protocol_v1" --preflight-only
8. Native CUDA experiment:
& "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.d4_equal_win_feedback_pilot --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 16 --max-query-events 16 --probe-groups-per-type 4 --threshold-registry "outputs/risk_safe_sprint_20261027/d2a_audiojack_s0_train64_fixed_thresholds/threshold_registry.json" --fixed-quantile 0.90 --reference-d2a-summary "outputs/risk_safe_sprint_20261027/d2a_audiojack_s0_q37_b16_q90_probe4/summary.json" --reference-d3-dir "outputs/risk_safe_sprint_20261027/d3_audiojack_s0_q37_b16_q90_probe4_fixedk" --d1b-manifest-dir "outputs/risk_safe_sprint_20261027/d1b_protocol_v1" --device cuda --math-sdpa --out-dir "outputs/risk_safe_sprint_20261027/d4_audiojack_s0_q37_b16_q90_equalwin_feedback_v1"
9. Read-only verify: append `--verify-only` to command 8; this never reruns GPU.
10. Only after successful verification: stage ONLY the NEW D4 output directory; inspect + commit + push; send HEAD. Do NOT use Probe B or overwrite any old outputs.

Current tests are committed but cannot be run in your Windows Real-IAD / CUDA environment by GitHub connector. Stop if CPU tests or preflight fail and provide traceback rather than executing more expensive GPU.
