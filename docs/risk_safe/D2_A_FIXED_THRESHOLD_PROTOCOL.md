# D2-A: fixed-threshold identifiability and online CDARG validation

Status (2026-10-28): code and preregistration only, Windows GPU outcomes pending. Immutable D1-A/B/C/D and first LOSO D2 evidence are not edited. The old audiojack Probe A pilot was FNR=1 and FPR=0 for all arms at LOSO theta=0.622621, with CDARG U_N=0. The purpose of Stage A is to change the fixed operating point using development NORMAL data only, and re-run fresh causal admissions rather than merely thresholding an old score trajectory.

## Frozen normal-only calibration rule

1. Reconstruct the identical strict 4-shot static VisionAD bank for audiojack support seed=0 with the original V0.2 / D1-D scorer and the same LOSO reference.
2. Use the pre-existing 64 mature-history TRAIN verified-normal views, one per distinct official Real-IAD sample key, selected by the original D1-A split seeds. This is not a new split. These 64 sample keys are disjoint from all 4-shot support seeds, the 3 streaming phases, Probe A and Probe B. History is never used in the online memory pilot.
3. Score all 64 training-normal views under the frozen static native image scorer. Use the HIGHER empirical order statistic. Freeze two thresholds before any new Probe result:
       theta_q90 = sorted_normal_scores[ceil(63*0.90)]
       theta_q95 = sorted_normal_scores[ceil(63*0.95)]
4. Freeze the sample std (ddof=1) of training-normal scores as the normal-score scale for the original CDARG risk normaliser and margin. Record original LOSO theta/sigma for diagnostic comparison. This changes both theta and sigma relative to LOSO, while comparing q90 against q95 holds sigma fixed. Do NOT attribute old-versus-new differences exclusively to theta without later holding sigma constant.
5. Store both q90/q95 in ONE immutable threshold_registry.json, with 64 verified-normal scores, chosen sample and camera paths, dataset/source SHA-256 and a separately verified source_manifest.json. The runner rejects nonregistered q, wrong support, changed dataset and missing manifest. Probe images/labels and actual online queried labels are forbidden for fitting.

Quantiles represent development normal-score operating points, not promised test FPR/FNR. The original fixed LOSO remains the historical baseline. Both q90 and q95 are preregistered sensitivity conditions, NOT alternatives to select by observing Probe accuracy.

## Controlled online experiment

Use exactly the D2 six methods and native scorer: Static, Append, Greedy, Reservoir, causal HITL-style Novelty and CDARG. Run separate independent bank trajectories for q90 AND q95, with audiojack, 4-shot seed 0, Q-A query seed 37, B=16 full 16 queries, RANDOM order and 4 normal plus 4 defect-containing Probe A sample groups (40 camera images). Query tape, image-specific feedback, support, native 8/16/24 candidate rules, frozen bank capacities and six-arm evaluation checkpoint positions t=0/875/1750/2625 remain the same. Sidecar and Signed Correction are OFF.

Gate pilot constants remain untuned: epsilon_defect=0.1; lambda_defect=1; tau_utility=0; defect margin =0.25*training-normal score std; risk denominator=max(training-normal score std,0.01); zero-defect-anchor => unavailable/reject. This is the full-image native-verified-view anchor variant, NOT the final spatial protected-defect-patch proposal. These conditions are research pilots and cannot establish a deployment risk guarantee.

Important interpretability note: fixed theta affects native candidate keep selection and CDARG U_N / H_D, hence each registered theta needs a NEW sequential replay. A retrospective binary threshold sweep on old banks is not equivalent. Neither q90 nor q95 should be chosen after inspecting Probe A; both must be reported. The evaluation probes remain excluded from admission and threshold calibration.

## Outputs and acceptance

Calibrator directory:
- threshold_registry.json (training-normal scores, sorted-higher quantiles, 64 identities, prior LOSO)
- source_manifest.json (source/Real-IAD/registry SHA-256)

Two NEW D2 directories, one per q90/q95, each with:
- summary.json: effective theta and sigma, LOSO reference, threshold-identifiability audit, per-arm measures, risk-prediction pair count
- baseline_comparison_dev.csv: admission counts, FNR/FPR, AUROC, AUPRC, score erosion, separated online/offline times
- probe_snapshots.csv: 4 independent fixed-position probe checkpoints
- candidate_decisions.csv: U_N/R_D/H_D, commit/reject, post-decision offline probe damage, SHA256
- query_ledger.csv: identical 16 Q-A image-specific labels
- source_manifest.json: pinned Real-IAD JSON, frozen D1-B and new threshold registry hashes

Gate for engineering GO:
- CPU unit tests and metadata preflight pass.
- Each q run verifies SHA and the exact normal training cohort; one read-only verify run per directory.
- Online six-arm replay runs with truly independent commits and no probes passed to gate; static unmodified.
- Distinct quantile runs may yield distinct candidate choices, and that is the intended sensitivity outcome.

Scientific stop rules:
- If FNR remains saturated (1.0), no success claim based on FNR; report continuous defect erosion and ranking.
- If U_N remains 0, do not silently change tau_U or epsilon_D after viewing Probe labels.
- If historical defect anchors give few paired predictions, state insufficient evidence.
- A small audiojack Probe A with correlated five-camera sample groups supports only development diagnosis, not confident multi-class performance ranking.
- Do not compare average latency before/after the D2 timing correction (offline probe rescoring previously contaminated online admission seconds).
- Native ContCore / FOADS and matched-admission-rate ablations remain separate later comparisons.

## Windows PowerShell sequence (each is one line)

From E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection with a clean working tree:
1. git pull --ff-only origin risk-safe-admission-sprint-20261027
2. Compile new threshold and runner modules with python -m py_compile.
3. Run python -m unittest discover -s tests -p "test_risk_safe_d2a.py" -v and test_risk_safe_d2.py.
4. python -m experiments.risk_safe.d2_threshold_calibration --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --preflight-only
5. Run the SAME calibrator with --device cuda --math-sdpa --out-dir "outputs/risk_safe_sprint_20261027/d2a_audiojack_s0_train64_fixed_thresholds", then read-only --verify-only.
6. For each registered q 0.90, 0.95, run d2_native_replay --threshold-registry "outputs/risk_safe_sprint_20261027/d2a_audiojack_s0_train64_fixed_thresholds/threshold_registry.json" --fixed-quantile QUANTILE, with the same frozen Q-A B16 and 40-probe-image config, but DISTINCT q90/q95 evidence directories.
7. Use --verify-only for each D2 run, stage ONLY the three new result directories, inspect, commit and push.

See the ChatGPT D2-A handoff for complete deterministic one-line commands. Do not start GPU until CPU preflight and tests pass.
