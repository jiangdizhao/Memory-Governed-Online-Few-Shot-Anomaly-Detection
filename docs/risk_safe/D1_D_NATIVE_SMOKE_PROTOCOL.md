# D1-D — Native v22 scorer / Greedy parity and bounded queried-event GPU audit

**Status:** source committed, actual Windows GPU results **pending**. Do not confuse passing the D1-C synthetic CPU experiment with authentic VisionAD risk-safety performance. This D1-D implementation tests the exact existing v22 image scoring entrypoint, original V0.2 Greedy update, and a very small trusted Q-A queried-image replay. Source v22, V0.2 and archived D1-A/B/C outputs are unchanged.

## Scientific scope

Goal: before running CDARG FNR experiments, verify that a full native v22 Shadow Bank produces the **same image score** as the live bank before mutation, and that a proposed shadow normal-patch update matches **the real V0.2 MemoryArm.update Greedy implementation** for a candidate selected with the native 8/16/24 patch selector.

Native scorer is the unchanged a0_a1_strict4_matched.score_bank, which calls PatchVMB.score_sample_multiview with the recorded topk/boost/fusion/aggregation settings. Synthetic CosinePatchScorer is NOT used. Native Greedy replacement uses unchanged v02_memory.greedy_slot. A genuine MemoryArm.update is executed on two isolated parity-only arms, never on the online arm: first the empty-bank append path, then a separately initialised forced-capacity K=2 per-layer case that must exercise the full-bank Greedy decision path without appending.

Second, select the pre-registered **Q-A / Random order / 4-shot / seed 37 / budget 16** tape, and process only its first four queried images. Query membership, camera identity, event positions and tape digest must match the existing immutable D1-B query_policy_manifest.json. For each selected queried image, the script computes its static/dynamic score before opening its true camera-image label and then logs the causal anchor state. Verified-normal images receive a proposed native top-8/16/24 write and a CDARG audit-only evaluation. Verified-defect images are retained as entire scored camera images with historical label times. This first full-native anchor diagnostic uses whole-image scores and **not calibrated defect-patch localisation**.

Unqueried frames are not encoded. This is legitimate for testing the state of a frozen, shared exogenous Q-A tape when *no updates are actually committed*, but is NOT a full prequential inference/throughput/performance replay.

**No production memory updates are allowed by the CDARG audit.** Any apparent admission decision uses clearly UNCALIBRATED coefficients copied solely for coding validation. No FNR, FPR, AUROC, CDARG benefit or statistically valid defect-risk prediction claims arise from this smoke run.

## Environment and invariant inputs

- Repo: E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection
- Branch: risk-safe-admission-sprint-20261027
- Interpreter: D:\anaconda3\envs\torch\python.exe
- Real-IAD root: G:/dataset/realiad
- Official JSON directory: G:/dataset/realiad/jsons/realiad_jsons
- Class: audiojack; support seed: 0; query seed: 37; budget B=16.
- Frozen exact D1-B query registry: outputs/risk_safe_sprint_20261027/d1b_protocol_v1/query_policy_manifest.json.
- Frozen VisionAD configs reused from V0.2: frozen support bank 25000, 5000/image, four augmentations, concat fusion, 32 scoring patches, image max/64/0.95, kNN=5. LOSO threshold multiplier=2.5.

Source files: experiments/risk_safe/native_bridge.py, experiments/risk_safe/d1d_native_smoke.py, tests/test_risk_safe_d1d.py.

## Local commands — Windows PowerShell, exactly one command per line

1. Check workspace before pulling:

    git status --short

2. Sync:

    git pull --ff-only origin risk-safe-admission-sprint-20261027

3. Compile:

    & "D:\anaconda3\envs\torch\python.exe" -m py_compile experiments/risk_safe/native_bridge.py experiments/risk_safe/d1d_native_smoke.py tests/test_risk_safe_d1d.py

4. CPU contract tests, including previous stages:

    & "D:\anaconda3\envs\torch\python.exe" -m unittest discover -s tests -p "test_risk_safe_d1*.py" -v

5. Very small GPU score/update parity + query evidence audit:

    & "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.d1d_native_smoke --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 16 --max-query-events 4 --device cuda --math-sdpa --d1b-manifest-dir "outputs/risk_safe_sprint_20261027/d1b_protocol_v1" --out-dir "outputs/risk_safe_sprint_20261027/d1d_audiojack_s0_q37_b16"

6. Read-only validation of resulting JSON, source and input hashes. This does not rerun DINO or test GPU bitwise determinism:

    & "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.d1d_native_smoke --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 16 --max-query-events 4 --device cuda --math-sdpa --d1b-manifest-dir "outputs/risk_safe_sprint_20261027/d1b_protocol_v1" --out-dir "outputs/risk_safe_sprint_20261027/d1d_audiojack_s0_q37_b16" --verify-only

7. After ALL commands pass, stage only the new outputs:

    git add -- outputs/risk_safe_sprint_20261027/d1d_audiojack_s0_q37_b16

8. Inspect staged evidence:

    git diff --cached --stat

9. Check whitespace:

    git diff --cached --check

10. Commit and push:

    git commit -m "data(risk-safe): archive D1-D native GPU parity and Q-A smoke"

    git push origin risk-safe-admission-sprint-20261027

    git rev-parse HEAD

Report the latest commit SHA. Review will be done directly against GitHub. Do not manually search for or upload source/JSON files.

## Pass/fail and scientific boundaries

**Mandatory pass:**
1. Python compilation and CPU tests pass.
2. No native scorer clone-parity error; the live score and untouched native-deepcopy score differ no more than abs tolerance 1e-5 plus rel tolerance 1e-4. The recorded numeric difference must be inspected.
3. Full true MemoryArm.update of BOTH the initially empty parity-only arm and the deliberately K=2 saturated parity-only arm equals GreedyBank.proposed on every layer, with torch.equal and exact bank SHA-256. The saturated arm must append zero candidates. A mismatch is a research-engineering blocker, not something to hide by loosening thresholds.
4. Frozen query event hash/stream hash/positions match D1-B; no support/probe overlaps.
5. Online audit arm remains entirely unchanged (no CDARG commit); fixed reference bank SHA remains identical.
6. Output JSON plus provenance file validate, and the source snapshot includes the original v22 and V0.2 files.
7. No claim of complete online replay, strong defect-risk calibration or improved FNR.

**Possible failure diagnosis:** Native greedy parity can fail if the synthetic shadow's use of torch.nn.functional.normalize differs from native v22 l2_normalize or in mixed dtype, if candidate selection layer ids differ, or if native codebook state has unexpected side effects. Preserve the full traceback and do not write incomplete evidence. Fix the mismatch in code after inspecting the true native semantics; never silently change registered B, query seeds, support or historical V0.2 code.

**Next gate:** once authentic scoring and update parity pass, implement a fully prequential native adapter with verified-label causal patch-anchor scoring, development-only CDARG hyperparameter calibration and identical frozen Query Tape baseline arms. Only then execute first real FNR/FPR comparison.
