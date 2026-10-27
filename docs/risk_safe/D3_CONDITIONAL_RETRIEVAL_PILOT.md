# D3 — Native VisionAD Conditional Retrieval Feasibility Pilot

Status: source committed, Windows GPU outputs NOT YET VERIFIED. This is a mechanism-feasibility audit, not the complete risk-aware learned retrieval controller. All D1/D2/D2-A experiments and original v22/V0.2 files remain unchanged.

## Frozen design

Real-IAD audiojack; strict four-shot support seed 0; query seed 37; original full B16 Q-A Random stream. Frozen q90 fixed normal-only calibration, no Sidecar or Signed Correction. Probe A first four normal and four defect-containing groups: 40 images. Probe B not accessed. Source baseline reproduction is **MANDATORY**: the immutable D2-A q90 summary.json provides original Greedy 104 and causal Novelty 48 dynamic prototypes, same support/queries, same Probe FPR/FNR. Each D3 native Global score must also equal the unchanged v22 score exactly. Deviations fail closed; do not publish an output directory.

The only intervention is RETRIEVAL of the same already-written dynamic prototype rows. No change in write admission, source candidate selection, feature extractor, static bank or fixed threshold. Each appended prototype retains physical Camera C1-C5, official sample key, original patch index, normalized (u,v), and original online query timestamp. Replace/evict cannot be faithfully attributed with this version, so any such operation fails closed.

Six retrieval rules, on BOTH identical Greedy and Novelty banks:
- global: unrestricted STM, exact original v22 behavior
- camera: only same physical Camera ID dynamic prototypes
- spatial_0125 / spatial_025: normalized patch L-infinity radius <= 0.125 or 0.25
- camera_spatial_0125 / camera_spatial_025: require both conditions

Frozen Static references remain global even if dynamic references are excluded. The adapter overrides _dN_min_fixed_dyn on a PRIVATE clone. It preserves native score_single_view / score_sample_multiview / topk / image aggregation. Native STM uses MEAN of min(5, M) nearest distances where M is the TOTAL dynamic prototype count, **not** the number of eligible prototypes. Corrected D3 uses a fixed effective k=min(5,M) for ALL patches; if a context leaves fewer than this k, it DISABLES dynamic retrieval for that query patch and falls back to Static. If at least k are eligible, the same mean-top-k is recomputed on eligible sources. This guarantees the restricted dynamic distance is no lower than native Global (within numeric precision). It is intentionally conservative and its side effect (small-context fallback to Static) MUST be reported, not mistaken for superior safety. The experiment now logs per-policy share of patches with k eligible prototypes, share actually using dynamic retrieval, and static-fallback rate. Physical camera is not v22 identity-augmentation view id.

**2026-10-28 failed GPU pilot / correction:** the first implementation instead used min(5, n_eligible) in each query patch, which let one very near allowed prototype falsely appear more similar than the original five-neighbour MEAN. The correct protective assertion raised NativeParityError: Scoped memory improved more than global memory. This is a numerical protocol bug, NOT a failure of the context-retrieval scientific hypothesis. The original failed run had NO completed evidence outputs, and the corrected protocol/summary is explicitly versioned v2 with a NEW _fixedk output directory. Do not remove or loosen the native-global score monotonicity check.

## Outputs and scientific limitations

D3 writes summary.json, retrieval_probe_scores.csv (including effective-native-k eligibility and actual dynamic-win shares), collision_patch_attribution.csv, native_write_provenance.csv, causal_query_ledger.csv, and source_manifest.json, published atomically after SHA-256 checks. Collision CSV is intentionally top-1000 censored, and top-1 matched dynamic prototype is ONLY diagnostic of the native top-5 average. Defect-image patch erosion is NOT GT-mask localization. Probe label is offline evaluation only; no Probe B or pixel-level ground truth used. No claims of statistical generality from audiojack alone.

Primary GO signal: at the SAME stored prototype count and identical Q-A tape, context restrictions decrease independent defect score erosion WITHOUT losing the significant normal-FPR correction obtained by global Greedy/Novelty. A policy that disables almost all dynamic retrieval and restores Static FPR is NO-GO. If context-only shows potential, next experimental phase should compare risk-aware dynamic retrieval-rights against equally context-constrained Greedy/Novelty and matched effective retrieval opportunities.

## Windows PowerShell, each command on one physical line

1. git pull --ff-only origin risk-safe-admission-sprint-20261027

2. Compile:
& "D:\anaconda3\envs\torch\python.exe" -m py_compile experiments/risk_safe/conditional_retrieval.py experiments/risk_safe/conditional_retrieval_pilot.py experiments/risk_safe/retrieval_collision_audit.py tests/test_risk_safe_d3.py

3. Unit tests:
& "D:\anaconda3\envs\torch\python.exe" -m unittest discover -s tests -p "test_risk_safe_d3.py" -v

4. Metadata preflight, no GPU:
& "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.conditional_retrieval_pilot --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 16 --max-query-events 16 --probe-groups-per-type 4 --threshold-registry "outputs/risk_safe_sprint_20261027/d2a_audiojack_s0_train64_fixed_thresholds/threshold_registry.json" --fixed-quantile 0.90 --d1b-manifest-dir "outputs/risk_safe_sprint_20261027/d1b_protocol_v1" --reference-d2a-summary "outputs/risk_safe_sprint_20261027/d2a_audiojack_s0_q37_b16_q90_probe4/summary.json" --preflight-only

5. GPU replay, new immutable output:
& "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.conditional_retrieval_pilot --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 16 --max-query-events 16 --probe-groups-per-type 4 --threshold-registry "outputs/risk_safe_sprint_20261027/d2a_audiojack_s0_train64_fixed_thresholds/threshold_registry.json" --fixed-quantile 0.90 --device cuda --math-sdpa --d1b-manifest-dir "outputs/risk_safe_sprint_20261027/d1b_protocol_v1" --reference-d2a-summary "outputs/risk_safe_sprint_20261027/d2a_audiojack_s0_q37_b16_q90_probe4/summary.json" --out-dir "outputs/risk_safe_sprint_20261027/d3_audiojack_s0_q37_b16_q90_probe4_fixedk"

6. Add --verify-only to the identical GPU command for read-only validation (GPU not used).

7. Read offline collision summary:
& "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.retrieval_collision_audit --result-dir "outputs/risk_safe_sprint_20261027/d3_audiojack_s0_q37_b16_q90_probe4_fixedk"

8. After all verification: stage ONLY the new D3 output directory, inspect diff, commit, push, send latest HEAD. Never overwrite historical experimental evidence.
