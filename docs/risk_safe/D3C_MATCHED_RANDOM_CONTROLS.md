# D3-C: Shuffled Context and Matched-Random Retrieval Attribution
Status 2026-10-28: CODE ONLY; Windows CPU/GPU validation outstanding.

## Question, frozen sources and interpretation
Does true Spatial(0.125) improve the frozen VisionAD normal adaptation / defect erosion trade-off because ORIGINAL spatial coordinates carry real geometric information, or is it only random restriction of dynamic memory access?

Use previous D3 v2 audiojack result at:
outputs/risk_safe_sprint_20261027/d3_audiojack_s0_q37_b16_q90_probe4_fixedk
and previous D2-A q90 fixed-threshold calibration and D1-B B16 Q-A tape. Same original four-shot static bank, DINOv2, 104 Greedy / 48 causal HITL-style Novelty dynamic prototypes, 40 Probe A camera images (25 NORMAL + 15 DEFECT), and original q90 fixed threshold. Probe B untouched. No modification to original D2-A/D3 outputs, v22/V0.2, query policy, Sidecar, Signed Correction, or online admission. The D3-C runner reconstructs ORIGINAL D3 through the same native scoring pipeline and verifies all 480 original D3 scores against the archived file within absolute 1e-6. It fails closed on changed query tape, q90 registry, Memory counts, support, Probe identities, FPR/FNR, and run requirements.

## EXACT pre-registered controls
Control repetitions use three fixed SHA256 seeds (0,1,2) PER Greedy or Novelty.

A: true_spatial_0125. Unchanged real original (u,v) coordinates, exact native distance. Repeat 0 only (reference, no Monte Carlo).

B: shuffled_spatial. Randomly PERMUTE prototype (u,v) locations among the bank's rows. Keep embeddings, physical camera, feature matrix, query labels and prototype count UNCHANGED. Preserves overall coordinate multiset but not the feature-position association. Each repeat has an independent deterministic permutation.

C: matched_count_random. For each query image and patch, compute original Spatial(0.125) eligible mask, then permute the BOOLEAN mask independently across all frozen dynamic prototype columns. This preserves EXACT per-patch number of eligible prototypes, hence EXACT share of query patches with enough candidates for native fixed k=5. The actual matching prototypes are random, independent of image labels, patch features or feature similarity. It does NOT guarantee matched dynamic wins.

D: source_calibrated_hit_random. Select the Bernoulli eligibility probability from FIXED grid [0.05,0.10,0.20,0.30,0.40,0.60,0.80,1.00]. For EACH memory arm and repeat, first compute true Spatial dynamic-win fraction and random-p dynamic-win fraction on first EIGHT sorted, reserved TRAIN mature-history NORMAL views (already separated from online stream and Probe A/B by original D1-A partition). Select p minimizing ABSOLUTE difference on those SOURCE normal views; ties choose SMALLER p. Then freeze p and apply unchanged on Probe A. Probe A scores, features and labels do NOT choose p. Achieved Probe dynamic-hit gap is recorded and should be interpreted; exact hit-rate matching on Probe is NOT guaranteed and should NOT be claimed.

The normal-history calibration cohort overlaps data previously used for the q90 normal-only threshold calibration, but not the heldout evaluation or online writes. That is acceptable for this DEV mechanism attribution; this cohort is NOT a new independent safety validation sample.

Native fixed-k (min(5, number of stored dynamic prototypes)) is maintained for EVERY condition. If context/random mask leaves fewer than k eligible, static-only fallback for that PATCH. This prevents artificially low anomaly scores from averaging fewer than five matches. Original score_single_view and image aggregation are untouched. Two protected invariants:
  global <= randomized score <= static (+/-1e-6 numerical tolerance)
  true_spatial_0125 reproduction exactly agrees with original D3 summary.
Randomness is derived only from arm/kind/repeat/immutable query image ID, never labels or matching results. No human-verified defect patch masks.

## Metrics and scientific STOP/GO
- Primary CONTINUOUS Pareto plane: mean reduction in static anomaly score over NORMAL probe images (benefit, higher better) vs mean positive static-to-updated erosion on DEFECT-labeled images (potential harm, lower better). Distinguish real pixel anomaly masks from these defect-image scores.
- Secondary FPR/FNR at registered q90, AUROC, AUPRC and share of patches with available native top-5 and share actually won by dynamic memory.
- Report each of 3 random repeats individually and the distribution across repeats, and pair over ORIGINAL official sample groups. Three repeats on the same eight Probe sample groups are NOT three independent datasets. Do not compute naive IID significance tests.
- For matched_count: exact eligibility cardinality equality is the targeted negative control, not necessarily identical dynamic-hit rate.
- For source_calibrated_hit_random: report source calibration target/gap and achieved Probe hit gap. If actual Probe hit gap is appreciable, label the result as an imperfect hit-rate control, do not assert a clean hit-matched advantage.
- GO toward a feedback-driven learned retrieval-rights mechanism only if true Spatial provides a nontrivial Pareto gain vs these controls, preserving genuine normal adaptation. If shuffled/count-matched random behaves equivalently, spatial coordinates have not demonstrated independent informativeness here; investigate prototype usage opportunities and representation-context collision instead.
- Note original true spatial policy 0.125 and methods were selected after examining previous Probe A development results; this is a DEV ablation, not independent hypothesis confirmation. Reserve Probe B and new categories/seed streams for later blinded confirmation.

## Outputs and provenance
New immutable directory:
outputs/risk_safe_sprint_20261027/d3c_audiojack_s0_q37_b16_q90_probe4_randomcontrols_v1

Files: summary.json; random_control_probe_scores.csv; random_control_comparison.csv; source_only_hit_calibration.json; source_manifest.json.
Cohort, repeated random seeds, threshold, original D3 CSV/hash, original D2-A summary and D1-B tape are recorded. All files are published ATOMICALLY after SHA-256 verification. No partial/failed run is published. Source manifest contains actual device argument and model execution status, instead of relying on generic metadata-preflight environment fields.

## Read-only verifier correction (post-first D3-C run)

The first D3-C `--verify-only` contained a field-name bug: the run correctly wrote
`summary.json.source_reference.passed`, but the verifier checked the nonexistent
`summary.json.reference_d3_native_replay.passed`. That produces
`NativeParityError: D3-C reference native replay not verified` even when the
480-source-score parity validation succeeded during the GPU run.

**Follow-up (historical Git vs Windows line endings):** A clean Windows
Git working tree can contain CRLF bytes while its stored git blob uses LF.
The historical source manifest recorded raw working-tree SHA256, so direct
hashing of `git show` bytes can incorrectly report
`Historical source does not match generating git commit`. The verifier
now accepts only the exact recorded raw Git blob OR its exact LF-to-CRLF
Windows checkout transformation. It does **not** ignore arbitrary whitespace,
source content changes, or any mismatched input/output digest. A failed run
is not silently approved. Regression tests cover both newline modes and
actual Python source edits. This remains a read-only check and preserves
the original source_manifest.json byte-for-byte.

**Recovery: do NOT rerun GPU, delete outputs or modify historical JSON.**
Pull the new code, rerun CPU unit tests and the ORIGINAL `--verify-only`
command. The verifier now validates `source_reference` (480 matched original
D3 scores and <=1e-6 max difference), 800 control score count, 20 summary
rows and the independent data flags. It checks the generated output hashes,
input hashes and config, then verifies each **historical recorded source
SHA-256 against the original generating Git commit's tracked blob** through
`git show`. Git ancestry is required; changed/dirty historical tracked
source is rejected. Current source hash changes from this verifier-only fix
are NOT silently substituted into the original manifest. This is a
historical-evidence verification, not a new simulation run. Do not rewrite
the original source manifest.

## Windows PowerShell single-line commands

From E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection with clean working directory:
1. git pull --ff-only origin risk-safe-admission-sprint-20261027
2. Compile:
& "D:\anaconda3\envs\torch\python.exe" -m py_compile experiments/risk_safe/retrieval_random_controls.py experiments/risk_safe/d3_controlled_retrieval.py experiments/risk_safe/conditional_retrieval.py experiments/risk_safe/conditional_retrieval_pilot.py tests/test_risk_safe_d3c.py
3. CPU tests:
& "D:\anaconda3\envs\torch\python.exe" -m unittest discover -s tests -p "test_risk_safe_d3c.py" -v
4. Existing D3 regression:
& "D:\anaconda3\envs\torch\python.exe" -m unittest discover -s tests -p "test_risk_safe_d3.py" -v
5. Metadata preflight; no DINOv2/CUDA:
& "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.d3_controlled_retrieval --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 16 --max-query-events 16 --probe-groups-per-type 4 --threshold-registry "outputs/risk_safe_sprint_20261027/d2a_audiojack_s0_train64_fixed_thresholds/threshold_registry.json" --fixed-quantile 0.90 --reference-d2a-summary "outputs/risk_safe_sprint_20261027/d2a_audiojack_s0_q37_b16_q90_probe4/summary.json" --reference-d3-dir "outputs/risk_safe_sprint_20261027/d3_audiojack_s0_q37_b16_q90_probe4_fixedk" --d1b-manifest-dir "outputs/risk_safe_sprint_20261027/d1b_protocol_v1" --preflight-only
6. Run GPU exactly same arguments but replace --preflight-only with:
--device cuda --math-sdpa --out-dir "outputs/risk_safe_sprint_20261027/d3c_audiojack_s0_q37_b16_q90_probe4_randomcontrols_v1"
7. Verify same GPU command with --verify-only appended, no GPU extraction.
8. Stage ONLY new D3-C output directory; inspect; git commit; git push; git rev-parse HEAD. Do NOT overwrite historical evidence.
