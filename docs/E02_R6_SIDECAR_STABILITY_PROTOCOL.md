# R6 — Audit-only Sidecar Stability and Extreme-Stream Protocol

Branch: e02-r6-sidecar-stability-audit
Entrypoint: experiments/e02_r6_sidecar_stability.py
Tests: tests/test_e02_r6_sidecar_stability.py
Base branch R5-D: commit 4526b3d2650cf36e0377af2981f79b58de46603e

## Scope and provenance

R6 is an OBSERVATIONAL mechanism audit, not a new Sidecar or full-v22 result.
It reuses canonical v22 initialization, the exact R5-D time-paced query
policy, E02 verified-normal-only STM updates, the 3% query cap, and four
existing Sidecars. No signed correction, LTM or maturation; the original
v22 and historical R3/R4/R5 are unchanged.

R6 questions:
1. Why does R5-D Balanced Rescue cease threshold moves after image 41?
2. What happens on normal-first, defect-first, alternating-block streams?
3. Are missing-class risk, selective labels and memory-state score staleness
   important? Keep hypothesis and verified evidence separate.

An audit wrapper records actual forced/periodic Sidecar method calls,
pre/post cooldown, verified normal/defect counts, actual candidate risk
and objective evaluations, and updated theta. Candidate rejection reasons
are classified POST HOC using the original code's exact guard predicates
and already-observed values; no additional Sidecar function is called.

Natural/reference order comes from the R5 data-index and RNG. Extreme
orders use EVALUATION labels once, OFFLINE, to construct a frozen
permutation; the online query controller cannot see future or unqueried
labels. These are designed stress cases, NOT deployment policies.
Fixed memory and support paths must match; complete image membership is
identical. Query indices and learned STM may differ after permutation.

In evaluation-only 256-step windows, absent normal => FPR null, absent
defect => FNR null, either absent => Balanced Accuracy null. Missing risk
does not mean zero risk. Observed queried-label risks are selected, not
automatically unbiased population risk estimates.

## Step 0: checkout and run CPU-only checks (PowerShell)

    git fetch origin
    git switch e02-r6-sidecar-stability-audit
    git pull --ff-only origin e02-r6-sidecar-stability-audit
    git status --short
    git rev-parse HEAD
    conda activate torch
    python -m py_compile experiments/e02_r6_sidecar_stability.py tests/test_e02_r6_sidecar_stability.py
    python experiments/e02_r6_sidecar_stability.py --self-test
    python -m unittest discover -s tests -p "test_e02_r6_sidecar_stability.py" -v

Do not modify the historical v22 code to fix failures.

## Step 1: 100-image Audiojack parity smoke

The R4/R5 smoke source files must exist. A smoke never establishes
scientific performance (only 3 feedback labels).

    python experiments/e02_r6_sidecar_stability.py --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --legacy-seed 0 --stream-seed 0 --query-seed 37 --query-rate 0.03 --max-samples 100 --math-sdpa --local-saliency --device cuda --print-every 25 --variants D --stream-order native --reference-r4-summary "outputs/e02_r4_abc_audiojack_smoke/r4_summary.json" --reference-native-summary "outputs/e02_r5_d_audiojack_smoke/r5_d_summary.json" --reference-native-csv "outputs/e02_r5_d_audiojack_smoke/r5_d_prequential.csv" --require-parity --out-dir "outputs/e02_r6_audiojack_native_smoke"

Acceptance: exactly 100 rows, 3 queries, prequential_parity.passed
and reference_check.native_full_parity both true; nonempty audit log.

## Step 2: complete 3,815-image Audiojack canonical R5-D parity

    python experiments/e02_r6_sidecar_stability.py --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --legacy-seed 0 --stream-seed 0 --query-seed 37 --query-rate 0.03 --max-samples 0 --math-sdpa --local-saliency --device cuda --print-every 100 --variants D --stream-order native --reference-r4-summary "outputs/e02_r4_abc_audiojack_full_seed0/r4_summary.json" --reference-native-summary "outputs/e02_r5_d_audiojack_full_seed0/r5_d_summary.json" --reference-native-csv "outputs/e02_r5_d_audiojack_full_seed0/r5_d_prequential.csv" --require-parity --out-dir "outputs/e02_r6_audiojack_native_full_seed0"

Required baseline: N=3815, K=114, 26 normal/88 defect verified,
theta0=0.5568159315735102, Fixed Bank hash
ceb275704cb1277e8816feede44e9a0957bdeab27916eedf25d254123f0a3f86,
Balanced Rescue moves on steps 1/24/41 only, FNR=0.806327,
Balanced Accuracy=0.594653, Mistake Feedback 37 moves and
longest no-query gap 182.

The script compares every R5 row (score, query, policy, prediction, theta)
within prescribed numeric tolerance, as well as full query indices,
support paths, bank hash, settings, metrics and Sidecar moves.
Do NOT analyze R6 results unless this parity passes.

## Step 3: Audiojack time-order stress (after Step 2 succeeds)

Execute in PowerShell (same 3,815 images in each condition):

    $COMMON = @("--dataset-root","G:/dataset/realiad","--json-dir","G:/dataset/realiad/jsons/realiad_jsons","--class-name","audiojack","--support-seed","0","--legacy-seed","0","--stream-seed","0","--query-seed","37","--query-rate","0.03","--max-samples","0","--math-sdpa","--local-saliency","--device","cuda","--print-every","100","--variants","D","--reference-native-summary","outputs/e02_r5_d_audiojack_full_seed0/r5_d_summary.json","--reference-native-csv","outputs/e02_r5_d_audiojack_full_seed0/r5_d_prequential.csv")
    python experiments/e02_r6_sidecar_stability.py @COMMON --stream-order normal_first --out-dir "outputs/e02_r6_audiojack_normal_first"
    python experiments/e02_r6_sidecar_stability.py @COMMON --stream-order defect_first --out-dir "outputs/e02_r6_audiojack_defect_first"
    python experiments/e02_r6_sidecar_stability.py @COMMON --stream-order alternating_blocks --block-size 256 --out-dir "outputs/e02_r6_audiojack_alternating"

Do not require original query-position parity: query actions naturally
respond to different preceding stream observations. Instead require the
same initial state, same N, K and exact image multiset.

## Step 4: select independent Real-IAD classes by prevalence only

No visual encoder or GPU execution is needed for the profile:

    python experiments/e02_r6_sidecar_stability.py --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --profile-classes --profile-out "outputs/e02_r6_class_prevalence.json"

The class profile selects extremes by label prevalence (not performance).
After recording the chosen category names, run native R6 for each class
with a unique output folder and without prior R4/R5 references. The
native R6 summary and CSV then serve as reference-native-summary and
reference-native-csv in that class's extreme-order runs. Use at least
two support seeds if GPU budget permits; each class/seed/order must have
a unique output path. Do not choose favorable seeds after reviewing metrics.

## Step 5: optional memory-score staleness check

After Step 2 passes, repeat its full native command in a NEW output
directory with flags:

    --rescore-every 256 --rescore-anchor-cap 8

This caches up to eight queried image descriptors in CPU memory; at
checkpoints it rescores them using the current STM, producing
r6_shadow_rescore.csv (score_then, score_now, delta, memory version).
All re-scores are evaluation-only, never fed to query, memory or Sidecar.
Require full original R5 parity again. This is descriptive and not a
probability-corrected population risk estimate.

## Outputs

- r6_summary.json — original metrics, audit outcomes, candidate failure
  reason histogram, full protocol and class-presence-aware windows.
- r6_sidecar_events.jsonl — event-by-event true method call/candidate records.
- r6_prequential.csv — predictions and query decisions before feedback.
- r6_shadow_rescore.csv — optional state-conditioned score drift.
- canonical/ — unchanged E01/E02 original v22 initialization dump.

## Git workflow for local outputs

After a successful local run, stage ONLY that specific new output folder:

    git status --short
    git add outputs/e02_r6_audiojack_native_smoke
    git diff --cached --stat
    git commit -m "experiment: R6 audiojack parity smoke results"
    git push origin e02-r6-sidecar-stability-audit
    git ls-remote origin refs/heads/e02-r6-sidecar-stability-audit

Check no Real-IAD image files, model weights, credentials, or large raw
feature caches are staged. Stop and report any parity mismatch, and do
not overwrite historical outputs. R6 cannot prove general SOTA or
independent theory by its single-class diagnostic outcome.
