# R0 — Opportunity and Mechanism Activation Audit (2026-10-28)

**Status:** R0 implementation pushed for local offline tests + CUDA validation; **no R0 GPU results have been run or claimed by this document**. This is NOT R1 Oracle, R2 feedback learning, or R3 stress testing.

**Parent:** `risk-safe-admission-sprint-20261027`. R0 branch: `r0-opportunity-audit-20261028`. Preserve all historical outputs, D2–D4 and the original v22 without modification.

## Scientific question

Is the proposed stream/budget/memory setting even capable of identifying an improvement over Greedy/Novelty? In D2-A on Audiojack, only 11 verified normal events, five verified defect images, seven events with available historical defect evidence, zero Greedy replacements, and no positive CDARG normal hinge utility were observed. A new controller cannot be scientifically judged if its key mechanism is absent.

R0 therefore has TWO deliberately distinct audits:

1. **Metadata preflight (CPU):** frozen Q-A tape, number/time order of queried image-level normal/defect labels, number of verified-normal events with 1/4/8/16 prior defect labels, query fractions using two denominators. **Upper bounds only** on real normal write opportunities.
2. **Native-v22 CUDA activation (GPU):** strict 4-shot initial static bank, native frozen feature scoring, original selected 8/16/24 candidate patch rows, independent dynamic Greedy memory at capacity 64/128/256/1024, true append/replacement/skip counters, read-only counterfactual rescoring on historical verified whole-image anchors, and final full Probe-A development snapshot. Query decisions are the precommitted external Q-A tape; Sidecar/Correction/LTM are disabled.

R0 **does not** prove a safe admission advantage and should not be used to choose a more favorable query seed after examining outcomes.

## Information firewall

- Query tape created and verified **before** true image labels are revealed. Exactly one queried camera-image label per official sample group. All 2,625 stream images are in the frozen D1-B sequence, but the condensed Q-A replay extracts and scores **queried images only**. Nonqueried images cannot update normal memory in this locked design. It is **not a whole-stream prequential benchmark**.
- R0 metadata may read authoritative test labels AFTER query tape construction to diagnose opportunity availability, and must never feed those diagnostics to the online algorithm.
- A verified-normal image may produce a candidate (subject to native selector); a verified-defect image **must never be inserted into normal memory**.
- For event t, shadow risk audits use only previously queried normal/defect images; current label becomes an anchor **after** every independent memory-capacity arm processes the event.
- The full Probe A is offline-only and used solely at final snapshot. Probe B metadata is used for exclusion in inherited frozen partitions, but Probe B images/features/labels are never used for R0 learning or probe scoring. R0 A is already development-adaptive; no confirmatory inference.
- Initial threshold is an explicit experimental arm: `--theta-mode loso` (four-support **LOSO** diagnostic, n=4, default) or `--theta-mode train64-q90` (64 **additional** reserved verified-normal training views scored against the same frozen support, empirical q90 per D2-A). The second arm is **NOT four-shot-total**. Neither gives an operating-point risk guarantee; do not compare FPR across the two regimes as if initialization matched.
- `official_sample_key` is the certified official Real-IAD camera-grouping unit; do not present it as guaranteed unique manufactured physical object identity.

## Activation vs performance

R0 native counters are **exact for every processed queried-normal event**:
`n_normal_events`, `n_candidate_events`, `n_candidate_ineligible`, `n_defect_history_eligible_candidates`, `n_defect_history_ge4_candidates`, `n_full_before_proposal`, `n_replacement_attempts`, `n_replacements`, `n_appended`, `n_skipped_patches`, `final_prototypes`.

Risk evidence is **audited on a defined subset**, at `--audit-every N` verified-normal events (default N=1). It uses most recent `--anchors-per-label K` prior, actually queried image-level anchors (default K=4 for each label). Report `n_shadow_audits`, `n_defect_risk_available`, `n_defect_risk_positive`, `n_normal_utility_positive`, `n_normal_score_gain_positive`, and `n_margin_crossings`. These are **historical-anchor** signals, not held-out defect-mask damage or calibrated safety probabilities.

The descriptive activation flags require at least 30 candidate events, 20 historical-defect-eligible candidates, 20 audited defect-risk candidates, 5 nonzero defect-risk events, 20 capacity-full proposal attempts, 10 actual replacements, and 5 positive normal hinge-utility events. **These are exploratory activation thresholds, not statistical power, not algorithm performance targets, and cannot guarantee activation.** An unmet flag is `underpowered` rather than algorithm failure. No changes to the fixed input cohort/seed/budget should be made after seeing performance merely to manufacture a positive result.

## Single-line PowerShell setup

Run from repository root. Check working tree first; do not discard local changes.

```powershell
git status --short
git fetch origin
git switch r0-opportunity-audit-20261028
git pull --ff-only origin r0-opportunity-audit-20261028
conda activate torch
python -m py_compile experiments/risk_safe/r0_opportunity.py experiments/risk_safe/r0_native_activation.py tests/test_risk_safe_r0.py
python -m unittest discover -s tests -p "test_risk_safe_r0.py" -v
```

## Archived query manifest vs R0 extension (B=96/128)

The committed D1-B `query_policy_manifest.json` contains **main budgets B=16,32,64,79 only**, while the frozen protocol also permits sensitivity budgets B=96,128 and diagnostic B=8. The prior D1-D helper `frozen_tape()` incorrectly indexes the archived table with `str(128)` and fails with `KeyError: '128'`. R0 now uses `r0_opportunity.frozen_r0_tape()`: recompute label-blind D1-B stream / sample priority / camera map, check **all four archived main budgets** against archived full hash/position/image/sample records, enforce nested consistency, then derive the approved extra budget without touching the old manifest. Record `tape_verification` in R0 summary and `tape_derivation` in R0 provenance. Unknown budgets, missing primary budgets and altered archived digests raise `R0TapeIntegrityError`.

GitHub Actions R0 offline tests additionally cover B128 extension, legacy B16 exact parity, historical manifest corruption, and missing archived main budget. **This is not a GPU validation**, and no original D1-D output or archived D1-B file was changed.

## Stage A: metadata-only across all six classes and all registered query seeds/budgets

```powershell
python -m experiments.risk_safe.r0_opportunity --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --classes "audiojack,bottle_cap,eraser,plastic_plug,switch,rolled_strip_base" --budgets "16,32,64,79,128" --query-seeds "37,101,211" --support-seeds "0,1,2" --out-dir "outputs/r0_opportunity_20261028/metadata"
python -m experiments.risk_safe.r0_opportunity --verify-only --out-dir "outputs/r0_opportunity_20261028/metadata"
```

Read `r0_metadata.csv` and `r0_metadata.json`. This audit does NOT run the backbone and cannot conclude that a Memory update was harmful or that capacity was reached.

## Stage B: first native CUDA smoke (4 queried images, intentionally INCOMPLETE)

```powershell
python -m experiments.risk_safe.r0_native_activation --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 128 --max-query-events 4 --capacities "64,128" --audit-every 1 --anchors-per-label 4 --probe-eval none --math-sdpa
python -m experiments.risk_safe.r0_native_activation --class-name audiojack --support-seed 0 --query-seed 37 --budget 128 --max-query-events 4 --capacities "64,128" --audit-every 1 --anchors-per-label 4 --probe-eval none --verify-only
```

The smoke is for source parity, correct label timing and no input/shape errors; do NOT use its negative/positive risk counts as scientific findings. Smoke uses the default LOSO regime, whereas full Stage C explicitly uses train64-q90, and they must not be compared as if threshold-matched.

## Stage C: first full R0 activation, then second development class

```powershell
python -m experiments.risk_safe.r0_native_activation --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 128 --theta-mode train64-q90 --capacities "64,128,256,1024" --audit-every 1 --anchors-per-label 4 --probe-eval final --probe-sample-groups-per-type 24 --math-sdpa
python -m experiments.risk_safe.r0_native_activation --class-name audiojack --support-seed 0 --query-seed 37 --budget 128 --theta-mode train64-q90 --capacities "64,128,256,1024" --audit-every 1 --anchors-per-label 4 --probe-eval final --probe-sample-groups-per-type 24 --verify-only
python -m experiments.risk_safe.r0_native_activation --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name bottle_cap --support-seed 0 --query-seed 37 --budget 128 --theta-mode train64-q90 --capacities "64,128,256,1024" --audit-every 1 --anchors-per-label 4 --probe-eval final --probe-sample-groups-per-type 24 --math-sdpa
```

These runs may be computationally expensive: exact native-v22 shadow scoring performs fresh scoring of historical anchors for each eligible normal event and capacity. Diagnose GPU time/VRAM first. If throughput is prohibitive, run an explicitly labeled *additional* diagnostic at `--audit-every 4` in a **new output directory**, keeping all candidate/replacement counters exact and clearly labeling the shadow risk sample as sparse. Do not rewrite the full result or silently change its settings.

## Outputs, traceability and failure handling

Each metadata run creates `r0_metadata.json`, `r0_metadata.csv`, `checksums.json`.

Each native run is named with its `theta-mode` and creates `r0_summary.json`, `r0_candidate_events.csv`, optional `r0_probe_scores.csv`, `r0_provenance.json`, `checksums.json`. Output names identify class, support/query seeds, budget, capacities, audit cadence and smoke/full. **Existing native output directories are never overwritten.** Verification rechecks file checksums without invoking CUDA.

If a CUDA run fails, STOP at the traceback. Do not delete existing output directories or bypass parity assertions. New experiments must be run in new declared directories. Historical D1-B, D2–D4, V0.2, v22 outputs and manuscripts are immutable.

## R0 acceptance gate before R1

1. Native smoke passes tests and reports an incomplete smoke status.
2. Metadata query counts and per-label audit exactly match label-blind tape.
3. Full GPU run writes ALL B queried events, with no current/future-label use in policy.
4. At least one capacity setting genuinely reaches and attempts replacements, or explicitly reports `underpowered`.
5. Historical defect risk and normal utility opportunities are counted with explicit missingness; never convert missing defect anchors to zero risk.
6. Probe A metrics are treated as development diagnostics; R0 does not consume Probe B.
7. Only after the R0 opportunity map is reviewed do we lock R1 Oracle strata and computational allocation. Do not automatically jump to R1 or selectively retain successful classes.
