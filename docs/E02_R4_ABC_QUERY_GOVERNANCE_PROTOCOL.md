# E02-R4: Bootstrap query-governance A/B/C protocol

Branch: e02-r4-abc-query-governance
Code: experiments/e02_r4_abc_query_governance.py

Original v22, R3, E02 and earlier experiments are unchanged. This is a
controlled mechanism audit rather than the complete original full v22 system.

## Three query-governance variants

All use the same v22-derived support images, Fixed Bank, original theta0,
query reference score, common test stream, verified-normal E02 STM writes,
and EXACT K=round(N*0.03) label limit. Each variant runs Frozen, Guarded,
Balanced Rescue and Mistake Feedback sidecars on its own ONE STM trajectory.
Changes between variants affect query labels and memory, so only comparisons
WITHIN each variant isolate threshold control.

A — unchanged R3 native Bootstrap-to-SimpleAIF querying:
The original v22 SimpleAIFState.observe(...) local query-cost adaptation,
strict R3 final label cap and terminal catchup, all unmodified.
An optional full CSV replay check must show exact identity to historic R3.

B — suppress Bootstrap-induced Query Cost inflation:
During native Bootstrap, real labels are still queried, counted in K,
reported, and used for supervised STM updates. However the adaptive
Query Cost controller does not update in Bootstrap. Afterwards,
Query Cost uses only post-Bootstrap query flags. This is an intervention
on adaptive-cost feedback, not free additional labels.

C — B plus predefined score-only region quotas:
Define z=(score-theta0)/sigmaN0. Below z=-1 is low_score with a nominal
45% budget; -1<=z<=1 is boundary with 35%; z>1 is high_score (20%).
During early Bootstrap, low_score candidates are preferentially queried
while fewer than 16 historical verified normals are available. This is
NOT an oracle normal selection. A deterministic pacing cap allows up to
40% of all queries in the first 400 steps, and afterward queries may
address score-region spending deficits. The global exact K limit still
applies. At end of stream, label-blind terminal catchup can exceed
regional quotas; overflows, denied proposals and forced queries are logged.

C's weights are exploratory, set before seeing R4 results, not calibrated
to Real-IAD ground-truth labels. None of these three arms directly addresses
calibration selection bias or memory-induced score scale staleness.

Original historical normal_dyn_plus_correction, correction bank, LTM,
maturation and other auxiliary writers are intentionally omitted, matching
the earlier R3 causal mechanism audit.

## Local Windows PowerShell setup

~~~powershell
git fetch origin
git switch e02-r4-abc-query-governance
git rev-parse --short HEAD
git status --short
conda activate torch
python experiments/e02_r4_abc_query_governance.py --self-test
python -m unittest discover -s tests -p "test_e02_r4_abc_query_governance.py" -v
~~~

GitHub CI verifies syntax and offline contracts, but only your local GPU
machine can test DINOv2 and Real-IAD end to end.

### Smoke test: 100 Audiojack images, three labels each

~~~powershell
python experiments/e02_r4_abc_query_governance.py --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --legacy-seed 0 --stream-seed 0 --query-seed 37 --query-rate 0.03 --max-samples 100 --math-sdpa --local-saliency --device cuda --print-every 25 --variants A,B,C --out-dir "outputs/e02_r4_abc_audiojack_smoke" --reference-r3-summary "outputs/e02_r3_audiojack_smoke/r3_summary.json" --reference-r3-csv "outputs/e02_r3_audiojack_smoke/r3_prequential.csv"
~~~

### Full run: 3815 Audiojack images, 114 labels each

~~~powershell
python experiments/e02_r4_abc_query_governance.py --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --legacy-seed 0 --stream-seed 0 --query-seed 37 --query-rate 0.03 --max-samples 0 --math-sdpa --local-saliency --device cuda --print-every 100 --variants A,B,C --out-dir "outputs/e02_r4_abc_audiojack_full_seed0" --reference-r3-summary "outputs/e02_r3_audiojack_full_seed0/r3_summary.json" --reference-r3-csv "outputs/e02_r3_audiojack_full_seed0/r3_prequential.csv"
~~~

Each output directory contains:
- r4_summary.json: methods, metrics, label counts, queries, score-region
  spending, query-cost diagnostics, Sidecar moves and stability windows.
- r4_prequential.csv: per-step pre-update scores, thresholds, predictions,
  query_cost_pre, score_region, selected label-blind query reason.
- canonical/canonical_v22_init.json: actual initial supports/theta0.
- r3_reference_replay in summary: passed=true, exact per-image A=R3
  replay for query positions, all controller scores and predictions.

If a three-arm run exceeds resources, run separate variants with
--variants A, --variants B, --variants C and different output dirs. Use
the R3 reference flags only for the A run. Confirm identical support
paths, stream hash and initial threshold in all three reports.

## How to judge results

- All arms must spend exactly K labels. A must reproduce the original R3
  Bootstrap prequential CSV exactly; otherwise B/C results are not
  interpretable as narrow interventions.
- Class-normal and class-defect queried feedback counts are audited
  separately from predicted score-region quotas. A low score does NOT
  guarantee a normal true label.
- For each variant, Frozen/Guarded/Balanced/Mistake must have identical
  AUROC and anomaly-score trajectory; FPR/FNR/BA can differ.
- Compare early (first 200/400) label counts, post-Bootstrap Query Cost,
  longest query-free gap, forced terminal catchup and actual STM sizes.
- Offline evaluation uses 256-image non-overlapping stability windows:
  FPR/FNR/Balanced Accuracy, class counts, queried labels per class.
  Missing class -> undefined risk (JSON null), NOT zero error risk.
- Compare worst-window error and recovery time before concluding one arm
  is robust. Overall AUROC/FNR alone is insufficient.
- Audiojack is exploratory: no hyperparameter tuning on these labels
  followed by a confirmatory Audiojack claim. For a publication, use
  fixed configs and independent categories/support seeds/held-out probes.

## Push results back to remote

~~~powershell
git add outputs/e02_r4_abc_audiojack_smoke outputs/e02_r4_abc_audiojack_full_seed0
git commit -m "experiment: E02-R4 ABC query governance results"
git push origin e02-r4-abc-query-governance
git ls-remote origin refs/heads/e02-r4-abc-query-governance
~~~

Only stage existing output directories; avoid accidental dataset, weight,
token, or sensitive configuration disclosure when .gitignore is unrestricted.
