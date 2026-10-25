# E02-R5: D = B + time-only query pacing

## Objective and attribution

R4 A/B/C found:
- A Bootstrap cost feedback over-suppressed queries;
- B removing Bootstrap cost penalty exhausted 114 labels near step 1965;
- C used both score-region query interventions AND temporal pacing.
C improved feedback timing, but we cannot attribute its gains to stratification
without a pure time-pacing control.

**D = B + EXACT C cumulative temporal query allowance.** D changes
no other part of B's query proposal policy, cost adaptation, verified-normal
STM writing, Fixed Bank, or all four Sidecar controllers. Unlike C, D
has no predicted score strata budgets, no low-score priority, and no
underfilled-stratum forced queries. The underlying R4 observation logger
still records score_region; that field is diagnostic only in D.

Pacing rule is imported verbatim from R4 ScoreRegionBudget.allowed_total:
- for the first 400 images, cumulative allowed labels: round(K * 0.40);
- later the allowance grows linearly from this early cap to K by the
  final image;
- the original R4 hard budget still triggers label-blind terminal catchup
  if necessary. Catchup can exceed the pacing allowance at the very end,
  particularly in the 100-image smoke (same limitation as C).

B/D isolates the total effect of time pacing (including downstream feedback
and memory differences). D/C isolates the extra combined effect of C's
score-region quotas, low-score priority and quota-deficit supplementation,
which may also alter Bootstrap termination and later adaptive query cost.
D/C does NOT isolate each of those individual components.

Uses E01/E02 canonical initialization, frozen theta for query decisions,
E02 verified-normal STM update, and R4 Frozen/Guarded/Balanced/Mistake
thresholds. The historical full correction/LTM/maturation system is not
tested. All R3/R4 code and results stay unchanged.

## Setup in Windows PowerShell

Execute inside your local repo root. Preserve any uncommitted work before
switching branches.

~~~powershell
git fetch origin
git switch e02-r5-d-temporal-pacing
git rev-parse --short HEAD
git status --short
conda activate torch
python experiments/e02_r5_d_temporal_pacing.py --self-test
python -m unittest discover -s tests -p "test_e02_r5_d_temporal_pacing.py" -v
~~~

CI only tests syntax and CPU-only contracts; full Real-IAD CUDA runs on
your local machine.

## Step 1: GPU smoke (100 images)

The R4 smoke summary must exist. This is required for matching the exact
support, stream and query-budget settings.

~~~powershell
python experiments/e02_r5_d_temporal_pacing.py --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --legacy-seed 0 --stream-seed 0 --query-seed 37 --query-rate 0.03 --max-samples 100 --math-sdpa --local-saliency --device cuda --print-every 25 --variants D --strata-early-steps 400 --strata-early-budget-fraction 0.4 --reference-r4-summary "outputs/e02_r4_abc_audiojack_smoke/r4_summary.json" --out-dir "outputs/e02_r5_d_audiojack_smoke"
~~~

Acceptance: r4_reference_check.passed=true, N=100, exactly three queries,
no traceback, schema e02.r5.d.time_only.v1. For this very short smoke,
the C pacing allowance is only one query before catchup; do not
interpret its classification metrics scientifically.

## Step 2: Full Audiojack (3815 images)

~~~powershell
python experiments/e02_r5_d_temporal_pacing.py --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --legacy-seed 0 --stream-seed 0 --query-seed 37 --query-rate 0.03 --max-samples 0 --math-sdpa --local-saliency --device cuda --print-every 100 --variants D --strata-early-steps 400 --strata-early-budget-fraction 0.4 --reference-r4-summary "outputs/e02_r4_abc_audiojack_full_seed0/r4_summary.json" --out-dir "outputs/e02_r5_d_audiojack_full_seed0"
~~~

D alone runs through the full stream; there is NO need to recompute
R4 B or C. Comparability gates run before output:
- old R4 B/C reference, matching parameters, exact four support paths;
- same initial Fixed Bank hash and initial theta;
- identical path sequence SHA and stream length;
- same 114-label final query budget.
If a gate fails, stop and inspect the mismatch rather than comparing
different data streams or initialization settings.

## Outputs

- r5_d_summary.json: D final AUROC/FPR/FNR/BA (four controllers),
  256-step stability windows, bootstrap label counts, worst-window BA,
  longest no-query interval, pacing diagnostics, and auto B/D/C
  comparisons from the archived R4 summary.
- r5_d_prequential.csv: every single D prequential prediction,
  query-cost pre-update, query decision, pacing status and
  evaluation-only true label.
- canonical/canonical_v22_init.json: original canonical initialization.

Interpretation rules:
- The early pacing cap for a full stream is 46 out of 114 labels by
  step 400; C used this SAME time allowance.
- D still has the native Bootstrap and SimpleAIF query preferences.
  Even if it spends 46 labels early, there is NO guarantee it obtains
  16 true normal samples. Labels must not be read to force a quota.
- Inspect queried normals/defects, effective STM prototypes, longest
  query-free gap, 256-step Balanced Rescue FNR/FPR, and threshold moves.
- D vs B is pacing-alone. C vs D is score-region intervention package,
  not only quota fractions.
- The time pacing rule was developed on Audiojack; independent classes,
  support seeds, and held-out probes are necessary to validate any
  publication-level generalization.

## Upload results for analysis

~~~powershell
git add outputs/e02_r5_d_audiojack_smoke outputs/e02_r5_d_audiojack_full_seed0
git commit -m "experiment: E02-R5 D time-only pacing results"
git push origin e02-r5-d-temporal-pacing
git ls-remote origin refs/heads/e02-r5-d-temporal-pacing
~~~

Only stage the directories actually produced. Since this repo's ignore
rules have been relaxed, verify that no private datasets, credentials
or model weights are included accidentally in the commit.
