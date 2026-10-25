# E02-R3 — bootstrap-aware matched-budget Sidecar audit

This branch **does not modify** original `test_phase3_AIF_simplified_v22.py`, E01, E02, or R1/R2. New entrypoint:
`experiments/e02_r3_bootstrap_sidecar.py`.

## Scientific aim

Separate two effects, without claim inflation:

1. **Within-query-schedule sidecar effect**: Frozen, original v22
   Guarded Dual Anchor, original v22 Balanced Rescue, and the already tested
   Mistake Feedback controller receive the **same online STM score trajectory**
   and the **same queried verified labels**. Their FPR/FNR/BA differences
   identify effects of threshold controller design under that schedule.
2. **Across-query-schedule effect**: seeded random label selection versus a
   native-v22-primitive Bootstrap → Simple-AIF selection sequence under a
   **matched hard final 3% budget**. These two schedules intentionally produce
   different queried labels and **different learned STM memories**; their
   result gap is a joint effect of querying and memory, **not** isolated
   Sidecar or Active Inference EFE efficacy.

We use original v22's actual
`choose_simple_aif_bootstrap_action()`,
`select_simple_aif_action()`, `SimpleAIFState`,
`NormalCoverageMemory`, `compute_two_channel_evidence()`, and
`GuardedDualAnchorThresholdController.observe_labeled()`.

For comparability with the prior E02 family, the memory update is still the
E02 verified-normal-only `update_from_topk` primitive with novelty fixed
at zero for write weighting; no signed correction, LTM, maturation, defect
bank, or unverified self-training. The original Simple-AIF post-Bootstrap
uses `ablation=normal_dyn_only` for EFE, **not**
`normal_dyn_plus_correction` from the historical 30-class command.

### Explicit differences from the historical run

- Historical test used `--baseline_compat`, correction-enabled memory,
  native local query budget adaptation and (possibly) other features.
  This experiment uses the canonical E01/E02 initialization, which has
  `baseline_compat=False`, and simplified STM-only updates.
- A strict global label cap K = round(N × qrate) has been added for **both**
  query schedules, to protect against the historical 123-vs-114 feedback
  mismatch. A deterministic, label-blind *terminal catchup* fills any
  unspent budget at the end. The counters
  `budget_suppressed` and `budget_forced_terminal` are reported.
  This cap/catchup is experimental and differs from native v22
  unconstrained Simple-AIF querying. High terminal catchup indicates
  the schedule needs redesign; it MUST NOT be quietly presented as an
  original v22 trajectory.
- The query controller uses a frozen canonical theta reference for
  every schedule, rather than coupling queries to the threshold controller
  being compared. Otherwise different sidecars would alter the query tape
  and STM content, invalidating within-schedule causal contrasts.
- Native warmup-budget override and FN audit are **off**: the tested early
  behavior is Simple-AIF **Bootstrap**, not the separate v22 warmup budget.
- No held-out probe split is implemented **in R3**; this is a prequential
  online-stream mechanism audit. Held-out R1 and cross-seed followups
  remain necessary.

## Run on Windows (PowerShell)

In repository root, after safely retaining any local uncommitted changes:

~~~powershell
git fetch origin
git switch e02-r3-bootstrap-sidecar-audit
git rev-parse --short HEAD
git status --short
conda activate torch
python experiments/e02_r3_bootstrap_sidecar.py --self-test
~~~

### First smoke: Audiojack, 100 images

~~~powershell
python experiments/e02_r3_bootstrap_sidecar.py --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --legacy-seed 0 --stream-seed 0 --query-seed 37 --query-rate 0.03 --max-samples 100 --math-sdpa --local-saliency --device cuda --print-every 25 --out-dir "outputs/e02_r3_audiojack_smoke"
~~~

Only 3 labels, so balanced rescue may have too few defect anchors. Do not
use this run for scientific conclusions. First confirm run completes and
reports exactly 3 queries **in both schedules**, no hidden label reads,
and the four controller AUROCs are identical **within each schedule**.

### Full run: Audiojack, 3815 images, 114 queries per schedule

~~~powershell
python experiments/e02_r3_bootstrap_sidecar.py --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --legacy-seed 0 --stream-seed 0 --query-seed 37 --query-rate 0.03 --max-samples 0 --math-sdpa --local-saliency --device cuda --print-every 100 --out-dir "outputs/e02_r3_audiojack_full_seed0"
~~~

The native query primitives default here to the historical
`query_cost=0.8`, `beta_entropy=0.3`, Bootstrap target=16 verified
normal global descriptors, max Bootstrap steps=400, absolute boundary
band floor=0.02. The strict budget guard is part of the comparison.

### Outputs

- `r3_summary.json` contains both schedules' query indices,
  first-200 query counts, normal vs defect feedback mix, suppression
  and terminal catchup counts, memory sizes, controllers' AUROC/FPR/FNR/BA,
  accepted threshold changes and diagnostics.
- `r3_prequential.csv` has 2 × N rows, grouped by step, one per schedule:
  score, theta and prediction **before** current-step label disclosure,
  policy action and budget reason.
- `canonical/canonical_v22_init.json` from unmodified v22.

When you provide results, upload `r3_summary.json` and
`r3_prequential.csv`, plus full traceback if a smoke test fails.

## Acceptance gate

1. All runs share the same canonical v22 support images / Fixed Bank hash.
2. Random query tape matches seeded E02 random positions exactly.
3. Both schedules consume precisely K=114 verified labels (full run).
4. Only queried verified normals enter STM and NormalCoverageMemory.
5. Within each schedule, all threshold variants have bitwise identical
   anomaly scores and thus equal AUROC, using the same query tape.
6. Frozen, Guarded and Balanced start from identical theta0.
7. No test labels leak into the selection policy; evaluation labels are read
   only after every prediction for that step is computed.

### Interpretation

Analyze within-schedule Balanced Rescue vs Guarded before comparing
across schedules. Compare `first_200_queries` with historical 32/200,
but do **not** expect exact matching: historical initialization and
correction-enabled policy differ. If Bootstrap query rate or feedback mix
is different, inspect `query_reasons`, `bootstrap_steps`,
`budget_suppressed`, and `budget_forced_terminal` rather than modifying
the controller to force a desirable FNR.
