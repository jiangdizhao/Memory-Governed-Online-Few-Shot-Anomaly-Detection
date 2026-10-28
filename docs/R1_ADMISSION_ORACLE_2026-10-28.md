# R1 — Counterfactual Sequential Admission Oracle (2026-10-28)

## Status and interpretation

**R1 code is committed for local tests + native CUDA runs. No GPU R1 results exist until the operator runs them.** Branch `r1-admission-oracle-20261028`, forked from `r0-opportunity-audit-20261028`. All R0, D1–D4 and v22 source/results remain immutable.

R0 observed actual replacement, risk erosion and significant Probe-A FN increases under B128. R1 asks whether there is an *offline hindsight* sequence of verified-normal admission choices that could avoid **new False Negatives** while rescuing normal False Positives. This tests **attainability**, not the existence of a deployable causal rule.

**Modes**
- `exact` enumerates all `2^N` accept/reject binary sequences for the actual `N` verified-normal Q-A events. Each branch replays native state-dependent Greedy. Exact optimum is ONLY for the fixed class/seed/candidate tape, memory capacity and **Oracle-search Probe-A subset** under the locked ranking. There is no guarantee on future images.
- `beam` maintains at most `--beam-width` states and evaluates both accept/reject extensions after each query-normal candidate. Heuristic **BEST FOUND FEASIBLE**, **NOT** a global bound or provably optimal result. Different choices alter future native Greedy replacements; no additive shortcut is used.
- `--max-normal-events N` truncates the normal admission sequence and is always `PREFIX_SMOKE_INCOMPLETE`; it is never the full protocol or an all-budget global optimum.

## Frozen causal and statistical contracts

- Same four-shot initial static VisionAD bank and original DINOv2/v22 top-5 scoring / static score-based patch candidate extraction and exact Greedy `proposed()` semantics. Original Q-A random queried-event tape is strictly label-blind; R1 verifies archived 16/32/64/79 manifests and derives B128 from the *frozen* D1-B generator.
- Only a queried **verified normal** image generates one indivisible admit/reject decision. Queried defects are not admitted, but the *offline Oracle knows the verified normal-event sequence* retrospectively. This is NOT legitimate online policy information, thus performance claims are diagnostic.
- No adaptive query controller, Sidecar, Signed Correction, LTM, or probe-label feedback to an online deployable policy.
- Exact same normal feature proposals are offered across all admission states. Each acceptance is applied to that state using the real native Greedy slot update and exact full-feature normalization. Rejection preserves the state byte-for-byte. No pseudo-distance surrogate.
- Use `--theta-mode train64-q90` for matched R0 B128: four fixed supports plus **64 separate normal training views** for threshold calibration, NEVER four-shot-total information. LOSO arm exists separately. Neither threshold is a proven safe deployed operating point.
- **Probe-A data firewall:** official sample-group split, deterministic, 24 normal +24 defect groups; first `K` per type for Oracle SEARCH and remaining `24-K` per type for **disjoint audit**. Both include five camera views, and some defect-group views are image-level normal. Search subset true labels are used ONLY by retrospective Oracle. Disjoint audit is evaluated only after the final policy is selected; still development-only, because Probe A has been inspected in prior experiments. Probe B stays unopened.
- Do NOT publish the Oracle Probe-A search result as unbiased model performance. No method has access to the hindsight search labels at deployment.
- Report static, Greedy-all (same capacity), and Oracle-found with same tape; **admission count is not matched** by default. Exact search provides `r1_frontier_by_count.csv` for retrospectively matched counts. Beam search does not exhaust matched-count frontiers.
- No category, seed, budget, capacity, penalty or Probe-A partition is re-chosen after seeing performance to make a method look better.

## Pre-registered primary selection objective

For each terminal state, on the **Oracle-search** subset, rank candidate sequences lexicographically:
1. Any state with zero **new** FNs relative to static outranks all states introducing FNs;
2. Then minimize number of new FNs;
3. Then maximize rescued FPs;
4. Then maximize `mean_positive_normal_score_relief - lambda_defect * mean_positive_defect_score_erosion` with `lambda_defect=1`;
5. Then maximize balanced accuracy; then prefer fewer admissions/effective changes and deterministic tie-breaking.

The first two entries prioritize **no new defect misses**. This is intentionally conservative and may select Static (no admissions). If there is no beneficial safe sequence, R1 should report it rather than changing the objective.

In addition to the chosen state report `tp/tn/fp/fn`, FPR, FNR, balanced accuracy, new FN, rescued FP, continuous score relief / defect erosion, signed normal hinge utility, accepted-event indices, dynamic bank digest, replace/append/skip counts, all exact leaves or terminal beam states, and disjoint audit Probe-A results.

## GPU memory / runtime management

Exact B16 usually has 11 normal query events in Audiojack seed37, so `2^11=2048` terminal states, each scored on (4+4) Probe-A official groups = 40 camera images. **This is expensive even on RTX 3090 Ti and has no time SLA**. Lower search groups or narrower prefixes are only separately labeled diagnostics, not replacements for the registered full exact run. Exact mode fails if `2^N > --max-leaves` (default 4096), rather than silently truncating.

Beam B128 has 85 normal events in the already recorded Audiojack seed37 test. `--beam-width 4` explores at most 8 states per depth on the search subset, and finally evaluates 200 disjoint audit images. Other classes may have different N. Beam cannot be described as exhaustive.

Default `--probe-cache-device cuda` keeps **40 Oracle-search image features** on GPU; for OOM use `--probe-cache-device cpu` in a separate run folder and report altered execution settings. This changes cache placement, not intended scoring; check numerical parity. R1 always processes the final disjoint audit image-wise without caching all its features.

## One-line Windows PowerShell commands

All commands below are **independent single lines**, run from `E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection`.

### 1. Synchronize branch

```powershell
git status --short
git fetch origin
git switch r1-admission-oracle-20261028
git pull --ff-only origin r1-admission-oracle-20261028
git log -1 --oneline
python -m py_compile experiments/risk_safe/r1_oracle_core.py experiments/risk_safe/r1_native_oracle.py tests/test_risk_safe_r1.py
python -m unittest discover -s tests -p "test_risk_safe_r1.py" -v
```

If local work is uncommitted, inspect it before switching. Never discard/reset the working tree to follow this document.

### 2. CUDA smoke — exact 3-normal-event prefix

```powershell
python -m experiments.risk_safe.r1_native_oracle --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 16 --capacity 128 --mode exact --max-normal-events 3 --search-groups-per-type 2 --theta-mode train64-q90 --math-sdpa
python -m experiments.risk_safe.r1_native_oracle --class-name audiojack --support-seed 0 --query-seed 37 --budget 16 --capacity 128 --mode exact --max-normal-events 3 --search-groups-per-type 2 --theta-mode train64-q90 --verify-only
```

Expected `PREFIX_SMOKE_INCOMPLETE`, `2^3=8` enumerated leaves; actual GPU score evaluation checks native-v22 and memory/state API. **Do not use smoke metrics for research claims.** If smoke errors, stop and report traceback, do not run the full jobs.

### 3. Full R1-A: B16 exact sequential Oracle, capacity 128

```powershell
python -m experiments.risk_safe.r1_native_oracle --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 16 --capacity 128 --mode exact --search-groups-per-type 4 --theta-mode train64-q90 --max-leaves 4096 --math-sdpa
python -m experiments.risk_safe.r1_native_oracle --class-name audiojack --support-seed 0 --query-seed 37 --budget 16 --capacity 128 --mode exact --search-groups-per-type 4 --theta-mode train64-q90 --max-leaves 4096 --verify-only
```

The run must report `EXACT_FINITE_PROTOCOL_OPTIMUM`, `globally_exhaustive=true`, and the exact `2^N` leaf count; do not infer from budget 16 that there are 16 admitted normal candidates.

### 4. Full R1-B: B128 beam, capacity 256; strict R0 parity gate

```powershell
python -m experiments.risk_safe.r1_native_oracle --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 128 --capacity 256 --mode beam --beam-width 4 --search-groups-per-type 4 --theta-mode train64-q90 --r0-reference-summary "outputs/r0_opportunity_20261028/native/audiojack_s0_q37_b128_cap64-128-256-1024_thetatrain64-q90_audit1_full/r0_summary.json" --math-sdpa
python -m experiments.risk_safe.r1_native_oracle --class-name audiojack --support-seed 0 --query-seed 37 --budget 128 --capacity 256 --mode beam --beam-width 4 --search-groups-per-type 4 --theta-mode train64-q90 --verify-only
```

This gate compares R1 Static and Greedy-all to the **original R0** on all Probe-A images, as well as the immutable static-bank fingerprint, threshold and Q-A tape. It must pass before interpreting Oracle results. `beam` should report `BEAM_BEST_FOUND_NOT_AN_UPPER_BOUND`, and never claim complete search among `2^85` choices.

### 5. Inspect summary and upload ALL R1 results to GitHub

```powershell
Get-ChildItem "outputs/r1_oracle_20261028" -Recurse -File | Select-Object FullName,Length
git status --short
git add -- outputs/r1_oracle_20261028
git diff --cached --stat
git commit -m "R1: record exact B16 and beam B128 oracle experiments and provenance"
git push -u origin r1-admission-oracle-20261028
git status --short
```

**Do not stage dataset images, checkpoints, CUDA caches or local credentials.** `.gitattributes` in this branch disables Git automatic newline rewriting for R1 outputs; R1 CSV/JSON use LF UTF-8 for cross-platform stable SHA-256. Archive all result files, including failed/successful smoke, but commit only complete directories with valid `checksums.json`.

After `git push` succeeds, tell the assistant only **“R1 结果已经推送 GitHub，请直接检查”**. The assistant should fetch files from this branch directly, verify result configs/checksums/provenance, and analyze them without requiring terminal copy/paste.

## Outputs per run

- `r1_summary.json`: exact/beam status, search/audit/full Probe-A metrics for Static, Greedy-all, Oracle-found, score calls, selections and causal/fairness caveats
- `r1_queries.csv`: all Q-A events, verified labels and static proposal sizes; labels are shown retrospectively only
- `r1_search_states.csv`: all exact terminal states or surviving beam states, admission bitstrings and metrics
- `r1_frontier_by_count.csv`: best balanced-accuracy point for each observed admission count; only exhaustive for exact mode
- `r1_search_probe_scores.csv` / `r1_audit_probe_scores.csv`: image-level paired scores, group IDs and labels for diagnosis, audit disjointness
- `r1_beam_trace.csv`: beam rank/width and best-state trajectory when beam mode
- `r1_provenance.json`: Git HEAD, source hash, dataset JSON digest, D1-B archived Q-A digest, disjoint Probe-A group IDs
- `checksums.json`: exact SHA-256 file-byte digests, UTF-8/LF stable

## R1 acceptance / research interpretation

R1 technical GO requires passing CPU tests and native CUDA smoke, complete B16 exact leaves, B128 beam labeled heuristic, unmodified original D1-B tape identities, disjoint Probe A audit and no Probe B, R0 native Greedy parity on B128, and verified Git-pushed outputs.

R1 scientific GO for a deployable admission controller additionally needs an **attainable safe trade-off**, robust non-overfit audit performance, and later R2 risk prediction from causal-only evidence. Oracle Probe-A search performance alone cannot meet this scientific GO.

If exact or beam optimizes away all updates, report it. If Oracle seems to dominate Greedy-all on SEARCH but not AUDIT, report selection overfitting. Do not change lambda or probe cohorts after reading R1 results without starting a separately declared exploratory protocol.
