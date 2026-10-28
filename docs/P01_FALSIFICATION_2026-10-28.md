# P0/P1 Small Falsification Study — 2026-10-28

## Purpose, before any further Bayesian controller development

**Research question P0:** Is the poor **fixed-threshold** FNR of the native Greedy bank primarily an out-of-date threshold artifact? For each final memory, estimate an independent **q90** threshold from the same 64 training-normal image identities (no Probe-A/B calibration). Report the unchanged R0 frozen q90 threshold *and* the bank-specific updated q90, plus AUROC/AUPRC unaffected by threshold.

**Research question P1:** At matched **22 verified-normal event admissions**, does R1 Hindsight Oracle dominate simple event selection (first K, hash K, frozen-static patch novelty top K, frozen-static image score top K)? Also execute original **causal historical-median novelty** with its natural unmatched event count, and greedy-all/static. Compare normal/defect trade-offs, admission counts, prototype counts, and calibration after final state.

### Strict interpretation

- Original frozen four normal support samples **plus 64 extra training-normal calibration images**, explicitly **NOT 4-shot-total**.
- Frozen D1-B Q-A camera/sample tape, single Real-IAD audiojack s0 q37 B128 run, capacity 256. Selected patch indices/sizes come from native static scorer. Native memory updates via original v22 Greedy, no replacement of source algorithms.
- P0 recalibration uses **64 train-normal images only** at the final memory state. Probe A and B never enter threshold fitting. All metrics are Probe A **development** measures. Do not inspect Probe B.
- Four K-matched controls use the COMPLETE sequence of already queried verified-normal metadata (future decisions). They are **hindsight controls**, not online-causal policies; they use no Probe labels. Their purpose is an exact event-count matched *counter-explanation* to R1.
- `r1_hindsight_beam_k` is even less causal: its decisions were trained on Probe-A SEARCH labels in R1. It must be reported as a hindsight reference, never as an unbiased benchmark. Its full-Probe-A confusion must match frozen R1.
- `causal_novelty_median` uses existing D2 baseline state, seen-only queried verified-normal features and an online prior median threshold; it does not enforce 22 admissions. Count and prototype usage are explicitly reported. It is not an independent recreation of the original HITL authors' system.
- Matching exactly 22 **events** does not match 208 stored prototypes. Always compare `n_dynamic_prototypes`, append/replace/skip counts. This is a preliminary falsification pilot, not final statistical identification.
- `static_score_top_k` and `static_novelty_top_k` use future queried-normal event scores to rank; for a causal online ablation, a separate preregistered stream-time budget policy is needed.
- The threshold after memory update is calibrated with fixed normal-only historical identities; full-stream prequential threshold control is NOT tested.
- Do not treat using the same 64 normal images for calibration of static and dynamic banks as proof that calibration generalizes; normal-domain shift should be audited later.
- R0 240-Probe-A static and greedy confusion **MUST** match to accept full run, including tape/static hashes and fixed theta; R1 Beam Oracle must match its saved full-Probe-A confusion.
- `--max-query-events` and reduced Probe-A groups always mark `PREFIX_SMOKE_INCOMPLETE`, never a full comparison.
- SHA-256 files use LF UTF-8 and `.gitattributes` disables platform newline conversion for this output folder. Never overwrite an existing output directory.

## P0/P1 command sequence (Windows PowerShell — EACH line independent)

Run from `E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection`, activated `(torch)`.

### 1 — Safe checkout and CPU contracts

```powershell
git status --short
git fetch origin
git switch p0p1-falsification-20261028
git pull --ff-only origin p0p1-falsification-20261028
git log -1 --oneline
python -m py_compile experiments/risk_safe/p01_falsification_core.py experiments/risk_safe/p01_falsification_native.py tests/test_risk_safe_p01.py
python -m unittest discover -s tests -p "test_risk_safe_p01.py" -v
```

Do not throw away uncommitted local work. Resolve uncommitted conflicts before switching.

### 2 — Native CUDA smoke, B128 first 4 Q-A events, first 2 probe-A groups/type

```powershell
python -m experiments.risk_safe.p01_falsification_native --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 128 --capacity 256 --target-admissions 2 --max-query-events 4 --max-probe-groups-per-type 2 --math-sdpa
python -m experiments.risk_safe.p01_falsification_native --class-name audiojack --support-seed 0 --query-seed 37 --budget 128 --capacity 256 --target-admissions 2 --max-query-events 4 --max-probe-groups-per-type 2 --verify-only
```

Expected `PREFIX_SMOKE_INCOMPLETE`, verified checksum, no scientific interpretation. If an error occurs, stop; provide the traceback for code correction. Never rerun an existing output path without first diagnosing the failure.

### 3 — P0/P1 full GPU study; frozen R0 and R1 parity mandatory

```powershell
python -m experiments.risk_safe.p01_falsification_native --dataset-root "G:/dataset/realiad" --json-dir "G:/dataset/realiad/jsons/realiad_jsons" --class-name audiojack --support-seed 0 --query-seed 37 --budget 128 --capacity 256 --target-admissions 22 --hash-seed 1701 --r0-reference-summary "outputs/r0_opportunity_20261028/native/audiojack_s0_q37_b128_cap64-128-256-1024_thetatrain64-q90_audit1_full/r0_summary.json" --oracle-summary "outputs/r1_oracle_20261028/audiojack_s0_q37_b128_cap256_beam_w4_thetatrain64-q90_P4_lambda1_full/r1_summary.json" --math-sdpa
python -m experiments.risk_safe.p01_falsification_native --class-name audiojack --support-seed 0 --query-seed 37 --budget 128 --capacity 256 --target-admissions 22 --hash-seed 1701 --verify-only
```

The native study produces 240 Probe-A scores and 64 reserved training-normal calibration scores **for each policy**. It must report R0 and R1 parity, `COMPLETE`, and per-policy fixed/calibrated FPR/FNR and AUROC. If it does not, do not claim success.

### 4 — Inspect and upload ALL validated results to GitHub

```powershell
Get-ChildItem "outputs/p01_falsification_20261028" -Recurse -File | Select-Object FullName,Length
git status --short
git add -- outputs/p01_falsification_20261028
git diff --cached --stat
git commit -m "P0/P1: archive calibration and admission falsification GPU results"
git push -u origin p0p1-falsification-20261028
git status --short
```

Never stage dataset images, checkpoints, conda secrets or credentials. Commit complete result directories with their `checksums.json`, plus smoke outputs, without rewriting original R0 or R1 files.

Once pushed, send only **“P0/P1 结果已推送 GitHub，请直接分析”**. Assistant should fetch the saved artifacts from `outputs/p01_falsification_20261028`, independently verify SHA-256, check R0/R1 parity, and report both the fixed-theta and independent normal-only calibration outcomes before choosing whether to continue Safe Admission.

## Results to interpret

- `p01_summary.json`: per policy fixed `theta=0.562028...` vs normal-only recalibrated theta; confusion counts, AUROC/AUPRC, balance and source metadata, provenance checks.
- `p01_query_tape.csv`: ground-truth Q-A metadata shown only retrospectively; normal-admission candidates are created after query disclosure.
- `p01_candidate_metadata.csv`: precomputed frozen-static novelty, native candidate patch counts and source identity.
- `p01_admission_decisions.csv`: selector decisions and actual native Greedy operations for matched and causal novelty arms.
- `p01_calibration_scores.csv`: independently scored 64 extra train-normal images for every final memory, enabling independent threshold reconstruction.
- `p01_probe_a_scores.csv`: paired static and dynamic image scores for all 240 Probe-A images; offline-only, no Probe-B.
- `p01_provenance.json`: source code hashes, Git HEAD, frozen D1-B Q-A and training/probe identities.
- `checksums.json`: byte-exact SHA-256.

## Predeclared decision criteria

**Calibration primary question:** Does Greedy 256 with *training-normal-only* recalibrated q90 substantially reduce FNR without greatly increasing FPR, compared with its original frozen q90? Preserve the AUROC/AUPRC distinction. If it does, complexity of a dedicated safety gate is less justified.

**Admission primary question:** Do simple K=22 controls achieve comparable error tradeoffs to R1 Beam Oracle, especially after matching prototype count? If yes, the R1 Beam result is not evidence of sophisticated risk assessment. Do not hide any baseline with a better result.

**Safe Admission NO-GO:** If normal-only calibration explains most fixed-threshold damage, and novelty/hash/first-K provide comparable matched-event outcomes, postpone CDARG/Bayesian controller; prioritize calibration and efficient memory capacity/retrieval. This single Audiojack seed is exploratory: any positive difference must later replicate across seeds/classes.

**Further study GO (not automatic):** Only if a residual harm persists after fair calibration AND a causal risk signal can predict held-out defects beyond novelty should a new Safe Admission model be considered.

