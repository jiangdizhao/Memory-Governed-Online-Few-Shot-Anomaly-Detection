# R2-SGRM-VAL1 - CPU Retrospective Development Diagnostic

**Status:** `COMPLETE`<br>
**Classification:** `RETROSPECTIVE_DEVELOPMENT_DIAGNOSTIC`<br>
**Run:** `r2_sgrm_val1_20261102_a5`<br>
**Source commit:** `e2e31be50ccfffbcc967fb1d68505b2a57da595c`<br>
**Runtime:** 40.84s / 600s hard ceiling<br>
**Execution:** CPU-only; CUDA visibility disabled; no model/DINO/GPU/image decoding.

This is development-only retrospective evidence, not independent confirmation, a new blind evaluation, or cross-category generalization. Historical aggregate development results had already been observed before this task. Earlier broad documentation searches surfaced historical outcome summaries; these were not used as online scoring inputs. No parameter search or correction rescaling was performed, and no unrelated R1 output was consulted during this run.

## Supervised attempt history

Fresh attempts `r2_sgrm_val1_20261102_a1` and `_a2` stopped before input files were opened because the runner referenced a nonexistent EXP2A path attribute. `_a3` stopped after verifying and loading the sealed EXP2B score receipts, before reading event shards, because a fractional beta cap was incorrectly converted to an integer. `_a4` stopped at switch event 35 after that event's score was fsynced and only its current queried diagnostic/feedback was read; the diagnostic schema has no event ID field. No attempt was resumed, all partial namespaces were preserved, and only fresh run `_a5` contributes results.

## Inputs and score-seal provenance

EXP2A input root: `E:/Memory-Governed-Online-Few-Shot-Anomaly-Detection/outputs/calibration/r1_digest_contract_v1_exp2a_r2exp2a_20261101_a4`<br>
EXP2B input root: `E:/Memory-Governed-Online-Few-Shot-Anomaly-Detection/outputs/calibration/r1_digest_contract_v1_exp2b_r2exp2b_20261101_b1`<br>
Authorized scope: support seed 0; switch, audiojack, regulator, rolled_strip_base; exactly 512 events each. The EXP2A regime was 4 deployed seed-0 supports plus 64 additional C64 training normals; this is not strict four-shot-total.

EXP2A Stage A seal SHA-256: `b23f552e4c18cccba12d52e3a978f2fe6973b03b58704d44fe0295ad8f5b74c1`<br>
EXP2A completion seal SHA-256: `8a93dc52b1d26f660241ac85fc2132ac83678873d358ec96190f283ebf1384c0`<br>
EXP2B Stage A score seal SHA-256: `864860113869af96a1e25aa4b7398d99dab21f191c8d02c9ad408a117974b167`<br>
EXP2B event identity receipt SHA-256: `206b7c5f426d79519910b73ef0301364bcbdb161beefaa398c21ac39f1de3c56`<br>
Stage A sealed all 2,048 new scores before the Stage B child process opened outcome metadata. During Stage A, the 67 permitted queried-feedback records were consumed sequentially only after the matching event score was flushed and fsynced. No unqueried outcome, future feedback, or mask was used.

The C1-NoBoth matched FIFO replay used the original dN keep mask, beta floor/clipping, signed beta, causal query position, Source Group identity, and capacity-256 policy. The EXP2B receipts report exact native candidate parity, no eviction/merge/replacement, and no FIFO/native memory divergence for these prefixes. The two readouts use the same pre-event anchors and one shared cosine-distance matrix; each full 784-patch map is max aggregated.

C0 scores matched exactly for all 2,048 events. Existing C1-NoBoth native parity receipts covered 512 events per category. Flat Gaussian versus EXP2B D maximum absolute patch-delta errors were 4.66e-8 to 1.20e-7; maximum image-score errors were 5.96e-8 to 8.94e-8, below the frozen atol=1e-5/rtol=1e-4 tolerance. Reconstructed FIFO inserts were 99 (audiojack), 136 (switch), 116 (regulator), and 100 (rolled_strip_base); no evictions occurred.

## Category metrics

Audiojack is normal-only in this fixed prefix, so AUROC, AP, FNR and BA remain undefined. The four-category AUROC macro is undefined; the three-evaluable-category macro is descriptive only.

| Category | Arm | AUROC | AP | Prevalence | FPR | FNR | BA | TP/TN/FP/FN |
|---|---|---:|---:|---:|---:|---:|---:|---|
| switch | C0 | 0.8070 | 0.9152 | 0.7012 | 0.0261 | 0.6017 | 0.6861 | 143/149/4/216 |
| switch | C1_NoBoth | 0.8525 | 0.9369 | 0.7012 | 0.0588 | 0.4791 | 0.7310 | 187/144/9/172 |
| switch | SGRM | 0.8522 | 0.9370 | 0.7012 | 0.0523 | 0.4847 | 0.7315 | 185/145/8/174 |
| switch | historical_C2_a0.02 | 0.8410 | 0.9305 | 0.7012 | 0.0588 | 0.4958 | 0.7227 | 181/144/9/178 |
| switch | matched_flat_Gaussian | 0.8525 | 0.9369 | 0.7012 | 0.0588 | 0.4791 | 0.7310 | 187/144/9/172 |
| audiojack | C0 | undefined | undefined | 0.0000 | 0.0059 | undefined | undefined | 0/509/3/0 |
| audiojack | C1_NoBoth | undefined | undefined | 0.0000 | 0.0039 | undefined | undefined | 0/510/2/0 |
| audiojack | SGRM | undefined | undefined | 0.0000 | 0.0039 | undefined | undefined | 0/510/2/0 |
| audiojack | historical_C2_a0.02 | undefined | undefined | 0.0000 | 0.0000 | undefined | undefined | 0/512/0/0 |
| audiojack | matched_flat_Gaussian | undefined | undefined | 0.0000 | 0.0039 | undefined | undefined | 0/510/2/0 |
| regulator | C0 | 0.6872 | 0.5888 | 0.3887 | 0.0096 | 0.9246 | 0.5329 | 15/310/3/184 |
| regulator | C1_NoBoth | 0.7180 | 0.6387 | 0.3887 | 0.0096 | 0.9347 | 0.5279 | 13/310/3/186 |
| regulator | SGRM | 0.7176 | 0.6376 | 0.3887 | 0.0096 | 0.9296 | 0.5304 | 14/310/3/185 |
| regulator | historical_C2_a0.02 | 0.7041 | 0.6089 | 0.3887 | 0.0064 | 0.9447 | 0.5244 | 11/311/2/188 |
| regulator | matched_flat_Gaussian | 0.7180 | 0.6387 | 0.3887 | 0.0096 | 0.9347 | 0.5279 | 13/310/3/186 |
| rolled_strip_base | C0 | 0.8983 | 0.9824 | 0.8613 | 0.0704 | 0.3129 | 0.8083 | 303/66/5/138 |
| rolled_strip_base | C1_NoBoth | 0.9069 | 0.9841 | 0.8613 | 0.1831 | 0.1701 | 0.8234 | 366/58/13/75 |
| rolled_strip_base | SGRM | 0.9066 | 0.9840 | 0.8613 | 0.1831 | 0.1701 | 0.8234 | 366/58/13/75 |
| rolled_strip_base | historical_C2_a0.02 | 0.9050 | 0.9836 | 0.8613 | 0.1549 | 0.1950 | 0.8250 | 355/60/11/86 |
| rolled_strip_base | matched_flat_Gaussian | 0.9069 | 0.9841 | 0.8613 | 0.1831 | 0.1701 | 0.8234 | 366/58/13/75 |

## SGRM minus matched flat Gaussian

| Category | Delta AUROC | Delta AP | Delta FPR | Delta FNR | Delta BA | Helpful reversals | Damaging reversals |
|---|---:|---:|---:|---:|---:|---:|---:|
| switch | -0.0003 | 0.0000 | -0.0065 | 0.0056 | 0.0005 | 125 | 141 |
| audiojack | undefined | undefined | 0.0000 | undefined | undefined | 0 | 0 |
| regulator | -0.0004 | -0.0011 | 0.0000 | -0.0050 | 0.0025 | 60 | 86 |
| rolled_strip_base | -0.0003 | -0.0001 | 0.0000 | 0.0000 | 0.0000 | 22 | 30 |

## Correction and Source Group diagnostics

The aggregate JSON reports paired Source Group bootstrap intervals (1,000 replicates, seed 20261101), conditional on the existing trajectories. It also reports patch RMS/mean absolute correction, image-score movement, signed evidence, active Source Group counts, eligible patch-count distributions, effective contributing groups, and pair-reversal diagnostics. Effective groups are the inverse Herfindahl count over absolute per-patch group contributions. No magnitude normalization was used.

The SGRM-versus-flat effect is descriptive. A null or negative result is valid. No safety guarantee, publication-level novelty, or independent generalization claim follows from these four previously exposed development prefixes.

## CPU regression checks

`py_compile` passed. Focused CPU regression suites: 62 passed, 1 skipped. The Windows process-tree termination test passed. The symlink-parent test was skipped because symlink creation was unavailable in this environment.

Stage A: `SEALED_COMPLETE`; total scored events 2048; queried feedback updates 67. Flat Gaussian versus EXP2B D parity used atol=1e-05, rtol=0.0001; per-category maximum patch and image-score errors are in `aggregate_results.json`.
