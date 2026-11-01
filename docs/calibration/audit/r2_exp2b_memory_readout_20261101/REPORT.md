# R2-EXP2B — CPU-Only Memory Management × Residual Readout Attribution

**Status:** `PASS`. **Evidence class:** conditional development analysis on existing EXP2A trajectories, not independent confirmation.

**Run:** `r2exp2b_20261101_b1`; input: `r2exp2a_20261101_a4`; source parent: `4a677553202df67f66cac855c97eb5d4e091d251`.

## Protocol and provenance

- CPU-only replay of the four existing 512-event EXP2A prefixes. No DINO loading, image decoding, new feature capture, new categories, seeds, windows, query policies, or masks.
- Stage A consumed only the existing feature/map shards, C0 trajectory, frozen query tape, and 67 already-disclosed query labels. Stage B opened only the first 512 identity-matched TEST outcomes per valid category after the score seal.
- Four-shot support plus additional C64 normals; this is not strict four-shot total. No C64 rescoring occurred.
- Supervisor: `COMPLETED`, Windows Job Object, 1800 second total ceiling; no automatic restart.
- Stage A seal: `SEALED_COMPLETE`; Stage B: `COMPLETE`; valid categories: switch, audiojack, regulator, rolled_strip_base.
- The original EXP2A initial manifest remains `INCOMPLETE`; its separate completion and Stage A seals were verified.
- Historical EXP2A runner hashes differ/current source status: Stage A exact match `False`, Stage B exact match `False`. Pinned v22 and R1 dependencies matched. The CPU replay is independently checked against every stored C1-NoBoth event delta, image score, update decision counts, occupancy, and source exclusion.

## Source and input provenance

Stage B's authoritative seal is `COMPLETE`. The first report rendering read a nonexistent `status` field from `stage_b_results.json` and displayed `None`; this was corrected to use `stage_b_seal.json`. The correction changed report assembly only. Stage A and Stage B result files were not modified or replayed.

| Fingerprint | SHA-256 |
|---|---|
| EXP2B module used for the replay | `ad997d87767db9de7d7a5f44cbf362ed9b2c4ebf373011d694b93260dd4c08d7` |
| Final committed EXP2B module (report-only correction after replay) | `13e84ec8e2d84816c9bb0e9224c04f881ce0a2ec22dc4b4fdfcfe6e5cc943deb` |
| Synthetic test source | `a4e5d83b1a40bf06d3224fb9d16f8d0e8fb8018b2b630fa21b2abe9b2c5a4e74` |
| R1 matched-correction pilot | `3208e2c52277ad76351eb86968e362aa315effe9e3c734bfe2b6ff7bab0db448` |
| Windows Job Object supervisor | `e6254a82d8dd24156cea65380b41b04cb1372744da64a085043aa6dd74df5bab` |
| Frozen v22 source | `bfc871d5a9a32b2f5bdb9d7aab5177e702dec2e1110a6632f20bd13c20fcba30` |
| EXP2A identity-projection module at replay | `4b798dcb1472e6dd630f58b3e62f3782dfb302f7579e942fcdb399885a5f33d3` |
| PERF-1 shared metric helper | `aaac635b5d52ba06ada7a9f719f4f8e0722310ccc359563d9d9da8d3994a90db` |
| EXP2A Stage A seal | `b23f552e4c18cccba12d52e3a978f2fe6973b03b58704d44fe0295ad8f5b74c1` |
| EXP2A completion seal | `8a93dc52b1d26f660241ac85fc2132ac83678873d358ec96190f283ebf1384c0` |
| EXP2A report | `b36d300f88f05d3cd82b9a5e7fe1e2642fd19ff854f2bc1ba21c1920f26c1f57` |
| EXP2A sanitized aggregate | `7a580145cae7e7101be5e43debd5ec4820c667d0339cf198990aa590e1b2c4e8` |

The EXP2A seal's recorded Stage A and Stage B runner hashes (`5f36962307f27c6a28cd97d3e9312a2b984da90cdd0cdf0e1bc17d0f33f6d840` and `1433d54e0083af9858d967267c4a60be964509c2d0ddb4706b906da4f8056c21`) do not match the present EXP2A module hash above. Those historical files were not substituted. Replay parity was checked against the sealed per-event outputs and update receipts.

## C1-NoBoth reference parity

| Category | Events | Patch delta max abs error | Score max abs error | Non-bitwise patch values | Update/query checks | State SHA exact matches (diagnostic) | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| switch | 512 | 0.00000022 | 0.00000006 | 13105 | 17 | 0 | PASS |
| audiojack | 512 | 0.00000007 | 0.00000006 | 13090 | 17 | 0 | PASS |
| regulator | 512 | 0.00000017 | 0.00000012 | 12756 | 16 | 0 | PASS |
| rolled_strip_base | 512 | 0.00000016 | 0.00000012 | 13118 | 17 | 0 | PASS |

CPU/GPU state SHA equality is diagnostic only; exact categorical Codebook update counts, occupancy, source exclusion, query timing, and event order are required. The primary D score and patch-delta tolerances were frozen at `atol=1e-5, rtol=1e-4`.

## Image AUROC by category

| Category | A FIFO Top-5 | B FIFO Gaussian | C Codebook Top-5 | D Codebook Gaussian | C0 | Historical C2 0.02 | C2 matched pool |
|---|---:|---:|---:|---:|---:|---:|---:|
| switch | 0.8426 | 0.8525 | 0.8426 | 0.8525 | 0.8070 | 0.8410 | 0.8415 |
| audiojack | undefined | undefined | undefined | undefined | undefined | undefined | undefined |
| regulator | 0.7054 | 0.7180 | 0.7054 | 0.7180 | 0.6872 | 0.7041 | 0.7056 |
| rolled_strip_base | 0.9145 | 0.9069 | 0.9145 | 0.9069 | 0.8983 | 0.9050 | 0.9102 |

AUROC and AP are undefined for audiojack because its authorized 512-event prefix contains normals only. The strict four-category AUROC macro is therefore undefined; the separately named three-evaluable-category descriptive macro is reported in `aggregate_results.json`.

## Fixed C64 threshold FPR/FNR

| Category | Arm | FPR | FNR | BA | TP/FP/FN/TN |
|---|---|---:|---:|---:|---:|
| switch | A_FIFO_matched_Top5_beta | 0.0784 | 0.4596 | 0.7310 | 194/12/165/141 |
| switch | B_FIFO_original_Gaussian | 0.0588 | 0.4791 | 0.7310 | 187/9/172/144 |
| switch | C_Codebook_matched_Top5_beta | 0.0784 | 0.4596 | 0.7310 | 194/12/165/141 |
| switch | D_Codebook_original_Gaussian | 0.0588 | 0.4791 | 0.7310 | 187/9/172/144 |
| switch | C0 | 0.0261 | 0.6017 | 0.6861 | 143/4/216/149 |
| switch | historical_C2_a0.02 | 0.0588 | 0.4958 | 0.7227 | 181/9/178/144 |
| switch | C2_matched_candidate_pool | 0.0588 | 0.4680 | 0.7366 | 191/9/168/144 |
| audiojack | A_FIFO_matched_Top5_beta | 0.0000 | undefined | undefined | 0/0/0/512 |
| audiojack | B_FIFO_original_Gaussian | 0.0039 | undefined | undefined | 0/2/0/510 |
| audiojack | C_Codebook_matched_Top5_beta | 0.0000 | undefined | undefined | 0/0/0/512 |
| audiojack | D_Codebook_original_Gaussian | 0.0039 | undefined | undefined | 0/2/0/510 |
| audiojack | C0 | 0.0059 | undefined | undefined | 0/3/0/509 |
| audiojack | historical_C2_a0.02 | 0.0000 | undefined | undefined | 0/0/0/512 |
| audiojack | C2_matched_candidate_pool | 0.0000 | undefined | undefined | 0/0/0/512 |
| regulator | A_FIFO_matched_Top5_beta | 0.0064 | 0.9497 | 0.5219 | 10/2/189/311 |
| regulator | B_FIFO_original_Gaussian | 0.0096 | 0.9347 | 0.5279 | 13/3/186/310 |
| regulator | C_Codebook_matched_Top5_beta | 0.0064 | 0.9497 | 0.5219 | 10/2/189/311 |
| regulator | D_Codebook_original_Gaussian | 0.0096 | 0.9347 | 0.5279 | 13/3/186/310 |
| regulator | C0 | 0.0096 | 0.9246 | 0.5329 | 15/3/184/310 |
| regulator | historical_C2_a0.02 | 0.0064 | 0.9447 | 0.5244 | 11/2/188/311 |
| regulator | C2_matched_candidate_pool | 0.0064 | 0.9497 | 0.5219 | 10/2/189/311 |
| rolled_strip_base | A_FIFO_matched_Top5_beta | 0.1690 | 0.1655 | 0.8327 | 368/12/73/59 |
| rolled_strip_base | B_FIFO_original_Gaussian | 0.1831 | 0.1701 | 0.8234 | 366/13/75/58 |
| rolled_strip_base | C_Codebook_matched_Top5_beta | 0.1690 | 0.1655 | 0.8327 | 368/12/73/59 |
| rolled_strip_base | D_Codebook_original_Gaussian | 0.1831 | 0.1701 | 0.8234 | 366/13/75/58 |
| rolled_strip_base | C0 | 0.0704 | 0.3129 | 0.8083 | 303/5/138/66 |
| rolled_strip_base | historical_C2_a0.02 | 0.1549 | 0.1950 | 0.8250 | 355/11/86/60 |
| rolled_strip_base | C2_matched_candidate_pool | 0.1549 | 0.1859 | 0.8296 | 359/11/82/60 |

## Factor effects, strength, and mechanism accounting

Factor effects use D−B, C−A, D−C, B−A, and D−C−B+A for AUROC, AP, FPR, FNR, and BA. Paired Source Group bootstrap intervals use 1,000 replicates, seed 20261101, with one common group draw across every arm.

| Category | Memory-policy status | First divergence | FIFO inserts/evictions | Codebook insert/merge/replace/capacity reject | dN rejects | beta-floor rejects | Final FIFO/Codebook occupancy |
|---|---|---:|---:|---:|---:|---:|---:|
| switch | INACTIVE: neither memory reaches capacity and Codebook has no merges/rejections | None | 136/0 | 136/0/0/0 | 0 | 408 | 136/136 |
| audiojack | INACTIVE: neither memory reaches capacity and Codebook has no merges/rejections | None | 99/0 | 99/0/0/0 | 0 | 445 | 99/99 |
| regulator | INACTIVE: neither memory reaches capacity and Codebook has no merges/rejections | None | 116/0 | 116/0/0/0 | 0 | 396 | 116/116 |
| rolled_strip_base | INACTIVE: neither memory reaches capacity and Codebook has no merges/rejections | None | 100/0 | 100/0/0/0 | 0 | 444 | 100/100 |

The common eligibility stream applies the stored native dN keep mask, original float32 patch beta, original beta maximum clip, and `abs(beta) < beta_min - 1e-5` floor. Codebook merges/replacements/capacity rejects are reported separately. A small final bank is not treated as evidence of retention quality.

### Observed factorial effects

The memory factor had zero observed AUROC, AP, FPR, FNR, and balanced-accuracy effect in every evaluable category. FIFO and Codebook states stayed aligned: no FIFO eviction, Codebook merge, replacement, or capacity rejection occurred. These prefixes therefore did not activate a memory-management contrast.

| Category | Gaussian minus Top-5 AUROC | AUROC paired 95% interval | Gaussian minus Top-5 AP | AP paired 95% interval |
|---|---:|---:|---:|---:|
| switch | 0.0099 | [-0.0097, 0.0311] | 0.0047 | [-0.0037, 0.0134] |
| audiojack | undefined | undefined | undefined | undefined |
| regulator | 0.0127 | [-0.0014, 0.0261] | 0.0273 | [0.0089, 0.0479] |
| rolled_strip_base | -0.0076 | [-0.0228, 0.0060] | -0.0015 | [-0.0044, 0.0011] |

The three-category descriptive AUROC macro was 0.8208 for A/C, 0.8258 for B/D, 0.7975 for C0, 0.8167 for historical C2 0.02, and 0.8191 for the matched-pool C2. Gaussian readout had higher descriptive AUROC on switch and regulator; matched Top-5 had higher AUROC on rolled_strip_base. All three paired Source Group AUROC intervals include zero. Regulator's AP interval is fully positive, but it is category-specific and unadjusted for multiple comparisons. This is category-heterogeneous evidence on fixed trajectories, not a general readout recommendation.

| Category | Top-5 patch RMS | Gaussian patch RMS | Retrospective alpha |
|---|---:|---:|---:|
| switch | 0.01442 | 0.01549 | 1.0744 |
| audiojack | 0.02128 | 0.01580 | 0.7425 |
| regulator | 0.01635 | 0.01652 | 1.0103 |
| rolled_strip_base | 0.01602 | 0.01624 | 1.0137 |

Across audiojack's 512 normal events, C0 FPR was 0.0059, A/C FPR was 0, and B/D FPR was 0.0039 at the fixed C64 threshold. Mean absolute score movement from C0 was 0.02048 for A/C and 0.01090 for B/D. The correction readout changes normal scores even where defect-side metrics are undefined.

### Retrospective magnitude-normalized diagnostics

The trajectory-wide alpha for each memory is computed from all 512×32 Gaussian and matched Top-5 patch deltas before evaluator outcomes are read. These scores use future score statistics and are not causal online algorithms, deployable baselines, or headline improvements.

### Ranking changes and category heterogeneity

Helpful and damaging normal–defect ranking changes are paired against C0 and exclude within-Source-Group pairs. Audiojack contributes only normal-side score movement and fixed-threshold FPR; no defect-side inference is made.

| Category | Arm | Helpful pair reversals | Damaging pair reversals | Net helpful |
|---|---|---:|---:|---:|
| switch | A/C Top-5 | 3,724 | 1,783 | 1,941 |
| switch | B/D Gaussian | 4,324 | 1,846 | 2,478 |
| regulator | A/C Top-5 | 2,625 | 1,498 | 1,127 |
| regulator | B/D Gaussian | 3,603 | 1,706 | 1,897 |
| rolled_strip_base | A/C Top-5 | 778 | 271 | 507 |
| rolled_strip_base | B/D Gaussian | 667 | 398 | 269 |

### Limitations and conclusion

This analysis is conditional on the same previously exposed development trajectories. It does not provide independent confirmation, does not include a separate Static+Correction trajectory, and does not establish broad Greedy/Correction complementarity. Gaussian versus matched Top-5 tests readout families (weighting, neighbor selection, normalization), not a pure distance-range effect. Memory F versus S tests the full management package, not sign-aware retention alone.

The supported mechanism conclusion is category-specific: interpret the factorial AUROC/AP effects together with the exact common-candidate activation, beta-floor rejects, merges, replacements, occupancy, and RMS-normalized diagnostic. No single memory or readout policy is declared beneficial unless the observed per-category evidence supports it.

## Runtime, storage, and tests

- Total child runtime ceiling: 1800s; output cap: 134217728 bytes with 65536 bytes reserved; file cap: 20; prelaunch free-space floor: 2147483648 bytes.
- Timings and the final output audit are reported in `aggregate_results.json`: 1,883,039 bytes across 18 files, within the 128 MiB and 20-file limits. Stage A took 12.47 s; Stage B took 17.69 s; the supervisor elapsed time was 33.22 s.
- Peak working set was not captured by the Windows measurement helper and is `null` in `aggregate_results.json`; memory use is unverified.
- Stage B uses `experiments.calibration.r2_perf1_stage_b._metric_record` and the authorized EXP2A identity-matched outcome projection.
- Focused synthetic tests: `tests/calibration/test_r2_exp2b_memory_readout.py`; 17 passed, 0 failed in 4.32s using Python 3.11.14 and pytest 9.2.0; bounded test ceiling: 300s.
- Only the four authorized tracked deliverables are included in this task.
