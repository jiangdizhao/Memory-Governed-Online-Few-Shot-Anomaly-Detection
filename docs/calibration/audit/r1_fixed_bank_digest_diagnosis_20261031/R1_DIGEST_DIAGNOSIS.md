# R1 Fixed Support Bank Digest Diagnosis

**Date:** 2026-10-31

**Branch / HEAD reviewed:** `calibration-stage0-contract-20261029` / `98b9419b7c300b7d67e091ee7ebe2c657e7c65e6`
**Status:** `ROOT_CAUSE_IDENTIFIED`

## 1. Executive finding

The R1 gate compared different digest formats. R1 computes `p0.fixed_bank_profile(bank)`, which hashes only concatenated prototype tensor bytes. The historical Stage 0.5 scoring implementation that survives in the repository obtains `fixed_support_sha256` from `isolate_native_bank_k256`, whose `_fixed_bank_digest` hashes each prototype tensor’s structural path, shape, dtype, and bytes. The Stage 0.5 `native.py` source hash recorded for score generation matches the current implementation. This is a digest-schema mismatch: the expected and rebuilt SHA-256 values are not a valid byte-for-byte tensor comparison when their preimage formats differ.

The exact score-time `stage05_runner.py` source blob is not present in local Git, so the field-to-producer link is strongly supported by the surviving runner and phase provenance rather than independently replayable from the exact runner bytes. More importantly, R1 did not persist its rebuilt digest or tensor metadata. The available evidence identifies the comparator defect; it does **not** establish whether the rebuilt prototype tensor itself was identical to the historical tensor.

The prior R1 report records zero stream events and zero queried labels. Its existing output remains untouched. This diagnosis used source and existing metadata only; it did not load DINO, decode images, read queried labels, or run GPU work.

**Recommendation:** pursue historical recovery through a narrowly scoped support-only verification using the historical digest format and runtime, if the historical DINO source/runtime can be restored. Do not reuse historical C64 scores until both fixed-bank identity and scorer/query-encoder identity are reconciled. If the historical source cannot be restored or the bank digest fails, register a fresh bank and obtain separate authorization before computing any new C64 receipt.

## 2. Confirmed and possible mismatch causes

| Area | Evidence and assessment |
|---|---|
| Digest calculation | **Confirmed R1 defect.** R1 compares the P0 raw-byte digest at [r1_matched_correction_pilot.py:885](../../../../experiments/calibration/r1_matched_correction_pilot.py#L885) with the receipt value captured at line 604. P0’s [fixed_bank_profile](../../../../experiments/p0_protocol_audit.py#L210) hashes only `x.tobytes()` in layer order. The surviving Stage 0.5 runner assigns `fixed_hashes[seed]` from `isolate_native_bank_k256` and writes that value as `fixed_support_sha256` ([stage05_runner.py:537,610](../../../../experiments/calibration/stage05_runner.py#L537)); [native.py:46-88](../../../../experiments/calibration/native.py#L46) shows that function uses a structured digest. The Stage 0.5 score-phase hash for `native.py` is `3a347d65aaf4bfd6b41eff2ea7339d878065bb2e08ed64795f57b0458135dd45`, matching current `native.py`. The exact score-time runner source hash is `1e40d9c5e48c252eef3d96686d2a315c4a87f1f325130a6c2961f7cc41cafaea`; its contents were not found in local Git, so the historical field mapping cannot be re-verified from that exact file. |
| Support IDs and order | **Confirmed match.** Seed 0 uses these four frozen supports in order: `OK/S0025/switch_0025_OK_C4_20231021112518.jpg`, `OK/S0161/switch_0161_OK_C3_20231021135812.jpg`, `OK/S0253/switch_0253_OK_C5_20231021144425.jpg`, `OK/S0281/switch_0281_OK_C2_20231021145359.jpg`. The support group digest is `ed48457bddd13cc15dd9199e08653f8a94d2fc1f652b14c078c92f82bdc5a83f`; the support image-identity digest is `6f831e8245fa4d78569c53b4c95624c420aa50dc5511c061eefdf3643409127e`. R1 consumed the ordered frozen list; the Stage 0.5 runner also builds each seed’s path list from the frozen support list. |
| Absolute path strings | **Root matches; exact strings not retained.** Stage 0.5 and R1 use the same normalized Windows dataset root (`G:\dataset\realiad`), and the recorded root identity matches the current resolved root. The frozen support record stores relative IDs, not the absolute strings passed into the encoder. The exact score-time path-builder source is unavailable locally, so equality of every encoder path string is likely, not proven. This matters because the P0 augmentation seed includes the path string. |
| Augmentation and bank sampling | **Configuration matches; runtime determinism differs.** Both paths use augmentation seed 0, bank seed 0, four augmentations with identity excluded, fixed capacity 25,000, per-reference cap 5,000, concatenated features, and the same locked scorer profile. P0 derives each augmentation seed from `deterministic_subseed(augmentation_seed, path)` and each per-view/layer sampling seed from `deterministic_subseed(bank_seed, path, view_idx, layer_idx)`; see [p0_protocol_audit.py:107-200](../../../../experiments/p0_protocol_audit.py#L107). Stage 0.5 calls v22 `set_seed(0)` and sets math-only CUDA SDPA; R1 did not call that setup. Backend determinism is a confirmed difference and a possible tensor-generation contributor. |
| Scorer profile | **No discrepancy found in locked values.** The R1 configuration and Stage 0.5 `NativeScorerConfig` agree on augmentation/bank seeds, augmentation count, identity-view policy, fixed-bank settings, feature fusion, KNN, Top-32, and image aggregation. The recorded profile hash is `7ff59...` in the historical provenance; current `p0`, `native`, v22, and VMB source hashes match their score-phase hashes. The profile agreement does not close runtime or path-string differences. |
| Checkpoint | **Current bytes match the recorded checkpoint.** Historical provenance records SHA-256 `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`. The cached checkpoint currently hashes to the same value. R1 recorded the cache path but did not persist its own checkpoint checksum, so the exact bytes loaded during that attempt are not independently bound by its output manifest. |
| DINO source and runtime | **Unresolved.** Historical provenance records DINOv2 ViT-L/14 with source-tree hash `1de67887ca330bdfaad8ccd45128d2b9313d63aa707bbe8e4f38310e52005b24`, Torch `2.0.1+cu117`, RTX 3090 Ti, and math-only SDPA. The current cached source tree hashes to `b7c0037f90671c4932393bd56de8c8a205b29a6eb22dea563b6c8a88c471da15` under the R1 helper; the historical record does not preserve its tree-hash procedure, so this value difference is a warning requiring reconciliation, not proof of a source mismatch. R1 did not persist its source-tree hash, loaded state hash, Torch/CUDA versions, or backend flags. Its report notes xFormers-unavailable warnings. |
| Prototype tensor properties | **Source-derived expected layout; actual bytes not recoverable from the failed attempt.** The DINOv2 ViT-L/14 path uses four 1,024-channel hooked layers; at the 392-pixel crop this gives a 28×28 patch grid. v22 concatenates and L2-normalizes the four layers into one [784,4096] feature layer ([vmb_visionad_new.py:337](../../../../vmb_visionad_new.py#L337), [v22.py:5919](../../../../test_phase3_AIF_simplified_v22.py#L5919)). Four non-identity augmentations per support yield 16 support references. The fixed-bank builder caps each reference at 512 patches, samples 8,192 total patches, and the codebook is float16 with capacity 25,000; source logic therefore implies one expected stored tensor of [8192,4096], float16, with insertion order determined by the seeded per-reference and final-pool sampling ([p0_protocol_audit.py:168-200](../../../../experiments/p0_protocol_audit.py#L168), [v22.py:3789](../../../../test_phase3_AIF_simplified_v22.py#L3789), [v22.py:1971](../../../../test_phase3_AIF_simplified_v22.py#L1971)). These are code-derived expectations, not measured receipt fields. The exact bytes and order produced during the historical run are not retained. R1 computed its P0 profile in memory before failing, but did not persist it; its actual digest is null in the output manifest. |

### Digest semantics

The P0 digest preimage is the concatenation of each stored layer’s contiguous CPU prototype bytes, in layer-list order. Its separate profile reports layer count, prototype count, feature dimension, capacity, dtype, and byte count, but those fields do not enter the digest.

The native digest recursively walks `normal_fixed` in sorted dictionary-key and list-index order. For each prototype codebook it adds the structural path, ASCII shape/dtype metadata, and contiguous CPU tensor bytes. It also avoids revisiting shared objects. `isolate_native_bank_k256` returns this digest before and after cloning. These are different serialization contracts. A matching tensor must be checked with the same contract used to create the expected receipt.

The historical expected digest is `8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558`. It is recorded in the Stage 0.5 C64 score tape manifest, whose receipt SHA-256 is `91fefe78fe54a2d933770b120df08894b4c8e6f69297d0cd2d2ac7b950e0aa9e`; the receipt records native parity at tolerance `1e-8`. R1’s report explicitly says the rebuilt digest was not retained. Thus no existing evidence can determine whether raw prototype bytes differed.

## 3. Source and artifact evidence

| Evidence | Recorded identity or location | What it establishes |
|---|---|---|
| Prior R1 report | [R1_ENGINEERING_REPORT.md](../r1_switch_s0_512_engineering_20261031/R1_ENGINEERING_REPORT.md) | `BLOCKED_INPUT`; support digest gate stopped execution; 0 stream events, 0 labels; actual rebuilt digest absent. |
| R1 output | `outputs/calibration/r1_switch_s0_512_engineering_20261031/run_manifest.json` and `attempt_history.jsonl` | Failed output exists and must remain unchanged. It is not eligible for the narrow empty-bootstrap resume path. |
| Historical score tape | `outputs/calibration/stage05_exploratory_20261029/score_tape_manifest.json` | Expected fixed support digest and C64 receipt binding. |
| Historical frozen inputs | `outputs/calibration/stage05_exploratory_20261029/frozen_inputs.json` | Frozen support identities, support ordering, group and image-identity digests. |
| Historical provenance | `docs/calibration/audit/stage05_exploratory_v1/source_and_protocol_provenance.json` and Stage 0.5 `run_manifest.json` | Checkpoint, model/runtime record, source hashes, profile, and support/C64 provenance. |
| Source hash limitation | Historical score-time hashes: `stage05_runner.py` `1e40d9c5...`; `stage05.py` `af1ef435...`. Current file hashes are `1d110e1c...` and `4b8aae25...`; exact historical source blobs were not found in local Git history. | Limits exact historical reconstruction, even though core P0/native/v22/VMB source hashes match their score-phase hashes. |

## 4. Option A — Historical-exact reconstruction

Reproducing the original fixed-bank digest requires the exact frozen support IDs and order; byte-identical support images; the exact absolute path strings used for path-derived augmentation seeds; the same P0, VMB, v22, scorer configuration, and deterministic seed behavior; the same DINO checkpoint and source tree; matching Torch/CUDA/PIL runtime and backend settings; the same generated augmentation views and per-layer sample order; the same fixed-codebook update semantics; and the historical structured native digest function.

Currently verified: frozen relative support identities/order and their identity digests; current checkpoint bytes matching the recorded checkpoint SHA; matching core source hashes for P0, native, v22, and VMB; and matching locked scorer values. Missing or unresolved: actual R1 tensor/digest, exact absolute support path strings in the historical encoder call, score-time Stage 0.5 runner/core source blobs, historical hub source tree contents (the provenance has a hash but no Git commit), and R1-time runtime/backend/model-state metadata. Those source/runtime records are not reconstructable from the local repository alone. Historical-exact recovery remains conditional on restoring the archived source/runtime or demonstrating exact compatibility.

## 5. Option B — Fresh matched-trajectory protocol

A fresh protocol can register one immutable bank under a new experiment identity. It may use the same four support images and order, but must record the exact source commit, path convention, checkpoint/source/runtime, augmentation and sampling seeds, bank tensor metadata, both named digest formats, and the new native digest before any score generation. The failed R1 output stays untouched; use a new output ID bound to the repair source commit.

With one newly registered bank, a new run can keep C0/C1/C2 on a common score trajectory, common Greedy writer, frozen query tape, and verified feedback, then measure controller replay differences against that shared trajectory. That is a fresh matched-trajectory experiment. It cannot be described as a reproduction of historical Stage 0.5 or as evidence that historical R1 scores were reproduced.

The existing label-blind Query Tape is potentially reusable because its query positions and schedule are scorer-independent. Before reuse, verify the same class, frozen stream identity, prefix positions, query-group digests, and tape binding. Reusing the tape does not authorize label access outside its existing verified-feedback protocol.

## 6. C64 compatibility

The protocol is four deployed supports plus **an additional 64 C64 calibration normals**; it is not strict four-shot-total. Historical C64 scores are compatible only if the fixed bank **and** query-scoring identity match the historical receipt. Correcting the digest comparison alone does not establish either condition. If the support-only check reproduces the expected native digest under the historical source/runtime, and query encoder/scorer identity is also reconciled, the existing receipt may be considered for reuse under its original profile.

If the new bank differs, historical C64 scores, the old initial threshold, and the old sigma cannot be carried over. Compute a new C64 receipt under the new bank/profile; derive threshold and sigma from that receipt using the locked rule. Since the initial score geometry changes, Greedy write decisions and the resulting memory trajectory may change. Any new C64 scoring requires separate researcher authorization and an explicit new provenance record. No C64 rescore is authorized by this diagnosis.

## 7. Proposed minimal code changes (not implemented)

1. Preserve the historical digest gate. In a versioned repair, compare the receipt’s expected digest to the same structured native digest implementation used by Stage 0.5. Keep the P0 raw-byte digest as a separately labeled diagnostic, not as a substitute comparator.
2. Before raising on a mismatch, persist the expected digest algorithm/value, both actual digest values, per-layer shape/dtype/count/capacity, support path strings and seed identities, checkpoint/source hashes, runtime/backend flags, and current source commit. This evidence must be written atomically to the new run’s own output directory.
3. Reapply and record the Stage 0.5 deterministic setup (`set_seed(0)`, deterministic backend settings, math-only SDPA) when attempting historical recovery. Do not change frozen support, augmentation, v22, or scorer semantics to force a match.
4. Replace the stale hard-coded `PARENT_COMMIT` expectation in a new version with an explicit expected source commit (or immutable source tag resolved to a commit). Require a clean worktree, record the commit/tree identity, and reject implicit source drift.
5. Require a distinct, non-overwriting output directory. Do not resume or edit `r1_switch_s0_512_engineering_20261031`; it contains the failed attempt and is not the permitted empty-bootstrap case.

## 8. Support-only GPU diagnostic

Yes. Because R1 did not retain its rebuilt digest or tensor profile, a support-only bank rebuild is necessary to establish whether the corrected historical digest matches. It should encode only the four frozen support images, build the bank once, compute the structured native digest and diagnostic P0 profile, and stop before C64 scoring, stream scoring, labels, or correction. Save its evidence in a new immutable output location.

This diagnostic can answer the bank-tensor question only under a reconciled historical source/runtime. A current-source diagnostic whose DINO source or numerical settings differ can characterize a fresh bank, but cannot by itself establish historical C64 compatibility.

## 9. Resource and authorization requirements

The current D0 authorization covers read-only documentation only. Any support-only diagnostic requires separate explicit GPU authorization and a bounded resource budget. Limit its inputs to the four frozen supports, retain no image copies or feature caches, and record time, peak memory, source commit, and hashes. It requires no C64 images, stream images, queried labels, masks, or correction execution.

A fresh full R1 execution requires its own authorization and distinct source-bound output. Any newly computed C64 receipt separately requires researcher authorization and a new provenance record. No C64, R1, or R2 execution is performed here.

## 10. Next action and recommendation

**Recommendation: historical recovery, beginning with the support-only verification described above.** First establish whether the historical DINO source tree and runtime can be restored. If so, rebuild only the four-support bank with historical deterministic settings and compare using the structured native digest. Reuse the old C64 receipt only after both bank and scorer/query-encoder identity pass review. If historical source/runtime cannot be restored or the digest differs, stop historical recovery and propose a separately authorized fresh-bank protocol with new C64 provenance.

**Disposition:** `ROOT_CAUSE_IDENTIFIED` for the R1 comparator’s digest-schema error; the actual prototype-tensor equivalence remains unverified. This diagnosis does not authorize a GPU run, C64 rescoring, R1 rerun, or R2.
