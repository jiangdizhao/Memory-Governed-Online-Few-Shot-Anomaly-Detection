# Stage 0 sixth-pass source provenance

**Status:** WIP / Stage 0 blocked under existing G6/G7 acceptance. This record
describes software and bounded synthetic/native diagnostics; it does not
establish empirical scientific validity.

## Checkout and execution identity

- Repository: `jiangdizhao/Memory-Governed-Online-Few-Shot-Anomaly-Detection`
- Authorized branch: `calibration-stage0-contract-20261029`
- Starting/local source commit during validation:
  `ced068258709841df11c59f6a4a1accafa24246d`
- Python: the existing `torch` Conda interpreter, Python 3.11.14
- PyTorch: 2.0.1+cu117; CUDA available
- Conda environment and installed dependencies were not changed.
- Stage 0 Runner output (local, non-committed):
  `outputs/calibration/stage0/stage0_sixthpass_final_20261029/`
- Runner checksum-manifest SHA-256:
  `81c838bc515cb50bc81a7d6b44526dd0cfb11390fd25793fabb7d099553f2c0c`
- Stage 0 source/config fingerprint SHA-256:
  `d6c9fe52d8948806af10aefe98ec696c8cd351c76a68a6d9231f995af6a60587`
- Full source hash map and protected-baseline result are in the compact audit
  package's `protocol_fingerprints.json` and
  `protected_source_verification.json`.

## Protected historical sources

`verify_protected_sources()` reports PASS for all 13 dependencies recorded in
`docs/calibration/PROTECTED_SOURCE_BASELINE.json`; there are no unresolved or
unrecorded dependency paths. In particular:

- Reviewed protected baseline commit:
  `9d2d20df0bd94629eaea0f8cbd6a6f84ebc28a44`
- `experiments/risk_safe/shadow_memory.py` baseline and worktree SHA-256:
  `70c2ba3d29b225f241d9ad147f5be168ee2e0613f452e1428cc585bbce3339b5`
- Byte equality to baseline: true
- Failures: none

No historical v22, VisionAD, P0/P1, risk-safe, V0.2, or archived experiment
source was edited. New implementation code and tests are confined to
`experiments/calibration/` and `tests/calibration/`; research candidates and
reports are under `docs/calibration/`.

## Native source and artifact provenance

The bounded native diagnostic used the cached DINOv2/VisionAD/v22 source path
and did not download model code or weights. The local cached hub source did
not expose a commit identifier (`dinov2_hub_source_commit: null`); the exact
hub tree digest was recorded as
`1de67887ca330bdfaad8ccd45128d2b9313d63aa707bbe8e4f38310e52005b24`.
Checkpoint SHA-256:
`36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`.

The native replay scoped itself to 34 disjoint audiojack training-normal
groups, exactly four deployed supports, 64 additional C64 normals, and one
Q-A-TS-v1 query. It encoded 102 feature images and made 326 native score calls
in 22.526 seconds. Score values were not exported. The query remained
label-blind; the one verified-normal disclosure caused the sole memory update.
All 64 C64 anchors were refreshed under the current memory version, with eight
queried-source rows excluded by LSO. Resume matched uninterrupted scores,
predictions, traces, controller states, and memory hashes.

Recorded opaque artifact digests:

- Four-support group set: `27152d5ae58919a13f8b0c93072f3b86cc14db4f69dc248c07ca32449d398eb4`
- Additional C64 group set: `9b2ca4ed0dc4bae36a751a8a6367f716e6863b774981d9b8ea5966660a60ad6c`
- Native replay query tape: `1c382f528aa6509df6b3d4b6cc452f71714c40978d671e04448612ac55dfc82f`
- Native replay score vector: `da0a4fcefc8c4e88b6598c0d43483c30fec0a6d06bd886dd1b02a6092d456d6c`
- Isolated static support: `c5f3dd6494f105a9197e7fc9b275d56d8a084b0eb4aa4052a157a9a6fed5b3d2`
- Post-write source map: `c3347956877efc82c15c0d4a2ced14a43abeba63b1e264fee2b6802c295bf46e`

The native score vector and AUROC are diagnostic engineering evidence only;
the replay's alternating binary labels are synthetic, not Real-IAD outcomes.
The separate Greedy operator matrix uses synthetic FP16 features and exact
`torch.equal` comparisons; it is not an image scorer.

## Method source and fidelity records

- **B2 historical sidecar:** imported from
  `test_phase3_AIF_simplified_v22.GuardedDualAnchorThresholdController`; the
  calibration wrapper disables its unqueried-label fallback and records its
  C64 initialization adaptation. Full native-loop parity is not claimed.
- **B4 adapted Dual-ACI:** source citation is Gibbs and Candès, NeurIPS 2021,
  https://papers.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html.
  The sparse-feedback/current-memory adaptation does not inherit source
  guarantees.
- **B5a SPOT:** Siffer et al., KDD 2017,
  https://doi.org/10.1145/3097983.3098144. The method card pins the libspot
  source tag and blob, but no compiled C numerical trace parity was run.
- **B5b dSPOT:** independently follows the KDD paper's Algorithm 3; the pinned
  libspot C source does not provide a dSPOT implementation. Its 50-event local
  window and 1,000-residual startup remain source-native unavailable.
- **B6 Sun-CS:** Sun et al., ICML/PMLR 235 (2024),
  https://proceedings.mlr.press/v235/sun24h.html and
  https://raw.githubusercontent.com/mlresearch/v235/main/assets/sun24h/sun24h.pdf.
  The paper and appendices were reviewed. No verified author implementation
  was located. Equation/trace reference tests pass with declared endpoint,
  empty-prefix, and reset-boundary conventions; author-code parity and
  physical-defect/dynamic-memory theorem transfer are not claimed.
- **B7 SCDAC-G:** source proposal SHA-256
  `5fe07256384351c1170d67183eae5a80e5aa289a03586ba767671f062f4fb977`.
  Version 0.1 parameters remain explicitly provisional; its Clopper–Pearson
  endpoints are descriptive only in this selected-feedback, dynamic-memory
  experiment.

Complete equations, permissions, defaults, adaptations, unresolved items, and
reference-test identifiers are in `controller_method_cards.json` in the audit
package. No method is represented as theoretically interchangeable with
another merely because each can emit a threshold or decision.

## Validation commands and artifact linking

Executed with the named interpreter:

```powershell
python -m compileall -q experiments/calibration tests/calibration
python -m experiments.calibration.runner --stage 0 --all --output-dir outputs/calibration/stage0/stage0_sixthpass_final_20261029
python -m experiments.calibration.runner --stage 0 --verify-only --output-dir outputs/calibration/stage0/stage0_sixthpass_final_20261029
python -m unittest tests.calibration.test_sun_cs_reference tests.calibration.test_registry_and_greedy tests.calibration.test_audit_export tests.calibration.test_runner_integrity tests.calibration.test_engine_and_ledger -v
git diff --check
```

The Runner's 85-test run and bounded native payloads are preserved in the
checksummed local output. After correcting report selectors and G9 key casing,
the source fingerprint and artifact checksums were recomputed; Runner
`--verify-only` then returned `ARTIFACT_INTEGRITY_VERIFIED`,
`source_and_config_match: true`, and protected-source PASS. The 3,815-event
Sun-CS synthetic complexity diagnostic was separately completed and inserted
into those linked outputs. The compact audit export records the checksum of
the Runner manifest and per-file SHA-256 links without committing the local
Runner output directory or any dataset, image, mask, feature cache, or weight.

## Stage boundary

The Stage 1 protocol and G6/G7/G9 semantic amendment are candidates requiring
independent researcher approval. Existing G6 and G7 remain blocked in the
operative gate matrix; the aggregate status is `BLOCKED_WIP`. Stage 1, Stage 2,
Real-IAD performance tuning, and confirmation-set inspection were not run.
