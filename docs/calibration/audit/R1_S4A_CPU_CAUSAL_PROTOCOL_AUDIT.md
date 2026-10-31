# R1-S4A CPU-only Causal Protocol Audit

**Status: `CPU_AUDIT_PASS`**
**R1-S4 GPU pilot: not run and not ready for researcher consideration pending resource-containment review.**
**This report is an engineering audit, not scientific performance evidence.**

## Scope and safety

This audit repaired and exercised synthetic CPU paths in `experiments/calibration/r1_matched_correction_pilot.py`. It did not call `run_pilot()`, initialize DINOv2, decode images, access Real-IAD labels or masks, calculate performance metrics, or modify historical outputs. The native v22 module was imported by small CPU Codebook tests only; its source was not changed.

Branch was `calibration-stage0-contract-20261029`. The starting local and remote branch HEADs both matched parent `5e9ac93b718f85d076ed5d2f5cb42166a1f12c4e`. The implementation/tests commit is `88db3b0d9af4bccf7489dc377b64e761c824575f`; the report is committed separately afterward. Nothing was pushed.

At the initial safety check, staged paths: none; tracked modifications: none; untracked paths: 82, all preserved. The pre-existing untracked paths are listed in the appendix. No unrelated tracked change or unexpected commit was present. The task added only the isolated pilot changes, focused tests, and this report.

The latest research handoff's Sections 12–14 were reviewed. Its cautions about unresolved method math, single-seed limits, and selective-feedback guarantees are consistent with treating this as an engineering-only audit.

### Source fingerprints

| File | SHA-256 before this task | SHA-256 after implementation |
|---|---|---|
| `experiments/calibration/r1_matched_correction_pilot.py` | `8a44b52ca51fbd576fbcb51441007e3b24316ff69f721056876c185eb32d2f27` | `ddfbab56c75c008c7b416b4c655418150add9ea8645fa85bb23a7223b216ff3c` |
| `tests/calibration/test_r1_matched_correction_pilot.py` | `29f5ae7eb1af61565666b4424ead13a7a2565c41da1c2bd8c23eaa3195f89f4c` | `dbe1e18f8f59cc32624e95336b4e949f2fe1da8e08f4dfab53b1c0fad392f9cb` |
| Protected `test_phase3_AIF_simplified_v22.py` | `bfc871d5a9a32b2f5bdb9d7aab5177e702dec2e1110a6632f20bd13c20fcba30` | unchanged |
| Protected `vmb_visionad_new.py` | `bd6545bf5ff63acce01792e5d42aa964c88e73fb59670febeb55f2d1139a409a` | unchanged |
| `experiments/calibration/stage1a_p12_label_provider.py` | `4e51255ae894c2d6ffcc7142e8fd59d9714df94b18ed26ac996528adb163ceaf` | unchanged |

The new test file SHA-256 at implementation commit: `857d813a3e6becdff1c6c9bd2ab0266a942bd19457d5b14a3f568b999b0b5e07`.

## C2 timing defect and repair

The runner appended up to 32 signed C2 anchors after a queried label was disclosed. Those anchors correctly had `position == t`. A later assertion used the pre-score condition `position >= t` on this already-updated bank, so every legal current-event anchor looked like future leakage. A second call to the pre-score selector after that assertion would also have rejected current-event anchors. Simply changing the comparison would have left the phase confusion in place.

The runner now enforces separate invariants through helpers used in the production score/update path:

- **Pre-score at `t`:** `signed_knn_delta` validates every C2 anchor's provenance and feature digest, rejects any `position >= t`, and uses only anchors from a source group different from the current event. The C1 score adapter likewise requires `last_step < t`, validates source-group history, and filters the current source group.
- **Post-update at `t`:** C1 and C2 state validators allow `last_step/position == t` but reject any value greater than `t`. A current-event anchor first becomes score-eligible at a later event, subject to source-group exclusion.

The false post-update call to the pre-score selector is removed. The runner validates post-update state directly after correction updates. C2's fixture remains 5 neighbors, amplitude 0.20, 256 anchors, FIFO; the capacity is now named by a constant and remains unchanged.

## Event order and label access

The enforced order is:

`Observe → C0/C1/C2 score → seal prediction → decide query → optional label reveal → C1/C2 update → common C0 Greedy write → finalize → next event`

`CausalEventOrder` is created before event scoring and does not know the query decision initially. `_seal_prediction_and_reveal` first calls the real `EvidenceTapeWriter.seal_event`; only after it returns does the gate enter `scores_sealed`, decide from the frozen query positions, and optionally call the label provider. A seal failure leaves the provider untouched. Query decision and query reference are recorded in the finalized transition, not the public score-shard metadata.

The production runner wraps the existing `QueriedPositionLabelReader` in `OrderedQueriedLabelReader`, which requires the exact next frozen query position. The original provider remains unchanged. A synthetic JSON test verifies that trying position 2 before position 0 does not call or advance the underlying provider, then reads only the two allowlisted synthetic positions in order.

## C1 source-group provenance

The native `SignedCorrectionCodebook.update` reports aggregate inserted/merged/replaced counts and updates `last_step`, but does not return the affected slot IDs. Merge updates a slot's step; insertion and replacement write a slot and its step. The pilot therefore attributes the current source group to every active slot whose native `last_step == t`.

For insertion, this records the inserted source. For a merge, unioning the current group with existing history records both sources. For replacement, native aggregate statistics cannot identify which prior slot contents were replaced; the old source history is deliberately retained. That history is **conservatively overinclusive** after replacement and can exclude a replacement anchor for a source that contributed only to the old slot. It cannot make ambiguous provenance eligible. Missing, malformed, negative-step, or inactive-slot provenance fails closed.

CPU tests use the native Codebook to exercise insertion, same-sign merges from different sources, and replacement, then pass its state through the production provenance helper and C1 score adapter. Same-source revisits are excluded. A current-event C1 slot is accepted by post-update validation and rejected by pre-score validation at the same position.

## Common C0 writer

The actual runner now uses `common_c0_greedy_writer_action`. Its arguments are C0 score, frozen threshold, sigma, C0 prediction, shared Top-32 IDs/features, and the query/label authorization. It does not accept C1/C2 outputs. The eligibility callback receives the original common C0 values; the update callback is invoked only for queried verified normals and uses the common feature layers and Top-32 IDs.

Synthetic runs with radically different injected C1/C2 outputs but identical C0 inputs, queried-normal feedback, and initial memory produce identical eligibility inputs, writer action, candidate IDs, and memory state. Queried defects and unqueried events do not call either eligibility or update. C2 remains an engineering fixture and is **NOT YET SCIENTIFICALLY BUDGET/AMPLITUDE MATCHED for R2**.

## Evidence tape and resource-control audit

Focused tests exercise seal-before-finalize, failed seal with no label read, duplicate/gapped positions, duplicate finalization, checksum tampering, forbidden public label/mask metadata, retained incomplete state after an injected post-seal exception, and sealed-versus-finalized counts. Existing tests also verify that a pre-existing output directory is refused by default; only the explicit, narrow empty `ModuleNotFoundError` bootstrap retry is accepted. Partial tapes remain `INCOMPLETE`; no general event-state resume is allowed.

Resource limits are cooperative rather than hard process controls. The 120-minute limit is checked at event boundaries, so it cannot interrupt a hung/long encode or scoring call. The 2 GiB output limit is checked as the writer updates its manifest and again after an event; writes can cross the ceiling before the check. Failure-manifest recording is best effort and its exception is suppressed. Subprocess calls in this runner are short Git identity/status probes; there is no external supervisor that can kill a stuck GPU process or enforce a hard storage quota. These risks are unresolved and must be addressed before a GPU pilot is considered. CPU tests do not establish GPU timing, memory, or storage behavior.

## Test results

CUDA visibility was disabled. The final command compiled the pilot and ran the focused protocol tests, existing R1 synthetic tests, and the synthetic queried-label provider tests:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
@'
import py_compile, sys, runpy
py_compile.compile(r'experiments/calibration/r1_matched_correction_pilot.py', doraise=True)
sys.path.append(r'D:\anaconda3\Lib\site-packages')
sys.argv = ['pytest', 'tests/test_r1_s4_cpu_protocol.py',
            'tests/calibration/test_r1_matched_correction_pilot.py',
            'tests/calibration/test_stage1a_p12_label_provider.py',
            '-q', '-p', 'no:cacheprovider']
runpy.run_module('pytest', run_name='__main__')
'@ | & 'D:\anaconda3\envs\torch\python.exe' -
```

| Check | Result |
|---|---:|
| Python compilation of isolated R1 pilot | PASS |
| New `tests/test_r1_s4_cpu_protocol.py` | PASS, included in combined run |
| Existing `tests/calibration/test_r1_matched_correction_pilot.py` | PASS, included in combined run |
| Existing synthetic `tests/calibration/test_stage1a_p12_label_provider.py` | PASS, included in combined run |
| Combined focused CPU run | **45 passed, 0 failed** |
| Warnings | 4 upstream torchvision/Pillow interpolation deprecations; no test failures |

Two intermediate test-fixture runs failed before the final combined run: first, the failed-seal test's gate position mismatched the requested tape position, so the gate correctly rejected the call before the intended tape failure; second, an early-finalize rejection assertion was placed after a successful seal, where finalization was valid. Both test setups were corrected. Neither failure indicated a production-code defect. The final combined run above passed. No GPU command, dataset path, image decoder, Real-IAD label, mask, or performance metric was used.

## Recommendation and remaining limits

**`CPU_AUDIT_PASS` for the causal protocol repair and synthetic CPU checks.** This is not `R1-S4 ENGINEERING_PASS` and says nothing about empirical performance. The C1 replacement provenance remains conservative/overinclusive. GPU execution paths, real label ordering against the frozen stream, actual model scoring, device behavior, and resource ceilings remain unverified. The proposed S4 GPU pilot is **not ready for researcher consideration** until hard wall-time supervision and a hard storage bound are specified and reviewed. No GPU pilot or R2 comparison was started.

Implementation commit: `88db3b0d9af4bccf7489dc377b64e761c824575f` (parent `5e9ac93b718f85d076ed5d2f5cb42166a1f12c4e`). This report is committed separately. The remote branch was not updated.

## Pre-existing untracked paths preserved

At the initial safety check there were no staged or tracked modifications. These 82 untracked paths existed before the task and were not opened, modified, staged, or deleted:

```text
docs/calibration/audit/queried_anchor_identifiability_v1_failed_attempt_repair1/
docs/calibration/audit/queried_anchor_identifiability_v1_finalization_attempt1/
docs/calibration/audit/queried_anchor_identifiability_v1_measurement_attempt_repair2/
docs/calibration/audit/stage075a_g2_integrity_cert_v1_20261030/
docs/calibration/audit/stage075b_synthetic_v2/
docs/calibration/audit/stage075b_synthetic_v3/
docs/calibration/audit/stage075b_synthetic_v4/
docs/calibration/audit/stage075b_synthetic_v5/
docs/calibration/audit/stage075b_synthetic_v6/
docs/calibration/audit/stage075b_synthetic_v7/
docs/calibration/audit/stage0_fifthpass_final_v1/
docs/calibration/audit/stage0_fifthpass_v1/
docs/calibration/audit/stage0_fifthpass_v2/
docs/calibration/audit/stage0_research_closure_v1/
docs/calibration/audit/stage1a_final_readiness_v1_20261030/
docs/calibration/audit/stage1a_final_readiness_v2_20261030/
docs/calibration/audit/stage1a_final_readiness_v3_20261030/
docs/calibration/audit/stage1a_final_readiness_v4_20261030/
docs/calibration/audit/stage1a_final_readiness_v5_20261030/
docs/calibration/audit/stage1a_p12_low_memory_recovery_v1_20261030/
docs/calibration/audit/stage1a_p12_low_memory_recovery_v2_20261030/
docs/calibration/audit/stage1a_p12_native_replay_v1_standing_authorization_01_20261030/
docs/calibration/audit/stage1a_p12_native_replay_v1_standing_authorization_02_20261030/
docs/calibration/audit/stage1a_p12_native_replay_v1_standing_authorization_04_20261030/
outputs/calibration/b3_state_lag_formal_v1_20261030/RUN_STATE.json
outputs/calibration/b3_state_lag_formal_v1_20261030/mechanism_diagnostics.json
outputs/calibration/b3_state_lag_formal_v1_20261030/per_method_metrics.json
outputs/calibration/b3_state_lag_formal_v1_20261030/predictions/
outputs/calibration/b3_state_lag_formal_v1_20261030/preflight.json
outputs/calibration/b3_state_lag_formal_v1_20261030/run_identity.json
outputs/calibration/queried_anchor_identifiability_v1_20261030/
outputs/calibration/queried_anchor_identifiability_v1_postmeasurement_gate_fix_tests_20261030/
outputs/calibration/queried_anchor_identifiability_v1_repair1_20261030/
outputs/calibration/queried_anchor_identifiability_v1_repair2_20261030/
outputs/calibration/r1_digest_contract_v1_support_only_supportcheck_20261031_a1/
outputs/calibration/r1_s1_launcher_supportcheck_20261031_a1/
outputs/calibration/r1_s2_c64_parity_switch_s0_20261031_c64parity_20261031_a1/
outputs/calibration/r1_s3_native_map_switch_s0_16_20261031_native-map-20261031-a1/
outputs/calibration/r1_s3_native_map_switch_s0_16_20261031_native-map-retry-20261031-b1/
outputs/calibration/r1_switch_s0_512_engineering_20261031/
outputs/calibration/stage0/stage0_31f7072d4a206d12/
outputs/calibration/stage0/stage0_341a074daa935971/
outputs/calibration/stage0/stage0_4832e2270c72c525/
outputs/calibration/stage0/stage0_bee149a5010fa818/
outputs/calibration/stage0/stage0_ceaf3aa9e3aaed6a/
outputs/calibration/stage0/stage0_e457ca9e96fed287/
outputs/calibration/stage0/stage0_fifthpass_final_20261029/
outputs/calibration/stage0/stage0_fifthpass_release_20261029/
outputs/calibration/stage0/stage0_fifthpass_run1_20261029/
outputs/calibration/stage0/stage0_fifthpass_run2_20261029/
outputs/calibration/stage0/stage0_final_protocol_correction_20261029/
outputs/calibration/stage0/stage0_fourthpass_run1_20261029/
outputs/calibration/stage0/stage0_sixthpass_final_20261029/
outputs/calibration/stage0/stage0_thirdpass_20261029/
outputs/calibration/stage0/stage0_thirdpass_20261029_final/
outputs/calibration/stage05_exploratory_20261029/
outputs/calibration/stage075a_b3_headroom_v1_20261030/
outputs/calibration/stage075a_b3_headroom_v1_20261030_verified2/
outputs/calibration/stage075a_greedy_ranking/
outputs/calibration/stage075a_greedy_ranking_repaired2_20261029/
outputs/calibration/stage075a_greedy_ranking_repaired3_20261029/
outputs/calibration/stage075a_greedy_ranking_repaired_20261029/
outputs/calibration/stage075b_synthetic_20261029/
outputs/calibration/stage075b_synthetic_v3_20261029/
outputs/calibration/stage075b_synthetic_v3_blocked_20261029/
outputs/calibration/stage075b_synthetic_v3_blocked_final_20261029/
outputs/calibration/stage075b_synthetic_v3_final_20261029/
outputs/calibration/stage1a_formal_controller_comparison_v1_20261030/
outputs/calibration/stage1a_formal_controller_comparison_v2_20261030/
outputs/calibration/stage1a_formal_controller_comparison_v3_20261030/
outputs/calibration/stage1a_formal_controller_comparison_v4_20261030/RUN_STATE.json
outputs/calibration/stage1a_formal_controller_comparison_v4_20261030/checksums.json
outputs/calibration/stage1a_formal_controller_comparison_v4_20261030/prediction_phase_complete.json
outputs/calibration/stage1a_formal_controller_comparison_v4_20261030/predictions/
outputs/calibration/stage1a_formal_controller_comparison_v4_20261030/preflight.json
outputs/calibration/stage1a_p12_low_memory_recovery_v1_20261030/
outputs/calibration/stage1a_p12_low_memory_recovery_v2_20261030/
outputs/calibration/stage1a_p12_low_memory_recovery_v3_20261030/
outputs/calibration/stage1a_p12_native_replay_v1_standing_authorization_01_20261030/
outputs/calibration/stage1a_p12_native_replay_v1_standing_authorization_02_20261030/
outputs/calibration/stage1a_p12_native_replay_v1_standing_authorization_04_20261030/
outputs/calibration/stage1a_p12_native_replay_v1_standing_authorization_05_20261030/
```
