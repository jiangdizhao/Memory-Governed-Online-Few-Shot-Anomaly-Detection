# R1 Engineering Report — switch / support seed 0

**Final status: `BLOCKED_INPUT`**

## 1. Executive conclusion

The CPU implementation gates passed, and the frozen Query Tape, C64 score receipt, image-path presence, and support/query group checks passed metadata preflight. The GPU run loaded the cached DINOv2 model and rebuilt the four-support bank, then stopped because the rebuilt fixed-support digest did not equal the digest bound to the frozen Stage 0.5 C64 receipt.

No stream event was encoded or scored. No queried label was decoded. Native C0 parity, dense-map/Top-32 parity, Greedy transitions, and correction activation were therefore **not reached**. This is not an engineering pass and provides no evidence about anomaly-detection performance.

## 2. Git and source identity

- Branch: `calibration-stage0-contract-20261029`
- Parent HEAD: `acc241299373d2f93d8c85ada8d2dbba2caf4f9f`
- Protected historical scorer, Greedy writer, Query Controller, and frozen experiment files were not modified.
- The new R1 runner imports native functions from the following source files. Their SHA-256 values at report preparation are:

| Source | SHA-256 |
|---|---|
| `test_phase3_AIF_simplified_v22.py` | `bfc871d5a9a32b2f5bdb9d7aab5177e702dec2e1110a6632f20bd13c20fcba30` |
| `vmb_visionad_new.py` | `bd6545bf5ff63acce01792e5d42aa964c88e73fb59670febeb55f2d1139a409a` |
| `experiments/p0_protocol_audit.py` | `b83918ec5aceb3ef63cca942fc9fb0e241559d9672d20bed422b15fcd48135e7` |
| `experiments/v02_memory.py` | `3c4e247e3a41dc9acc8147a35aa892cf3f3571aca85beb4169809bd70ad48ae0` |
| `experiments/risk_safe/native_bridge.py` | `9575214ea8d14ebe32a537fa15afa332446b41d66c1c8c85f890d2251eaef8f7` |
| `experiments/a0_a1_strict4_matched.py` | `e916659681607bf75e097dd62379dcc363ccc3adf6f991d4108592997e1b53bf` |
| `experiments/calibration/profiles.py` | `7ff59b6a3783bf4061ab75c6b0782b23a0f6e456b40457a022e5d0de13d6f7a5` |
| `experiments/calibration/stage05.py` | `4b8aae252f5f1e096b127906794064da269aa79f7cd5514f6f738cda585c017e` |
| `experiments/calibration/stage1a_p11_production.py` | `77e84fa11c83eca683b0b95c4849b175053c0b24298e78dfe6482c91bdf6ea7d` |
| `experiments/calibration/stage1a_p12_label_provider.py` | `4e51255ae894c2d6ffcc7142e8fd59d9714df94b18ed26ac996528adb163ceaf` |
| `experiments/calibration/stage1a_formal_comparison.py` | `a9391ab94e9a994198484cd3fb5efc7f534b3515c870723c141c3417ea717c84` |

Cached DINOv2 source and checkpoint were present at `C:\Users\rico\.cache\torch\hub\facebookresearch_dinov2_main` and `C:\Users\rico\.cache\torch\hub\checkpoints\dinov2_vitl14_reg4_pretrain.pth`. The loader emitted xFormers-unavailable warnings and continued with its available fallback. Whether that runtime difference caused the support digest mismatch is unknown.

## 3. Exact R1 authorization scope

The run used the authorized limits: Real-IAD `switch`, four deployed supports at seed 0, stream positions 0–511 (maximum 512), frozen Q-A-TS-v1 seed 37 at 3%, the DINOv2 ViT-L/14 profile, flat Greedy memory at capacity 256, and C0/C1-Greedy-adapted/C2. C64 was read only from its existing receipt for the fixed C0 threshold. No other class, seed, confirmation class, mask, or outcome evaluation was used.

## 4. Implementation files

- `experiments/calibration/r1_matched_correction_pilot.py` — isolated native dense-map adapter, causal event gate, deterministic evidence shards, C1 source-aware filtering, bounded C2 interface, preflight, and resource/status controls.
- `tests/calibration/test_r1_matched_correction_pilot.py` — CPU-only synthetic gates.
- This report.

The runner includes an explicit empty-bootstrap recovery path. The first invocation stopped before preflight because direct script execution omitted the repository root from `sys.path`; it created only an empty manifest. The corrected invocation retained that attempt in `attempt_history.jsonl` and used `--resume-empty-bootstrap-failure`. No event or feedback state was resumed.

## 5. CPU synthetic tests

**PASS: 10 tests.** Command:

```text
python -m pytest tests/calibration/test_r1_matched_correction_pilot.py -q
```

The tests cover dense-map reconstruction and max aggregation after a corrected patch changes rank; Top-32 persistence; deterministic NPZ/checksums; zero-correction identity; label-provider gating; queried/unqueried event order; same-source and future-anchor exclusion; frozen baseline non-mutation; rejection of duplicate/gapped events and unsafe resume; and shard checksum verification.

Compilation also passed:

```text
python -m py_compile experiments/calibration/r1_matched_correction_pilot.py tests/calibration/test_r1_matched_correction_pilot.py
```

## 6. Query Tape preflight and disclosure causality

**Metadata preflight: PASS.** The Q-A-TS binding SHA-256 was `dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183`; protocol SHA-256 was `6f1d8cf40e5e7b2d98dcfabe5836b49c6cd57069d3c8377122dba64e3fc2da1c`. The frozen tape SHA-256 was `b48928b1b547d68f2978769191662362c8fd5f57bd40d39b0126e5800648cc16`; stream SHA-256 was `db7d78465d135c7a27e4ba695d2d52cdeb60864afe534b98545b9699a80f7a09`; positions SHA-256 was `899edd0f87d93da37cbf23ab044fbe49a9923d5b09e2558d99946d746369f930`.

There are 17 fixed query positions in the authorized prefix. The switch label JSON path was checked for existence only. The queried-position reader was never called: **0 queried labels decoded**. No unqueried label, mask, or confirmation label was read.

## 7. Frozen support and C64 fidelity gate

The frozen score receipt file hash was `91fefe78fe54a2d933770b120df08894b4c8e6f69297d0cd2d2ac7b950e0aa9e`; its recorded native parity passed at tolerance `1e-8`. The existing 64 C64 scores produced `theta0 = 0.5165398120880127` and `sigma0 = 0.021652716174389672` with the 0.02 floor. The frozen four-support bank digest expected by that receipt was `8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558`.

The newly rebuilt native support bank failed equality against that expected digest. The failed invocation did not persist the rebuilt digest, so its actual value is unavailable. The gate was not relaxed, and no stream event was started. This is the exact blocker. The present evidence does not identify whether the difference comes from configuration, support feature generation, environment/runtime, or another reproduction detail.

## 8. Native C0, dense map, and Top-32 parity

**NOT REACHED.** The support digest gate runs before the native scoring smoke. No stream feature tensor, dense base map, Top-32 ID set, or C0 event score was produced. Parity-failure count `0` means no parity event ran; it is not a pass. The runner’s required stream tensor shape is 784×4096 FP32, but that shape was not observed in this run.

## 9. Shared Greedy memory transitions

**NOT REACHED.** The fixed-support check failed before the flat Greedy writer was constructed. There were 0 writer events and no dynamic memory transitions. The common-C0 writer input invariant is implemented and synthetically covered only at the interface level; it has not been validated on this stream.

## 10. C1 and C2 status

There were 0 C1/C2 score replays, 0 correction writes, and 0 admitted positive or negative anchors. This is not `INSUFFICIENT_ANCHOR_EVIDENCE`; the stream never began. C1 is explicitly named `C1-Greedy-adapted`, not a faithful historical v22 replay, and uses the existing signed codebook with direct `strength_scale=1.0` in the runner. C2 is an unvalidated interface fixture with cosine kNN, five neighbors, 0.20 scale, capacity 256, and FIFO retention. Those C2 settings are not approved scientific hyperparameters; a future comparison would require preregistration of the method identity, locality, capacity/retention, and matched supervision budget.

## 11. Evidence tape and replay integrity

The output manifest is `outputs/calibration/r1_switch_s0_512_engineering_20261031/run_manifest.json`, SHA-256 `ca6e1603cd2c559166be4a862f6205eb824aec392f7e2227579c2f64cf48a27e`. The retained bootstrap history is `outputs/calibration/r1_switch_s0_512_engineering_20261031/attempt_history.jsonl`; its SHA-256 is recorded in the run manifest. The directory contains no event shard, score row, feature row, or feedback payload. Sealed events: 0; finalized events: 0. Full-tape readback/replay was not reached.

The report-preparation update to the run manifest records the metadata preflight result and the missing rebuilt digest explicitly. It does not add event or label data.

## 12. Resource use

- GPU wall time before the support fidelity stop: **83.016 seconds**; under the 120-minute ceiling.
- Stream events: **0 / 512**.
- Newly retained output: **1,993 bytes**, under the 2 GiB limit.
- Peak CUDA allocated/reserved memory: **not measured**; the run stopped before per-event telemetry. The manifest’s null fields mean unavailable, not zero.
- Peak CPU RSS: **not measured**.
- Feature encoding and stream scoring time: not applicable; only model/support setup ran.

## 13. Prohibited evaluation and data access

No AUROC, AUPRC, classification metric, threshold tuning, or performance comparison was computed. No test mask or confirmation data was accessed. The existing C64 score receipt was read as authorized; the switch label file was only checked for presence and was not opened.

## 14. Gate matrix

| Gate | Result |
|---|---|
| CPU synthetic tests | **PASS** — 10 passed |
| Frozen Q-A-TS identity and prefix | **PASS** — 17 authorized queries in 512 positions |
| Existing C64 receipt | **PASS** — checksum and recorded native parity valid |
| Rebuilt fixed-support digest | **FAIL / BLOCKER** — differs from frozen expected digest; actual digest not retained |
| Native C0 parity smoke | **NOT REACHED** |
| Dense map / Top-32 parity | **NOT REACHED** |
| Query label causality on stream | **NOT REACHED** — 0 labels decoded |
| Shared Greedy transitions | **NOT REACHED** — 0 writer events |
| C1/C2 replay and source exclusion on stream | **NOT REACHED** |
| Complete evidence-tape replay | **NOT REACHED** — 0 sealed events |
| GPU time / storage ceilings | **PASS** — 83.016 sec / 1,993 bytes |
| Scientific outcome evaluation | **NOT RUN** |

## 15. Limitations and recommendation

The R1 infrastructure has passing CPU tests and validated label-blind metadata inputs, but the required native support bank could not be reproduced byte-for-byte from the frozen receipt. The native parity smoke and all stream-dependent checks remain unverified.

**Recommendation: do not authorize R2 yet.** First independently trace the frozen Stage 0.5 bank construction/runtime identity and reproduce the expected fixed-support digest. Then review this blocked report and decide whether a fresh R1 execution is authorized. No scientific conclusion or anomaly-detection improvement is supported by this run.
