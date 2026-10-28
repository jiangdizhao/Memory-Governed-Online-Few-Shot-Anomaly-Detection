# Stage 0 fifth-pass source provenance

This file describes the final fifth-pass source snapshot represented by the
checksummed Runner output and compact audit package. Hashes use SHA-256 unless
the row explicitly identifies a Git blob SHA-1. The Runner source checkout was
`a083738556424563cba39d8df246623f96a74fd1`; the final implementation commit is
reported in the task completion message.

## Reproducibility fingerprint

- Runner output: `outputs/calibration/stage0/stage0_fifthpass_release_20261029/`.
- Compact audit: [`audit/stage0_fifthpass_release_v1`](audit/stage0_fifthpass_release_v1/).
- Full-source fingerprint SHA-256:
  `507266e10c1306dae95315bf5d6e5d5f7f44bbb40022afd15818b99ae2e99bfd`.
- Hashed entries: **61**, including transitive historical Python imports,
  calibration modules, tests, contracts, B6 specification, B7 proposal PDF,
  and the static protected-source baseline. The complete privacy-safe map is
  in `protocol_fingerprints.json`; the full map is in Runner
  `source_hashes.json`.
- Runner artifact checksum-manifest SHA-256:
  `be2431f00c58a0e78dc9ffa76ecf95dd461bc9e33c24b35d044b4609992a888d`.
  The audit manifest records this value and hashes every selected Runner
  source artifact. Its verifier recomputes the source fingerprint from the
  exported 61-entry map and verifies source-to-audit linkage.

## Protected historical dependency baseline

The closure is resolved from the Native Scorer, event replay, Greedy matrix,
native parity smoke, Anchor Ledger, method registry and Runner, plus explicit
dynamic-import dependencies. The reviewed baseline is fixed at
`9d2d20df0bd94629eaea0f8cbd6a6f84ebc28a44`. All 13 resolved protected files
matched the static hashes and the transitive set exactly; no dependency was
missing from or stale in the manifest.

| Protected source | Baseline SHA-256 |
|---|---|
| `experiments/a0_a1_strict4_matched.py` | `629251fe41386fef35bd7f5d92a9099970803a1996ff98607550bb43c6ad3741` |
| `experiments/p0_protocol_audit.py` | `03586ae40f79e9e49ad6545a4131c411a5f6adef9d085feae0b885f2f98966da` |
| `experiments/risk_safe/__init__.py` | `f95084f037129dd9818a1cd78688b3d5da2a2e3f8f86869600ab7fd80e67a416` |
| `experiments/risk_safe/d2_threshold_calibration.py` | `417a55a41b3ee9fcba97a4c34662506bc228b4a774fccd0c2c5a6666ada3abc2` |
| `experiments/risk_safe/native_bridge.py` | `313b94e7a3df82a4996b0c8a97a1e2464380f0e4cded09ae2e1647866859dfb4` |
| `experiments/risk_safe/p01_falsification_native.py` | `377a3d8a3d6b168e5712692d53ac83ddfd0004046c904a24b5e985f082da8ea1` |
| `experiments/risk_safe/preflight.py` | `65d62b73a2835ee9625a26dc140c1c5b062be6292bc0b0981cd53218a424e0bb` |
| `experiments/risk_safe/query_protocol.py` | `aa1c67b9550d181a3b8f674c10ff01c073674434da720eef4e4fae9cf0c503ec` |
| `experiments/risk_safe/shadow_memory.py` | `70c2ba3d29b225f241d9ad147f5be168ee2e0613f452e1428cc585bbce3339b5` |
| `experiments/v02_architecture_core.py` | `38c8e31b831614400c2d461538d637a0a0dc215fee19893791187b0b229c38de` |
| `experiments/v02_memory.py` | `d4cf38e75fad9c3cef317e07370c283d660ae495da11a6d759c1f0be8b789d68` |
| `test_phase3_AIF_simplified_v22.py` | `a79b6a78ccf918c9314025ab0acef2d9e8d84cf44600615a4085cd82802bda4f` |
| `vmb_visionad_new.py` | `0430959e86ba75b029341aef003655e6a3ac7d08df22f3f48a04e0b2d7fc39c9` |

The restored `shadow_memory.py` raw bytes and the Git blob at the baseline
commit both hash to
`70c2ba3d29b225f241d9ad147f5be168ee2e0613f452e1428cc585bbce3339b5`.
`test_historical_shadow_memory_is_byte_identical_to_pre_fourth_pass` checks
this byte-level equality. Calibration-owned prototype content hashing and
version transitions live in `experiments/calibration/memory_state.py`; the
legacy Greedy version and source-map revision remain independent axes.

## Model and external-source provenance

- Runtime: Python from the existing `torch` Conda environment; PyTorch
  `2.0.1+cu117`; CUDA available. No package installation or dependency change.
- Cached DINOv2 checkpoint SHA-256:
  `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`.
- Cached DINOv2 Python source-tree SHA-256:
  `1de67887ca330bdfaad8ccd45128d2b9313d63aa707bbe8e4f38310e52005b24`.
  The cache has no Hub Git commit metadata; source-tree content is pinned by
  this digest.
- SPOT/dSPOT provenance: Siffer et al., KDD 2017; pinned
  `asiffer/libspot` commit `d57e921a08930b960d07adcbec3a839bd228fe94`,
  `src/spot.c` blob `fdebc8c216fffeea93afcc51efab83a2d6013351`. The pinned
  file includes SPOT, not dSPOT. No compiled C trace parity is claimed.
- Sun-CS provenance: Sun, Sankararaman and Narayanaswamy, ICML 2024, PMLR
  235, paper <https://proceedings.mlr.press/v235/sun24h.html> and full PDF
  <https://raw.githubusercontent.com/mlresearch/v235/main/assets/sun24h/sun24h.pdf>.
  Full paper and appendices reviewed; Appendix G says implementation would be
  open sourced, but title, `sun24h`, author and publication-page searches did
  not identify a verified author repository or commit.
- SCDAC-G proposal PDF SHA-256:
  `5fe07256384351c1170d67183eae5a80e5aa289a03586ba767671f062f4fb977`.

Full test identifiers and result details are in Runner `test_results.json`;
the compact audit includes the 76-test summary and passed test IDs. The audit
verification is a cryptographic and provenance check only. It does not
establish scientific validity, external baseline equivalence, or empirical
performance.
