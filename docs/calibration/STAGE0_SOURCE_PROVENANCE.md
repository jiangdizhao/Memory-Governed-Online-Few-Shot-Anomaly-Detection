# Stage 0 source provenance

> **Historical report:** this records the prior fourth-pass provenance and is
> retained as an archive. The current fifth-pass hashes and dependency
> baseline are in
> [`STAGE0_FIFTHPASS_SOURCE_PROVENANCE.md`](STAGE0_FIFTHPASS_SOURCE_PROVENANCE.md).

## Current fourth-pass provenance (authoritative)

Hash algorithm: SHA-256. Runner implementation snapshot: source commit
`9d2d20df0bd94629eaea0f8cbd6a6f84ebc28a44`; working source fingerprint
`693b17200af009da4bf682ea0f25827e02d36d82f3fc30d8615d421d18fc7537`.
The current source manifest is in the checksummed local runner artifact
`outputs/calibration/stage0/stage0_fourthpass_run1_20261029/source_hashes.json`
and its privacy-safe committed counterpart is
`docs/calibration/audit/stage0_fourthpass_v1/protocol_fingerprints.json`.
The runner hashes 48 source, test and scientific-contract files. This report
and the implementation report are omitted from the source fingerprint to avoid
self-reference. Their current content is committed as documentation.

The compact audit's `audit_manifest.json` records the SHA-256 of the full
runner checksum manifest and the selected full runner artifact files. The
package's `checksums.json` verifies every exported file. Verification passed
both against the local runner artifact and as a standalone privacy/checksum
scan. The compact package contains no Real-IAD image, mask, feature tensor,
absolute score vector, absolute local path, or unredacted sample identifier.

### Newly implemented source hashes

| File | SHA-256 |
|---|---|
| experiments/calibration/spot_dspot.py | 540c7b2ba68db6deca16687ae7c208918462b8dce72ee9ef968cec0289023414 |
| experiments/calibration/native_greedy_matrix.py | 298f100b1d12e6c2c973d934b737bc040a639b0aaa22bfda82a4ef3fbdaf0937 |
| experiments/calibration/native_event_replay.py | 66b3d9d7d801405b012f77761e2335d4e262eedf72f683746272a5686c1f2134 |
| experiments/calibration/audit_export.py | 887ba9bc2dc48f6024355277e9af3b7c1549a20f98087932656c5dbcd3e308d9 |
| docs/calibration/SPOT_DSPOT_IMPLEMENTATION.md | 652211533b509fc9541a8e3409d1475e1752f59daeec67fa8ae7b4b53cbc0fff |
| docs/calibration/BASELINE_FIDELITY_AND_REFERENCES.md | 9ed226a3f97c8d56ec41d0ca0b275930b3b1ba6264cec8e4ccbea032edc95740 |

### External and runtime provenance

- Siffer et al. KDD 2017 primary paper: `https://doi.org/10.1145/3097983.3098144`.
- Inspected `asiffer/libspot` commit: `d57e921a08930b960d07adcbec3a839bd228fe94`; `src/spot.c` blob SHA-1 `fdebc8c216fffeea93afcc51efab83a2d6013351`. SPOT source exists; dSPOT does not. No compiled C numerical trace parity is claimed.
- Sun et al. ICML 2024 / PMLR 235 primary paper: `https://proceedings.mlr.press/v235/sun24h.html`. No implementation source is claimed; B6 remains blocked.
- Environment used: Python 3.11, PyTorch `2.0.1+cu117`, NumPy `1.24.1`, SciPy `1.11.2`, CUDA available. No dependencies were installed or changed.
- Cached DINOv2 checkpoint SHA-256: `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`.
- Cached DINOv2 Python source tree SHA-256: `1de67887ca330bdfaad8ccd45128d2b9313d63aa707bbe8e4f38310e52005b24`. The cache has no Git commit metadata.
- `historical_sources_unchanged_by_this_work` is `true` in the runner manifest. No v22, VisionAD, P0/P1, V0.2, native bridge or archived experiment source was modified.

## Historical third-pass source manifest (superseded)

The table below is preserved as the third-pass snapshot only. It must not be
used as current provenance; use the current fingerprint and compact audit
package above.

Hash algorithm: SHA-256. File hashes below are taken from the checksummed Stage 0 runner source manifest for the third-pass source snapshot. The implementation report itself and this provenance file are omitted to avoid self-reference; the final runner artifact records their hashes.

Runner source commit before implementation commit: 42aecc32f865b5b5a9cf782c32e14755df305106.
Checkpoint SHA-256: 36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51.
Cached DINOv2 Python source tree SHA-256: 1de67887ca330bdfaad8ccd45128d2b9313d63aa707bbe8e4f38310e52005b24.
Cached DINOv2 Hub Git commit: None (cache has no Git metadata; content is pinned by the source-tree hash).

## Calibration code and tests

| File | SHA-256 |
|---|---|
| docs/calibration/ARCHITECTURE_AND_CAUSAL_PROTOCOL.md | f3c357dc930be2e8fc7a84349b566896ab676944d03349d6d10b1cebb89d78bf |
| docs/calibration/AUTOMATION_AND_ACCEPTANCE.md | a4f42686505eb60c93373481efe3505a9d501e2987f0d0c5bdad9a6941a20b97 |
| docs/calibration/BASELINE_FIDELITY_AND_REFERENCES.md | 6233524d63bfe3c548c806654529205bdd1fefc364518eefdb5bb97afeeefae4 |
| docs/calibration/CODEX_IMPLEMENTATION_TASK.md | 372f9f07c8567187beb7b4bdded82e6cf6e37d67a4a2a22cff2fd37149224295 |
| docs/calibration/README.md | 47ac272ec4cea83c64514164fab2f4923de1c3fc09b287917a3f79a1f5b74dd5 |
| docs/calibration/SCDAC_G_V0_METHOD_SPEC.md | ca1cd5f23d57545dce37622c9bf75ce939d38a00e3fe49d67ea944f33b77aef6 |
| docs/calibration/STAGE0_RESEARCH_CONTRACT.md | 101d15bba543783d6f48004bd0c1b6cf9cd1e2ef75d238b0baa1a77cb04aae66 |
| experiments/a0_a1_strict4_matched.py | e916659681607bf75e097dd62379dcc363ccc3adf6f991d4108592997e1b53bf |
| experiments/calibration/__init__.py | 4bd9454236d33c974c3251757a34108d8f2ff1a2e93ea9b095241ba69823f430 |
| experiments/calibration/anchor_ledger.py | 84d6ec03b5a181274f1655586d9e321d59ca80365dbef5f2a11ce8b74d5e83c7 |
| experiments/calibration/controllers.py | 4f0cdf429a5c3286b67abd9f34a72b9e529945a13dc1d70dc07e23d0f9e78cf4 |
| experiments/calibration/engine.py | e8d6ed37e13e2c0409a3f01e079e703fa79481363a9c397f5767b70ab8822680 |
| experiments/calibration/method_registry.py | 8726f5f2583300e46622a923ded47043d6012ca5633ebe4ffdb0542aab94040d |
| experiments/calibration/metrics.py | e879c170cfe5ec364d4479f73d8c09b5a7c9b6008260f1592968117c56f78dee |
| experiments/calibration/native.py | e4fea9b8b7f6cdb1ac6e92705f80da8c459816908240c2475d7f518202c52c1f |
| experiments/calibration/native_parity_smoke.py | fc5f312d527c719a47b505cb528f7ec8772f64b7bebddb96621c3000194e4742 |
| experiments/calibration/plugins.py | 5c57f7870cfd4041893dfff3b477eb2aa755a285185c0355a94f4d469b99d208 |
| experiments/calibration/profiles.py | 7ff59b6a3783bf4061ab75c6b0782b23a0f6e456b40457a022e5d0de13d6f7a5 |
| experiments/calibration/protocol.py | d012f8c96c4355f377c7263ebd46234423e8176f22bf2a22d730996892a03c2f |
| experiments/calibration/query_tape.py | ed6f77c2a4de1bd05969ae3c09f2c98c13a744481ca436f80e4633bd6c412c6d |
| experiments/calibration/real_iad_preflight.py | b3b75b395b7e7ca0734f5796aa8e2365d2acaaef50a942c22c32c4d502c03fd8 |
| experiments/calibration/runner.py | 07466254eedb49189fc1536ff7f4ba7b885e7c35ca825ccf9593dcf2242afa5f |
| experiments/calibration/scdac_g.py | 4c69aadeff6ba74bd55f3664702b5dfb068aeffb823c2197c1c9ce2d53fbb6ef |
| experiments/p0_protocol_audit.py | b83918ec5aceb3ef63cca942fc9fb0e241559d9672d20bed422b15fcd48135e7 |
| experiments/risk_safe/d2_threshold_calibration.py | 874bf3e1dc40785a240ad7267141b68b0fe58362620a9c8cf5b795fbcb375d93 |
| experiments/risk_safe/native_bridge.py | 9575214ea8d14ebe32a537fa15afa332446b41d66c1c8c85f890d2251eaef8f7 |
| experiments/risk_safe/p01_falsification_native.py | 6c9cd2cc6540033d413d67c3805d1c3c2603bbbe011943dfed548c43a7bdc7f2 |
| experiments/risk_safe/preflight.py | 2070ba770ac36af06f688b0450c711d67b7bca526e3939178d559216535c25ba |
| experiments/risk_safe/query_protocol.py | 83cfdff6867e9b8c2afe0ab4047f177e5b2814b4deafd63a6035a9dd06e16b2f |
| experiments/v02_memory.py | 3c4e247e3a41dc9acc8147a35aa892cf3f3571aca85beb4169809bd70ad48ae0 |
| test_phase3_AIF_simplified_v22.py | bfc871d5a9a32b2f5bdb9d7aab5177e702dec2e1110a6632f20bd13c20fcba30 |
| tests/calibration/test_controllers.py | d70d23eb7debc9e11d105db051fecf1fb45b532c2ccbff90ef89d11a01064302 |
| tests/calibration/test_engine_and_ledger.py | e7f135397b3c9136d7251170fc29f18ced79cba6d115f452cda1b6c6653abde0 |
| tests/calibration/test_query_tape.py | 406df95b32fb90eb063206fb55793b60a05bf2b60c2bb243d7aa79fb173fa8ae |
| tests/calibration/test_real_iad_preflight.py | 17306739c2ed2b8d7ee3de8c5a9cb78975d09c5f85bb66f29736ea9bb4e028ff |
| tests/calibration/test_registry_and_greedy.py | 195a14036c9b5c3bc97be092867badcde53037ae886cd56bca1b57dde85525e2 |
| tests/calibration/test_runner_integrity.py | 08cddea80a5a27c451b887c8fd3a10415b87a67518aa8c16b315a25011904025 |
| tests/calibration/test_scdac_g.py | 82b910910987e929df169ded411b1a6984a655f4dc5fe04bb5e66d2730bbf90e |
| vmb_visionad_new.py | bd6545bf5ff63acce01792e5d42aa964c88e73fb59670febeb55f2d1139a409a |

## Historical and scientific source identity

SCDAC-G proposal PDF: 5fe07256384351c1170d67183eae5a80e5aa289a03586ba767671f062f4fb977.
Pinned libspot source audited for B5: commit d57e921a08930b960d07adcbec3a839bd228fe94; src/spot.c blob fdebc8c216fffeea93afcc51efab83a2d6013351 (SPOT present; dSPOT absent).

Historical source hashes above establish the read-only inputs used by the native bridge; no historical file is part of the implementation diff.
