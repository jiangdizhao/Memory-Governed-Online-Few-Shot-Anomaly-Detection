# R2-G1A Switch Pilot — Execution and Authorization Report

**Status: `WAITING_RESEARCHER_AUTHORIZATION`**

**Pilot status: `NOT_STARTED`**

**S1: `NOT_RUN`**

**S2: `NOT_RUN`**

## Scope and decision

The requested CPU adapter, synthetic tests, and operation-specific authorization package were prepared. Real-IAD metadata and images were not opened, DINO was not instantiated, no GPU feature extraction or geometry calculation was run, and no output run directory was created. The task's stop condition applies because the fresh signed G1-A scope and independently certified switch A identity/image inventory are absent.

The old L2C Mode B authorization is limited to the TRAIN metadata census. It does not authorize image decoding or feature extraction. The reported L2C aggregate of 7,293 TRAIN candidate groups (minimum 167/category) establishes numerical possibility only; it does not certify the switch A membership, camera-1 image identities, or availability for this pilot.

## Repository and existing work

- Repository: `E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection`
- Branch: `calibration-stage0-contract-20261029`
- Parent HEAD at task start: `d085cf9fa58f1eab88ed66cdfc8168890a15dcdb`
- Tracked working tree at task start: clean.
- Pre-existing untracked audit/output paths were preserved and not staged.
- No branch switch, worktree creation, historical output edit, or push occurred.

The new adapter keeps the R1 and native sources read-only. The image access function requires a signed-scope-derived in-process permit, a reviewed per-image identity tuple, the selected source group, camera 1, and the S1 membership when applicable. The separate image identity inventory receipt binds category, A role, TRAIN-normal population, source-manifest digest, every image/group/camera tuple, and an independent Ed25519 reviewer attestation. Missing or tampered inventory evidence fails closed before the image loader.

Projected data are exactly the 49 fixed R1 cell-center rows. Full maps are released after one-image projection. The adapter connects the projected cohort to the existing exact `compare_source_disjoint()` and category summary methods; it does not alter their distance math. S1 is frozen to the first ten ranked groups; S2 is frozen to all 100. The output store uses the existing R1 pre-write byte budget, refuses reused directories, and is not created before authorization.

## Authorization and A-identity gates

The read-only preflight was run against `%USERPROFILE%\.r2_g1a_switch_authorization`. It reported these six exact missing artifacts:

1. `authorization.signed.json`
2. `trust_directory.json`
3. `trust_directory.sha256` — prospective digest pinned independently of the trust file
4. `source_fingerprint.json`
5. `source_fingerprint.sha256` — prospective digest pinned independently of the source file
6. `a_reference_bundle.json` — certified switch A availability and identity evidence, plus the reviewed camera/image identity inventory and its separate reviewer attestation

No `PREFLIGHT_READY_NO_DATA_IO` result is available. The scope requires a new trust root whose directory ID begins `r2-g1a-switch-access-`; an old L2C trust root or authorization is not substituted. The A bundle must bind the certified A availability receipt and source manifest, contain exactly the certified normal TRAIN image/group counts, identify camera 1 for each selected ranked group, and be signed by a trusted reviewer independent of the receipt producers and the A identity reviewer.

**A identity/availability: `PENDING_A_REFERENCE_RECEIPT`.** No group membership was assigned or certified here. **Image inventory: `PENDING_A_IMAGE_IDENTITY_REVIEW`.** No filename/camera census was run in this task.

## Source, model, and resource evidence

The frozen profile values remain the acceptance targets:

- Checkpoint SHA-256: `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`
- DINO hub source-tree SHA-256: `b7c0037f90671c4932393bd56de8c8a205b29a6eb22dea563b6c8a88c471da15`
- Expected loaded model-state SHA-256: `1f26464ba4055403e550f1d5c1333fb012631d98661df6993cabea730d17e3f9`
- Representation SHA-256: `06237800ef7044857bb617677625f9c2be79c0ce837ef83175ed915b23a91c28`

The checkpoint and DINO source-tree file fingerprints were checked without instantiating DINO and matched the frozen profile. Canonical-LF hashes for the five inherited R1 profile source files match their pinned values; raw Windows file-byte hashes differ for three CRLF files, so the canonical-LF comparison is the source-profile check. The **loaded model-state hash is not verified** because no model was loaded. The final G1 source-fingerprint digest must be generated and independently pinned against the committed adapter before any future access.

Read-only host checks reported an RTX 3090 Ti with 23,014 MiB free of 24,564 MiB VRAM at check time, approximately 278.8 GiB free on E: and 245.3 GiB free on G:. These are transient host snapshots, not pilot resource measurements. The proposed run target `outputs/calibration/r2_g1_pilot/r2_g1a_switch_20261101_a1` was absent. No output bytes were written.

## CPU verification

Using `D:\anaconda3\envs\torch\python.exe`:

- `py_compile` for the new adapter and focused test module: passed.
- Focused adapter suite: **32 passed**.
- Combined adapter, R2-L2A/L2B/L2C/L2D/L2E, and R1-S4A resource-containment regression suite: **211 passed in 13.77s**.

Synthetic coverage includes exact 49-row projection, representation and receipt binding, deterministic 100-group ranking, signed A/image-inventory receipt verification, malformed/tampered evidence rejection, pre-loader authorization and membership denial, camera-1 enforcement, S1 ten-image cap, exact geometry-engine adapter, output-budget refusal, non-overwrite behavior, and full-map release. Cryptographic test keys and identity digests are generated in memory and are not production evidence.

## S1/S2 results

Neither stage ran. S1 cannot start until the fresh signed scope, source pin, trust-root pin, and reviewed A image inventory pass. S2 additionally depends on S1 completing and passing its predeclared engineering checks. Therefore there are no real distance statistics, projected-cache digests, GPU/CPU experiment timings, experiment memory measurements, or pilot output sizes to report.

## Classification and next action

- `G1A_ADAPTER_CPU_TESTS_PASS`
- `G1A_SIGNED_SCOPE_PENDING`
- `G1A_A_IDENTITY_AND_IMAGE_INVENTORY_PENDING`
- `G1A_MODEL_STATE_UNVERIFIED`
- `G1A_S1_NOT_RUN`
- `G1A_S2_NOT_RUN`
- `G1A_REAL_DATA_PILOT_NOT_STARTED`

The next action is to place the six listed artifacts in a new private authorization directory, with researcher/reviewer supplied prospective pins and the signed A image inventory bundle, then run the read-only verification command in `G1A_AUTHORIZATION_STEPS.md`. Do not run S1 or access images until that command returns `PREFLIGHT_READY_NO_DATA_IO` and a separately reviewed execution plan starts the pilot under the existing 120-minute Windows Job Object supervisor.
