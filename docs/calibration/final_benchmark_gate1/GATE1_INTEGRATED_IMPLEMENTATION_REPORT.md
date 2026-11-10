# FINAL-GATE1 Native Backend and Scientific Validity Report

**Status:** `GATE1_PARTIAL_NATIVE_BLOCKER_REMAINS`

**Branch / base HEAD:** `calibration-stage0-contract-20261029` / `6cdd20fcb8510792371f4719cc046d1fabbc37b0`

**Frozen Gate 0 commit:** `a34b870fd4700bfe970e459d2963a602f04b1341`

**Working tree:** Gate 1 repair source, tests and documents are uncommitted. The prior `attempt_01/gate1_source_identity.json` binds the earlier implementation and is preserved as historical evidence; it does not identify these edits.

## Scientific access correction

The original Real-IAD census used `json.JSONDecoder` with an object-pairs filter. Python decoded TEST `anomaly_class` and mask metadata values before the filter discarded them. The census covered all 30 categories, including protected Mint and PCB. The projected identity records did not return those fields, and the checked private manifests do not contain them; this was nevertheless an unintended access contrary to the identity-only restriction. The original VisA CSV reader also decoded complete lines containing TEST label/mask cells into text buffers, although it did not assemble or return those cells. These facts supersede the original receipt's `NOT_ACCESSED` wording. No claim is made that the incidents establish a particular scientific result or that the full history has been forensically audited.

The corrected Real-IAD parser scans opaque JSON boundaries and decodes only allowed metadata, TRAIN normal-eligibility classes and TEST image identities. It does not decode forbidden TEST values or nested mask keys. The corrected VisA parser reads bytes and decodes only allowed fields. Synthetic sentinel tests cover both corrections. No real census, TEST outcome read, mask-file access, DINO inference, GPU operation, or scientific Stage A/B execution was performed during this repair. The researcher and independent reviewer should assess the prior exposure and its implications for provenance and claims before any scientific execution.

## Native implementation

`native_backend.py` adds a source-bound DINO input encoder and commissioning adapter, registered TEST identity/content verification, and a `NativeFourArmBackend` with separate F-B/K-B/F-M/K-M banks, Greedy writers, correction state, coverage state, C64 scores and B3 controllers. It uses the native R1 dense C0 scorer and writer, R9/Exp2A Full C1, R10 signed FIFO, R7 C64/B3, and E6/C BN+Band components. K-B's objective uses its own corrected score, FIFO residuals, C64 and coverage state. Support and C64 groups and image identities are rejected from the TEST scoring path. The registered model and local source receipt are checked when the future runtime is loaded; no model was loaded here.

N-shadow has an independent negative-only C1 codebook and reads F-B's Greedy state. Following durable group sealing and a legal F-B disclosure, its C1 update runs before F-B's Greedy write so that both consume pre-feedback C0/Greedy evidence. A real native CPU fixture compared F-B detector, Greedy and correction state with shadow enabled and disabled; an additional synthetic backend test compared scores, predictions and query decisions. Defect feedback leaves shadow C1 unchanged.

`label_broker.py` adds a Real-IAD query-only broker and private fsynced feedback receipt. The sink records executed query identities only after the full group is fsynced, limits each arm to one query per group, and consumes a receipt once. The broker validates the annotation hash and original position, checks the selected official image identity, and decodes only that purchased image's `anomaly_class`; a synthetic NG group with normal and defect views verifies image-level semantics. The broker and sink do not yet provide restart/recovery of unconsumed receipts after a process failure; an incomplete category must remain incomplete and its journal be preserved.

## Static baselines and metrics

Static-4+64 uses an independently built native fixed scorer whose fixed-bank digest must equal the principal four-Support bank; its C64 q90 threshold is frozen. Static-36+32 validates the P0 36/32 schedule, calls A4's candidate construction with its A3b quotas and identity/tie conventions, calls the A3b CUDA K-Center source, checks the 8,192 x 4,096 FP16 / 64 MiB bank, and calibrates a frozen C32 higher-q90 threshold. The full CUDA constructor and numerical source parity have not been run. Static-36+32 is requested only for P0; P1/P2 require only Static-4+64.

Stage B checks each static prediction against its own constant threshold and reports AUROC, AP, balanced accuracy, BER, FPR and FNR. N-shadow remains AUROC/AP only. Synthetic tests exercise P0/P1/P2 schema and distinct static thresholds.

## Data and outstanding work

- Real-IAD's local census remains 30 categories, 36,465 TRAIN-normal images in 7,293 groups and 114,585 TEST events in 22,917 complete five-view groups. One byte-identical TEST pair spans `button_battery` and `phone_battery`; it remains recorded and was not removed. The 30 annotation artifact-set hash is `8652d842d26785fff780fa3ac429592a39da72675c8dacb5fe56eabf512ce420`; the local identity snapshot hash is `d44a059bf998231fcdc2a296b97ca45544a2554075e1f92b4dfc15c89a5fe4e2`. These identify a reproducible local snapshot, not an independently authenticated publisher release.
- No publisher-authenticated release checksum was present in the reviewed local records. A narrow researcher-approved convention could define a canonical local snapshot digest as the frozen `dataset_release_sha256` input, explicitly identified as a local convention. Its algorithm, source scope and exact digest must be approved before generating role/stream hashes; this work does not substitute a digest.
- VisA's 1cls split has no recognized physical-group key. `prepare_visa_exception_candidate` validates caller-supplied image/duplicate-unit proposals and 4+64/TEST separation, but marks them non-executable. The current online runner and journal still require five-view Real-IAD groups; a variable-size VisA unit path remains to be implemented after the researcher decides the exception. No physical-independence claim is supported.
- A complete four-arm initialization and P0 static construction have not been exercised end-to-end: the production path includes GPU-only K-Center and native C64 rescoring. The components and source-call boundaries have CPU coverage, but their combined GPU numerical/source fidelity and resources remain unverified. This, the VisA unit path, and review of the prior access deviation prevent a completion claim.

## Verification and Gate 2 boundary

The focused suite passes 27 CPU synthetic tests. It covers actual native CPU C0 writer/Greedy parity, signed C1 updates and source exclusion, 4,096D native F/K scoring with R10, native C1/C64/B3 refresh on a small CPU fixture, real C1/Greedy shadow noninterference, R10 and independent B3 state, arm-specific BEA inputs, group sealing/query disclosure, identity-only projections, P0-only static schema and static metrics. Some full-backend integration tests use synthetic bank or C64 stand-ins; they are not numerical GPU parity tests. Compilation passes. Reproduce with:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& 'D:\anaconda3\envs\torch\python.exe' -m unittest discover -s tests/calibration/final_benchmark -p 'test*.py' -v
& 'D:\anaconda3\envs\torch\python.exe' -m compileall -q experiments\calibration\final_benchmark tests\calibration\final_benchmark
git diff --check
```

A separately authorized Gate 2 should verify the real DINO/preprocessing/model receipt and 4,096D features, four-bank initialization and independent score/update parity, full C64 C1/R10 rescoring and B3 transitions, N-shadow noninterference, A4 candidate identities and CUDA K-Center 8,192 FP16 bank/C32 q90, and end-to-end runtime, VRAM, RAM and disk costs. These are pending checks, not measured results or authorization to run a scientific campaign. No supervisor sign-off is asserted.

## Local source identity

The following SHA-256 values identify the uncommitted repair at report time:

- `native_backend.py`: `8b7c2ccd7db68dfc5c5dd0f6eda8ad52b7377456f0e7ea36d85f36b3fbe4f9f4`
- `label_broker.py`: `ddcc296e15a932a38b8f67d6354b0d4c48182a2f1269f8898a5e233d02892132`
- `identity.py`: `5611d3314351415c6c91aebe58b08bf49f1f3075e47a178c1bd3d444efb88bb9`
- `runner.py`: `9328c95be4f68b70f03984e3ae207b3b6a838cc01ce7b5ca07cc7fd2c030bddf`
- `sink.py`: `17f22c33c2475f7763da68d548b525e508b1628cf1a61eca11097f7883765fa8`
- `evaluation.py`: `15ca48487b1409ccd862b0b91648aecd3dfef141c45853618562e271d5ff80fe`
- `static_baselines.py`: `f12727c2348cbc3454af0a3afc44243fffe5123ccc00efb017ba2c008369cf73`
- `test_gate1_synthetic.py`: `3a49e09407934168e10de677299a086f1a47411de89ffabde3ca2ad5da9e4cc1`
- `test_native_backend_cpu.py`: `3d7cf004cdfb6a602f2f2a8e6f7c9543fb17686d71594b212fb481095cf7dd5a`
