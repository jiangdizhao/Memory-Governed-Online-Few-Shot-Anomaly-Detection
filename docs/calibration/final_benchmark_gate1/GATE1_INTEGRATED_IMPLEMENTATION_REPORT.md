# FINAL-GATE1 Native Backend and Scientific Validity Report

**Status:** `GATE1_IMPLEMENTATION_COMPLETE_GATE2_NUMERICAL_REVIEW_PENDING`

**Branch / base HEAD:** `calibration-stage0-contract-20261029` / `f9b3ce3919aadd1129472967bb14cb0cabb9f50d`

**Frozen Gate 0 commit:** `a34b870fd4700bfe970e459d2963a602f04b1341`

**Working tree:** Gate 1 closeout source, tests and documents are uncommitted. Existing unrelated research artifacts remain untouched. The prior `attempt_01/gate1_source_identity.json` binds an earlier implementation and remains preserved as historical evidence; it does not identify these edits.

## Scientific access correction

The original Real-IAD census used `json.JSONDecoder` with an object-pairs filter. Python decoded TEST `anomaly_class` and mask metadata values before the filter discarded them. The census covered all 30 categories, including protected Mint and PCB. The projected identity records did not return those fields, and the checked private manifests do not contain them; this was nevertheless an unintended access contrary to the identity-only restriction. The original VisA CSV reader also decoded complete lines containing TEST label/mask cells into text buffers, although it did not assemble or return those cells. These facts supersede the original receipt's `NOT_ACCESSED` wording. No claim is made that the incidents establish a particular scientific result or that the full history has been forensically audited.

The corrected Real-IAD parser scans opaque JSON boundaries and decodes only allowed metadata, TRAIN normal-eligibility classes and TEST image identities. It does not decode forbidden TEST values or nested mask keys. The corrected VisA parser reads bytes and decodes only allowed fields. Synthetic sentinel tests cover both corrections. No real census, TEST outcome read, mask-file access, DINO inference, GPU operation, or scientific Stage A/B execution was performed during this repair. The researcher and independent reviewer should assess the prior exposure and its implications for provenance and claims before any scientific execution.

## Native implementation

`native_backend.py` adds a source-bound DINO input encoder and commissioning adapter, registered TEST identity/content verification, and a `NativeFourArmBackend` with separate F-B/K-B/F-M/K-M banks, Greedy writers, correction state, coverage state, C64 scores and B3 controllers. It uses the native R1 dense C0 scorer and writer, R9/Exp2A Full C1, R10 signed FIFO, R7 C64/B3, and E6/C BN+Band components. K-B's objective uses its own corrected score, FIFO residuals, C64 and coverage state. Support and C64 groups and image identities are rejected from the TEST scoring path. The registered model and local source receipt are checked when the future runtime is loaded; no model was loaded here.

N-shadow has an independent negative-only C1 codebook and reads F-B's Greedy state. Following durable group sealing and a legal F-B disclosure, its C1 update runs before F-B's Greedy write so that both consume pre-feedback C0/Greedy evidence. A real native CPU fixture compared F-B detector, Greedy and correction state with shadow enabled and disabled; an additional synthetic backend test compared scores, predictions and query decisions. Defect feedback leaves shadow C1 unchanged.

`label_broker.py` adds a Real-IAD query-only broker and private fsynced feedback receipt. The sink records executed query identities only after the full group is fsynced, limits each arm to one query per group, and consumes a receipt once. The broker validates the annotation hash and original position, checks the selected official image identity, and decodes only that purchased image's `anomaly_class`; a synthetic NG group with normal and defect views verifies image-level semantics. The broker and sink do not yet provide restart/recovery of unconsumed receipts after a process failure; an incomplete category must remain incomplete and its journal be preserved.

`entrypoint.py` now connects the certified identity/release/role bundle, `NativeInputEncoder`, four-arm native backend, query-only Real-IAD broker, dataset-aware runner, fsync journal, and Stage B evaluator. It verifies separate private locator sets for the 4+64 TRAIN roles and the registered TEST stream. It has a one-category, 5-or-10-event numerical preflight path that creates no Stage A seal and has no label access. Both entry paths require source- and manifest-bound authorizations; Stage B invokes its label projection callback only after a separate seal-bound grant passes. The entrypoint rejects the current uncertified local release identity. No entrypoint or model path was executed against real data in this task.

## Static baselines and metrics

Static-4+64 uses an independently built native fixed scorer whose fixed-bank digest must equal the principal four-Support bank; its C64 q90 threshold is frozen. Static-36+32 validates the P0 36/32 schedule, calls A4's candidate construction with its A3b quotas and identity/tie conventions, calls the A3b CUDA K-Center source, checks the 8,192 x 4,096 FP16 / 64 MiB bank, and calibrates a frozen C32 higher-q90 threshold. The full CUDA constructor and numerical source parity have not been run. Static-36+32 is requested only for P0; P1/P2 require only Static-4+64.

Stage B checks each static prediction against its own constant threshold and reports AUROC, AP, balanced accuracy, BER, FPR and FNR. N-shadow remains AUROC/AP only. Synthetic tests exercise P0/P1/P2 schema and distinct static thresholds.

## Data and outstanding work

- Real-IAD's local census remains 30 categories, 36,465 TRAIN-normal images in 7,293 groups and 114,585 TEST events in 22,917 complete five-view groups. One byte-identical TEST pair spans `button_battery` and `phone_battery`; it remains recorded and was not removed. The 30 annotation artifact-set hash is `8652d842d26785fff780fa3ac429592a39da72675c8dacb5fe56eabf512ce420`; the local identity snapshot hash is `d44a059bf998231fcdc2a296b97ca45544a2554075e1f92b4dfc15c89a5fe4e2`. These identify a reproducible local snapshot, not an independently authenticated publisher release.
- No publisher-authenticated release checksum was present in the reviewed local records. The identity audit now records a precise local-snapshot identifier and domain-separated digest proposal for researcher review. Gate 0 was not amended, no digest was substituted, and no role/event manifests were issued under the proposal.
- VisA's 1cls split has no recognized physical-group key. The candidate adapter now binds each image to either a single-image unit or a supplied exact-duplicate cluster, rejects role-crossing or content-inconsistent clusters, and computes a candidate digest. The runner, fsync sink, and Stage B verifier support variable-size candidate units in synthetic-only mode; one query per unit is enforced after unit sealing and the category cap is `floor(0.10*T_c)` over images. Real VisA execution still requires a separate researcher approval bound to the candidate and role/release identities. No physical-independence claim is supported.
- The Real-IAD native path is source-connected and ready for a bounded Gate 2 numerical preflight after release/role certification and authorization. The end-to-end runtime was not executed: Native DINO loading, complete four-arm commissioning, P0 GPU K-Center for Static-36+32, C64 rescoring and measured resource parity remain Gate 2 work. The earlier TEST-field access deviation also remains a researcher/independent-review dependency for claims and subsequent data execution.

## Verification and Gate 2 boundary

The focused suite passes 34 CPU synthetic tests. It covers the existing native CPU C0 writer/Greedy parity, signed C1 updates and source exclusion, 4,096D native F/K scoring with R10, native C1/C64/B3 refresh, real C1/Greedy shadow noninterference, independent arm state, arm-specific BEA inputs, Real-IAD group sealing/query disclosure, VisA image and duplicate-cluster unit timing/caps, durable variable-unit seals, bounded entrypoint wiring, release-certification rejection, Stage B authorization gating, P0-only Static-36+32, and static metrics. Some full-backend checks use synthetic bank or C64 stand-ins; none are numerical GPU parity tests. Compilation passes. Reproduce with:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& 'D:\anaconda3\envs\torch\python.exe' -m unittest discover -s tests/calibration/final_benchmark -p 'test*.py' -v
& 'D:\anaconda3\envs\torch\python.exe' -m compileall -q experiments\calibration\final_benchmark tests\calibration\final_benchmark
git diff --check
```

A separately authorized Gate 2 should execute only the bounded checks enumerated in `GATE1_GATE2_HANDOFF.md`: one certified Real-IAD category/condition, 64 commissioning images, at most 10 complete TEST events, no labels, and no Stage A seal. Verify model receipt and DINO/preprocessing descriptors; independent native bank/scorer/write parity; C64/C1/R10/B3 state; N-shadow noninterference; P0 Static-36+32 candidate IDs and CUDA K-Center 8,192 FP16/64 MiB bank with C32 q90; and measured runtime/VRAM/RAM/disk costs. These remain pending, not measured results or authorization to run a scientific campaign. No supervisor sign-off is asserted.

## Closeout source identity

These SHA-256 values identify the current uncommitted closeout files:

- `entrypoint.py`: `2fe77693d4008b65e4ed4cf86cddba5c0137cf2e62c5cec6b7fdded3000236ac`
- `identity.py`: `8c235a57386a75056ef35bc9f24464be5e7133997332a5285db1442ab5eccb59`
- `runner.py`: `e0d56ba4541314aa75132c294852f905642dc82f0a0150bf61d718d76fe6d4a2`
- `sink.py`: `98b5c0e1c463a79627ed259dce67a0d425988e9070e1cc1b58996d4bea08c943`
- `evaluation.py`: `9b7b9369a83899703c095daded8e8f93a00f4002ec07fb1429655c5c89c6a112`
- `__init__.py`: `5749b609cce57974a522207cc2142cbb2bc5329dfc4f3c1ccae81e84cac1b82c`
- `test_gate1_synthetic.py`: `31fad4699dc4dea0e4e314d0145f15e7518d58e4ad5818508da8bbb717eb6fbb`

The prior repair hashes and `attempt_01/gate1_source_identity.json` remain historical evidence and were not overwritten.
