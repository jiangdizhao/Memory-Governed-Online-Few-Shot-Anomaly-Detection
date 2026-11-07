# FS4 final joint campaign freeze - engineering report

**Task:** `FS4_FINAL_JOINT_CAMPAIGN_FREEZE_V1`
**Branch:** `calibration-stage0-contract-20261029`
**Starting HEAD:** `0bb42b8c363342c8bc10b34c7a706d3f458ff579`
**Implementation commit:** `354711967bc05ad4232798bd25e6f4d3a5bedd07`
**Compatibility-scope commit / final source freeze:** `ac60c09b0078ba22fe7d6d88b847cb15439cf6ea`
**Frozen campaign spec SHA-256:** `1c96077bfdaa58f9ff93a67a9349a4cc6bc8b63578ae8915ff23762b50aee896`
**Scientific execution authorized:** No.
**Disposition:** Ready for independent review under the documented trusted-local model; not authorization to stage data or run the campaign.

## Frozen population, arms, and identities

The population is development data only: audiojack 3,815 events / 763 five-view groups; regulator 3,855 / 771; switch 3,820 / 764; rolled_strip_base 3,805 / 761. Total: **15,295 events in 3,059 Source Groups**. No confirmation categories or independent external-validation claim is included.

The exact arm registry is:

- **S: `SHARED_TRAJECTORY_THRESHOLD_ONLY`:** one score, query, feedback and Memory trajectory; simultaneous L, E and FrozenTheta readouts, each on G and GC. FrozenTheta remains fixed while using the same evolving Memory.
- **CL: `COUPLED_QUERY_ONLY_FROZEN_WRITER`, controller L:** independent query, feedback, Greedy, C1 and Coverage state; evaluate G and GC.
- **CE: `COUPLED_QUERY_ONLY_FROZEN_WRITER`, controller E:** independently coupled state; evaluate G and GC.

The primary estimand is category-macro `BER(S/E/GC) minus BER(S/L/GC)`, paired within category. The secondary coupled estimand is `BER(CE/E/GC) minus BER(CL/L/GC)` and is a whole-system contrast, not a threshold-only effect.

Initialization and identity freeze:

- Four original registered TRAIN-normal Support images/category, Support seed 0, original absolute path strings and native path-dependent augmentation/subsampling semantics; no extra normal calibration images. Native fixed bank is 8,192 x 4,096 FP16; Greedy and C1 start empty.
- Historical R1 V3 Support reference receipt: `a709e088477002ecc8690841b2cbcfc0e02929f7e7369108df9f1b19daa91482`. It supplies reference identity only; actual four-Support construction and parity are still required later.
- Fixed-bank SHA-256 by category: audiojack `837fa5153e852b617e1d0ecf563509bca037f5c0cab29c48b06befd21ce02673`; regulator `b162fe852cdc4c7dd5c5a25e3560263427cde588d42f12364bac0d3c79df48bb`; switch `8a710d028509350568e976bc3ede66aa59603499cbea525bdd8579322b1a1558`; rolled_strip_base `12c1dcacba90b1f0c576480b0afddaabf2d80466845ac61f4346aebfd968d4f3`.
- Model: `dinov2_vitl14_reg`; checkpoint SHA-256 `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`; model-state SHA-256 `1f26464ba4055403e550f1d5c1333fb012631d98661df6993cabea730d17e3f9`; hub source-tree SHA-256 `b7c0037f90671c4932393bd56de8c8a205b29a6eb22dea563b6c8a88c471da15`; native-config SHA-256 `ba3fa736734c2c6c3789f485f3288c5ae77177c68b6b3e24de7e257e447ced87`.
- Stream: category order audiojack, regulator, switch, rolled_strip_base; historical R7 deterministic Source Group shuffle, seed 0, with registered view order 1-5 inside each group. The code freezes the reconstruction and bijection check; the real-data R7 order receipt has not been constructed in this task.
- Query: stochastic BEA group-slot policy, seed 37; target 0.037/image (not a guarantee); initial cost 0.8, learning rate 0.02, 256-event cost window, bounds [0.05, 3.0], candidate-slot probability 0.2, exploration 0.05 when below cost, at most one query/group. The frozen score scale, clipped weights, minimum 12 normal groups/ESS, weighted 0.9 quantile and plus/minus 0.01 threshold step are in the machine spec. No catch-up queries or C64 initialization.

## Information access and causal order

Stage A receives metadata-free lossless RGB resources, the four registered Support paths needed for native bootstrap, opaque event/group/view IDs, and frozen model/source/spec identities. It does not receive original TEST filenames, defect codes, full labels, masks, private mapping, unqueried outcomes, future events, hidden class histograms, historical query positions/scores or C64 outcomes. Each event is scored and predicted before any current-label disclosure. Only an executed query with durable query receipt and `LABEL_INTENT` can reach that trajectory's spawned query-only broker. Only legally queried verified normals can update the shared Greedy bank.

All arms share one opaque registration and source-to-opaque mapping, but S, CL and CE have distinct run IDs, ledgers, disclosure namespaces, Memory state and checkpoints. A label disclosed in one coupled branch is unavailable to another branch unless that branch independently queries it. All three trajectories finish and seal before Stage B opens full outcomes. The trusted-local implementation does not claim isolation from malicious code running as the same account.

## Evaluation and temporal analysis

Stage B uses pre-event predictions and thresholds. For every registered readout it reports TP/TN/FP/FN, FPR, FNR, BER = 0.5 * (FPR + FNR), balanced accuracy, image AUROC and average precision; missing denominators remain NA. Category macros weight categories equally. Shared L/E AUROC and average precision must match exactly for each common G/GC score sequence or the campaign is invalid. Coupled score rankings are evaluated separately.

Temporal reports split each category by Source Group arrival order using `[0,floor(n/3))`, `[floor(n/3),floor(2n/3))`, `[floor(2n/3),n)`. Sliding windows are 50 consecutive groups (250 events), stride 10 groups. One-class window BER is NA with class counts and exclusions. Worst-window BER uses valid windows only; worst FPR and FNR are separate. Query rate, normal feedback, ESS, controller updates, rescoring, query deserts, threshold movement and native state diagnostics are retained. No inferential interval is preregistered; adaptive-trajectory bootstrap is not treated as an exact counterfactual policy interval.

## Resource and checkpoint feasibility

The frozen upper-bound model has 69 immutable snapshots across the three trajectories, 45,885 event encodings total, up to 1,171,243 exact E rescore invocations per E-bearing trajectory (S and CE), or 2,342,486 campaign-wide. Each cached feature is 784 x 4,096 FP32; capacity 1,024 is 13,153,337,344 bytes (about 12.25 GiB).

Summed checkpoint feature caches are 222,964,482,048 bytes. The non-cache allowance is 256 MiB/snapshot, or 18,522,046,464 bytes total. That models 241,486,528,512 bytes before reserve (about 224.9 GiB), and 301,858,160,640 bytes with the 25% reserve (about 281.1 GiB), plus blinded image resources and a separate 2 GiB ledger/output reserve. The per-snapshot allowance is an unmeasured ceiling that must be demonstrated by the required synthetic benchmark receipt. It is not an observed payload size or completed feasibility result.

The resource gate requires at least 32 GiB free host RAM, an NVIDIA GeForce RTX 3090 Ti with 24 GiB total VRAM, disk for the full estimate, measured synthetic native encoder/rescore timing and host/GPU peaks, and the checkpoint-payload allowance check. The RTX 3090 Ti was present for engineering tests. Full-dimension campaign resource benchmarking and intended-host storage feasibility remain open. Per-trajectory and whole-campaign wall limits are seven and 21 days.

## Validation matrix

| Check | Result |
|---|---|
| `py_compile` for all changed campaign/integration modules and the new test | PASS |
| `python -m experiments.calibration.fs4_joint_campaign freeze-only` | PASS; inert, 15,295 events / 3,059 groups; execution disabled |
| `python -m experiments.calibration.fs4_joint_campaign validate-contract` | PASS; exact S/CL/CE registry and 15,295-event population |
| `source_fingerprints()` | PASS; 63 required source fingerprints present |
| `pytest -q tests/calibration/test_fs4_joint_campaign.py` | PASS; 9 passed |
| `pytest -q tests/calibration/test_fs4_hardening_cuda.py tests/calibration/test_fs4_hardening_r2_cuda.py` | PASS; 6 passed on RTX 3090 Ti, synthetic CUDA only |
| `pytest -q tests/calibration/test_fs4_r4.py tests/calibration/test_fs4_research_execution.py tests/calibration/test_fs4_threshold.py` | PASS on final source commit; 42 passed, 4 Pillow deprecation warnings |
| `pytest -q tests/calibration/test_fs4_hardening_r2.py::test_production_rejects_preconstruction_encoder_substitution` | FAIL, preserved known high-security regression: expected `FS4BoundaryError`, but none was raised |
| Actual four-Support reconstruction/parity; real-data resource preflight; Stage A; Stage B | NOT RUN, as required by task authorization |

The first committed regression attempt exposed two compatibility issues. A follow-up source commit scoped the new Support receipt and FrozenTheta readout to the joint campaign and restored the legacy projection path; the complete 42-test regression suite then passed on the final source commit. The high-security substitution failure is separate and remains explicit; this freeze makes no malicious same-account security claim.

## Source and output protection audit

The final source freeze is `ac60c09b0078ba22fe7d6d88b847cb15439cf6ea`; implementation was introduced in `354711967bc05ad4232798bd25e6f4d3a5bedd07`. Starting HEAD matched the task. The source changes are confined to the listed calibration campaign and FS4 integration files. No v22, vmb, V0.2, P0/P1, R0/R1, D1 through D4 historical code, datasets, model weights, credentials, Real-IAD images, or historical outputs were committed. Pre-existing untracked audit/output directories were left untouched and unstaged. The tracked worktree was clean at the final source freeze. No push was performed; branch remains ahead of origin.

## Changed files

The implementation and compatibility-scope commits changed these 26 files:

- `docs/calibration/fs4_joint_campaign/ARM_REGISTRY.json`
- `docs/calibration/fs4_joint_campaign/CAMPAIGN_SPEC.json`
- `docs/calibration/fs4_joint_campaign/DATA_ROLES_AND_LABEL_FIREWALL.md`
- `docs/calibration/fs4_joint_campaign/ENGINEERING_AUTHORIZATION.md`
- `docs/calibration/fs4_joint_campaign/FINAL_INDEPENDENT_REVIEW_HANDOFF.md`
- `docs/calibration/fs4_joint_campaign/FINAL_PREREGISTRATION.md`
- `docs/calibration/fs4_joint_campaign/METRICS_AND_TEMPORAL_ANALYSIS.md`
- `docs/calibration/fs4_joint_campaign/QUERY_AND_FEEDBACK_CONTRACT.md`
- `docs/calibration/fs4_joint_campaign/RESOURCE_AND_CHECKPOINT_BUDGET.md`
- `docs/calibration/fs4_joint_campaign/SCIENTIFIC_ESTIMANDS.md`
- `docs/calibration/fs4_joint_campaign/SOURCE_MODEL_SUPPORT_FREEZE.md`
- `docs/calibration/fs4_joint_campaign/STAGE_A_STAGE_B_EXECUTION_PLAN.md`
- `docs/calibration/fs4_joint_campaign/SYNTHETIC_ACCEPTANCE_MATRIX.md`
- `experiments/calibration/fs4_durable_runtime.py`
- `experiments/calibration/fs4_joint_campaign.py`
- `experiments/calibration/fs4_joint_campaign_contract.py`
- `experiments/calibration/fs4_joint_campaign_evaluator.py`
- `experiments/calibration/fs4_joint_campaign_resources.py`
- `experiments/calibration/fs4_r2_context.py`
- `experiments/calibration/fs4_r4_blinded.py`
- `experiments/calibration/fs4_r4_mapping_verify.py`
- `experiments/calibration/fs4_r4_worker.py`
- `experiments/calibration/fs4_research_ledger.py`
- `experiments/calibration/fs4_research_stage_a_verify.py`
- `experiments/calibration/fs4_threshold_integration.py`
- `tests/calibration/test_fs4_joint_campaign.py`

This report is the sole artifact in the report-only commit.

## Open gates and documented research commands

Still required after independent review and separate campaign-bound authorization: real source manifest/category/group/view/count validation; R7 seed-0 order bijection and one-time blinded mapping; actual four-Support native bank, score, theta and Writer-sigma parity receipt; complete synthetic native resource benchmark and intended-host resource preflight; explicit Stage A authorization; complete three-arm Stage A seals; separate Stage B authorization and full label projection. No scientific result or hypothesis conclusion exists.

The following are documented in `STAGE_A_STAGE_B_EXECUTION_PLAN.md` and were **not executed**: `prepare-real-data`, `verify-support-preflight`, `resource-preflight`, `run-stage-a`, `verify-stage-a`, `run-stage-b`, and `verify-complete`. Each requires its separate authorization. This task did not stage real data, reconstruct actual Supports, access TEST outcomes, or execute scientific evaluation.

Limitations include four development categories, one query seed, Source Group correlation, no actual R7 receipt yet, no actual campaign-host resource benchmark, and the preserved adversarial same-account regression failure. The freeze records design and engineering checks only; it does not establish empirical superiority, ACI inheritance, coverage, finite-sample FPR control or external generalization.

FS4_FINAL_CAMPAIGN_FREEZE_READY_FOR_INDEPENDENT_REVIEW
