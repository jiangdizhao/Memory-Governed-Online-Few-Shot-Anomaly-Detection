# Experiment A4 scoped authorization — A4_EB68_v1

**Authorization date:** 2026-11-07
**Parent:** `84e434b7b89e892a55ad610408cd11161f7de3a5`
**Study:** Previously exposed four-category Real-IAD development study; not independent confirmation.

## Data rights and boundary

This document records the research lead's explicit authorization for one integrated Experiment A4 campaign. It extends Experiment A3b only for the registered equal-budget comparison. The permitted categories are `audiojack`, `regulator`, `switch`, and `rolled_strip_base`; the only condition is P0 (Support seed 0, group-shuffle seed 0).

Stage A may read TRAIN-normal images from the exact A1 seed-0 Support-4 and historical C64 identities, TEST pixels and label-blind event metadata for these four categories, and queried TEST labels only through the registered Q-A-TS-v1 disclosure gate. It may score the four frozen methods, construct native fixed banks, use the original native Greedy writer, and update the registered empirical q90 B3 states. Stage B may access TEST outcome labels only after all four complete Stage A category seals validate. TEST masks are prohibited.

No other TRAIN-normal image may be used for model construction. No other category, confirmation outcome, Experiment B/C, new query policy, Signed Correction, BEA, parameter search, online backbone training, hidden TEST histogram, unqueried online label, or pixel mask is authorized. Historical Stage 1/Stage 2 programs remain outside this scope.

## Frozen identity and budget accounting

For every category the permitted candidate universe is exactly the A1 seed-0 Support set `S4` plus the exact A1 historical additional C64 set. The sets contain 4 and 64 unique TRAIN-normal image identities from distinct official Source Groups, respectively; their union is 68. All are disjoint from TEST Source Groups. The A1 draw-0 C32 membership partitions C64; `R32` is its exact complement. No TEST outcome, score, prevalence, image difficulty, or later model output may influence that partition.

All methods use a realized fixed normal bank of 8,192 × 4,096 fp16 prototypes (67,108,864 bytes). A/B share the exact native Support-0 bank; C uses A3b K-Center on Support augmentation descriptors; D uses the same K-Center implementation on Support plus R32 descriptors with the frozen 65,536-candidate group-balanced cap. C32 is excluded from every fixed bank. A is Support-4 plus additional C64 calibration; it is not strict four-shot total. Equal fixed-bank bytes do not imply equal total system memory.

## Frozen method matrix and causal order

- A: native Support-4 bank; B3-C64.
- B: the same bank and the same writer/memory trajectory as A; B3-C32.
- C: K-Center Support-4 bank; B3-C32.
- D: K-Center Support-4 plus complementary R32 bank; B3-C32.

All methods use native v22/VisionAD scoring, K=1024 original native Greedy memory, the same Q-A-TS-v1 P0 query identities, and the A4 frozen C32 M0 writer reference for each distinct fixed bank. The writer reference does not use an arm's live B3 threshold. Only queried verified normals can update Greedy. Each event is scored and predicted before its query action; the score/decision row is fsynced before any queried label is disclosed; threshold refreshes apply only to later events. A/B share one exact score and writer state trajectory. C and D are separately identified calibration-coupled trajectories because their fixed banks differ.

Stage B is a separate CPU-only evaluator and cannot start until every category Stage A seal is complete and verified. The full 68-image comparison is a development sensitivity study and does not support independent confirmation, population-risk, or universal superiority claims.

## Stopping conditions

Stop affected execution on a data-role mismatch, historical receipt/hash mismatch, source or model mismatch, invalid native score/update parity, causal/query-gate violation, failed category seal, less than 30 GiB free on a required volume, category deadline over 120 minutes, campaign deadline over 8 hours, output-retention limit above 4 GiB, or any protected-file modification. Do not substitute identities, alter the candidate/prototype budget, access another category, or relax a gate. Partial/smoke artifacts remain `INCOMPLETE`.

## Git and output scope

Run on `calibration-stage0-contract-20261029` without switching/resetting/cleaning the worktree or pushing. The authorized additions are the A4 source, A4 tests, this document, the minimal A4-specific `AGENTS.md` exception, and new A4 report/run namespaces. Preserve all historical source, research documents, and outputs. Freeze the executable source in a local commit before the full empirical run; after execution, commit only verified reports, figures, and decisions.
