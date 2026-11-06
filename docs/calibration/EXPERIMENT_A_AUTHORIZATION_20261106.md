# Experiment A scoped authorization — 2026-11-06

## Authority and scope

The research lead has explicitly authorized Experiment A as a scoped follow-up to the historical Stage 0 work. This authorization supersedes the Stage-0-only execution restriction **only for Experiment A** and does not replace or rewrite `STAGE0_RESEARCH_CONTRACT.md`. Experiment A is a development study, not independent confirmation.

The permitted experiment components are A1 (initial operating point), A2 (readout under shared trajectories), A3a (calibration-coupled learning), and A3b (equal-storage expanded reference). Execution must follow the frozen Experiment A request and preserve its estimands, conditions, support identities, C64, nested calibration pools, query schedule, scorer, memory and Correction configurations, metrics, and resource limits. No new algorithm, hyperparameter search, query policy, or dataset may be introduced.

## Permitted categories and data access

Only these previously exposed development categories are authorized:

- `audiojack`
- `regulator`
- `switch`
- `rolled_strip_base`

Within those categories, permitted access is:

- TRAIN-normal metadata and images for Support, calibration, H, and reference construction.
- Existing verified Support and historical C64 identities.
- Additional legitimate TRAIN-normal images for C128/Cmax and the registered reference-building pool, subject to independently verified provenance and source-group disjointness.
- TEST images and label-blind event identities for prospective scoring.
- A TEST label only after the registered query action and only for the legal online feedback path.
- Unqueried TEST labels only inside the isolated Stage B evaluator after that branch's Stage A predictions have been sealed and integrity-checked.

Additional TRAIN-normal access in this study is a separate authorization. It must not be inferred from or conflated with the historical metadata-only H-SOURCE authorization. Verify training split, normal status, official sample/group identity, and provenance independently. Preserve the distinction between four deployed supports plus **additional** C64 calibration normals and strict four-shot-total.

## Prohibited access and use

Do not access:

- Confirmation categories or outcomes, including `mint` and `pcb`.
- Any other Real-IAD category.
- Ground-truth segmentation masks.
- Unqueried outcomes in online scorer, memory, writer, correction, query, or calibration state.
- Future outcome information, hidden class histograms, or any protected data outside the four authorized development categories.

Do not use TEST outcomes to choose a threshold, calibration size, support/reference selection, or algorithm configuration. H is a normal-only development audit subset, not an independent test or production-representative sample. Do not claim confirmation, population-risk guarantees, or independent validation from this study.

## Causal and state protocol

Preserve the exact event order:

1. Score and predict using the pre-event state.
2. Apply the already registered, label-blind query-tape decision.
3. Reveal a label only when that query is executed.
4. Update only the state permitted by that disclosed label and the registered writer/correction rules.
5. Seal Stage A predictions and provenance before Stage B accesses unqueried outcomes.

Calibration-isolated A2 variants must share the registered native score trajectory, query IDs, and common writer state. A3a may allow calibration to affect legal memory and Correction updates, but its divergent trajectories must be separately named, frozen, hashed, and evaluated as a distinct profile. The registered fixed tape is not BEA and may not be changed based on outcomes.

## Engineering boundary

All new Experiment A source, tests, reports, and outputs must use the isolated namespaces specified by the task. Preserve historical runners, source, manifests, outputs, and unrelated untracked work. Use unique non-overwriting output directories, manifest-lock and checksum artifacts, and mark partial or smoke outputs `INCOMPLETE`. Do not stage unrelated files or push this work without separate explicit authorization.
