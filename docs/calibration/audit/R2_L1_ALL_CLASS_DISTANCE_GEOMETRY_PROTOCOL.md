# R2-L1 All-Class Distance Geometry Protocol

## 1. Status and authorization boundary

This is a **future protocol specification only**. No Real-IAD inventory, image, feature cache, class JSON, label, mask, private feedback, correction-anchor record, or model was read for this document. No all-class scan is authorized by it. The R2-L1 mathematical task authorizes only synthetic CPU work in Step 1; the controlled online Shadow Correction study (Step 2), the matched scientific evaluation (Step 3), and every real-data scan require a new researcher authorization.

All final radii, radius multipliers, (\lambda), and kNN distance cutoffs remain:

`UNSELECTED_PENDING_ALL_CLASS_GEOMETRY_AUDIT`

The target statistical universe is the **complete 30-category Real-IAD universe**. The four historical development categories are not the population definition. No category names or counts are asserted here because the inventory procedure has not been run.

## 2. Data roles and protection

Before any future data access, create a versioned partition manifest assigning each official source group to exactly one role. Role membership is disjoint at source-group level; where source groups cannot be verified, use a separately frozen dataset-level split. A source group, including its camera views, may not appear in more than one role.

### A. Parameter-design reference data

Purpose: estimate permitted feature geometry and, after separate authorization, describe online Correction geometry for method design. This role may include legally accessible training/reference groups from all 30 categories. If all 30 categories inform radius priors, they are **not unseen at the category level**. The scientific claim must state this. Parameter-design summaries may not use development evaluation labels or confirmation outcomes.

G1 may use frozen feature descriptors from this role. G2 requires queried feedback and actual causal Correction Anchors, and therefore requires a separately authorized online Shadow Correction run; G1 cannot stand in for G2. The current R1 private feedback and correction-anchor ledgers are not authorized inputs to this protocol specification and must not be read without explicit authorization.

### B. Development evaluation data

Purpose: later development-only outcome evaluation under a separately frozen scientific protocol. Its labels and masks are evaluator-only. They are not inputs to A's radius or distance statistics. The four previously exposed development categories remain exposed information; they cannot be described as unseen confirmation. If the same category occurs in A and B, the source groups must be disjoint and the category-level exposure must be disclosed.

### C. Sealed independent confirmation data

Purpose: independent confirmation after all method choices, preprocessing, source code, and analysis are frozen. No C group or outcome may influence parameter design. Preserve independence by a source-group or dataset-level holdout even if all 30 category identities were used in A. If protected classes cannot be accessed under existing permissions, mark their inventory, descriptors, and outcomes `PENDING_AUTHORIZATION`; do not omit them while claiming all-category coverage. Mint, PCB, and any other protected confirmation data remain sealed unless a new authorization explicitly permits access.

### Partition audit

The future run must verify that A, B, and C contain no overlapping source-group digest. It must also record which categories appear in each role. If all-category A geometry consumes groups that were intended for C, stop and repair the partition before calculating any statistic. Do not infer a holdout from filenames or directory layout alone.

## 3. Reproducible 30-category inventory procedure

This procedure runs only after the relevant dataset-access authorization is granted.

1. Record dataset release/version, approved root, official inventory metadata source, filesystem snapshot or manifest digest, and authorization identifier. Use only the authorized inventory source; do not guess category names from cached paths.
2. Enumerate the official category universe from that source and preserve the exact canonical category strings. Produce a sorted manifest of category name, official source-group count, image count, and role eligibility. Do not hard-code the four historical development categories as the scan population.
3. Verify exactly 30 distinct category identifiers, no duplicate aliases, no missing or extra category, and a valid source-group identity for every counted image. Any unrecognized alias or count disagreement blocks the scan pending reconciliation.
4. Separately flag categories/groups that are unavailable or protected. If access is not authorized, record the exact missing inventory rows as `PENDING_AUTHORIZATION`; the result is not an all-30 audit.
5. Record per-category totals for source groups and images, plus role assignments and their provenance. Seal the manifest before feature extraction or distance calculation. A later change in dataset release or inventory requires a new manifest and run identity.

The task calls for category names and counts to be verified during that future audit. They are deliberately not copied into this protocol because none were inspected here.

## 4. Geometry populations (do not conflate)

### G1: Background representation geometry

G1 measures distances between patch descriptors from source-disjoint query and reference images in the authorized A training/reference pool. The encoder, layer, preprocessing, patch grid, normalization, view policy, and descriptor selection rule must be frozen and hashed. A query descriptor may not compare against its own image, its official sample group, another view of that group, or any source group assigned to B/C.

G1 characterizes the frozen representation's background geometry. It can show category, viewpoint, or source heterogeneity and can provide descriptive distance scales. It does not measure the distribution of feedback-derived Signed Correction Anchors or their sign mix. It cannot establish online correction coverage or effectiveness.

### G2: Online Correction geometry

G2 measures distances from legally admitted, historical C1/C2 correction Anchors to **future** query patches. It requires a separately authorized causal online shadow run with an immutable stream order and query schedule, score-before-disclosure ordering, explicit queried-label permissions, and the official source-group exclusion. Only Anchors available before the query score may be considered. The current event's label, current/future anchors, same-source anchors, and protected confirmation groups are excluded.

G2 is essential for actual correction support coverage. It is not identifiable from G1 alone and must not be reconstructed by treating background descriptors as feedback Anchors. No G2 data will be opened or generated as part of this Step-1 task.

## 5. Required measurements

For each population and category where the relevant quantities are identifiable, report both query-image/group-level and patch-descriptor-level denominators.

### Distance and support coverage

- Nearest eligible-anchor cosine distance.
- (k)-th eligible-anchor cosine distance, with (k=5) reported for Signed kNN; mark unavailable when fewer than five eligible anchors exist.
- Distance divided by the registered candidate (R) or (R_j), with the raw distance retained. A Gaussian width (r_j) is not an effective support radius by definition.
- Fraction of query patches with zero eligible anchors; fraction with at least one eligible anchor; fraction with at least five.
- For each descriptive candidate support radius, fraction within at least one compact support and the distribution of the number of overlapping supports per query patch.
- Coverage curves by category and source group, not just one pooled scalar. Radius values used to draw a descriptive curve must be frozen before looking at development outcomes; they are not selected hyperparameters.

### C1- and C2-specific geometry

- C1: per-Anchor normalized distance and support overlap; separate C1-Gated and C1-Compact coverage. For C1-Gated, also report the distinction between output support and global legacy-mixture influence. Do not imply that gating removes far-anchor influence inside another Anchor's support.
- C2: nearest and fifth-nearest local-anchor distances, local neighbor count, zero-neighbor fraction, and cutoff coverage. Retain the original ungated Signed kNN as the same-neighbor-count strong reference; report its distance distribution even though it has no cutoff.
- G2 only: nearest positive- and negative-sign Anchor distances, queries with both signs inside support, sign-conflict/overlap counts, and positive/negative Anchor proximity. Do not infer signs for G1 reference descriptors.
- Where authorized and geometrically meaningful, report background-reference to correction-Anchor distances as a separate cross-population statistic. Keep G1 and G2 source-role metadata and denominators visible; do not pool them as if they were one population.
- Online G2 drift by frozen event-time bins, showing anchor age, position, and source group. Bin boundaries are chosen before results and may not depend on label outcomes.
- Category, source-group, and viewpoint summaries where metadata exists; missing viewpoint/source metadata is reported rather than imputed.

## 6. Per-category reporting and uncertainty

For each of the 30 inventory entries, report:

1. Number of source groups and images in the assigned role.
2. Number of selected query patch descriptors.
3. Number of reference descriptors or admitted Correction Anchors.
4. Number of eligible cross-source query/reference comparisons (and number excluded as self/same-source/current/future).
5. Nearest-distance quantiles and (k=5) distance quantiles, with the eligible-query denominator and the count with fewer than five candidates.
6. Support-coverage curves and zero/one/at-least-five local-evidence rates.
7. For G2, overlap multiplicity and positive/negative conflict measures.
8. Source-group-cluster uncertainty intervals and an explicit status: `ESTIMABLE`, `INSUFFICIENT_GROUPS`, `NO_ELIGIBLE_COMPARISONS`, or `PENDING_AUTHORIZATION`.

Report category-equal macro summaries separately from sample-weighted pooled summaries. The category-equal macro gives every category equal weight. The pooled summary weights each query unit according to a declared denominator (image, source group, or selected descriptor) and must name that denominator. Do not present a patch-weighted pooled number as category-equal evidence.

Resample source groups, not individual patches. In each bootstrap replicate, sample source groups with replacement within category and retain the sampled group's images/views/patches together. For cross-source comparisons, maintain the group exclusion in every replicate. A category with fewer than two eligible source groups cannot support a group-resampling interval and is `INSUFFICIENT_GROUPS`; a larger minimum needed for stable intervals must be preregistered after inventory but before seeing distance results. Report both point estimates and interval method, seed, replicate count, and effective group count. Millions of correlated patch distances are not millions of independent observations.

## 7. Sampling, exclusions, and computational design

### Deterministic sampling

If full descriptor enumeration is infeasible, sample by a deterministic, label-blind hierarchy: category, source group, image/view within group, then spatial patch location. Set quotas and seed before inspecting distances. Use hashed group/image/location identifiers as deterministic priorities (reservoir sampling is suitable), and record selected counts and inclusion weights. Do not prioritize high anomaly scores, feedback outcomes, or visually selected cases. If sampling rates differ, use known inclusion weights for pooled estimates and still show unweighted category-group summaries.

Sampling parameters are resource controls, not radii. Freeze them after the inventory and compute-cost estimate, before reading any distance value. If the sample misses a category or leaves too few independent groups, stop or report the insufficiency; do not silently change the population.

### Source disjointness

Build an explicit query-group/reference-group exclusion map. Remove all descriptors from the query's official source group, including multiple views. For G2 also enforce strict past position before source-disjoint filtering. Count each excluded comparison by reason. Any inability to establish source identity blocks the affected estimate.

### Streaming distance calculation

Do not allocate a full query-by-reference distance matrix. Normalize frozen descriptors once, process query and reference blocks with matrix multiplication, clip similarities to ([-1,1]), convert to cosine distances, and maintain per-query nearest/fifth-nearest heaps or top-(k) arrays. Accumulate support-coverage and overlap counters online for the preregistered radius grid. Store per-query summaries and provenance, not every pairwise distance. Use stable ordering `(distance, reference_index)` for ties. Validate with synthetic exact cases and record any approximate search recall if approximate neighbor search is separately authorized.

For category quantiles, retain the final per-query nearest and fifth-nearest values, then compute query-level quantiles. This storage is linear in sampled queries. Bootstrap by group from these grouped summaries. For exact support-overlap counts, stream all authorized references inside each query block and update counts without retaining the full matrix.

### Resource estimate (planning illustration, not an inventory claim)

Actual image/group counts and feature availability are unknown until an authorized metadata inventory. The following scenario is a sizing illustration only: 30 categories, 500 source groups per category, one image per group, 32 selected descriptors per image, descriptor dimension 768, and float16 stored descriptors.

- 480,000 descriptors; descriptor storage (480{,}000\times768\times2\) bytes = 737,280,000 bytes, about 0.69 GiB, before metadata and indexes.
- At an illustrative 20% query-group / 80% reference-group split, each category has (3{,}200\) query and (12{,}800\) reference descriptors, or 40.96 million cross-products; all categories total about 1.23 billion descriptor comparisons.
- Direct dot products require about (0.94\times10^{12}) multiply-accumulate operations (about (1.89\times10^{12}) scalar FLOPs if multiply and add count separately). Runtime depends on hardware, BLAS, and extraction; no runtime guarantee is made.
- A 64-by-2,048 float32 distance block occupies 0.5 MiB. Streaming nearest/fifth-nearest summaries avoids a full all-pairs distance artifact, which in this illustration would be about 4.6 GiB at float32.
- Keep raw feature caches in their separately governed location; do not copy them into an audit output. A planning allowance of 0.7 GiB for float16 descriptors plus up to 0.1 GiB for inventory, indexes, checksummed summaries, bootstrap output, and logs is a scenario budget, not a task limit. Float32 descriptors double the feature storage. If full spatial grids or more images/groups are used, scale using the formulas above before authorization.

If features are not already available under an authorized frozen representation, feature extraction adds storage and compute not included in this distance-only estimate. It would require its own explicit model/data execution authorization; this protocol does not authorize DINO/VisionAD loading.

## 8. Radius-selection strategies to compare (specification only)

No strategy or numerical parameter is selected here. If later reviewed, compare these under identical data roles and descriptive coverage criteria before any outcome evaluation.

| Strategy | Data needs | Stability and heterogeneity | Leakage / contamination risks | Outlier or mislabeled-Anchor sensitivity | Sparse-feedback and deployment limits |
|---|---|---|---|---|---|
| One global effective radius | Broad source-disjoint G1 and G2 distributions across all categories; category and group counts retained | Simple and stable with many groups, but a pooled radius can over-cover dense categories and under-cover sparse ones; category-equal and pooled curves may disagree | A radius fit on all 30 categories uses those category identities for design; category-level novelty is lost | An atypical category/group can move pooled quantiles and distort coverage elsewhere; report robust summaries but do not select them without preregistration | Easy to deploy, but one radius cannot adapt to per-Anchor density; a global coverage rate is not a per-category guarantee |
| Category-specific radii | Adequate legal reference geometry for every category | Can follow heterogeneity; high variance or undefined when a category has few independent groups; missing category cannot borrow evidence without an explicit rule | Requires reliable category identity at inference and risks category-specific holdout leakage if groups cross roles | A mislabeled or atypical group has greater influence in a small category sample; group-level diagnostics and minimum-data status are necessary | Sparse online anchors may not support stable per-category estimates; fallback behavior must be specified before use |
| Hierarchical/shrinkage radii | Global and category-level geometry plus a frozen shrinkage model | May trade global stability against local heterogeneity; shrinkage strength itself needs calibration and uncertainty reporting | Learning the hierarchy from B/C outcomes contaminates evaluation; all-30 A data removes unseen-category claims | Outliers can affect both category estimates and the shared prior; quantify influence by group without choosing a robustification rule post hoc | More complex state and versioning; sparse categories depend strongly on prior/global contribution |
| Per-Anchor adaptive radii | Local legal neighborhood distances for each Anchor, with source-group exclusions and causal cutoffs | Can reflect local density but is unstable with few neighbors and may react to anomalous/mislabeled Anchors | Future/self/same-source neighbors or confirmation data can leak into radius construction; validate per-Anchor provenance | An atypical/mislabeled Anchor may create an excessively broad or narrow local radius; influence controls require separate specification | Dynamic update, storage, and staleness rules are needed; sparse feedback can leave radii unavailable |

For each strategy, characterize coverage intervals and failure states without optimizing AUROC, AUPRC, BA, FPR, or other supervised outcomes. Report sensitivity to mislabeled/atypical reference items as a synthetic or separately authorized robustness analysis, not as evidence for a preferred value. No radius quantile, multiplier, (\lambda), or C2 cutoff may be selected because it improves the existing switch/seed0 tape.

## 9. Future execution sequence and authorization gates

1. Obtain separate written authorization for the exact dataset release, approved category inventory source, role-specific roots, and any protected-class access. Mint/PCB remain sealed unless named explicitly.
2. Produce and review the 30-category inventory and source-group partition manifest. If a protected class is not authorized, mark the all-category goal blocked/pending and do not relabel the scope as complete.
3. Freeze representation identity, descriptor selection, G1 sampling, G2 shadow stream/query schedule, source exclusions, group-bootstrap plan, candidate coverage curve construction, and resource budget. Confirm A/B/C disjointness.
4. Run G1 only on authorized A reference descriptors. G2 is a separately named, separately authorized online Shadow Correction data collection; it is not part of an inventory scan and must not access B/C outcomes.
5. Produce per-category and macro/pooled tables, group-resampling intervals, missing-data status, provenance hashes, and a non-overwriting output manifest. Have a researcher review the geometry and holdout impact before considering any parameter-design decision.
6. If radius priors are formed from all 30 categories, update the study claim to state category-level exposure and retain an untouched source-group/dataset-level confirmation set. Do not proceed to scientific outcome evaluation until a separate Step-3 authorization.

## 10. Required future artifacts and stop conditions

The authorized run should emit:

- Dataset release and exact 30-category inventory manifest with counts, role assignment, and source hash.
- Group-disjoint A/B/C partition manifest and protection status.
- Frozen feature/descriptor identity, layer, preprocessing, normalization, and patch-selection manifest.
- G1 and, if separately authorized, G2 query/reference counts, exclusions, nearest/kth distances, support curves, overlap/conflict statistics, time drift, and category-specific availability.
- Category-equal macro and clearly denominated sample-weighted pooled summaries.
- Source-group bootstrap method, seed, replicate count, interval outputs, and insufficient-category flags.
- Sampling-priority seed, selection counts, inclusion weights, chunk sizes, code/dependency versions, SHA-256 provenance, and resource accounting.
- `INCOMPLETE` status until all 30 inventory rows and all mandatory category reports are reconciled.

Stop before distance computation if category count is not 30, any name/count is unreconciled, source-group identity is missing, roles overlap, protected access is unresolved for an all-30 claim, representation identity changes, a query can match itself or a same-source group, future/current anchors enter G2, or output would require opening B/C labels. Report exact missing categories/counts from the authorized inventory manifest only; never infer or fabricate them.

This protocol does not itself authorize Step 2, Step 3, feature extraction, a Real-IAD scan, protected data access, or empirical parameter selection.
