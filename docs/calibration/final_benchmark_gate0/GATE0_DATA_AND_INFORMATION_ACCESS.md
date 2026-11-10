# Gate 0 — Data and Information-Access Contract

**State:** `DRAFT_NOT_AUTHORIZED`. R1 freezes the proposed metadata-only role-selection rule, while actual identity generation remains a separately authorized Gate 1 operation. Nothing here authorizes opening data now.

## 1. Dataset scope and role table

| Dataset / scope | Proposed stream and target | Predeployment normal roles | Online label boundary | Identity/exposure state |
|---|---|---|---|---|
| Real-IAD, all 30 official categories | Official TEST images; `rho=0.03`; proposed category cap `floor(.03*T_c)` | 4 verified normal Supports plus 64 additional disjoint normal C64 images | Only a label for the queried image may be disclosed after its prediction/query row is sealed. Unqueried labels are Stage B only after all Stage A seals. No mask access. | Project-level category/population outcomes are historically exposed; exact identity overlap is unknown. Not wholly unseen confirmation. |
| VisA, all 12 official categories | Official `1cls` TEST only; `rho=0.10`; proposed cap `floor(.10*T_c)` | Proposed same explicit four Support + separate 64 C64 roles only if identity and data-role feasibility are verified. Do not import unused TRAIN-normal images into the stream. | Same query-only disclosure and post-seal Stage B boundary. No mask access is needed for the registered image-level metrics. | Local D3-P0 document reports metadata/header audit, 2,162 1cls TEST events, and partial duplicate hashing only. Project outcome exposure, official acquisition provenance, physical grouping and full duplicate/source relationships are unverified. |

`T_c` is the exact count of official TEST image events in the frozen category manifest, not the number of groups or the number of eligible queries. It is not to be copied from planning estimates. The cap is per category and per independently evolving querying arm; N-shadow inherits F-B opportunities and has zero query authority of its own. Source Group eligibility is an additional constraint and may leave the cap unused.

## 2. Role separation and permitted information

| Role | Permitted use | Prohibited use |
|---|---|---|
| Four Support normals | Build that paired condition's immutable native Fixed Bank under frozen augmentation/sampling. The four deployed images and group digests are predeployment inputs. | They do not establish a strict-four-total experiment when C64 is also available. Do not choose among a larger pool by TEST outcomes. |
| C64 commissioning normals | Separate 64-image, role-disjoint normal calibration set for initial/current-state q90/B3 and only explicitly registered normal commissioning quantities. Cache frozen features read-only; rescore under each arm's own scorer state when state-consistent B3 requires it. | Never append C64 to Fixed Bank, Greedy normal memory, or Signed correction memory. No use of TEST labels to choose the q90 or threshold. |
| TEST image before query | Current image bytes and frozen encoded descriptors only; arm-specific pre-event scorer state; opaque image/group identity; prior legally acquired feedback receipts. | No current image label, future label, mask, hidden class histogram, or another arm's private feedback. |
| Executed queried TEST image | Its official image-level label only after score and decision are durably sealed and query-specific authorization succeeds. | No label transfer to arms that did not query that image, except the explicitly defined N-shadow inheritance from F-B. No automatic disclosure for other views of the same group. |
| Unqueried TEST image | Label remains unavailable to all online state. After Stage A seals, Stage B may access official image-level labels for the preregistered evaluator. | No unqueried label or mask in query, correction, memory, threshold, or stopping decisions. |
| Masks | None required for the registered image-level outcomes. | No pixel masks or protected Mint/PCB outcome access in Gate 0 or online control. |

All four principal arms receive the same Support/C64 identities inside a paired condition, but retain arm-specific mutable state. Each arm has its own one-query-per-Source-Group set and private disclosure record. The N-shadow's query ledger is an explicit read-only view of F-B's executed opportunities; it has no query decision or feedback influence on F-B.

## 3. Gate 0 role selection; Gate 1 identity records

The Gate 0 candidate selection algorithm uses metadata only. After separate Gate 1 authorization, enumerate official TRAIN-normal image metadata and authoritative group keys. Do not read candidate pixels, compute features, or rank by anomaly scores to choose roles.

For each dataset release, category and Support Seed `s`, rank eligible official physical groups by ascending SHA-256 of UTF-8:

```text
Gate0-role-v1\0<dataset_release_sha256>\0<category>\0<s_decimal>\0<official_group_key>
```

Within each ranked group, select one officially normal TRAIN image by ascending SHA-256 of:

```text
Gate0-image-v1\0<dataset_release_sha256>\0<category>\0<s_decimal>\0<official_group_key>\0<official_image_key>
```

Use the canonical NFC UTF-8 official keys, lowercase release digest, unsigned decimal seed, SHA-256 digest comparison and tie rules in the Scientific Contract. The first four distinct groups/images are Supports; the next 64 are C64. Require at least 68 eligible distinct groups and verify disjointness from TEST groups; otherwise fail closed rather than silently broadening the pool. Support Seed 0 is reused by P0 and P1, so their exact Supports/C64 roles are identical. Support Seed 1 selects a separate role set for P2. Stream Seed does not change role selection. Feature extraction is allowed only after the 68 selected identities are fixed; feature inspection of a larger candidate pool is additional predeployment information and is excluded from this protocol.

This is a new deterministic protocol candidate, not an assertion that historical P0 Support construction used the same ranking. No historical image IDs are guessed. Do not claim historical event/score parity unless later identity/source verification establishes exact equivalence.

Actual image/group hashes, role membership, annotation checksums, and manifest digests are Gate 1 metadata artifacts. Their absence does not prevent Gate 0 design review. Gate 1 requires a separate researcher authorization for exact roots/metadata fields and review of the resulting checksummed manifests; this R1 task does not authorize that operation.

## 4. Group rules and event timing

Real-IAD's existing E6 contract assumes five camera views per official Source Group and checks camera 1 through 5, Support/C64/TEST group disjointness, and one query per group. R1 recommends **Group-atomic feedback** as a new protocol: use a group-major stream with all five views contiguous in camera order 1–5; score and seal all views from one pre-group state; permit at most one query per arm/group; after group sealing account for event-level actions, disclose only the purchased view's label, and update state for the next group. Missing, duplicate, incomplete, or noncontiguous membership fails closed. This is `RECOMMENDED_RESEARCHER_DECISION_REQUIRED` and is not historical R7/D2/B/C event parity.

Rank eligible official TEST groups by ascending SHA-256 of UTF-8 `Gate0-stream-v1\0<dataset_release_sha256>\0<category>\0<stream_seed_decimal>\0<official_group_key>`; within each group retain registered camera order 1–5. P0 and P2 share Stream-0 order; P1 uses Stream Seed 11. The eventual event digest binds dataset release, category, stream seed, group digest, camera/image identity and within-group position. Only opaque digests appear in public score records.

Do not transplant Real-IAD's five-camera grouping rule to VisA. The tracked D3-P0 summary says no official five-view Source Group structure was established; physical groups, complete duplicate relationships, and Support/C64/TEST group disjointness remain uncertified. Use an official VisA object/group key if documented. If none exists, the researcher must approve one explicit exception: a complete identity/duplicate-cluster query unit with no physical-independence claim, or no online VisA feedback. No image-as-independent-physical-group assumption is silent. The local D3 summary found no cross-partition exact duplicates only among partially hashed candidates; that does not establish full-dataset absence. Exact byte identity also does not establish physical-object identity.

## 5. Exposure, terminology, and provenance register

| Evidence in inspected tracked text | What is established | What remains unknown / claim limit |
|---|---|---|
| `docs/calibration/audit/p0_benchmark_integrity_and_exposure_20261101/REALIAD_HISTORICAL_EXPOSURE.md` and `P0_AUDIT_DECISION.md` | A full 30-category historical Real-IAD table was reported. Four-category Stage 0.5/0.75 and Stage 1-A results, Audiojack/Probe-A hindsight analysis, and additional Switch engineering exposure are documented. The four development categories are exposed. | Exact image and Source Group intersections are unknown where only aggregate/digests are reported. Prior use for model selection is not fully established. Category names/digests alone do not prove independent sample identities. Do not call the 30 categories wholly unseen confirmation. |
| Same Real-IAD ledger | The tracked summary says Mint/PCB test outcomes and masks were not accessed in later named phases; original all-category paper results exposed category-level results. | Historical mask exposure and exact protected identity overlap remain `UNKNOWN`. This task did not open those outcomes, masks, images, or artifacts. |
| `docs/calibration/STAGE05_DATA_EXPOSURE_LEDGER.md` | Stage 0.5 accessed test outcomes after query/support/tape freeze for the four development categories; category-level outcome analysis is therefore exposed. It documents group counts and five views per group. | Exact group lists were not reopened. These artifacts do not establish identities for a new 30-category campaign. |
| `docs/calibration/audit/d3_p0_local_dataset_feasibility_20261105/REPORT.md` | Researcher-provided report summary says a local metadata/header feasibility audit covered VisA and MVTec AD 2, including partial exact duplicate hashing, without feature extraction or outcome evaluation. VisA 1cls TEST had a reported 2,162 events. | The full local receipts are absent from this tracked summary; exact duplicate lists were not independently re-examined. Project-specific historical VisA outcome exposure, original dataset release provenance, and physical group identities are not established. No identity may be inferred here. |
| `docs/calibration/B3_CALIBRATION_PAPER_RECOVERY_MASTER_PLAN_AND_PROGRESS.md` | C0 strict-four-shot and independent-confirmation authorization is explicitly `HOLD` / not authorized, requiring dataset permission, a data-access comparison, and model freeze. It distinguishes four Supports plus additional C64 from strict-four-total. | This task does not lift the HOLD. Existing 4+C64 results cannot answer a strict-four-total claim. |

No distinct external handoff/report for the proposed 11 November campaign was identified in the locally inspected tracked documents. The 0.01 Macro BER practical threshold is supplied in the current task context but its external handoff provenance and researcher/supervisor approval are not present in the inspected record.

Recommended terminology for review: **Real-IAD — full-category benchmark reassessment on a historically exposed dataset. VisA — cross-dataset external validation, with independent-confirmation status conditional on exposure and identity/provenance certification.** Do not describe VisA as completely untouched without supporting evidence. Historical absence of a recorded outcome read is not proof of independent data exposure. Protected Mint/PCB outcomes and masks remain under their existing authorization boundaries.

## 6. Gate 1 identity-manifest schema (identities pending)

Before any future Gate 1 data open, publish a checksummed manifest for every dataset/category/paired condition with at least:

- dataset release/version and provenance receipt; category; official split/role; official JSON/annotation file SHA-256 (without placing labels in public Stage A records);
- source image opaque identity and content SHA-256; Source Group/object-group digest; camera/view or dataset-specific physical-group relation; duplicate-candidate audit and disposition;
- role=`SUPPORT`, `C64`, or `TEST`; four Support and 64 C64 identity digests; role and group disjointness evidence; Support Seed and Stream Seed; event-order/tape digest and expected image-event count;
- frozen scorer/feature and preprocessing identities; query/cap/controller version; authorized label-reader identity and scope; which labels may be queried; Stage A seal and Stage B evaluator boundary;
- explicit `UNKNOWN` for any provenance, group relation, or overlap not certified by a permitted source.

The existing E6/C/D2 contracts are for four Real-IAD development categories and Support Seed 0 / Shuffle Seed 0. They do not supply authenticated 30-category Real-IAD or 12-category VisA identities, P1/P2 final manifests, or hard-cap trajectories. Gate 0 freezes the selection, grouping, order, and checksum schema only. No actual sample IDs or hashes were created or read for this draft; Gate 1 remains separately unauthorized.
