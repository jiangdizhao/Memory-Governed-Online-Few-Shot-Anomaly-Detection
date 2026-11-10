# Gate 0 — Data and Information-Access Contract

**State:** `DRAFT_NOT_AUTHORIZED`. This document records permitted roles needed by a future separately authorized campaign; it does not authorize opening any data now.

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

## 3. Group rules and event timing

Real-IAD's existing E6 contract assumes five camera views per official Source Group, checks camera 1 through 5, checks Support/C64/TEST group disjointness, and suppresses a second query by an arm from a group already queried. No one query reveals the other four labels. Source Group identities are opaque in public event records.

The current event loop seals each event before a query, then reads and applies a queried label before the next event. It does not defer updates to a five-view group boundary. Thus an earlier queried view's verified feedback can alter the scorer/threshold state used on a later view from that same group. This is existing behavior, not a new rule endorsed here. Researcher decision is required whether to retain it or define a group barrier. The independent evaluation/bootstrap unit should not be the image view when multiple views share a Source Group.

Do not transplant Real-IAD's five-camera grouping rule to VisA. The tracked D3-P0 summary says no official five-view Source Group structure was established for VisA; physical groups, exact duplicates, and support/C64/TEST group disjointness must be defined from a permitted official identity source before any future access. The local D3 summary reports no cross-partition exact duplicates among hashed VisA candidates but explicitly says hashing was partial and does not establish full-dataset absence. It also states exact identical-file hashing alone would not prove physical independence.

## 4. Exposure and provenance register

| Evidence in inspected tracked text | What is established | What remains unknown / claim limit |
|---|---|---|
| `docs/calibration/audit/p0_benchmark_integrity_and_exposure_20261101/REALIAD_HISTORICAL_EXPOSURE.md` and `P0_AUDIT_DECISION.md` | A full 30-category historical Real-IAD table was reported. Four-category Stage 0.5/0.75 and Stage 1-A results, Audiojack/Probe-A hindsight analysis, and additional Switch engineering exposure are documented. The four development categories are exposed. | Exact image and Source Group intersections are unknown where only aggregate/digests are reported. Prior use for model selection is not fully established. Category names/digests alone do not prove independent sample identities. Do not call the 30 categories wholly unseen confirmation. |
| Same Real-IAD ledger | The tracked summary says Mint/PCB test outcomes and masks were not accessed in later named phases; original all-category paper results exposed category-level results. | Historical mask exposure and exact protected identity overlap remain `UNKNOWN`. This task did not open those outcomes, masks, images, or artifacts. |
| `docs/calibration/STAGE05_DATA_EXPOSURE_LEDGER.md` | Stage 0.5 accessed test outcomes after query/support/tape freeze for the four development categories; category-level outcome analysis is therefore exposed. It documents group counts and five views per group. | Exact group lists were not reopened. These artifacts do not establish identities for a new 30-category campaign. |
| `docs/calibration/audit/d3_p0_local_dataset_feasibility_20261105/REPORT.md` | Researcher-provided report summary says a local metadata/header feasibility audit covered VisA and MVTec AD 2, including partial exact duplicate hashing, without feature extraction or outcome evaluation. VisA 1cls TEST had a reported 2,162 events. | The full local receipts are absent from this tracked summary; exact duplicate lists were not independently re-examined. Project-specific historical VisA outcome exposure, original dataset release provenance, and physical group identities are not established. No identity may be inferred here. |
| `docs/calibration/B3_CALIBRATION_PAPER_RECOVERY_MASTER_PLAN_AND_PROGRESS.md` | C0 strict-four-shot and independent-confirmation authorization is explicitly `HOLD` / not authorized, requiring dataset permission, a data-access comparison, and model freeze. It distinguishes four Supports plus additional C64 from strict-four-total. | This task does not lift the HOLD. Existing 4+C64 results cannot answer a strict-four-total claim. |

No distinct external handoff/report for the proposed 11 November campaign was identified in the locally inspected tracked documents. The 0.01 Macro BER practical threshold is supplied in the current task context but its external handoff provenance and researcher/supervisor approval are not present in the inspected record.

## 5. Identity manifest requirements (schema only; identities pending)

Before any future Gate 1 data open, publish a checksummed manifest for every dataset/category/paired condition with at least:

- dataset release/version and provenance receipt; category; official split/role; official JSON/annotation file SHA-256 (without placing labels in public Stage A records);
- source image opaque identity and content SHA-256; Source Group digest; camera/view or dataset-specific physical-group relation; duplicate-candidate audit and disposition;
- role=`SUPPORT`, `C64`, or `TEST`; four Support and 64 C64 identity digests; role and group disjointness evidence; Support Seed and Stream Seed; event-order/tape digest and expected image-event count;
- frozen scorer/feature and preprocessing identities; query/cap/controller version; authorized label-reader identity and scope; which labels may be queried; Stage A seal and Stage B evaluator boundary;
- explicit `UNKNOWN` for any provenance, group relation, or overlap not certified by a permitted source.

The existing E6/C/D2 contracts are for four Real-IAD development categories and Support Seed 0 / Shuffle Seed 0. They do not supply verified 30-category Real-IAD or 12-category VisA identities, P1/P2 role assignments, or new hard-cap tapes. No actual sample IDs or hashes were created or read for this draft.
