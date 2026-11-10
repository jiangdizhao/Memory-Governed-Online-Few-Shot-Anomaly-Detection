# Gate 0 — Scientific Contract Draft

**State:** `DRAFT_NOT_AUTHORIZED` · R1 recommendations await researcher and ChatGPT review
**Planning date:** 2026-11-10. The proposed 11 November campaign has not been approved or executed.

## 1. Scientific scope and prior evidence

The proposed campaign compares four independently evolving online systems on Real-IAD and VisA. It asks whether verified defect feedback adds value beyond normal-only correction; whether simplified BN+Band acquires useful evidence relative to a Margin policy at a comparable realized budget; and whether Full Gaussian C1 is worth its added complexity relative to Signed kNN when both use BN+Band.

The task brief reports that prior Experiment B found incremental discrimination from queried defect feedback on four exposed Real-IAD development categories, while Signed kNN remained competitive with Full C1. It reports Experiment C macro BER 0.1696 for BN+Band at 523 queries and 0.1969 for Margin37 at 522 queries in its four-category development setting. These are prior development findings supplied by the researcher; this audit did not reopen result artifacts or independently recalculate them. They do not establish equal per-category counts, independent generalization, or the proposed 3%/10% capped policy.

Real-IAD has extensive project-level historical exposure, including a reported 30-category result. The campaign must be called a full benchmark reassessment or development/replication study unless a separate identity-level holdout and exposure audit supports a narrower independent claim. VisA project-specific performance exposure and duplicate/source identity remain unknown.

## 2. Principal system matrix

| ID | Correction | Query policy | Role |
|---|---|---|---|
| F-B | Full Gaussian C1 | Simplified BN+Band | Principal system |
| K-B | Signed kNN | Simplified BN+Band | Principal system |
| F-M | Full Gaussian C1 | `Margin_rho` proposal, with dataset-specific `rho` | Principal system |
| K-M | Signed kNN | `Margin_rho` proposal, with dataset-specific `rho` | Principal system |
| N-shadow | C1 normal-only correction | No autonomous policy; inherits F-B legal query opportunities and exact normal-writer inputs | Matched-feedback diagnostic |

The four principal arms share the frozen visual substrate, the same Support/C64 identities, stream order, and permitted predeployment information within each paired category/condition. Each owns its query, Greedy normal-memory, correction, C64/B3 threshold, and controller state. Distinct legally acquired query identities are expected and must be preserved. A shared immutable encoded image may be reused only under a verified feature identity; mutable state is never shared across principal arms.

N-shadow is a passive matched-feedback mechanism control, not a fifth independently querying system. For each group it receives F-B's exact executed query opportunity and legally disclosed label. It reads the same pre-event F-B Greedy Normal Memory state, frozen image features, and base C0 scoring inputs; keeps its own initially empty normal-only C1 state; and adds a negative C1 anchor only after verified-normal feedback. It never adds positive anchors, requests labels, or influences F-B. A read-only reference to F-B's Greedy state is preferred over a duplicate writer. It scores before current feedback can affect it. Its primary outputs are prequential anomaly scores for AUROC/AP; an independent B3 threshold is unnecessary unless a separately named thresholded metric is approved. The source-compatible update uses Experiment B N's registered common C0 evidence inputs. Its implementation remains a Gate 1 source-review requirement, not an implemented feature.

## 3. Questions, estimands, and interpretation

For each category and condition, compute paired outcomes and contrasts. Average the three condition-level contrasts within category, then give every category equal weight within each dataset. Report every condition and category separately. Real-IAD is the primary confirmatory population for H_Q; VisA is a separate cross-dataset external validation population, not pooled with Real-IAD. The interval and final decision rule are recommendations pending review.

- **Primary confirmatory question — evidence acquisition:** `H_Q = BER(F-M) - BER(F-B)`. Positive favors BN+Band under comparable supervision expenditure. Test only this preregistered primary contrast on Real-IAD; do not promote another contrast after seeing results. Use the recommended paired category bootstrap described in the metrics contract. VisA's H_Q is reported separately as external validation.
- **Supporting mechanism question — verified defect feedback:** `H_D = AUROC(F-B) - AUROC(N-shadow)`. Positive favors F-B. It estimates the conditional effect of enabling defect-side correction under matched F-B query and normal-memory opportunities. It may include effects mediated by C1-bank occupancy, replacement and competition; it does not isolate defect information independently of correction-state interactions.
- **Transferability question:** `H_Q_transfer = BER(K-M) - BER(K-B)`. Positive favors BN+Band. This is an independently evolving whole-policy contrast: query identities and later states may differ.
- **Architecture decision:** `H_C = BER(K-B) - BER(F-B)`. Positive favors Full C1 on BER. This is a complete-system comparison under the same named BN+Band policy, not a feedback-matched pure architecture effect. If K-B is competitive within approved practical and resource margins, the simpler scorer may be preferable. F-B is not required to defeat K-B.

Report each contrast with its sign convention and uncertainty. Keep Real-IAD and VisA separate. A score/algorithm comparison is not an independent confirmation claim; a backbone/supervision-mismatched external comparison is descriptive system evidence, not pure causal evidence.

## 4. Dataset conditions and common initialization

- **Real-IAD:** all 30 official categories; official TEST image stream; proposed `rho=0.03` and per-category cap `floor(0.03*T_c)`.
- **VisA:** all 12 categories; official `1cls` TEST only; no unused TRAIN-normal images in the stream; proposed `rho=0.10` and cap `floor(0.10*T_c)`. It is a short-stream external validation condition, not evidence that 10% VisA feedback transfers to 3% long-stream Real-IAD. No forced query is added at stream start.
- **Common normal information:** four verified normal deployed Supports plus 64 additional, disjoint, verified normal C64 commissioning images. This is not strict-four-shot-total. C64 can support initial/current-state calibration and registered controller quantities, but cannot enter Fixed Bank, Greedy writer, or Signed correction memory. Any broader normal-pool selection or feature-based search is a separate, explicitly described information allocation.
- **Paired conditions:** P0 `(Support Seed 0, Stream Seed 0)`; P1 `(Support Seed 0, Stream Seed 11)`; P2 `(Support Seed 1, Stream Seed 0)`. P0/P1 use the same Support and C64 roles; P2 reselects both using Support Seed 1. Stream Seed changes only group order. These are design conditions, not authenticated identity manifests. Existing E6/C establishes only four exposed Real-IAD categories at Support Seed 0 / Shuffle Seed 0; the new deterministic selection/order rule below does not claim historical identity or score parity.

### Deterministic Support/C64 candidate-selection rule

After a separately authorized Gate 1 metadata operation, enumerate only official TRAIN-normal metadata and official group keys; do not read candidate image pixels or extract features to choose roles. For each dataset release, category and Support Seed `s`, rank eligible official physical groups by the ascending SHA-256 of the UTF-8 string:

```text
Gate0-role-v1\0<dataset_release_sha256>\0<category>\0<s_decimal>\0<official_group_key>
```

Use the official group key only locally; publish its opaque digest. Require one or more officially verified normal TRAIN images in each selected group. Within each selected group, choose one normal image by ascending SHA-256 of:

```text
Gate0-image-v1\0<dataset_release_sha256>\0<category>\0<s_decimal>\0<official_group_key>\0<official_image_key>
```

Encode official keys as Unicode NFC UTF-8 without path normalization; encode the release digest as lowercase 64-character hex and seeds as unsigned decimal without leading zeroes. Compare SHA-256 digests as lowercase ASCII hex, then break ties by the normalized official key's UTF-8 bytes. The first four distinct ranked groups/images are deployed Supports; the next 64 are C64. Fail closed if 68 eligible groups are not available or if any selected group overlaps TEST. Support/C64 role selection is independent of stream seed, model features, outcomes, and performance. P0/P1 therefore share roles; P2 uses a separate Support/C64 draw. Extract features only after the 68 identities are frozen. This is a new selection protocol; historical event/score parity is not claimed unless later evidence proves exact identity and source equivalence.

For stream order, rank eligible official TEST groups by ascending SHA-256 of the UTF-8 string `Gate0-stream-v1\0<dataset_release_sha256>\0<category>\0<stream_seed_decimal>\0<official_group_key>` using the same NFC key and digest rules. For Real-IAD, keep each complete five-view group contiguous and order its views by registered camera number 1–5. P0/P2 share group order; P1 uses Stream Seed 11. The exact identity list and resulting checksums are Gate 1 artifacts, not Gate 0 deliverables.

For VisA, use an official, independently documented physical-group key if the release provides one. The current local record does not establish it. If absent, the researcher must explicitly approve an image-level/duplicate-cluster exception with weakened physical-independence claims, or remove online VisA feedback; no silent five-camera or image-as-independent-group assumption is permitted. This dataset-specific group-key decision remains `RESEARCHER_DECISION_REQUIRED`.

## 5. Candidate query policy and the cap intervention

The historical simplified BN+Band objective is `U_t=B_t+N_t+D_band,t`, using the registered risk/uncertainty, normal novelty, and predicted-normal asymmetric band terms. Preserve the E6/C utility and its price parameters: initial price 0.80, learning step 0.02, up-to-256 actual-action history, bounds `[0.05,3.0]`, strict `objective > price`. Target `rho` is 0.03 for Real-IAD and 0.10 for VisA; the latter target and the hard cap are new interventions, not historical E6/D2 results.

For each arm/category, distinguish `proposed`, `executed`, `Source Group suppressed`, and `hard-cap suppressed`. At each view, evaluate the proposal from the arm's current pre-group detector and policy state. Within a group, the first proposal that is not group- or cap-suppressed executes and reserves that group's eligibility. Give group-suppression precedence in the recorded reason; if the category count is already at `floor(rho*T_c)`, record cap suppression. Increment query count only for an executed label. No top-up or post-hoc action is allowed. Identical caps do not guarantee identical realized query counts. Candidate reporting criterion: pairwise realized query-rate differences must be at most 0.003 both per category and dataset macro to be called rate-matched; this is a reporting proposal, not an approved guarantee.

BEA's 256-action history uses final executed actions, not proposals. For each group, after every view row has been sealed, process the view action flags in registered view order (one `1` at most; all other views, including suppressed proposals, are `0`). Apply the historical recurrence `cost_next=clip(cost+0.02*(mean(last<=256 executed_flags)-rho),0.05,3.0)` after each event flag. All proposals in the same group use the pre-group price; price changes take effect in the next group. This preserves event-level action accounting without within-group policy changes.

Candidate `Margin_rho` is `READY_FOR_RESEARCHER_REVIEW`. Let `T_c` be official image-event count and define `L_c=min(64,max(16,floor(0.2*T_c)))`. For the first `L_c` image events, propose iff the identity-hashed uniform value `u` is `<rho`. Once the group containing event `L_c` is complete, later groups use the past-only empirical `1-rho` quantile of the preceding at most 256 observed standardized-margin utilities `psi=exp(-0.5*m^2)`, calculated in float64 with the registered NumPy linear quantile method (method `linear`, equivalent to Hyndman–Fan type 7). The current event never enters history before its decision. Compare exactly in float64: `psi>q` proposes, `psi<q` does not; at `psi==q`, propose iff the identity-hashed tie uniform value is `<p_tie`. Given `n` prior utilities, `n_gt` strictly above q and `n_eq` equal to q, use `p_tie=clip((rho*n-n_gt)/n_eq,0,1)` (zero when `n_eq=0`). Append every view's pre-group utility once after the entire group is scored/sealed, regardless of proposal, query execution, group suppression, or cap suppression. If a group straddles `L_c`, keep the whole group in cold-start mode; quantile mode starts at the next group. Pin the NumPy version and quantile method at Gate 1.

For either domain, the deterministic uniform value is the first eight bytes of SHA-256 interpreted as an unsigned big-endian integer, divided by `2^64`. Hash UTF-8 bytes of:

```text
Gate0-Margin-rho-v1\0<dataset_id>\0<category>\0<arm_id>\0<support_seed_decimal>\0<stream_seed_decimal>\0<event_id_sha256_lowercase>\0<cold-start|tie>
```

Apply the same function independently for each arm/category/paired condition; arm, category, dataset, both seeds, event digest and domain distinguish draws. Suppressions affect neither the fact that an image utility was observed nor the utility history; only an executed query increments query count. The group/cap rules then determine executed actions. This is not historical MARGIN37 (3.7%, 64-event cold start, q=.963). No forced query is introduced. The exact candidate remains subject to researcher approval.

Canonical `dataset_id` values are `realiad` and `visa_1cls`; category and official keys use the NFC UTF-8 form defined above, arm IDs are exactly `F-M` or `K-M`, and seed fields are unsigned decimal without leading zeroes. Define `event_id_sha256` as lowercase hex SHA-256 of UTF-8 `Gate0-event-v1\0<dataset_release_sha256>\0<category>\0<stream_seed_decimal>\0<official_group_key>\0<camera_id_or_dataset_view_key>\0<official_image_key>\0<within_group_ordinal_decimal>`, with the same canonical key encoding. This binds each image event without publishing source identifiers. `domain` is exactly `cold-start` or `tie`.

## 6. Event causality and stage separation

### Recommended Real-IAD Group-atomic sequence

`Group-atomic feedback` is `RECOMMENDED_RESEARCHER_DECISION_REQUIRED`, a scientific change relative to historical R7/D2/B/C; do not claim event-level parity with those trajectories.

1. The frozen stream is group-major: all five registered camera views are contiguous and ordered camera 1 through 5. Group order is deterministically ranked by Stream Seed. Gate 1 fails closed on a missing, duplicate, noncontiguous, or unexpected view; it does not silently repair or reorder a partial group.
2. At group start, snapshot each arm's detector and query-policy state. Encode and score every view and form each view's query proposal from that same pre-group state. No labels or updates are available within the group.
3. Seal each view's score, prediction and proposal before any label from that group is disclosed.
4. Traverse views in registered camera order to determine action status. At most one proposal executes per arm/group; the first proposal not suppressed by an exhausted category cap executes and reserves the group. Record group and hard-cap suppression separately. A later view's proposal cannot see an earlier same-group action or label.
5. After all five rows are sealed, append each view's observed Margin utility and replay BEA final-action price flags in image-event order. The first view's executed action contributes one `1`; all other views contribute `0`, including suppressed proposals. Price changes are calculated after each event flag but take effect only in the next group's proposal; all views in this group used the pre-group price. Query count increments once, only for the executed view.
6. Only after group completion, disclose the one purchased view's label. Apply the relevant arm's supervised C1 update; update its Greedy normal memory only if the queried label is verified normal, and never write a defect to normal memory. Refresh current-state C64/B3 for the next group. N-shadow reads F-B's resulting shared Greedy state but applies its own negative-only C1 update from the common C0 evidence only when the label is normal. Never disclose any other view's label or update N-shadow from a defect label.

If a group is incomplete or violates its manifest, fail the category closed and do not disclose a pending group label. Noncontiguous views are not permitted by this proposal; no open-group state is carried across another group. This group-major order is a new stream-order construction and not historical image-shuffle parity.

For VisA, do not invent five-camera groups. The metadata audit must establish an official group key or the researcher must approve an explicit identity-level exception before defining atomic units and stream order. If neither is established, the online VisA condition remains blocked; do not infer physical independence from filenames.

After all required Stage A group and category seals pass, Stage B may evaluate official image labels. Unqueried labels and masks remain unavailable to Stage A.

Source Group deduplication means at most one query per group per arm. Source-excluded Correction retrieval excludes same-source anchors from eligible neighbors. Neither property makes five views independent evaluation units. Category is the primary cross-category resampling unit; any image/view-level dependence analysis must cluster within Source Group rather than treat views as independent.

## 7. Supplementary references and falsification

- **Static-4+64:** matched static DINO/VisionAD-style scorer, four deployed Supports and separate C64 calibration; no online supervised update. This is not automatically a source-faithful reproduction of original VisionAD. A faithful VisionAD comparison needs its own explicit reference contract.
- **Static-36+32:** a declared allocation of the same 68 normal identities into 36 reference and 32 calibration roles, with equal registered reference-memory bytes where applicable. This is a normal-information allocation sensitivity, not a same-reference mechanism ablation. Identity membership and bank-construction algorithm are pending. Existing A4 results are not silently relabelled as this static method.
- **K-NG/FOADS:** optional source-faithful unsupervised online reference. Preserve its native feature extractor, pseudo-normal selection, and all-arrived-stream update rights. Backbone and supervision differ; comparison cannot establish pure neural-gas superiority. No local implementation/fidelity record was established; status `EXTERNAL_FIDELITY_PENDING`. No download, implementation, or run is authorized here. The manuscript must either include at least one credible source-faithful external online-method comparison or explicitly limit complete-system competitive claims. Static-36+32 remains optional allocation sensitivity, not a replacement for the principal 4+64 mechanism comparison or a new factorial.

Negative, mixed, undefined, and resource-cost results are retained. No algorithm or target is tuned on these campaign outcomes. No finite-time/anytime guarantee or safety guarantee is inferred. A superiority claim is limited by an approved BER practical margin, category heterogeneity, FPR/FNR trade-offs, actual query mismatch, and measured resource costs. The task-supplied 0.01 absolute Macro BER threshold is provisional and unapproved; no local external decision handoff was identified to establish its provenance. Missing-class metrics remain `UNDEFINED`, never zero.
