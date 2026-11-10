# Gate 0 — Scientific Contract Draft

**State:** `DRAFT_NOT_AUTHORIZED` · protocol freeze pending researcher and ChatGPT review
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

N-shadow follows F-B's executed-query ledger. It receives the same F-B verified feedback opportunity and the exact normal Greedy writer inputs; it admits only the verified-normal negative C1 correction. It ignores defect correction updates, cannot query, and cannot affect F-B. It is not an independent policy. Its purpose is to diagnose the incremental defect-correction mechanism under matched opportunities; its implementation and precise C64/B3 state contract are not present in the inspected sources and remain a P0 blocker.

## 3. Questions, estimands, and interpretation

Compute category-condition paired outcomes first. For each category, average the paired condition results as preregistered; then give every category equal weight within Real-IAD and within VisA. Report each condition separately as well. The exact interval procedure remains subject to the decision register.

- **RQ1 — verified defect feedback:** `H_D = AUROC(F-B) - AUROC(N-shadow)`. Positive favors F-B. This is a matched-feedback mechanism diagnostic only if the shadow truly inherits F-B's queries and writer inputs. It does not measure the independent closed-loop effect of changing the whole policy.
- **RQ2 — evidence acquisition:** `H_Q = BER(F-M) - BER(F-B)` and `H_Q_transfer = BER(K-M) - BER(K-B)`. Positive favors BN+Band. These are independently evolving whole-policy contrasts: realized query identities and later states can differ. Call budgets comparable only by prespecified actual expenditure rules; do not infer query matching from nominal targets.
- **RQ3 — architecture parsimony:** `H_C = BER(K-B) - BER(F-B)`. Positive favors Full C1 on BER. This is a complete-system comparison under the same named BN+Band policy, not a feedback-matched pure architecture effect. If K-B is competitive within approved practical and resource margins, the simpler scorer may be preferable. F-B is not required to defeat K-B.

Report each contrast with its sign convention and uncertainty. Keep Real-IAD and VisA separate. A score/algorithm comparison is not an independent confirmation claim; a backbone/supervision-mismatched external comparison is descriptive system evidence, not pure causal evidence.

## 4. Dataset conditions and common initialization

- **Real-IAD:** all 30 official categories; official TEST image stream; proposed `rho=0.03` and per-category cap `floor(0.03*T_c)`.
- **VisA:** all 12 categories; official `1cls` TEST only; no unused TRAIN-normal images in the stream; proposed `rho=0.10` and cap `floor(0.10*T_c)`. It is a short-stream external validation condition, not evidence that 10% VisA feedback transfers to 3% long-stream Real-IAD. No forced query is added at stream start.
- **Common normal information:** four verified normal deployed Supports plus 64 additional, disjoint, verified normal C64 commissioning images. This is not strict-four-shot-total. C64 can support initial/current-state calibration and registered controller quantities, but cannot enter Fixed Bank, Greedy writer, or Signed correction memory. Any broader normal-pool selection or feature-based search is a separate, explicitly described information allocation.
- **Paired conditions:** P0 `(Support Seed 0, Stream Seed 0)`; P1 `(Support Seed 0, Stream Seed 11)`; P2 `(Support Seed 1, Stream Seed 0)`. The existing E6/C path establishes only the four development categories with Support Seed 0 and Shuffle Seed 0. P1/P2 for all categories and VisA are not established by the inspected source and require role/identity manifests and explicit construction review.

## 5. Candidate query policy and the cap intervention

The historical simplified BN+Band objective is `U_t = B_t + N_t + D_band,t`, using the registered risk/uncertainty, normal novelty, and predicted-normal asymmetric band terms. Its source implementation is E6/C. The hard category cap is a new outer intervention; historical soft price learning does not enforce it.

Candidate rule for review: each arm first makes its ordinary policy proposal from its own pre-event state. A query executes only if proposed, the event's Source Group has not already been queried by that arm, and the arm's category count is below `floor(rho*T_c)`. A denied query is recorded with its reason. There is no top-up, future-length use, forced bootstrap, or post hoc rate adjustment. Report proposals, group suppressions, cap suppressions, and executed queries separately. For BEA, the recommended price update observes the final executed action; this means a cap-suppressed proposal is a non-execution. Existing `observe_final_action()` only models Source Group suppression, so the new cap reason/update interaction requires an implementation change after approval.

`rho` is 0.03 for Real-IAD and 0.10 for VisA. Historical E6 BEA uses target 0.03; the 10% VisA price target is a deliberate protocol change. The candidate controller retains initial price 0.80, rate-learning step 0.02, prior executed-action window up to 256, bounds `[0.05,3.0]`, and strict `objective > price`, with next-event price effect. The first window is naturally shorter on VisA.

For Margin, the proposed `Margin_rho` uses the arm's own standardized-margin utility `psi=exp(-0.5*m^2)`; a hashed Bernoulli draw with probability `rho` for the first 64 events; thereafter a past-only empirical `1-rho` quantile from up to the previous 256 utilities; strict `utility > quantile`; and an identity-hashed tie lottery with probability `clip((rho*n - n_gt)/n_eq,0,1)` when equal to the threshold. It has no forced query. Then apply the same one-query-per-group and hard-cap gate. This is a candidate extrapolation of D2 Margin37, not the historical Margin37 (which uses 3.7%, q=.963). The tie rule, history update under suppressions, cold-start draw, and cap interaction require researcher approval. If this rule is not approved, Margin remains `RESEARCHER_DECISION_REQUIRED`; do not call historical Margin37 3% or 10% matched.

## 6. Event causality and stage separation

For every image: encode frozen features; score all relevant arms using pre-event state; seal the prediction and query proposal without outcomes; disclose a label only for a legitimate executed query; apply only the registered queried-label Correction and normal-memory updates; refresh each changed arm's C64/B3 state; then advance. Same-event feedback cannot affect its sealed prediction. Unqueried TEST labels and masks stay outside every Stage A controller. Stage B may evaluate official image labels only after every required Stage A seal and independent integrity check.

The inspected E6 event loop applies a queried feedback update after the current row is sealed and before it scores the next stream event. It suppresses any second query from that Source Group, but it does not wait for all five views or defer the first view's update until a group boundary. Therefore a later view of the same Real-IAD Source Group can be scored under state changed by the first view's feedback (also for later-in-stream views if the group is not contiguous). Whether to retain this online timing or add a group-level feedback barrier is a P0 researcher decision; this draft does not change it.

Source Group deduplication means at most one query per group per arm. Source-excluded Correction retrieval excludes same-source anchors from eligible neighbors. Neither property makes five views independent evaluation units; Source Group must be the resampling/independence cluster where appropriate.

## 7. Supplementary references and falsification

- **Static-4+64:** same frozen native DINO scorer, four deployed Supports and separate legal C64 calibration; no online supervised update. Hold its threshold/score path fixed as declared in the final implementation contract.
- **Static-36+32:** a declared allocation of the same 68 normal identities into 36 reference and 32 calibration roles, with equal registered reference-memory bytes where applicable. This is a normal-information allocation sensitivity, not a same-reference mechanism ablation. Identity membership and bank-construction algorithm are pending. Existing A4 results are not silently relabelled as this static method.
- **K-NG/FOADS:** conditional unsupervised online reference. Preserve its native feature extractor, pseudo-normal selection, and all-arrived-stream update rights. Backbone and supervision differ; comparison cannot establish pure neural-gas superiority. No local implementation/fidelity record was established in this bounded audit; status `EXTERNAL_FIDELITY_PENDING`. No download, implementation, or run is authorized here.

Negative, mixed, undefined, and resource-cost results are retained. No algorithm or target is tuned on these campaign outcomes. No finite-time/anytime guarantee or safety guarantee is inferred. A superiority claim is limited by an approved BER practical margin, category heterogeneity, FPR/FNR trade-offs, actual query mismatch, and measured resource costs. The task-supplied 0.01 absolute Macro BER threshold is provisional and unapproved; no local external decision handoff was identified to establish its provenance. Missing-class metrics remain `UNDEFINED`, never zero.
