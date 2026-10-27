# D1-B — Risk-Safe Admission experiment protocol freeze (v1)

**Registered:** 2026-10-27. **Status:** implementation committed for metadata-only local validation; **no new admission performance results**, no GPU runs, no CDARG/SAEG code. D1-A-R2 is archived separately and never overwritten.

**Authority:** docs/RISK_SAFE_ADMISSION_7DAY_PLAN_2026-10-27.md (§2.1–2.6, D1). **Implementation:** experiments/risk_safe/d1b_protocol.py. **Immutable outputs after local verification:** outputs/risk_safe_sprint_20261027/d1b_protocol_v1/{protocol_v1.json,query_policy_manifest.json,stream_order_audit.json,source_manifest.json}.

## 1. Scientific question and invariants

The primary target is whether memory admission prevents actual **defect-score erosion** arising from writes of *individually verified-normal camera images* under partial human feedback. Do not use threshold movements to mask raw-memory score deterioration. The first causal isolation comparison uses **Q-A shared exogenous random tapes**, fixed LOSO threshold, fixed support and the same frozen scorer, candidate patches, memory capacity and evaluation probes for every admission arm.

Do not claim CDARG or SAEG works until measured. Do not make a safety guarantee for unobserved defect modes. Do not use oracle probe labels in admission or query selection.

The original v22 AIF-inspired Query Controller, LOSO initial threshold and Guarded Theta Sidecar (T1) are preserved, with T0=fixed theta0; future confidence-sequence threshold sidecar (T2) is a distinct replacement experiment, not a modification to this registration.

## 2. Data and supervision units (locked)

- Six development classes: audiojack, bottle_cap, eraser, plastic_plug, switch, rolled_strip_base.
- Official Real-IAD sample key is the test unit and uniqueness unit; one key has exactly five camera images C1..C5. Directory serial S#### is **only** a residual potential-specimen correlation diagnostic.
- Cold: 4 verified-normal support images per support seed, seeds **0, 1, 2**. Mature is **auxiliary**, using an additional 64 normal training samples (explicit supervision).
- Per-class online stream: **525** sample groups, **2,625** camera images, 3 fixed phase cohorts of **175** groups. These cohorts, probe groups and the support/history split never change when stream order changes.
- Probe A and B are each 24 normal session groups + 24 defect session groups = 48 groups/240 images. Each view's ground-truth label is read separately, only for evaluation. An NG capture session can contain image-level OK views; do not propagate session NG to all views.
- Query cost: **one image-level human verification**, maximum **one queried camera image per official sample** in a stream. At B=79, image-query fraction 79/2625=3.0095%, unique-sample audit fraction 79/525=15.0476%. Neither denominator is a fraction of known independent manufactured products.
- The six classes are for development; all remaining Real-IAD classes and MVTec-AD are confirmation-only, not hyperparameter sources.

## 3. Budgets, seeds and run priority (locked before new metrics)

**Primary Q-A budget levels per class:** B = **{16, 32, 64, 79}** absolute verified image labels. B=16 is deliberately evidence-starved; the default Guarded Sidecar cannot activate under a minimum recent-label gate of 24. B=32 is an early threshold; B=64 supplies a medium supervision intensity; B=79 maintains comparability to the prior V0.2 3%-of-image reference. This is not a claim these are optimal budgets.

**Auxiliary high-budget sensitivity:** B={96,128}. **B=8** is a metadata/cold-start diagnostic, not in the initial performance matrix. High budgets require separate cost approval before GPU runs.

**Q-A random query seeds:** **37, 101, 211** (arbitrary predetermined; not selected using new admission performance). Stream-order RNG seed: **1701**. Bursty sample run length: **16**. Historical V0.2 support seeds are **0,1,2**. Each seed family is independent.

**Compute hierarchy:** D1-C/D initial smoke = audiojack + bottle_cap × support seed {0} × query seeds {37,101} × B {16,32,64,79} × **random** stream only. This does NOT authorize executing all combinations; it defines the bounded highest-priority set. D2 primary extends paired shared-tape Q-A arms (Static, Append, Greedy, Reservoir, causal Novelty, SAEG, CDARG, if implemented and tested) under the random mode. D4 adds Q-B informative feedback smoke. D5 adds non-IID stress and capacities; D6 adds Q-C native AIF and A/C/T/Q factorial.

**Important:** the protocol is frozen now; coding and experimental execution remain separately gated by user authorization and smoke tests.

## 4. Stream orders: exact semantics and exposure labels

We distinguish the 3 pre-existing *phase cohorts* from the order *within a phase*. All five variants use **exactly the same 175 sample keys per phase and the same five camera images per sample**.

**Primary = random.** For each phase, shuffle official sample keys with independent seeded permutation. Emit each group's C1, C2, C3, C4, C5 consecutively. This mode does not access labels to order samples. This is a randomized offline simulation and is not claimed to reproduce the Real-IAD physical acquisition process.

**Stress 1 = normal_first.** Shuffle deterministic within the normal and defect subsets; then place all normal groups before defect-containing groups **inside each phase**. An official sample is called 'defect-containing' for constructing this offline stress if **any** of its five camera images has an NG image-level label.

**Stress 2 = defect_first.** Exactly the inverse within each phase.

**Stress 3 = bursty.** Draw deterministic shuffled lists of normal/defect-containing sample groups, then alternate chunks of up to **16** sample groups of each type within the phase. When one subset is exhausted, drain the other. The normal-vs-defect group type is **evaluation metadata at schedule construction**, never an online oracle label.

**Diagnostic = acquisition_time_blocked.** Preserve D1-A's original within-phase image sorting by timestamp, camera, path. Raw timestamps can overlap across the three phases; it is not a globally monotonically timed stream. This is a **separate diagnostic**, not a replacement for the new randomized primary stream.

**Critical disclosure:** normal_first, defect_first and bursty use full hidden ground truth to **construct synthetic environment schedules**. That does not make them legitimate real-world unsupervised query policies. They are explicitly artificial shift/interference stress tests. The eventual model, admission gate and Q-B/Q-C controller never receive these hidden labels.

## 5. Q-A — shared exogenous random tape (primary causal attribution)

A fixed deterministic pseudorandom *sample priority* is defined once per (class, query seed, exact official-sample universe). It is **independent of stream order** and does not perform label-stratified sampling. For every B the first B priority keys define the same queried official sample set; a separate per-sample deterministic RNG chooses one of its five camera images.

The **same sample key and the exact same camera image** must be queried across all five stream orders, for every class/seed/B. Only time position changes. Each order's tape is stored with its distinct stream digest and event positions. Every admission arm in a condition receives the identical tape and label disclosures. Changing B creates a full independent experiment replay from identical initial state, not an extension of a smaller-budget learned memory.

The Q-A offline tape generator may know the finite stream membership in advance; this is an **exogenous benchmark supervision schedule for causal isolation**, not a fully prospective deployed controller. It must never be presented as the native AIF controller. The actual controller receives opaque observation handles (no raw Real-IAD sample key, path, label or defect metadata). A trusted adapter alone maps handles to image tensors/features.

## 6. Q-B / Q-C pre-registered rules (not implemented yet)

**Q-B, shared frozen-reference informative query:** uses only fixed-support static anomaly score s_fixed, fixed support-only LOSO theta0, the query policy's **own past acquired labels** and budget ledger. One query opportunity per official sample on arrival of camera C1. Define delta=max(1e-6,MAD of normal support-only LOSO scores), u=1/(1+|s_fixed-theta0|/delta), p=min(1,(B_remaining/N_remaining)*(0.5+1.5*u)); draw U(0,1) from separate seeded per-sample pseudorandom stream. Query if draw < p. Record unspent budget; do not fill using future labels. **This is a pre-registered prospective score-uncertainty reference, not a reproduction of the full v22 native AIF utility.** A trusted sampler tracks sample eligibility without giving a label-bearing sample key to the controller. No Q-B implementation or score results exist yet.

**Q-C, native closed-loop:** each arm's original v22-inspired Q controller reads its own evolving state (M,C,theta and lawful queried-label history), never other arms' state or evaluation probes. Q-off is a **budget-cap-matched random controller**, not a no-query condition. Equal cap B does not imply equal actual queries; always plot actual spend, verified normal/defect evidence and controller activation.

**CAUTION:** The Q-B weights, MAD normalization and C1 opportunity are design choices registered for later verification; they are not empirically justified or performance-tuned. If later found infeasible they require an explicit **new protocol version**, not an invisible change to v1.

## 7. Threshold/admission/firewall and metrics

- T0 fixed LOSO theta0; T1 exact unchanged v22 Guarded threshold controller; T2 future independent CS+Risk-aware Sidecar. No threshold version changes during primary mechanism isolation.
- A-off Greedy; A-on CDARG. C-off disables Signed Correction **without discarding independent defect anchors**; T-off uses fixed theta0; Q-off uses capped random feedback.
- Per step: (a) trusted adapter obtains observation; (b) pre-query score/prediction; (c) query decision using only permitted past; (d) only selected camera gets image-level label; (e) verified-normal admission evaluated against anchors verified **strictly before the current event**; (f) feedback history, correction and threshold then updated for future steps.
- When no historical verified-defect anchor exists, CDARG risk is **unavailable, not zero**. A conservative pre-registered fallback must be implemented before GPU.
- Main admission evidence: signed raw anomaly-score changes on held-out defect probes; FNR-degradation AUC under fixed theta0; change in FPR under the same theta0; image-level AUROC/AUPRC; admission-rate / FPR-benefit-matched comparisons. Query composition, anchor age/count, time-to-first-defect, q spend and capacities/latencies are required. Do not pool camera views as independent samples for confidence intervals: class is primary clustering unit.

## 8. R2 continuity, output immutability and limitations

D1-A-R2 manifest and JSON are immutable and were verified after their output commit. D1-B changes query_protocol.py to tag stream orders; this *changes new source hashes and D1-A replay digests*. **Do not rerun D1-A-R2 into its old output directory, and do not rewrite or relabel R2 results.** The D1-B output directory is new. Git commit provenance in D1-B is pinned to the producing run's ancestral commit; source/data/config hashes remain strict.

The following limitations stay explicit: actual manufactured specimen identity cannot be inferred from S#### alone; Q-A finite-stream shared tape is not a live policy; stress schedules use oracle labels **only for offline environment assembly**; sample-group-level normal/defect membership is not the same as per-image NG feedback; LOSO theta0 itself can be biased (addressed by a parallel, later threshold-sidecar study, not retroactively altered here).

**Stop after D1-B metadata verification and archive its outputs. Do not start GPU, D1-C, Q-B or CDARG automatically.**
