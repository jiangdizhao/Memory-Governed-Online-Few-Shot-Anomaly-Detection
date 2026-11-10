# Gate 0 — Metrics, Query Budgets, Inference, and Resources

**State:** `DRAFT_NOT_AUTHORIZED`. Counts below are planning estimates unless explicitly described as source-history context. No benchmark, data census, or hardware measurement was performed here.

## 1. Query accounting

For category `c`, let `T_c` be its exact official TEST **image-event** count in the approved manifest. Candidate category budgets are:

```text
Real-IAD: cap_c = floor(0.03 * T_c)
VisA:     cap_c = floor(0.10 * T_c)
```

This hard cap is new protocol, not implied by historical E6's adaptive soft price. Apply independently to each autonomous principal arm after its proposal and Source Group check. The executed count cannot exceed the cap. One paid query reveals only that view's label. At most one query per arm per verified Source Group. N-shadow inherits F-B's receipt and has no additional count. Report actual image-query count/rate, nominal cap, number and reason for suppressed proposals, and queried normal/defect composition separately. There is no catch-up, forced early query, unregistered top-up, or post-hoc matching. The exact BEA action/price update under cap suppression and the Margin cold-start/tie/history behavior are researcher decisions before implementation.

Realized counts need not equal nominal targets: floor rounding, group suppression, the policy, and the hard cap can all lower them. `T_c` counts images; group eligibility does not redefine the denominator. A comparison is budget-comparable only under an approved per-category actual-count/rate rule; report each arm's realized budget even when it fails that rule.

## 2. Conditions and statistical units

| Dataset | Categories | P0 | P1 | P2 | Principal sampling unit |
|---|---:|---|---|---|---|
| Real-IAD | 30 | Support 0 / Stream 0 | Support 0 / Stream 11 | Support 1 / Stream 0 | Category; for view dependence/uncertainty, Source Group cluster |
| VisA | 12 | Support 0 / Stream 0 | Support 0 / Stream 11 | Support 1 / Stream 0 | Category plus a yet-unverified dataset-specific physical group |

The proposed 126 category/condition trajectories are paired repeated conditions, not 126 independent categories. Report all condition-specific values, then a category result averaged over paired conditions and a category-equal dataset macro. Preserve paired contrast vectors by category. The suggested uncertainty analysis is a paired category-cluster bootstrap, resampling categories and preserving all arms/conditions within each selected category; report condition variation separately. Do not resample image views as independent samples or claim three seeds are independent factories/categories. The exact interval level, replicate count/seed, family-wise handling for the three primary questions, and whether a secondary hierarchical seed resampling is needed require approval.

## 3. Outcomes and undefined values

**Primary:** per-category full-stream prequential `BER_c=(FPR_c+FNR_c)/2`, with every prediction sealed before any current queried label is disclosed. Dataset macro is the equal-weight mean of category BER values (after the preregistered paired-condition summary), never an unexplained Real-IAD/VisA pooled mean.

Report per-category and macro AUROC, AP, BA, BER, FPR, FNR; improved/worsened category counts; complete 256-event window BER summaries and undefined-window counts; actual query counts/rates; queried-label class composition; first defect-query position/time; segment-wise query allocation; maximum no-query gap; C1/KNN state/occupancy; persistent bytes; peak temporary cache; inference/update latency; and GPU/CPU/I/O/runtime.

If a required class is absent, its rate and any metric requiring both classes are `UNDEFINED`; do not substitute zero. Worst-window BER uses complete, non-overlapping 256-image windows in stream order; windows lacking either class are `UNDEFINED` and counted. A trailing incomplete window is separately counted but excluded from the complete-window worst-case summary. This reuses the existing C reporting horizon; confirm it for the new datasets before freeze. Report exact denominators. No selective deletion of difficult windows or categories.

Primary contrasts and signs are `H_Q=BER(F-M)-BER(F-B)`, `H_Q_transfer=BER(K-M)-BER(K-B)`, `H_D=AUROC(F-B)-AUROC(N-shadow)`, and `H_C=BER(K-B)-BER(F-B)`. Positive signs favor the named candidate. Also report AP and FPR/FNR for every contrast. H_Q/H_Q_transfer/H_C are independent whole-policy/system contrasts; H_D is a matched-feedback mechanism contrast only if N-shadow fidelity passes.

The task supplies `0.01` absolute Macro BER as a provisional practical-effect proposal. It is not approved and has no verified local handoff provenance. Do not use it as a binary success gate until researcher/supervisor approval. Also predeclare whether increased FPR or FNR can negate a BER gain, the tolerated actual query mismatch, resource-overhead criterion, and multiplicity interpretation. If no practical/safety margins are approved, report estimates and intervals without a formal “superior” declaration.

## 4. Workload arithmetic (not audited counts)

Planning inputs supplied in the task are approximately 114,585 Real-IAD TEST images and 2,162 VisA 1cls TEST images. They imply 116,747 images per paired condition, 126 category/condition trajectories, and 350,241 event encodings across three conditions before static/external extras, assuming one feature encoding is shared across arms. Four principal arms still perform independent stateful scoring; N-shadow is an extra read-only comparison. These figures are estimates only. Exact category counts and eligible groups must come from a later approved manifest.

The intended workstation has an RTX 3090 Ti with nominal 24 GiB VRAM. No device inspection or benchmark was done in Gate 0. Do not extrapolate Experiment B/C durations into a guarantee for this campaign.

## 5. Memory and resource accounting plan

Use byte-level component accounting and distinguish physical shared read-only buffers from per-arm logical state:

- **Encoded TEST feature:** E6 expects one FP32 `(784,4096)` map, 12.25 MiB for one current image. Do not retain a full TEST dense feature cache. Encode in stream order and release per-category temporaries at a declared boundary.
- **C64 feature cache:** if the full `(64,784,4096)` FP32 tensor is cached, it is `822,083,584` bytes = 784 MiB per category, shared read-only among arms; 42 categories resident would be about 32.2 GiB. This is a shape-based upper estimate, not a measured allocation. Keep it per-category/CPU-backed or prove a different bounded cache plan before Gate 2. Each arm's current-state C64 scores and threshold evidence remain separate.
- **Fixed bank:** current R7/B record gives 8,192×4,096 FP16 = 67,108,864 bytes (64 MiB) per physical bank. Record logical per-arm references and actual physical sharing; do not call equal fixed-bank bytes equal total method storage.
- **Greedy and C1:** count every tensor by shape, dtype, device, plus source group, identity, position, signs, indexes, allocator reserve, and staging copies. Greedy is K=1024 per native layer; C1 is K=1024. Capture actual before/after bytes rather than relying on a nominal K alone.
- **Signed kNN:** account for R10's FP32 feature FIFO at K=1024 (16 MiB of feature values alone), plus sign, position, 32-byte source digest per slot, indexes, and temporary distances.
- **Other buffers:** current image, each arm's native maps/top-k patches, C64 scores, query receipts, private/public logs, checkpoint/serialization buffers, framework/model allocations, and allocator high-water marks. Distinguish persistent memory from peak temporary memory.

Do not materialize all dataset features or hold every category's C64 patch tensor on GPU. Process a category under a measured limit and release category-specific state/caches before the next one, with persisted checksums and recovery semantics frozen later.

## 6. Gate 2 resource preflight requirements

Before Gate 1 execution planning freezes ceilings, a separately authorized representative preflight must measure: model-load and per-image encode time; per-arm native scoring, C64 rescoring, query update and checkpoint/log-write time; CPU/GPU utilization and contention; peak allocated/reserved VRAM; host RSS; C64 cache format/size; persistent fixed/dynamic state bytes; maximum temporary memory; bytes/event and all-arm output scale; per-category disk peak, free-space/failure reserve; end-to-end category throughput; and cleanup/release behavior. Measure the actual 42-category/three-condition schedule, including isolated per-category resource cleanup. Set stop thresholds from these measurements, not guessed numbers. No such preflight occurred in Gate 0.
