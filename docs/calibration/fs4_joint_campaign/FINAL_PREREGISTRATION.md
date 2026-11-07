# FS4 final joint campaign preregistration

**Protocol:** `FS4_FINAL_JOINT_CAMPAIGN_FREEZE_V1`
**Machine-readable specification:** `CAMPAIGN_SPEC.json`
**Specification SHA-256:** `1c96077bfdaa58f9ff93a67a9349a4cc6bc8b63578ae8915ff23762b50aee896`
**Status:** frozen for independent implementation review; scientific execution is not authorized by this freeze task.

## Question and population

The campaign compares selection-aware historical threshold L with current-state source-excluded threshold E under the existing four-Support native FS4 system. It has three trajectories: a shared threshold-only trajectory S, an L-coupled query-only trajectory CL and an E-coupled query-only trajectory CE. H1 is the paired category-macro E-minus-L BER difference on S/GC. H2 is the distinct whole-system CE/E/GC minus CL/L/GC contrast. They are not pooled.

The population is the four registered development categories: audiojack (3,815 images, 763 groups), regulator (3,855, 771), switch (3,820, 764), and rolled_strip_base (3,805, 761), totaling 15,295 events in 3,059 five-view Source Groups. These are development categories, not independent external validation. No category may be dropped or replaced.

The stream order is category-local R7 deterministic Source Group shuffle, seed 0, preserving camera views 1–5 inside each group. The complete event order must be independently reconstructed as a bijection before one opaque registration is sealed and shared by all three arms.

## Frozen initialization and runtime

Every category starts from exactly four registered TRAIN-normal Support images, support seed 0, the original paths and path-dependent native augmentation/subsampling semantics, a native 8,192 × 4,096 FP16 fixed bank, and empty Greedy and C1 memories. The initial threshold is the four-Support self-score mean plus 2.5 times the sample standard deviation (ddof=1), floored at 1e-6. This is a cold-start heuristic without a population guarantee. C64 is not used.

The encoder is the frozen registered DINOv2 ViT-L/14 configuration and checkpoint named in `SOURCE_MODEL_SUPPORT_FREEZE.md`. The stochastic BEA policy uses seed 37 and the exact registered lottery, utility, cost, lagged scale, selection probabilities and one-query-per-group limit in `CAMPAIGN_SPEC.json`. No budget catch-up, forced query or algorithm change is permitted.

## Event protocol and endpoints

At each event, the worker verifies the opaque identity, reads one metadata-free lossless RGB resource, encodes it, scores native G and GC, seals all pre-event predictions and query inputs, then seals the stochastic query. Only an executed query may trigger a broker disclosure. Genuine feedback and native memory updates occur afterward. Threshold updates affect future events only.

The primary endpoint is category-macro `BER(S/E/GC) − BER(S/L/GC)`. Secondary H2 is category-macro `BER(CE/E/GC) − BER(CL/L/GC)`. For every registered readout, report confusion counts, FPR, FNR, BER, Balanced Accuracy, AUROC and Average Precision. Category macros weight the four categories equally. Missing denominators remain NA with reasons.

Shared L and E use exactly one score sequence per scorer; their AUROC and Average Precision must match exactly for G and GC. A mismatch invalidates the campaign. No C64-based image metric and no pixel-mask metric is part of this protocol.

Temporal analysis uses Source Group arrival order. Early/middle/late boundaries are `[0,floor(n/3))`, `[floor(n/3),floor(2n/3))`, and `[floor(2n/3),n)`. Sliding windows contain 50 consecutive groups, stride 10, or 250 events. One-class window BER is NA; worst valid BER and independent worst FPR/FNR report their valid-window counts.

## Interpretation limits

There are four development categories and one query seed. The 15,295 views are clustered in 3,059 Source Groups. No inferential interval is preregistered, and a bootstrap of an adaptive trajectory would not be an exact counterfactual-policy interval. No finite-sample FPR, conformal, unbiasedness or broad industrial generalization guarantee is claimed. A negative result, inactive controller, low query rate, insufficient ESS or expensive E trajectory is a valid outcome.

The separate task authorization governs this freeze only. Real-data preparation, actual Support parity construction, Stage A and Stage B each require a separate campaign-bound authorization after independent review.
