# Stage 0 researcher decision — 2026-10-29
**Status: APPROVED_METHOD_DESIGN / STAGE1_EXECUTION_HOLD.**  
**Parent reviewed commit:** `cf79855e87b9a9ea3eac84b280fa6807961d9c13`.  
**Superseding support-seed design:** [Stage 1 approved design v3](STAGE1_EVALUATION_PROTOCOL_APPROVED_DESIGN_v3.md).  
**Prior candidate:** [Stage 1 frozen candidate v2](STAGE1_EVALUATION_PROTOCOL_FROZEN_CANDIDATE_v2.md) (archived unchanged).

## Decision and authorization provenance
The researcher explicitly selected **Option B (multiple support seeds)** and authorized a small protocol revision and formal design approval on 2026-10-29, while requiring **no Stage 1 execution** until the proposed Real-IAD query-composition / Native Scorer-headroom statistical analysis is discussed and approved separately. This record documents that decision; it is not a signed ethics authorization or a scientific validation result.

### Approved and frozen now
1. **Five support configurations:** support seeds `(0,1,2,3,4)`, each drawing **four distinct** training-normal official sample groups; historical Seed 0 remains exactly reproducible. C64 remains the **same 64 additional**, disjoint, protected train-normal groups/views across all five support seeds, selected with the original history seed `190021` and historical camera-view convention. For Seeds 1–4, support groups are drawn from the remainder after C64 exclusion; view selection uses the historical deterministic convention. No additional support or labeled C64 budget is granted by the five-seed design.
2. **Attribution and fair comparison:** a seed changes the frozen support bank and derived native score/initial threshold; C64 group identities remain fixed. Across all controllers for the same class, seed and query profile, freeze one label-blind query tape and one shared native score/Greedy trajectory. Main budget stays **3%** with query seed 37; Q-B is the separately defined early-allocation profile, not a rate increase. Report seed-level paired differences, per-class metrics and variation across the five precommitted configurations. Do not report five seeds as five independent test datasets.
3. **G6/G7/G9 evidence-taxonomy principles APPROVED FOR RESEARCH REPORTING**, as proposed in [Stage 0 gate amendment](STAGE0_GATE_AMENDMENT_CANDIDATE.md): distinguish executable/paper-equation fidelity, author-code parity, assumption compatibility, theoretical-guarantee transfer, synthetic/native engineering integration and scientific validation. Missing verified author code and nontransferable source assumptions **are recorded limitations**, not automatic disqualification from a transparently labeled empirical comparator. SPOT/dSPOT/Sun-CS remain included; fallback hybrids remain separately named. B7 remains provisional; descriptive CP bounds are **not** certified online risk guarantees.
4. The existing Stage 0 artifacts and actual G6/G7 `BLOCKED` fields remain historical truth for their pinned revisions; this research approval does **not** retroactively turn them into PASS, claim author-code parity, or rewrite the prior audit. Any implementation-facing gate status change requires its own versioned and tested reporting update.

### Explicitly NOT approved / outstanding
- **NO-GO Stage 1 and Stage 2 execution.** This is a design approval only.
- No Real-IAD test-label scan, query-class count analysis for 3%/5%/7%, Native Scorer performance assessment, or test-mask decoding **at this step**. Those activities must be scoped and approved under a separate analysis protocol; only the four development classes may be considered for such a future plan. Mint/PCB confirmation outcomes remain sealed.
- The **5% and 7% query rates** are discussion candidates for the statistical feasibility task, not frozen Stage 1 experimental arms.
- Seeds 1–4 are **not yet supported or validated** by the current Seed-0-only calibration preflight. Before any Stage 1 run, extend calibration-owned selection/preflight only, verify Seed-0 exact identity, normal train-only disjointness for each new seed and full profile/source fingerprints. Do not alter protected historical D2/v22 code.
- No pre-run labeling-informed query allocation, hyperparameter selection, controller redesign, B7 threshold change, defect-rate-aware stopping, or confirmation-set use.

### Next decision gate
First agree a *separate*, versioned protocol for (a) a descriptive, full-stream default-order **queried normal/defect composition audit at hypothetical 3%/5%/7% rates** and (b) **native-scorer ceiling/headroom analysis**. Specify allowable label access, temporal/group-aware query construction, paired seeds, performance estimands, leakage protection, and how this reconnaissance changes the independence of any subsequent Stage 1 evaluation. Neither audit is performed here.

Only after discussion and explicit authorization may that statistical task proceed. A further express `STAGE1_EXECUTION_GO` decision is required, independently of this approval.
