# Stage 0 software architecture and causal invariants (v1)

## 1. Architecture objective

Build **one composable online research engine**, not eight independent detectors and not a monolithic future agent. Stage-0 activates only `CalibrationIsolatedProfile`; inactive signed correction and BEA are represented through future-compatible interfaces, not fabricated implementations. Existing original scripts remain immutable. Treat class names below as *required behavior*, not enforced spelling.

```text
DatasetManifest -> FrozenFeatureProvider -> NativeScoreService (Fixed Bank + one shared Greedy M_t)
                 -> (raw_score_t, feature_ref, anomaly_map_ref, score_state_digest)
                 -> 8 independent CalibrationControllers.predict BEFORE query label disclosure
FrozenQATSTape  -> QueryDeliveryGate -> VerifiedNormalSharedWriter (one common policy)
                                  \-> CalibrationController.observe_feedback (only queried)
Changed M_t   -> VersionedAnchorLedger + Source-awareRescorer (read-only)
              -> CalibrationControllers.observe_score/observe_memory_changed
              -> theta_(t+1), diagnostics, separate evaluator-only labels
```

## 2. Recommended code boundaries

```
experiments/calibration/
    __init__.py
    profiles.py                 # immutable stage/profile config, disabled/enabled plugin declarations
    protocol.py                 # typed event records and information capabilities
    native_adapter.py           # native P0/0.2/v22 references; NO replacement scorer
    shared_memory.py            # one shared Greedy writer; stable state hash
    query_tape.py               # NEW Q-A-TS / Q-B; archival old Q-A left untouched
    anchor_ledger.py            # source identity, version, legal history, immutable frozen features
    state_rescorer.py           # exact native score per current M, LSO and cache
    controller_api.py           # prediction, unlabeled score, feedback, memory change
    controllers/
        frozen.py
        rolling_quantile.py
        original_sidecar.py
        scq.py
        dual_aci.py
        spot_dspot.py
        sun_cs.py
        scdac_g.py
    evaluator.py                # holds labels; NO algorithm access
    metrics.py                  # prequential BA/FPR/FNR windows and counts
    provenance.py               # hashes, code refs, checkpoint metadata
    prequential_engine.py       # causal event order, checkpoint state restoration
    runner.py                   # --stage 0 --all | --verify-only | --resume
tests/calibration/
    test_protocol.py
    test_score_memory_parity.py
    test_query_tape.py
    test_label_firewall.py
    test_rescorer_lso.py
    test_controller_reference.py
    test_causal_order.py
    test_runner_restart.py
docs/calibration/
    ...                         # already authored contracts
```

If a code module is small, combining files is acceptable; behavior, tests, and separation of roles are not optional. Do not copy original VisionAD/v22 source into a second implementation. Import the originals and use the existing `experiments/risk_safe/native_bridge.py`, `experiments/risk_safe/preflight.py`, `experiments/risk_safe/query_protocol.py`, `experiments/v02_memory.py`, `experiments/p0_protocol_audit.py`, `experiments/risk_safe/p01_falsification_native.py`. Preserve original outputs and do not modify them to produce parity.

## 3. Explicit interfaces

```python
@dataclass(frozen=True)
class Observation:
    t: int
    official_sample_key: str
    camera_id: int
    image_ref: str               # no label or defect mask
    score: float                 # native score BEFORE write
    score_memory_version: int
    score_state_digest: str

@dataclass(frozen=True)
class Feedback:
    t: int
    sample_key: str
    queried: bool
    label: Optional[int]         # y=0 normal, y=1 defect; nonqueried None

@dataclass(frozen=True)
class EvidenceSnapshot:
    memory_version: int
    normal_anchors: tuple        # (score, key, stored_version, source metadata...)
    defect_anchors: tuple
    available_normal: bool
    available_defect: bool
    source_exclusion_audit: dict
    # no true labels from unqueried stream samples

class CalibrationController(Protocol):
    name: str
    def initialize(self, source_normals, theta0, config): ...
    def predict(self, observation) -> int: ...
    def observe_score(self, observation): ...    # unlabeled method permitted
    def observe_feedback(self, observation, feedback): ... # queried only
    def on_memory_change(self, evidence_snapshot): ...
    def finalize_step(self) -> 'ThresholdUpdate': ...  # theta_(t+1)
    def state_dict(self) -> dict: ...             # serializable, deterministic resume
    def load_state_dict(self, state): ...
```

Exact signatures may be adapted; formal permissions and timing cannot. All calibrators get the **same observation**, and queried ones the **same Feedback**. `predict` cannot mutate shared Memory or Query. Controller state must own all algorithm parameters, cooldown, anchor counts and error counters.

Other future interfaces:
- `FeatureProvider`: frozen DINOv2 outputs with image/view IDs and hash.
- `ScoreService.score(image, memory_snapshot, correction_snapshot)`: returns `raw_score` and optional `corrected_score`; explicit score kinds.
- `MemoryPolicy.update_verified_normal`: source and query label validation; returns a new `MemoryVersion` ONLY on content change.
- `QueryPolicy.decide(observation, decisions_snapshot)`: Q-A-TS currently ignores decisions; BEA later may use pre-disclosure predictions and own state. Never allow offline outcome fields.
- `CorrectionPolicy.apply`: OFF in Stage 0; future experiments return corrected scores under separately hashed correction state.
- `EvaluationRecorder`: retains true labels separately and constructs reports after the event; never injected into another component.

## 4. Reuse and parity hazards

**Native source ancestry:**
- `vmb_visionad_new.py`: frozen feature extraction / VisionAD-style reference.
- `test_phase3_AIF_simplified_v22.py`: `PatchVMB`, multiview score and reference normal-write selection.
- `experiments/p0_protocol_audit.py`: native `p0.Config`, feature/support bank construction, true self-match-avoiding LOSO reference.
- `experiments/a0_a1_strict4_matched.py:score_bank`: native score wrapper.
- `experiments/v02_memory.py:greedy_slot`: actual Greedy replacement (not an official ContCore reproduction).
- `experiments/risk_safe/native_bridge.py`: `dynamic_codebooks`, `snapshot`, `install_on_clone`, `normalized_native_candidate_rows`, `verify_untouched_parity`, `verify_native_update_parity` for source-state preservation.
- `experiments/risk_safe/d2_threshold_calibration.py`: C64 construction and source q90 reference.
- `experiments/risk_safe/p01_falsification_native.py`: bank K=1024 default vs requested K=256 mismatch handling, scorer and hash parity, but final-state Probe-A results are not prequential evidence.

**No fresh hand-written scorer.** In particular do not silently substitute top-1%-pixel pooling, 1-NN, Euclidean distance, or altered image preprocessing for `p0.Config(image_aggregation=max, knn=5,...)`. Run native/shadow score parity at minimum before formal experiments, using a locked dtype and `math-sdpa` option when required on Windows GPU.

**Normal patch writer:** `v02_memory.MemoryArm.update` invokes `select_informative_normal_write(score, theta, sigma,...)` -> patch keep {8,16,24}. The shared writer must call this *once per queried verified normal* with a precommitted **reference** score/threshold independent of controller predictions, then use exact original Greedy append/replace operator. Original mutable v22/v02 code is unchanged. If prior reference policy has ambiguous native behavior, report blocker; do not select arbitrary keep.

**Patch provenance caveat:** existing Greedy codebook may store only prototype tensors, not per-row source identity. Source-aware LSO must track `event_id/official_sample_key` for each appended or replaced prototype, per layer, with synchronized swap/skip bookkeeping; verify against native tensor changes. If perfect source attribution cannot be proven, allow **C64-only protected** normal anchor main path without self-match and mark queried-normal-augmented LSO as `BLOCKED`, not falsely passed.

## 5. Versioned anchor evidence

C64 normal anchors: 64 precommitted official sample groups, disjoint from four supports, stream, probe; their features cached once; protected/read-only and never admitted. Query normal/defect anchors: only queried after label disclosure; record t, source key, camera, source memory version, contribution to mutable codebooks, full feature reference, current rescore version.

Anchor score stored at memory_version v is **invalid** for calibration state M_w (v != w) unless explicitly recomputed or proven score-state invariant. Use keyed cache `(anchor_key, camera, memory_state_digest, scorer_config_digest, score_kind, source_exclusion_policy)`, invalidate on changed hashes. Recompute score under exact `score_bank` pathway with corrected per-source source exclusion where applicable. A memory event that does not change a prototype does NOT increase version and does not trigger rescoring.

Eligible normal = protected C64 + any legally query-verified normal with correct LSO or excluded; eligible defect = prior query-verified defects (not written into normal bank). Distinct class evidence: `NORMAL_ONLY`, `DEFECT_ONLY`, `BOTH`, `NONE`. Under C64 `DEFECT_ONLY`/`NONE` likely rare but must be tested on synthetic Strict4/empty evidence.

`FPR_UNAVAILABLE` and `FNR_UNAVAILABLE` are explicit diagnostic states; never replace unavailable side by 0. Small-sample confidence upper bounds may be descriptive unless proven valid under adaptive selection and dependence; do not label them certified without assumptions.

## 6. Query tape contract and evaluator separation

Historical `risk_safe.query_protocol.QueryTape` is useful for identity validation, label-blind record design, timestamps and replay but **its flat priority/nested-budget construction differs** from proposed new **Q-A-TS 16-bin time-stratified 3%** schedule. Implement a new protocol version, separate from historical Q-A. Calculate budget using full number of image observations; reject impossible one-per-group schedules before scores/labels. Seed/group/view selection must be a pure function of public metadata and frozen seed; do not use class prevalence or future outcomes. Q-B uses approximately 20% of budget in first 400 images, if feasible, with a published exact rounding/redistribution rule.

Keep `PrivateEvaluationOutcome` outside the controller namespace; in tests, attempt to pass current/future label to controller via Observation should raise an error. Stage 2 labels must not be opened during Stage 0/1 research (apart from label-blind data integrity metadata).

## 7. Profiles and state ownership

| Profile | Memory | Score | Query | Intended inference |
|---------|--------|-------|-------|-------------------|
| `calibration_isolated` | one shared M_t | one shared s_t | common Q-A-TS/B tape | conditional calibration effect; identical AUROC |
| `factorial_isolated` | each configured arm may differ | **recompute for each state** | typically matched tape | memory/correction interaction, not shared score |
| `full_closed_loop` | independently evolving | independently evolving | BEA can react to controller output | total policy effect with query/memory mediation |

In Stage 0 implement only first profile, with typed registries permitting later components. No extra speculative POMDP controller, new memory system or online feature fine-tuning.

## 8. Engineering cost and caching

Freeze feature embeddings, reuse them across controller arms. Avoid repeated DINOv2 forwards at each memory version; redo only original native NN scoring against changed memory for protected/query anchors. Track actual rescore calls, time, persistent anchor bytes, maximum VRAM, CPU threshold computations. AUC equality should compare the **identical original-score vector**, not independent rounding from eight scorer instances.

Checkpoints must contain: stream index, query tape digest, feature/source config digests, original fixed support hash, current mutable bank state + versions + prototype source map, each controller `state_dict`, anchor ledger and RNG states, emitted rows checksum. Resume must be idempotent, fail on mismatched provenance, and never overwrite complete outputs.
