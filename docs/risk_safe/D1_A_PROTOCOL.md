# D1-A — Dataset and Query-Budget Preflight Protocol (Risk-Safe Memory Admission)

**Stage:** D1-A **revision R1** (metadata-only preflight; official Real-IAD sample identity).
No feature extraction, no DINOv2 loading, no CUDA allocation, no anomaly-detection metric,
no Admission algorithm.
**Branch:** `risk-safe-admission-sprint-20261027`
**Starting HEAD:** `4bcd8d389b4aacf4c654167ec1e29bbc609955ea`
**Identity version:** `realid-offical-sample-key-v2` (primary grouping `official_sample_key`,
sensitivity grouping `directory_serial_key`)
**Code:** `experiments/risk_safe/{__init__,query_protocol,preflight}.py`, `tests/test_risk_safe_d1a.py`
**Outputs:** `outputs/risk_safe_sprint_20261027/d1a_preflight/{dataset_preflight,query_budget_preflight,query_tape_manifest,source_manifest}.json`
**Supersedes:** `outputs/risk_safe_sprint_20261027/_superseded_d1a_preflight_serial_identity/`
(serial `S####` grouping, preserved unmodified with a `SUPERSEDED.md`).
**Authority:** `docs/RISK_SAFE_ADMISSION_7DAY_PLAN_2026-10-27.md` (v1.1), §2.1–2.5 and §4;
official identity semantics from
<https://github.com/Tencent/AnomalyDetection_Real-IAD> (README §3, `--sample_key_pat`).
**Status:** implemented and executed locally. Nothing in D1-A establishes CDARG effectiveness, Sidecar
activation or any online-detection performance. D1-B and later milestones are **not** started.

---

## 1. Purpose and scope

D1-A builds the metadata and query-budget infrastructure that every later milestone depends on:

1. how many independent official samples and original camera observations actually exist;
2. whether support / mature-history / stream / evaluation-probe partitions are sample-disjoint;
3. how to generate deterministic, label-blind query events under **absolute** human-label budgets;
4. whether two Admission methods can be guaranteed to receive *identical* query events and labels;
5. whether candidate budgets can provide enough normal-memory and defect-anchor opportunities;
6. whether the resulting manifests are reproducible and auditable.

Explicitly **out of scope**: feature caches, DINOv2, GPU work, Greedy/CDARG/SAEG/HITL implementations,
Sidecar execution, Q-B/Q-C controllers, any AUROC/FPR/FNR number.

---

## 2. Metadata protocol

For each of the six development classes

```
audiojack, bottle_cap, eraser, plastic_plug, switch, rolled_strip_base
```

the preflight reads the released per-class JSON through the **existing** parser
`vmb_visionad_new.RealIADJSONIndex` (unchanged) and derives identity only from paths:

| Field | Source | Used for |
|---|---|---|
| `path` / `rel_path` | JSON `image_path` + `meta.prefix` | manifest identity and digests |
| `is_good` | JSON `anomaly_class == meta.normal_class` | offline diagnostics, probe composition |
| `official_sample_key` | `<category>_<NNNN>_<CONDITION>` (official rule) | isolation unit, query uniqueness |
| `directory_serial_key` (`S####`) | sample directory | sensitivity diagnostic only |
| `capture_session` (`OK`, `NG_<CODE>`) | directory + file name | diagnostics, legacy comparison |
| `camera_id` (`C1..C5`) | file name | camera-view accounting |
| `timestamp` (`YYYYMMDDHHMMSS`) | file name | intra-phase chronological order |

The label is **not** part of any identity, ordering or selection rule.

---

## 3. Real-IAD identity: three distinct identifiers

The preflight keeps **three** identifiers and never lets them collapse into one:

| Identifier | Definition | Role |
|---|---|---|
| `official_sample_key` | `<category>_<NNNN>_<CONDITION>`, CONDITION = `OK` or `NG[_<DEFECT>]` | **primary grouping** (partitioning, query uniqueness, leakage decisions) |
| `directory_serial_key` | the `S####` directory name | identity-sensitivity diagnostic **only** |
| `image_id` | class-relative path of the original image | exact image identity in manifests and digests |

The primary key is extracted with the expression used by the official evaluation:

```
--sample_key_pat "([a-zA-Z][a-zA-Z0-9_]*_[0-9]{4}_[A-Z][A-Z_]*[A-Z])_C[0-9]_"
```

(Tencent/AnomalyDetection_Real-IAD, README §3; the same argument is used for every Real-IAD
setting.) Examples: `audiojack_0001_OK`, `audiojack__0001_NG_ZW`. The extracted key was verified to be
**byte-identical to the repository's historical `final_decision_core.object_id`** on 151,050 released
records across all 30 categories (0 misses, 0 mismatches): the historical V0.2 grouping already used
official sample semantics.

**Verified parser error in the D1-A first pass.** The first implementation used `S####` as the
physical object. One directory hosts the OK session *plus one session per defect code*, so that rule
merges distinct official samples. Measured on the six development classes, 123–168 directories host
more than one official sample, up to **5** per directory (`OK`, `NG_AK`/`NG_BX`, `NG_HS`, `NG_QS`,
`NG_ZW`), all captured at different timestamps; 127–171 of them cross the OK/NG boundary. Example:

```
S0001  ->  audiojack_0001_OK      OK/S0001/audiojack_0001_OK_C1_20231021130235.jpg
           audiojack__0001_NG_BX  NG/BX/S0001/audiojack__0001_NG_BX_C1_20231023111538.jpg
           audiojack__0001_NG_HS  NG/HS/S0001/audiojack__0001_NG_HS_C1_20231023091121.jpg
           audiojack__0001_NG_QS  NG/QS/S0001/audiojack__0001_NG_QS_C1_20231023093914.jpg
           audiojack__0001_NG_ZW  NG/ZW/S0001/audiojack__0001_NG_ZW_C1_20231023100717.jpg
```

Consequences of the wrong rule (`dataset_preflight.json -> identity_comparison`):

| Class | official test samples | serial "samples" | serial dirs shared with train | dirs crossing OK/NG |
|---|---|---|---|---|
| audiojack | 763 | 315 | 58 | 127 |
| bottle_cap | 757 | 311 | 53 | 123 |
| eraser | 767 | 321 | 62 | 126 |
| plastic_plug | 756 | 344 | 79 | 161 |
| switch | 764 | 348 | 90 | 167 |
| rolled_strip_base | 761 | 346 | 93 | 168 |

### 3.1 Three findings that must not be conflated

1. **Verified error in an identity parser** — grouping by `S####` merges distinct official samples.
   Proved by the counts above; corrected here.
2. **Possible physical-specimen correlation** — several official samples share one `S####`, and 53–93
   serials appear in both the train (OK session) and the test (defect sessions) part of a class. The
   released metadata does not state whether these are the same manufactured specimen. Recorded as
   **residual uncertainty**, not as leakage.
3. **Demonstrated evaluation leakage** — requires an overlap of *official sample keys* across
   partitions. Measured overlap: **0** in every class, asserted by the partition audit and by
   `all_classes_train_test_sample_disjoint`.

### 3.2 Fail-closed validation

`MalformedImageName` is raised when any of the following fails: the structural name pattern
`<class><_|__><NNNN>_<OK|NG>[_<DEFECT>]_C<1..5>_<YYYYMMDDHHMMSS>.<ext>`; the official sample-key
expression; agreement of the key with the class and condition in the name; a `S####` directory that
agrees with the numeric token (validated for layout, never used as identity); a real capture
timestamp; `OK` without a defect code and `NG` with one; and uniqueness of image path, file name and
class-relative path within a class.


## 4. Camera-view semantics

* One image = one original camera observation of `(official sample, capture session, camera, timestamp)`.
* **Every official sample has exactly five views, `C1..C5`**: `images_per_official_sample` is `{"5": N}`
  for all six classes, with zero samples missing a camera and zero samples with a duplicated camera id
  (in the D1-A-R1 outputs both counters are 0). The distribution is exactly uniform per class and split
  (audiojack test: 763 images per camera; audiojack train: 247 per camera).
* The superseded serial grouping produced "groups" of 5/10/15/20/25 images; that mixture was an
  artefact of merging sessions and is no longer used.
* **No label propagation.** Inside an `NG` session, individual views are labelled `OK` when that view
  shows no defect. Measured (test split, official samples):

  | Class | images labelled OK inside defective samples |
  |---|---|
  | audiojack | 1,289 |
  | bottle_cap | 924 |
  | eraser | 1,117 |
  | plastic_plug | 1,151 |
  | switch | 596 |
  | rolled_strip_base | 40 |

  Therefore a sample-level defect label does **not** label its sibling views, and evaluation keeps
  image-level truth. Symmetrically, a sample whose images are all normal-labelled never contains a
  defect-labelled image (checked, fails closed).
* Mixed-label official samples number 30–475 per class (audiojack 406).

---

## 5. Query accounting: counts are not images, and groups are not capture sessions

Historical V0.2 used a fixed 3% *image-level* rate with at most one queried view per group. Reproduced
unchanged from `v02_architecture_core` + `final_decision_core` for **every** class and support seed:

```
525 samples × 5 camera views = 2,625 stream observations
N_queries = round(0.03 × 2,625) = 79
image-query fraction        = 79 / 2,625 = 3.0095 %
unique-sample audit fraction = 79 / 525  = 15.0476 %
```

`query_budget_preflight.json → historical_accounting` stores this per class and asserts
`all_classes_reproduce_525_2625_79 = true`. The 3% figure therefore means **79 queries, never ~16**.
Because the official sample key *is* the historical grouping unit, the same 525/2,625 denominators are
restored in D1-A-R1, and the new generator reproduces them exactly (`§12`).

The new generator takes an **absolute** budget `B` (image-level human confirmations per class) and
reports three quantities side by side:

```
image_query_fraction       = N_queries / N_images
unique_sample_audit_fraction = N_unique_queried_samples / N_samples
human_verification_cost    = N_queries image-level confirmations
```

---

## 6. Partition construction (official-sample-disjoint)

Per class, with the historical V0.2 seeds and sizes (`SplitConfig`):

1. `mature_history`: 64 verified-normal train samples, `random.Random(190021)`, one camera image each.
2. `support`: 4 strict-Cold samples per support seed (0, 1, 2), `random.Random(seed)`, one camera
   image each, `random.Random(seed + 891)` for the view; support samples must not be history samples.
3. `reserved`: the **union** of every support sample across all registered seeds and the 64 history
   samples. This is a defensive guard: the released Real-IAD train/test splits are already disjoint at
   official-sample level, so it removes nothing (`reserved_samples_that_existed_in_test_split = 0`) and
   the historical partition is reproduced exactly. The guard would exclude a genuine overlap if a
   future dataset had one; a *shared `S####`* never triggers it.
4. `probe_A`, `probe_B`: 24 normal + 24 defect samples each (`random.Random(2201)`), all their images,
   **evaluation-only**.
5. `stream`: the remaining samples, shuffled with the same RNG, split into three contiguous phases of
   exactly **175 samples** each, as in the historical protocol.

Capacity is verified per class *before* building
(`dataset_preflight.json → classes.<name>.partition_feasibility`): the historical design needs
`621 = 2×48 probes + 3×175 stream` samples, `≥48` normal and `≥48` defect samples. Measured test
samples per class: 756–767 (all feasible), with 242–250 all-normal and 511–521 defect-containing
samples. If any requirement is impossible the preflight fails with the required and available counts
rather than shrinking a partition.

Disjointness is asserted pairwise across `support | mature_history | stream | probe_A | probe_B`; any
shared official sample key raises `DatasetProtocolError`.

---

## 7. Q-A — shared label-blind random query tape

For a class, stream, seed and budget:

1. take all eligible official samples (each with ≥1 stream image);
2. build **one** deterministic random permutation
   `random.Random(sha256("seed|Q-A|class|sample-priority|stream_digest"))`;
3. take the first `B` samples;
4. for each chosen sample draw **one** camera image with its own RNG identity
   `random.Random(sha256("seed|Q-A|class|camera|<sample_key>"))` over that sample's stream images sorted
   by `(timestamp, camera_id, path)`;
5. map the chosen images to their stream positions and sort ascending;
6. digest the tape (SHA-256 over canonical JSON of policy, class, seed, budget, `N_images`,
   `N_samples`, and the event list).

Structural label blindness: the selector receives `StreamImage` records that have **no label field**, so
`is_good`, `anomaly_class`, masks, probe labels, scores and adaptive state are unreachable by
construction — not merely by convention. Labels enter only afterwards, through
`label_diagnostics` (offline accounting) and `replay_shared_tape` (disclosure after the decision).

`B > N_samples`, `B ≤ 0` and non-integer budgets raise `QueryBudgetInfeasible`; the budget is never
silently reduced.

---

## 8. Nested budgets

One permutation and one camera-choice map serve **all** budgets of a class/stream/seed, so
`tape(8) ⊂ tape(16) ⊂ tape(32) ⊂ tape(64) ⊂ tape(79) ⊂ tape(96) ⊂ tape(128)`, and the selected image of
any shared sample is byte-identical across budgets. `assert_nested` verifies sample subset **and**
`(position, rel_path, camera_id)` equality and is re-checked on every preflight run
(`nesting.all_nested = true` for all six classes).

Nesting is a *sampling* device only. Each budget must be replayed independently from the same initial
memory state: high-budget trajectories may not be reused or leaked into low-budget replays.

---

## 9. Q-B and Q-C — declared interfaces only

**Q-B — Shared Frozen-Reference Informative Query (not implemented in D1-A).** A separate reference
query policy may read only the current image identity, its own frozen reference scorer, the labels it
has itself already obtained (with their ordering) and a shared initial `theta0`. It may **not** read any
compared method's STM/LTM/Buffer, Signed Correction or dynamic Sidecar state, nor future or probe
labels. It emits one frozen tape that is replayed unchanged to every Admission method, so any
difference between methods is attributable to Admission, not to supervision. Unused budget is saved and
reported, never backfilled with future labels. Q-B is **not** a reproduction of the original fully
dynamic v22 controller; AIF utility terms that require dynamic memory must be explicitly disabled and
recorded.

**Q-C — Native AIF-inspired closed loop (not implemented in D1-A).** The original controller is
restored so each method queries with its own legal state and may therefore follow a different
trajectory. Budget cap `B`, stream, initial support and supervision permissions are shared; actual
label use must be reported and the comparison is called **budget-cap-matched**, not count-matched.

The contracts are emitted verbatim in
`query_budget_preflight.json → query_policy_interfaces`.

---

## 10. Causal label access

The replay contract (`replay_shared_tape`) implements, per time step `t`:

1. the image arrives and **every** method produces its pre-query prediction;
2. the **pre-registered tape** (not the method) decides `q_t`;
3. only then is the *selected image* label revealed;
4. the label is delivered to every method in the same deterministic order;
5. it becomes evidence for `t+1` only.

Guarantees enforced by code and tests:

* every method receives the same ordered events (position, sample key, path, label) — parity digests are
  equal;
* `labels_visible_at_prediction` equals the number of earlier disclosures at every step
  (`causal_label_visibility.N_violations = 0` on all six classes);
* flipping the label of the first queried image leaves every prediction up to and including that step
  unchanged — a label cannot influence the decision that reveals it;
* a missing label raises instead of being skipped.

Two simulated consumers (`PassiveAdmission`, `RunningMeanAdmission`) are provided for the parity test;
no research Admission algorithm is implemented.

---

## 11. Sidecar activation feasibility (metadata only)

The declared defaults are read from `test_phase3_AIF_simplified_v22.GuardedThetaSidecarConfig` by
`ast.literal_eval` — the module is *not* imported, so no torch/backbone side effect occurs, and the
documented values (32 / 16 / 24 / 6) are asserted against the source. No threshold is changed; a
silently edited value makes the preflight raise.

Mechanisms, their real gates, and the smallest budget for which the *count* requirements could be met:

| Mechanism | Enabled by default | Gates | Min budget |
|---|---|---|---|
| `dual_anchor_full` | yes | recent ≥ 24, ≥ 6 normals, ≥ 6 defects | 24 |
| `normal_dominant_fallback` | yes | recent ≥ 24, normals ≥ 16, defects < 6, `bufA` ≥ 2, recent FPR ≥ 0.25 | 24 |
| `fast_upward_rescue` | **no** | ≥ 2 verified FP normals, recent FPR ≥ 0.35 | 2 |
| `balanced_rescue` | **no** | normal anchors ≥ 8, defect anchors ≥ 3, cooldown 150 | 11 |

Audit findings recorded in `sidecar_feasibility.declared_threshold_roles`:

* `min_recent_total` and `min_recent_per_class` are **hard gates** on the recent labelled window;
* `min_normal_anchor` / `min_defect_anchor` are **not gates** in the current code: they select a
  quantile branch in `_candidate_thetas`, and the fallback branch computes the same quantile whenever
  `bufN > 0` / `bufA > 0`. Exceeding them is neither necessary nor sufficient;
* update opportunities are capped by `update_every_labeled = 8`, so `floor(B/8)` controller evaluations
  at most, and the recent window retains at most 96 labels.

Per budget: 8 and 16 → **mathematically impossible** for both enabled mechanisms; 24 and above →
**potentially feasible**, subject to realised label composition. `actual_activation` is always
`not_established_without_online_controller_execution` — D1-A cannot and does not claim activation.

---

## 12. Measured D1-A-R1 stream and budget consequences

`query_tape_manifest.json` (seed 37, seven nested budgets, six classes; the stream is 525 official
samples × 5 views = 2,625 images in every class):

| Class | stream samples | stream images | B=79 defect labels | B=79 normal labels | B=79 image-query | B=79 sample-audit |
|---|---|---|---|---|---|---|
| audiojack | 525 | 2,625 | 27 | 52 | 3.0095 % | 15.0476 % |
| bottle_cap | 525 | 2,625 | 33 | 46 | 3.0095 % | 15.0476 % |
| eraser | 525 | 2,625 | 34 | 45 | 3.0095 % | 15.0476 % |
| plastic_plug | 525 | 2,625 | 35 | 44 | 3.0095 % | 15.0476 % |
| switch | 525 | 2,625 | 42 | 37 | 3.0095 % | 15.0476 % |
| rolled_strip_base | 525 | 2,625 | 51 | 28 | 3.0095 % | 15.0476 % |

* The absolute generator reproduces the historical 3.0095 % / 15.0476 % denominators exactly at
  `B = 79`, and 79 queries hit 79 distinct official samples (one label per sample).
* Image-query and sample-audit fractions for the other budgets (identical in all six classes because
  the stream size is identical): `B=8` 0.3048 % / 1.5238 %, `B=16` 0.6095 % / 3.0476 %, `B=32`
  1.2190 % / 6.0952 %, `B=64` 2.4381 % / 12.1905 %, `B=96` 3.6571 % / 18.2857 %, `B=128` 4.8762 % /
  24.3810 %. **79 is the only candidate that reproduces the historical 3 % operating point**; it is a
  reference, not a preferred budget.
* Under the official grouping the random tape yields far more defect evidence than under the
  superseded serial grouping (27–51 vs 12–27 defect labels at `B = 79`), so the declared
  `min_defect_anchor = 16` is reached in every class at `B ≥ 64` for this tape.
* Sidecar feasibility verdicts are unchanged: `B = 8, 16` mathematically impossible for both enabled
  mechanisms; `B ≥ 24` potentially feasible, composition-dependent; `actual_activation` is always
  `not_established_without_online_controller_execution`.

---

## 13. Known limitations

1. The stream is **phase-blocked**, not one global capture timeline: phases hold disjoint sample
   cohorts photographed in overlapping wall-clock windows, so raw timestamps are not monotone across
   phase boundaries. Ordering is deterministic and chronological inside each block.
2. Probe composition is sample-label stratified (evaluation-only) and therefore depends on labels; the
   *query tape* given a fixed stream does not.
3. Only one query seed (37, the historical V0.2 tape seed) is registered. Seed sensitivity is tested
   but not pre-registered as an experimental factor.
4. Budget candidates are feasibility candidates, not approved experimental budgets; the first-round
   grid must still be registered before any new results are read.
5. **Physical manufactured-specimen identity is undocumented.** Several official samples share one
   `S####` directory (up to 5), and 53–93 directories appear in both the train (OK session) and test
   (defect-session) parts of a class, captured days apart. Whether these are the same product is not
   stated by the dataset. The official protocol treats them as different samples; a stricter
   specimen-level protocol would be a separate, explicitly registered experiment and would cost up to
   half of the test groups.
6. Real-IAD label structure (view-level `OK` inside `NG` sessions) is preserved, not "cleaned".
7. D1-A says nothing about Admission quality, Sidecar activation, query informativeness or online FPR.

---

## 14. Reproducibility and validation

```powershell
& "D:\anaconda3\envs\torch\python.exe" -m unittest discover -s tests -p "test_risk_safe_d1a.py" -v
& "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.preflight `
  --dataset-root "G:/dataset/realiad" `
  --json-dir "G:/dataset/realiad/jsons/realiad_jsons" `
  --out-dir "outputs/risk_safe_sprint_20261027/d1a_preflight"
& "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.preflight ... --verify-only
```

* Outputs are written only if absent; an identical rerun reports `reused`, `--verify-only` reports
  `verified`, and any semantic or byte difference raises `OutputMismatch` instead of overwriting
  evidence. A preserved manifest carrying a different `identity_version` is rejected, so the archived
  serial-identity outputs can never be mistaken for current evidence.
* JSON is serialised deterministically (stable key order, `indent=2`, UTF-8, LF, `allow_nan=False`);
  `source_manifest.outputs_sha256` equals the SHA-256 of the exact file bytes.
* Every payload carries `identity_version` / `grouping_unit`, and tape digests bind the identity
  version, so Q-A tapes from different identity definitions cannot collide.
* `source_manifest.json` records branch, HEAD, tracked modifications, interpreter/torch versions,
  SHA-256 of every D1-A source and of the legacy modules it imports, per-class JSON digests, a
  semantic index digest over 30,225 records, the configuration digest and the output digests.
* No output contains wall-clock timestamps, so byte-identical reruns are possible.

Test coverage (`tests/test_risk_safe_d1a.py`, 52 tests, CPU only, no GPU, no model): official sample
identity (five-view grouping, OK vs NG, defect codes, equality with the official expression and with
`final_decision_core.object_id`), separation of the three identifiers, five-view structure reporting,
malformed names, duplicate paths, sample-disjoint splitting, capacity verification, duplicate-sample
failure, label-structure guard, query determinism, seed sensitivity, label blindness, nested budgets,
one image per sample, chronological positions, historical 525/2,625/79 parity of the new generator,
replay parity, post-query-only disclosure, impossible budgets, identity-version supersession,
source/manifest consistency, idempotent output handling and a full six-class synthetic end-to-end
preflight.

---

## 15. Superseded evidence

`outputs/risk_safe_sprint_20261027/_superseded_d1a_preflight_serial_identity/` holds the four manifests
of the first D1-A pass together with `SUPERSEDED.md`, which records the SHA-256 of every archived file
and why the identity rule changed. A second, earlier archive
(`_superseded_d1a_preflight_first_pass/`) holds a wording-only predecessor of that same serial run.
Neither directory is valid evidence for the current protocol; both are preserved unmodified.
