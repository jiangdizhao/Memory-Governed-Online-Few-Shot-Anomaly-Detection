# Stage 0.75-B synthetic audit package v8

**Primary contract:** candidate v3. **Runner status:** `BLOCKED`.
The package links to `outputs/calibration/stage075b_synthetic_v3_blocked_final_20261029/`, which is local and not committed. The full suite passed 133 tests; G1–G15 and independent artifact checks passed, while G16/Control E is BLOCKED. The repeated matrix contained 48 rows with identical trajectory digests. Four checkpoint cuts replayed exactly.

`results_compact.json` contains all per-seed arm metrics, paired BA differences, shared ranking results, source counts and trajectory hashes. `scenario_status.json` records S0–S14 plus control A; `control_status.json` records controls A–E and the Control E blocker. `report_provenance.json` binds the written reports to this package. `frozen_configuration.json`, `method_cards.json`, `gates.json`, `test_results.json`, `source_provenance.json`, and `runner_artifact_linkage.json` capture the audit inputs and evidence. No raw event trace, image, model, checkpoint, or private label data is included.

The full B8-Full threshold trace equaled actual B3 in all 48 trajectories. It showed no incremental synthetic BA. Control E is blocked because B8-M/S/P are not one-toggle Full variants, and dropping the two-class condition would leave the paired gain undefined. This is a negative mechanism observation with incomplete ablation acceptance, not proof that no B8 formulation can work. Candidate v1/v2 artifacts remain preserved; v2 is superseded because its optimizer did not consume the percentile state. See `STAGE075B_VERSION_LINEAGE.md` and the reports under `docs/calibration/`.

`checksums.json` lists SHA-256 for every package file except itself. Recompute those hashes to verify integrity. No Stage 0.75-C, Stage 1/2, Real-IAD performance evaluation, Mint/PCB, masks, or model tuning was performed.
