# Stage 1-A P1.1 Production Adapter Hardening

**Branch:** `calibration-stage0-contract-20261029`

**Base commit:** `40a50d4234b70b677d77e3db91e520bff13ef8e5`

**Scope:** CPU-only production integration checks; no new Real-IAD images or labels, GPU inference, or Stage 1-A evaluation.

## Decision

The P1.1 CPU adapter checks pass. Real Native replay remains **BLOCKED** until researcher approvals bind both Native/data access and queried-feedback access to the exact four verified Q-A-TS tape identities. This report does not authorize replay.

Historical Stage 0.75-A G2 remains **BLOCKED**. C64 verification establishes serialized artifact linkage only. The missing historical executed-source hash and query-response digest have not been recovered.

## Implementation and evidence

- **Historical writer schema:** the production adapter calls `stage075a_runner._writer_record` directly after applying the real `SharedNativeGreedyWriter`. CPU tests use that writer and its result object, checking the full historical schema, eligibility and action, versions, source revisions, layer counts, source-map digest, resident-source statistics, append, replacement, and skip behavior.
- **Frozen query tapes:** all four stored Stage 0.5 Q-A-TS-v1, seed-37, 3% tapes were verified from frozen metadata and tape artifacts. Verification checks class identity, exact events and positions, stored digests, stream identity, official-group uniqueness, and group identity at each queried event before feature or label providers run. Public bindings retain counts and SHA-256 identities; raw positions and group identities remain private.
- **Approval binding:** the P1.1 front door rejects missing approvals and any authorization whose tape bindings differ. CPU mutation tests confirm this. No researcher-approved access record was supplied to this run.
- **All-layer LSO:** the adapter calls `AnchorLedger._without_source`, audits source rows against actual native memory rows in every layer, checks layer count and complete removal or pre-existing absence, and binds the score proof to the exact current memory/source map and excluded-bank digest. A negative test deliberately retains rows in one layer and verifies failure before scoring. The score callback in CPU tests is synthetic; no Native Scorer computation was performed.
- **Causal evidence snapshots:** the P1.1 runner builds an initial snapshot from 64 protected C64 scores before replay events. After each queried disclosure it verifies the historical writer record, post-event memory identity, and stored C64 identity, then refreshes `VersionedAnchorEvidenceCache` and produces the complete `EvidenceSnapshot`. Scores are reused when memory identity is unchanged. CPU tests verify feature ? label disclosure ? writer ? evidence ordering. Snapshots carry all 64 protected C64 scores, queried-normal LSO scores, queried-defect current-memory scores, and exact memory/evidence provenance. Synthetic input compatibility passes for B4, B7, and Stage075BOffset; controller code and hyperparameters are unchanged. No real queried-anchor scores were generated.
- **Checkpoint:** private state includes actual writer/source rows, queried-label cache, feature/static-map/evidence cache payloads and identities, causal anchors and LSO proofs, event cursor, response-chain digest, and compute counters. A serialized checkpoint is reconstructed in a child process and resumes to the same event/query-response chains and writer state as one-shot execution. Causal anchor/proof cache restoration and mutated response-chain rejection are covered by CPU tests. The package contains only digests and summaries, never the private checkpoint.
- **C64:** the stored Stage 0.5 starting vectors link to Stage 0.75-A pre-update vectors; per-transition before/after vector digests, q90 values, memory-version transitions, trajectory checksums, root audit-linkage checksum, and protocol-manifest checksum pass. Verified totals are 20 initial vectors and 1,190 post-write vectors/transport rows. This is `PASS_HISTORICAL_ARTIFACT_LINKAGE_ONLY`; independently recomputed Native C64 scoring is `NOT_RUN`.

### Frozen Q-A-TS counts

| Class | Queried positions | Tape SHA-256 | Position-set SHA-256 |
|---|---:|---|---|
| audiojack | 114 | `cb1b8b4342b23dc520a45ef72c014ceb3ffa1e976046aef726fcc0aaa45504d5` | `cdb256e0f4c07b32c87f46b557f4ce6290efd5d91b1c4e26a433097e7d6f5609` |
| regulator | 116 | `fe4dae9b355bad1ec7cd836293ee445b353f1a6168fb85bac997601355139ca5` | `6d71a90656b9aad91b0159df4522081b9478de9b3f5adc15704795928fcd2d3f` |
| switch | 115 | `b48928b1b547d68f2978769191662362c8fd5f57bd40d39b0126e5800648cc16` | `899edd0f87d93da37cbf23ab044fbe49a9923d5b09e2558d99946d746369f930` |
| rolled_strip_base | 114 | `c5068752da92dfe17ad198d1103e5b18f8433d6bed057007a13f712ccef3e8a2` | `505dbec8986e24647585bc5202a40d9f20f91594ee1deb8e632611282c28eefa` |

The aggregate verified tape-binding digest is `dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183`.

## Validation

See `p11_gate_matrix.json` and `test_results.json` for machine-readable status. Python compilation passed. The adjacent Stage 1-A regression set ran **33 tests, all passed**. Historical query-tape and C64 metadata preflights passed. `git diff --check` passed for tracked changes.

## Remaining blockers and limits

1. Researcher approval is still required before construction of any real-data or queried-label provider. Approval must bind the exact four tape identities recorded in `query_tape_bindings.json`.
2. Real Native Scorer numerical parity, feature encoding, and memory trajectory parity have not been exercised. The LSO source-filter path is integrated; the CPU score callback is synthetic.
3. No real evidence revision with newly queried anchors was produced. The causal snapshot adapter was verified with synthetic anchors and verified stored C64 values only.
4. Historical Stage 0.75-A executed source bytes and queried-response digest remain unverified; original G2 stays blocked.
5. Stored C64 transport linkage is not independent numerical Native scorer parity.

The frozen Stage 1 v3 protocol and focused addendum were not modified. No controller algorithm or hyperparameter was changed. No Stage 1-A performance evaluation was run.
