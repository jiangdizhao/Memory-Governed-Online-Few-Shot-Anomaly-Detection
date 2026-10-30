# B3 state-lag Gate A0 verification

Status: PASS for frozen-input feasibility and historical A2 recurrence parity.

This is an artifact and threshold/prediction replay audit. It is not a performance evaluation and does not test H1.

## Verified counts

- Trajectories: 20/20
- Content-changing memory updates: 1190
- Versioned C64 q90 states: 1210
- Historical A2 events: 76475
- Exact pre/post threshold mismatches: 0/0
- Exact prediction mismatches: 0
- Event/memory identity mismatches: 0

A2 uses q_v, A3 uses q_max(0,v-1), and A4 uses q_max(0,v-5); each arm uses the fixed Delta=0.01. The q90 uses the higher order statistic at zero-based index 57 of the fixed 64-score vector. A2 initialization equals q0. A state change is applied after the current prediction and affects the next prediction.

## Feasibility matrix

| Class | Seed | Distinct versions | Content changes | q90 history | Status | Evidence SHA-256 prefix |
|---|---:|---:|---:|---|---|---|
| audiojack | 0 | 74 | 73 | COMPLETE | PASS | 2395014d8e3b |
| audiojack | 1 | 75 | 74 | COMPLETE | PASS | c55e10c20646 |
| audiojack | 2 | 75 | 74 | COMPLETE | PASS | 66db8745a9cd |
| audiojack | 3 | 75 | 74 | COMPLETE | PASS | 8149b8aed858 |
| audiojack | 4 | 74 | 73 | COMPLETE | PASS | d0905d2b2503 |
| regulator | 0 | 81 | 80 | COMPLETE | PASS | 3d493b7f1823 |
| regulator | 1 | 79 | 78 | COMPLETE | PASS | 730614a1da84 |
| regulator | 2 | 78 | 77 | COMPLETE | PASS | 998ddab119c4 |
| regulator | 3 | 79 | 78 | COMPLETE | PASS | 632e28da102f |
| regulator | 4 | 77 | 76 | COMPLETE | PASS | 4d51b719613a |
| switch | 0 | 54 | 53 | COMPLETE | PASS | bc0a8d3410b4 |
| switch | 1 | 55 | 54 | COMPLETE | PASS | 4a8366e848dc |
| switch | 2 | 55 | 54 | COMPLETE | PASS | 04d13a3754cc |
| switch | 3 | 55 | 54 | COMPLETE | PASS | 9b9b43e10ae2 |
| switch | 4 | 54 | 53 | COMPLETE | PASS | e2b1e534c817 |
| rolled_strip_base | 0 | 34 | 33 | COMPLETE | PASS | 33954a05edc4 |
| rolled_strip_base | 1 | 34 | 33 | COMPLETE | PASS | 30cf2d7ce111 |
| rolled_strip_base | 2 | 34 | 33 | COMPLETE | PASS | 02ff8ae00b98 |
| rolled_strip_base | 3 | 34 | 33 | COMPLETE | PASS | 2d18a72762fd |
| rolled_strip_base | 4 | 34 | 33 | COMPLETE | PASS | f29c466a7faa |

## Provenance limits

- Original Stage 0.75-A/G2 source-byte parity remains BLOCKED_UNCHANGED; this audit does not retroactively validate historical source bytes.
- Frozen C64 vectors and memory transitions were verified from stored hashes and receipts; no Native rescoring or GPU inference was run.
- Outcome and metric artifacts were not read or used. Historical per-event prediction rows contain no label fields.
- Access count: 20 initial C64 vectors, 1,210 distinct versioned C64 vectors, 1,190 transitions, 76,475 event rows, 76,475 historical prediction rows, and 20 P1.2 evidence files hashed without decoding. No images, outcome/label files, or performance metrics were opened. One aggregate receipt schema-probe object was read outside this verifier and is disclosed in the manifest.
- No BA/FPR/FNR was calculated. H1 remains untested.

Full input hashes and memory identities are in feasibility_matrix.json and source_and_input_manifest.json.
