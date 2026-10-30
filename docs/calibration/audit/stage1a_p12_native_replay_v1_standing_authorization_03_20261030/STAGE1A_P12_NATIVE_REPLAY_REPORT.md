# Stage 1-A P1.2 bounded Native replay and dense evidence

**Decision: `P12_BLOCKED`**

**Branch:** `calibration-stage0-contract-20261029`<br>
**HEAD:** `5cfa3d0f64e51850457ae0494c180809f7b064e6`<br>
**Q-A-TS binding:** `dbe72a2f989ee314becc07e0c8b8c7a8353ffacabbfc93863d5666cff9d28183`

## Stop reason

The run stopped in metadata/source preflight before opening Real-IAD image bytes,
decoding any label, initializing the DINOv2 model, or issuing a Native scoring
call. The Stage 0.75-A protocol and Stage 0.5 score manifest agree with each
other on the older source hashes, but two current source files differ. The exact
three-way comparison is recorded below and in `source_provenance.json`.
Expected older source bytes were not found in reachable Git history. This is a
provenance/parity failure, not an approval-record issue; the check was not
bypassed. The P1/P1.1/P1.2 runtime source files are SHA-256 bound at runner
entry in `source_provenance.json`; this identifies the blocked P1.2 execution
code but does not recover the older historical source bytes.

| Source | Stage 0.75-A protocol | Stage 0.5 manifest | Current source | Manifests agree |
|---|---|---|---|---|
| `experiments/calibration/stage05.py` | `af1ef435addfe2f08cb2e40a1d141b617fb15c852ed469c8aa7062663ccf8626` | `af1ef435addfe2f08cb2e40a1d141b617fb15c852ed469c8aa7062663ccf8626` | `4b8aae252f5f1e096b127906794064da269aa79f7cd5514f6f738cda585c017e` | True |
| `experiments/calibration/stage05_runner.py` | `1e40d9c5e48c252eef3d96686d2a315c4a87f1f325130a6c2961f7cc41cafaea` | `1e40d9c5e48c252eef3d96686d2a315c4a87f1f325130a6c2961f7cc41cafaea` | `1d110e1ca813a064767ebeab597b609b3d863f8afd4c8085b06948ad88347e14` | True |

The user's standing authorization was accepted as the authorization basis;
no approval records were created. The verified query-tape aggregate is recorded
above. Actual queried-label reads, image feature encodings, static-map calls,
anchor-score calls and replay events are all zero.

## Gate status and scope

Phase A exact parity, all 20 trajectory parity, dense causal anchor evidence,
and checkpoint/resume are **BLOCKED_NOT_RUN**. Historical C64 status, if
verified, is artifact linkage only and not independent numerical rescoring.
Historical G2 remains BLOCKED. No Stage 1-A performance comparison, metric
calculation, Mint/PCB access, or mask access was performed.

The final decision is `P12_BLOCKED`. Proceeding requires recovery or researcher
review of byte-identical source provenance; no historical audit evidence was
modified.

## CPU validation

Python compilation passed and the P1/P1.1/P1.2 CPU suite passed **35 tests**. Query-tape metadata preflight and historical C64 artifact-linkage checks passed. The real Native replay remained blocked before any event, feature encoding, queried-label access, or Native scoring. See `test_results.json` and the machine-readable gate matrix.
