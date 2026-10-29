# Historical writer source analysis

Reviewed source: `experiments/calibration/stage075a_b3_headroom_runner.py` at reference commit `9c7ec2757e7eec731211bef2b3939a8e76cbe7cd`.

- `_checksums` (lines 41-44) recursively hashes sorted files except `checksums.json` and `complete.json`; it does not exclude `INCOMPLETE`.
- The runner creates `INCOMPLETE` with bytes `Actual-B3 replay/headroom diagnostic is incomplete.\n` (line 53).
- It writes the first `checksums.json` from that state (lines 128-130).
- It removes its marker and then writes `complete.json`, recording the first checksum-manifest SHA-256 and `files_checked=len(checksums)` (lines 136-141).
- It then runs `_checksums` again and rewrites `checksums.json` (lines 144-146). With the marker removed, the resulting manifest has four entries.
- `_atomic_json` writes `canonical_json(value) + newline`; `canonical_json` sorts keys, uses compact separators, UTF-8 and disallows NaN.

Independent reconstruction uses the four current verified entries plus the exact historical marker bytes in memory. The reconstructed five-entry canonical file has the exact SHA-256 referenced by `complete.json`. This certifies the checksum-linkage inconsistency and its reproducible writer sequence.

## Provenance limitation

The artifact records completion at `2026-10-29T19:12:36.101944+00:00` (2026-10-30 06:12:36 +11:00). Git history records runner source commit `1e8e66c512984ea34d74564b461a6a9072b7a02e` at `2026-10-30T06:21:44+11:00`, about nine minutes later; its parent is the artifact's `head_before_diagnostic` value `d699ad63a19e207e8b58e7cc42aee8c58abc48cc`.

The run manifest does not carry a source hash for the executed runner, and no execution log or pre-refresh manifest is retained. Therefore the exact executed source bytes cannot be established from available evidence. The source analysis explains the observed output but does not claim proof of byte-identical execution source.
