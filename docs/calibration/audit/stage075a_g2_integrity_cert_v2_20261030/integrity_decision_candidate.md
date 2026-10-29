# G2 integrity decision candidate

**Independent certificate: PASS.** The current four payload checksum entries and run-manifest content links verify. The five-entry manifest reconstructed in memory, including the exact transient `INCOMPLETE` marker bytes, matches the completion pointer. The one-byte mutation-negative checks and before/after historical hash comparison are recorded in `independent_tests.log` and `integrity_verification.json`.

**Original G2 status: BLOCKED.** This certificate documents package integrity and reproduces the completion-manifest discrepancy. It does not itself close the original G2 gate.

**Acceptance: RESEARCHER_REVIEW_REQUIRED.** Exact executed runner source bytes remain unverified because no executed-source hash or retained execution log is available and the repository source was committed after the recorded run. Human review is required for closure.
