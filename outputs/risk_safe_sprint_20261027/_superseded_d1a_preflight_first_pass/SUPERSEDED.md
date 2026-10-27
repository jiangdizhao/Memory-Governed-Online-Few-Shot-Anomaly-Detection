# Superseded D1-A outputs (preserved, do not use)

These four JSON files were produced by the **first** execution of the D1-A preflight in the same
session, before one wording-only change in `experiments/risk_safe/preflight.py`
(`_tape_manifest_entry.stream.label_blind_construction`).

They are kept rather than deleted, in line with the rule "never delete or overwrite a previous output
automatically". The preflight refused to overwrite them (`OutputMismatch`), which is the intended
behaviour; they were then moved here manually and the canonical directory was regenerated.

**Do not cite these files.** The authoritative outputs are in
`outputs/risk_safe_sprint_20261027/d1a_preflight/`. The numbers inside are identical to the
authoritative run except for the wording of that single string and the resulting digests.
