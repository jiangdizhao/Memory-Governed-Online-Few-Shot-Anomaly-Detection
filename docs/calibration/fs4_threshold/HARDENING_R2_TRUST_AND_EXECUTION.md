# FS4 R2 trusted execution: blocked

Source revision: `f39f31b53856ffddfbcce5f7733a0795d1ffd7a7` on `calibration-stage0-contract-20261029`. This is an engineering-only revision; no scientific experiment was run for R2.

The production manifest and context implement signed binding of the registered dataset and event order, Support identities and preflight bank digest, frozen checkpoint and model state, native configuration, image identity and geometry, and the full model feature and CLS outputs. The production executor accepts event positions, not caller-provided feature tensors. Production recovery re-encodes registered inputs and checks recorded provenance. The explicitly synthetic executor and ledger retain test-only interfaces.

These checks do **not** establish an independent producer boundary. A caller able to execute Python code in the same process can substitute `native.encode_pil` before the production context captures it. In the synthetic regression `test_production_rejects_preconstruction_encoder_substitution`, the forged encoder preserves valid Support outputs but changes an event output, which is accepted. The test is intentionally failing and is not a scientific run.

`ProductionContext` therefore raises `FS4_R2_BLOCKED` unconditionally at construction. This is a fail-closed hold, not a secure isolation mechanism: same-process code can patch the hold itself. Do not use this implementation for production or interpret its synthetic checks as proof of trusted execution. Completing the requirement needs an independently controlled encoder/executor process or service, protected control and IPC boundaries, and an independently verified producer identity, followed by the negative test and recovery tests against that real boundary.

The earlier source commit `59db97d485888d1e3ac37ca5056872989af06a54` was followed by the WIP revision above when review identified this unresolved vulnerability. Consequently the requested single source-freeze commit could not be preserved without concealing a real failure. Both revisions are retained; no history was rewritten and no push was performed.
