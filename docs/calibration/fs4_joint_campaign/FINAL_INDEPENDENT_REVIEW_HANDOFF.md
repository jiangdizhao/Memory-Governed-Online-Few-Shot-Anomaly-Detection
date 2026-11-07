# Final independent review handoff

Review this freeze as a complete preregistration and trusted-local execution implementation. The machine-readable source of scientific settings is `CAMPAIGN_SPEC.json`, SHA-256 `1c96077bfdaa58f9ff93a67a9349a4cc6bc8b63578ae8915ff23762b50aee896`. `ARM_REGISTRY.json` and the nine operation names in `STAGE_A_STAGE_B_EXECUTION_PLAN.md` are the public arm and lifecycle contracts.

## Required review checks

1. Recompute the spec digest and compare all four populations, 3,059 groups, Support identities, model/config hashes, R7 seed/order, seed 37 query equations, L/E behavior, H1/H2 metrics and temporal boundaries.
2. Review the one-time common registration and three per-arm manifest wrapper; confirm the private mapping is never sent to Stage A and the broker uses distinct disclosure namespaces.
3. Review the new R4 integration and all source fingerprints. Confirm actual Support parity is checked against a separate receipt before first TEST event.
4. Review stage completeness, checkpoint selection/replay, resource upper bounds and result checksum verification.
5. Review the synthetic CPU/CUDA matrix and preserve the known high-security encoder-substitution failure as a separate adversarial result.

## Activation gates still open

- Independent review and separate `STAGING`, `SUPPORT_PREFLIGHT`, `RESOURCE_PREFLIGHT`, `STAGE_A` and `STAGE_B` authorizations remain required.
- Real-data registration, R7 source-order bijection, opaque mapping/pixel equality, and original Support path verification have not been run by this freeze task.
- The actual four-Support native-bank construction and per-category score/theta parity receipt have not been produced.
- Intended-host disk, RAM, GPU and complete synthetic encoder/rescore benchmark evidence must pass the resource gate. The conservative checkpoint estimate is 241.5 GB before reserve and 301.9 GB with the 25% reserve, before blinded image resources and the separate 2 GiB ledger/output reserve.
- No Stage A trajectory, complete-label projection or scientific Stage B metrics have been run. Therefore there is no empirical result or hypothesis conclusion.
- Trusted-local execution assumes the researcher controls the local Python environment. This implementation makes no malicious same-account isolation claim.

The next authorized research action after independent review is one complete S + CL + CE campaign with all four categories and one Stage B evaluation after all twelve Stage A category seals. Do not inspect outcomes to alter algorithms, parameters, population, initialization or metrics. Report a valid negative, inactive or operationally expensive outcome as observed.

Authorization records are JSON with schema `FS4_JOINT_CAMPAIGN_AUTHORIZATION_V1`, unique `authorization_id`, `authorized_by`, `authorized_at`, exact `scope`, campaign ID, exact specification SHA, a nonempty `authorization_statement`, and only the scope-specific binding fields listed by the launcher. They are an operator review record, not a cryptographic signature or same-account security boundary.
