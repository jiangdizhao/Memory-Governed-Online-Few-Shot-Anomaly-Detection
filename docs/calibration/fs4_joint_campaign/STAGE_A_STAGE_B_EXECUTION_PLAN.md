# Stage A and Stage B execution plan

## Inert interface and staged gates

`python -m experiments.calibration.fs4_joint_campaign` defaults to `freeze-only`; it reads only the frozen text specification and performs no data access. `validate-contract` checks machine/human arm identities. `prepare-real-data` requires a `STAGING` authorization, original registered metadata and a private outcome input; it reconstructs R7 seed-0 group order, verifies identities, then calls the reviewed R4 lossless pixel-preserving preparation once. The private original-to-opaque mapping is written once and shared by all arms.

`verify-support-preflight` requires a separate `SUPPORT_PREFLIGHT` authorization and an independently produced actual four-Support native receipt. It verifies exact paths, support content/groups, fixed-bank digests, four self-scores, the cold-start formula, Writer sigma, empty Greedy/C1, model and native config. It then binds the same registration and receipt into the three trajectory manifests. This task does not produce the actual native receipt.

`resource-preflight` requires a separate authorization and records the synthetic benchmark and host/disk checks. Stage A is refused unless its immutable receipt is PASS. `run-stage-a` then requires a distinct `STAGE_A` authorization bound to registration, Support receipt, resource receipt and all three manifest hashes. The arms run sequentially with separate ledgers, broker disclosure namespaces, memories and checkpoints. No Stage B outcomes are projected between arms. Every 1,000 events and category boundary is checkpointed. Recovery replays only from the latest committed checkpoint; a corrupt or ambiguous tail fails closed. Existing ledgers are never truncated.

`verify-stage-a` independently validates all S/CL/CE ledgers, twelve category seals, global seals, identities, pre-event threshold/prediction chronology, query/feedback receipts and Support bootstrap. A subset cannot authorize Stage B.

`run-stage-b` requires a separate `STAGE_B` authorization bound to all three manifest hashes, Stage A seals, campaign registration and private mapping identity. It verifies all Stage A streams before launching the R4 mapping projector once. It then compares every queried label against its branch feedback and computes the registered CPU-only metrics. The projection, Stage A score projections, telemetry, metrics and completion receipt are checksummed in a fresh protected output root. `verify-complete` verifies file hashes and recomputes the full metric report.

## Commands (documented, not executed in this freeze)

```powershell
python -m experiments.calibration.fs4_joint_campaign freeze-only
python -m experiments.calibration.fs4_joint_campaign validate-contract
python -m experiments.calibration.fs4_joint_campaign prepare-real-data --campaign-id <id> --campaign-root <fresh-root> --private-root <fresh-private-root> --image-root <fresh-image-root> --original-registration <file> --original-ledger <file> --labels <private-file> --authorization <STAGING-auth.json>
python -m experiments.calibration.fs4_joint_campaign verify-support-preflight --campaign-root <root> --support-receipt <receipt.json> --authorization <SUPPORT_PREFLIGHT-auth.json>
python -m experiments.calibration.fs4_joint_campaign resource-preflight --campaign-root <root> --benchmark-receipt <synthetic-benchmark.json> --authorization <RESOURCE_PREFLIGHT-auth.json>
python -m experiments.calibration.fs4_joint_campaign run-stage-a --campaign-root <root> --authorization <STAGE_A-auth.json>
python -m experiments.calibration.fs4_joint_campaign verify-stage-a --campaign-root <root>
python -m experiments.calibration.fs4_joint_campaign run-stage-b --campaign-root <root> --grant <STAGE_B-auth.json> --result-root <fresh-protected-root>
python -m experiments.calibration.fs4_joint_campaign verify-complete --result-root <root>
```

For a verified deterministic resume, `run-stage-a` may be invoked with `--resume`. Each authorization is a separate review artifact and must bind the exact campaign identity and inputs described in the CLI contract. The authorization record is an operator-controlled gate under the trusted-local researcher model; it does not provide isolation from malicious same-account code.
