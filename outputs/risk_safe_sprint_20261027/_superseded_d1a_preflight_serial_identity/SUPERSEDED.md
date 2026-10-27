# Superseded D1-A outputs -- serial (S####) sample identity

These four manifests were produced by the **first D1-A** implementation,
which treated the Real-IAD directory name `S####` as the physical-object
identity.  That is a **verified parser error**: one `S####` directory hosts
the OK capture session plus one session per defect code, so distinct
official Real-IAD samples were merged into a single group.

The official evaluation (Tencent/AnomalyDetection_Real-IAD, README sec. 3,
`--sample_key_pat "([a-zA-Z][a-zA-Z0-9_]*_[0-9]{4}_[A-Z][A-Z_]*[A-Z])_C[0-9]_"`)
and the repository's own historical `final_decision_core.object_id` rule
both group by `<category>_<NNNN>_<CONDITION>`.  D1-A-R1 restores that
primary identity and keeps the `S####` analysis only as a sensitivity
diagnostic.

Consequences of the superseded identity (recorded, not corrected here):

- the test split appeared to have 315 audiojack groups instead of 763;
- 53-93 train samples appeared to leak into test per class, which was an
  artefact of merging distinct official samples;
- the stream was reduced to 201 objects / 2,165 images, so the absolute
  budgets covered 35-40% of samples instead of 15.05%.

**Do not cite these files.**  The authoritative outputs are in
`outputs/risk_safe_sprint_20261027/d1a_preflight/`
(identity version `realid-offical-sample-key-v2`).

## Preserved file digests (SHA-256 of the exact bytes)

- `dataset_preflight.json`  146085 bytes  `8362ece03abd35841265d5d884455cbc2691c6939a4962f4ff4ea9ca814ae8be`
- `query_budget_preflight.json`  261131 bytes  `07e7991ccb4e43381af56ad252decffe9d49f23e6d333fafa96a7a6c876eb38a`
- `query_tape_manifest.json`  1144517 bytes  `7ef751f31fc76ea5b5c9ef0e48f4b20ffaba6077586155af75d5a9877f8d0fe7`
- `source_manifest.json`  3838 bytes  `5422df43b6a3a611fc0c47122b0a13b23ac1c35fec5509a93266aabc971c5758`

The archived files were moved, never rewritten or deleted.
