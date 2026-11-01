# G1-A Switch — Local Authorization and Read-Only Verification

This is a preparation guide only. It does not grant access. Do not open Real-IAD images, instantiate DINO, or launch S1/S2 unless a future local verification returns `PREFLIGHT_READY_NO_DATA_IO` and the separate execution action remains authorized.

## Required private artifacts

Use a new, private directory outside Git, by default:

`%USERPROFILE%\.r2_g1a_switch_authorization`

Place these files there without overwriting older authorization material:

1. `trust_directory.json` — fresh G1-specific trust root; `directory_id` must begin `r2-g1a-switch-access-`.
2. `trust_directory.sha256` — trust-directory digest independently pinned by the reviewer out of band. Do not compute a new file hash and call it a historical pin.
3. `source_fingerprint.json` — output of the committed source/checkpoint/hub fingerprint command below.
4. `source_fingerprint.sha256` — its digest independently pinned by the reviewer out of band.
5. `a_reference_bundle.json` — signed switch A availability receipt, L2B-certified A identity receipt, reviewed normal-only manifest and attestation, plus `image_identity_receipt` and its independent `image_identity_attestation`.
6. `authorization.signed.json` — the operation-specific researcher-signed scope, based on `G1A_ACCESS_SCOPE_PROPOSAL.json` and bound to the exact committed source fingerprint, pins, A receipt, image inventory receipt, and a fresh unique output path.

The image identity receipt contains only SHA-256 image identity, Source Group SHA-256, and camera ID tuples. It must bind the same A source manifest and availability receipt, contain no raw sample keys or paths, cover every image view count in the certified A receipt, and be independently attested by a trusted reviewer distinct from the inventory producer, A identity producer, and A identity reviewer. The reviewer must verify that the selected switch images are TRAIN-normal and that the camera-1 filename convention agrees with the identity contract. A missing or unreviewed receipt is a hard stop.

If no previously authorized identity source can produce this inventory, do not reuse the historical L2C census signature to create it. Obtain a separately scoped TRAIN-normal identity-projection authorization first; it must explicitly forbid TEST parsing, labels, and masks. Then have the independent reviewer certify the resulting identity-only receipt before requesting the image-decode/feature scope represented by the proposal. This task did not perform that metadata access.

## Prepare prospective source identity

After the adapter and tests have been committed with a clean scoped source tree, run:

```powershell
$PY = 'D:\anaconda3\envs\torch\python.exe'
& $PY -m experiments.calibration.r2_g1a_switch_pilot --print-source-fingerprint --repo-root $PWD
```

This hashes the committed source, checkpoint, and DINO hub `.py` tree. It does not instantiate the model or inspect dataset paths. Save the JSON output to a new `source_fingerprint.json` in the private authorization directory and arrange for its `source_fingerprint_sha256` to be independently pinned before signing. The loaded-model-state digest remains only an expected value until a separately authorized model initialization verifies it.

The source and trust pins must be communicated through an independent review channel and copied verbatim into their `.sha256` files. The researcher must not derive a pin from an adjacent file and treat that as independent evidence.

## Canonicalize and sign the scope

Copy `G1A_ACCESS_SCOPE_PROPOSAL.json` to a new private draft. Fill all `FILL_*` values, use the exact execution commit and source fingerprint, bind both reviewed A receipt digests, and select a new nonexisting run ID under `outputs/calibration/r2_g1_pilot/`. The proposal's current output path is only a proposed target; check that it remains absent before signing.

Canonicalize to a new filename (the tool uses exclusive creation and will refuse to overwrite):

```powershell
$AUTH = Join-Path $env:USERPROFILE '.r2_g1a_switch_authorization'
& $PY -m experiments.calibration.r2_g1a_switch_pilot `
  --canonicalize-draft (Join-Path $AUTH 'authorization.draft.json') `
  --canonical-payload-output (Join-Path $AUTH 'authorization.canonical.json')
```

Use the existing protected local Ed25519 signer to sign the exact canonical payload bytes. Keep the private key and password local; do not send them to Codex or place them in the repository. Add the resulting signature and the printed canonical payload digest to `authorization.signed.json`. The fresh trust root must list that researcher as an authorized policy approver and include the independent reviewer public keys needed by the A and image-inventory attestations.

## Verify without image access

Once all six files exist, provide the independent pin values to this read-only command:

```powershell
$AUTH = Join-Path $env:USERPROFILE '.r2_g1a_switch_authorization'
& $PY -m experiments.calibration.r2_g1a_switch_pilot `
  --authorization-dir $AUTH `
  --expected-trust-directory-sha256 '<independently pinned trust-directory SHA-256>' `
  --expected-source-fingerprint-sha256 '<independently pinned source-fingerprint SHA-256>' `
  --repo-root $PWD
```

This command validates the scope signature, canonical payload digest, trusted researcher key, prospective trust/source pins, exact execution commit, A availability/identity, reviewed camera/image inventory, output uniqueness, and free-space floor. It does not open any dataset image, load DINO, or start the pilot. Continue only if it returns `PREFLIGHT_READY_NO_DATA_IO`; otherwise preserve the report and resolve the stated gate without weakening it.

The S1/S2 execution itself must be separately launched under the existing Windows Job Object supervisor with the signed 120-minute wall-clock limit and 1-GiB output budget. S1 is exactly the first 10 ranked groups; S2 is the same frozen 100-group cohort. This document does not provide a command that bypasses those gates.
