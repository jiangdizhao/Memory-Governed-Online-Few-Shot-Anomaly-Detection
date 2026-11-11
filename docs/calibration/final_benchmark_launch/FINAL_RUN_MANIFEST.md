# Final Real-IAD Run Manifest

**Plan status:** `SCHEDULE_VERIFIED_SOURCE_BINDING_PENDING`
**Machine-readable template:** [FINAL_RUN_CONFIGURATION_TEMPLATE.json](FINAL_RUN_CONFIGURATION_TEMPLATE.json)
**Schedule digest:** `a475907f4a198a96b83dcdf5b42bc76ad10cc19c9dbb101bed735832c4cdfb76`

The metadata-only verification validated the 30 registered categories, 90 principal category-condition trajectories, all 30 P0 Static-36+32 allocation schedules, and all 30 FOADS P0 registrations. It checked bundle digests, exact role counts, complete ordered five-view groups, event counts, query caps, and the approved local dataset-release binding. It decoded no TEST labels and opened no masks. The configuration contains category-level fingerprints, not image paths or image-level identifiers.

## Principal trajectory schedule

For each category below, execute all three conditions P0 `(Support 0, Stream 0)`, P1 `(Support 0, Stream 11)`, and P2 `(Support 1, Stream 0)`. Each trajectory runs F-B, K-B, F-M, K-M, and passive N-shadow. Static-4+64 accompanies all three conditions; Static-36+32 accompanies P0 only. The machine-readable file records the role-manifest, stream-manifest, Static-36+32, and FOADS projection fingerprints for each trajectory.

| Category | TEST images, each condition | Five-view groups | Hard cap per principal arm |
|---|---:|---:|---:|
| audiojack | 3,815 | 763 | 114 |
| bottle_cap | 3,785 | 757 | 113 |
| button_battery | 3,790 | 758 | 113 |
| end_cap | 3,805 | 761 | 114 |
| eraser | 3,835 | 767 | 115 |
| fire_hood | 3,770 | 754 | 113 |
| mint | 4,470 | 894 | 134 |
| mounts | 3,800 | 760 | 114 |
| pcb | 3,775 | 755 | 113 |
| phone_battery | 3,790 | 758 | 113 |
| plastic_nut | 3,755 | 751 | 112 |
| plastic_plug | 3,780 | 756 | 113 |
| porcelain_doll | 3,790 | 758 | 113 |
| regulator | 3,855 | 771 | 115 |
| rolled_strip_base | 3,805 | 761 | 114 |
| sim_card_set | 3,785 | 757 | 113 |
| switch | 3,820 | 764 | 114 |
| tape | 3,850 | 770 | 115 |
| terminalblock | 3,830 | 766 | 114 |
| toothbrush | 3,780 | 756 | 113 |
| toy | 3,775 | 755 | 113 |
| toy_brick | 3,770 | 754 | 113 |
| transistor1 | 3,755 | 751 | 112 |
| u_block | 3,740 | 748 | 112 |
| usb | 3,820 | 764 | 114 |
| usb_adaptor | 3,860 | 772 | 115 |
| vcpill | 3,765 | 753 | 112 |
| wooden_beads | 3,915 | 783 | 117 |
| woodstick | 3,750 | 750 | 112 |
| zipper | 3,750 | 750 | 112 |

The schedule totals 114,585 images and 22,917 Source Groups per condition. The P0 hard caps sum to 3,424 per principal arm across categories; the same event totals and category caps apply to P1 and P2. These are maximum allowances, not realized query counts. Report actual expenditure, suppression, obtained label composition, and FPR/FNR trade-offs. No method receives forced queries or top-ups.

## FOADS and Stage B separation

The verified external K-NG checkout is `D:\Whishing-K-NG-63dd8e7c9ade5619771d4b5544f032206818bb22-lf`, detached at commit `63dd8e7c9ade5619771d4b5544f032206818bb22`. Its three registered source-file hashes passed `verify_upstream_source()`: `Klink.py` `402f9bd969693207aedc5286029b2e51f9bd0000016eeacfab3510cfbf76d440`, `Konline.py` `a2bd1823cc1bb1e45b3749108bc2865816852fa493ae32ad201fcb153fef72f1`, and `datasets/mvtec.py` `5a1598b202e5ed3e797126e08827ef0b5215e0d530dbc8d79514fc7cc84596b6`. Future FOADS launch commands must pass this path explicitly using `--kng-source-root`; tests set `KNG_SOURCE_ROOT` explicitly.

FOADS is a separate P0-only stream over the same 30 categories: 114,585 images in 22,917 ordered five-view groups, with two complete groups scored before each native ten-image update. For odd group counts, the final complete group is scored without a partial update. The configuration records 30 category-specific FOADS registration-projection hashes. A separate per-category launcher now wraps `FOADSKNGP0Runner`, verifies the exact projection, role/stream/release and locator hashes, requires the pinned K-NG source and ResNet18 checkpoint identities, and writes independent prediction/update journals and a category seal. FOADS has no paid-label access; its full-category authorization and output receipts have not been generated.

Stage B is a separate post-seal operation. The intended dependency is: all 90 Stage A category-condition seals and durable attempt receipts pass integrity verification; then a separate Stage B authorization binds each category-condition seal and authorizes outcome projection. The Stage B API now also requires the matching durable `SEALED_COMPLETE` Stage A attempt receipt before invoking its label projection callback. FOADS receipts have a separate verifier and do not authorize outcome access. No unqueried labels or masks were opened, and no Stage B execution was performed.

## Output and resource policy

Planned non-overwriting layout:

```text
outputs/calibration/final_benchmark/<campaign_run_id>/
  stage_a/<category>/<condition>/attempt_<nn>/
  foads_kng/<category>/P0/<attempt-id>/
  stage_b/<category>/<condition>/attempt_<nn>/
```

The `launch` CLI validates the source-bound plan, lists the 90 schedule rows, runs one authorized category, or runs a separately authorized explicit sequential schedule that stops on first failure. Attempt directories are non-overwriting. Stage A adds a durable `RUNNING`/`INCOMPLETE`/`SEALED_COMPLETE` attempt status around the existing fsynced group journal, private feedback ledger, category seal, and report. Early startup/runtime failures record phase, completed-event count, exception and source/configuration identity. Stage B receipt verification rejects incomplete or mismatched attempts.

FOADS writes a separate fsynced per-image score journal and native Batch-10 update journal. Each image record contains its registered identity/group, raw score, update index and the model-state digest before scoring. Each update receipt records the ten-image boundary, the two group identities and post-update model-state digest. The full-stream seal checks exact order/count, complete groups, floor(group_count/2) updates, any final unupdated group, zero label access and both journal checksums. It does not evaluate AUROC/AP.

Run one category at a time on the GPU and release category model/features before the next category. Operational stop rules use the actual bounded preflights: principal 5,989 observed output bytes/event, 6,708,342,784-byte host RSS peak and 17,263,755,264-byte reserved VRAM peak; FOADS 1,731 observed output bytes/event, 11,010,330,624-byte host RSS peak and 1,411,383,296-byte reserved VRAM peak. The launcher budgets 10x observed bytes/event, 10 MiB per attempt, and preserves at least 10 GiB free disk. It requires host available memory at least 1.25x measured peak and, before launch, free VRAM at least measured reserved peak plus 1 GiB; runtime reserved VRAM above 1.25x the observed peak stops the attempt. Full-category output growth and campaign runtime remain unmeasured; these are fail-closed guardrails, not resource guarantees.

At the final read-only resource snapshot on 2026-11-11, 381,233,139,712 bytes disk, 41,316,900,864 bytes host RAM and 23,894,949,888 bytes RTX 3090 Ti VRAM were available. The 10x all-campaign output guard estimated 23,221,070,420 bytes plus the 10 GiB reserve; that estimate is an extrapolation from bounded preflight receipts, not a measured full-run total. No campaign output root or attempt leaf was created.

There is no automatic resume contract. A failed or interrupted trajectory requires a fresh, non-overwriting attempt after evidence review. No output directories were created by this task.
