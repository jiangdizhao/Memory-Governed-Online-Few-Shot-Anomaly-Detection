# FINAL-GATE1 Dataset Identity Audit

**Status:** local identity census completed; official release and role certification pending.  
**Access:** identity/provenance metadata and opaque image-byte hashing only. No TEST labels, masks, image decoding, inference or evaluation.

## Source inventory and private receipts

Private checksummed manifests are stored under:

```text
outputs/calibration/final_benchmark_gate1/attempt_01/
```

Files:

- `realiad_identity_manifest.private.json`
- `visa_1cls_identity_manifest.private.json`
- `identity_census_receipt.json` and its `.sha256` sidecar

The private manifests store opaque image/group/content/locator digests, original metadata position, role eligibility and sanitized category summaries. They do not store raw image paths, TEST labels, mask paths or pixel data. The report does not reproduce image-level identity rows.

## Real-IAD

Registered source roots: `G:\dataset\realiad` and `G:\dataset\realiad\jsons\realiad_jsons`. The annotation inventory contains 30 category JSON files. The parser projects only the permitted metadata fields and official TRAIN image-level class needed for normal eligibility; TEST `anomaly_class` and all mask fields are dropped before records are returned. Official TEST image membership and camera/group metadata are derived from the official JSON identity keys; the filename/capture-session string is never used as a defect label.

| Census item | Result |
|---|---:|
| Official category JSON files | 30 |
| Categories with complete local identity census | 30/30 |
| Sanitized category projection errors | 0 |
| Complete five-camera TEST groups | 22,917/22,917 |
| TEST image events | 114,585 |
| TRAIN-normal eligible images / groups | 36,465 / 7,293 |
| TRAIN-normal groups per category | 167–252 |
| TRAIN-normal / TEST official group overlap | 0 |
| Exact duplicate content sets in inventoried TRAIN-normal + TEST images | 1 (two TEST images; no cross-split duplicate) |

**Certification result:** `IDENTITY_CENSUS_LOCAL_ONLY_OFFICIAL_RELEASE_PROVENANCE_PENDING`. All 30 categories have at least 167 locally identified normal TRAIN groups, complete five-camera TEST groups, no TRAIN-normal/TEST group overlap and no sanitized projection error. Local annotation/image hashes establish a reproducible local snapshot, not the identity of an official release. No externally verified release receipt or official release SHA-256 was supplied. Therefore the frozen `dataset_release_sha256` role, event and Static-36+32 hashes were deliberately not generated. Support/C64 identities and P0/P1/P2 stream manifests are not certified.

**Exact duplicate review:** one pair of byte-identical TEST files occurs under different official image identities and different local locator digests in `button_battery` and `phone_battery` (camera IDs 1 and 5; distinct opaque Source Group digests). No duplicate crosses TRAIN/TEST. The images were not decoded and no TEST outcome field was accessed. The local metadata establishes distinct category/identity/group keys but does not establish whether the duplicate is a legitimate shared source, a release-copy artifact or a provenance error. This cross-category dependency remains an unresolved Gate 1 provenance item; the pair is retained in the private manifest and must not be silently deduplicated from evaluation.

| Real-IAD category | TRAIN-normal groups | Complete TEST groups | TEST events | Invalid TEST groups | TRAIN/TEST overlap | Local status |
|---|---:|---:|---:|---:|---:|---|
| audiojack | 247 | 763 | 3,815 | 0 | 0 | census pass |
| bottle_cap | 246 | 757 | 3,785 | 0 | 0 | census pass |
| button_battery | 242 | 758 | 3,790 | 0 | 0 | census pass; duplicate pair noted |
| end_cap | 241 | 761 | 3,805 | 0 | 0 | census pass |
| eraser | 246 | 767 | 3,835 | 0 | 0 | census pass |
| fire_hood | 251 | 754 | 3,770 | 0 | 0 | census pass |
| mint | 167 | 894 | 4,470 | 0 | 0 | census pass |
| mounts | 250 | 760 | 3,800 | 0 | 0 | census pass |
| pcb | 247 | 755 | 3,775 | 0 | 0 | census pass |
| phone_battery | 242 | 758 | 3,790 | 0 | 0 | census pass; duplicate pair noted |
| plastic_nut | 251 | 751 | 3,755 | 0 | 0 | census pass |
| plastic_plug | 243 | 756 | 3,780 | 0 | 0 | census pass |
| porcelain_doll | 242 | 758 | 3,790 | 0 | 0 | census pass |
| regulator | 249 | 771 | 3,855 | 0 | 0 | census pass |
| rolled_strip_base | 249 | 761 | 3,805 | 0 | 0 | census pass |
| sim_card_set | 249 | 757 | 3,785 | 0 | 0 | census pass |
| switch | 246 | 764 | 3,820 | 0 | 0 | census pass |
| tape | 250 | 770 | 3,850 | 0 | 0 | census pass |
| terminalblock | 244 | 766 | 3,830 | 0 | 0 | census pass |
| toothbrush | 245 | 756 | 3,780 | 0 | 0 | census pass |
| toy | 250 | 755 | 3,775 | 0 | 0 | census pass |
| toy_brick | 248 | 754 | 3,770 | 0 | 0 | census pass |
| transistor1 | 249 | 751 | 3,755 | 0 | 0 | census pass |
| u_block | 250 | 748 | 3,740 | 0 | 0 | census pass |
| usb | 246 | 764 | 3,820 | 0 | 0 | census pass |
| usb_adaptor | 231 | 772 | 3,860 | 0 | 0 | census pass |
| vcpill | 252 | 753 | 3,765 | 0 | 0 | census pass |
| wooden_beads | 218 | 783 | 3,915 | 0 | 0 | census pass |
| woodstick | 251 | 750 | 3,750 | 0 | 0 | census pass |
| zipper | 251 | 750 | 3,750 | 0 | 0 | census pass |

## VisA 1cls

Registered source root: `G:\dataset\VisA`, using the official `split_csv/1cls.csv` and dataset license. The CSV identity projection retains object, split and image fields; it reads TRAIN `label` only to identify TRAIN-normal eligibility and skips the TEST `label` and `mask` cell contents. Mask files are not opened.

| Census item | Result |
|---|---:|
| Official 1cls categories | 12 |
| TRAIN-normal images | 8,659 |
| TEST images | 2,162 |
| Exact image-content duplicate sets | 0 |
| Recognized official physical-group columns | none |
| 68 disjoint physical TRAIN-normal groups certifiable | no |

| Category | TRAIN-normal images | TEST images |
|---|---:|---:|
| candle | 900 | 200 |
| capsules | 542 | 160 |
| cashew | 450 | 150 |
| chewinggum | 453 | 150 |
| fryum | 450 | 150 |
| macaroni1 | 900 | 200 |
| macaroni2 | 900 | 200 |
| pcb1 | 904 | 200 |
| pcb2 | 901 | 200 |
| pcb3 | 905 | 201 |
| pcb4 | 904 | 201 |
| pipe_fryum | 450 | 150 |

**Certification result:** `IMAGE_MEMBERSHIP_LOCAL_ONLY_ONLINE_GROUPING_BLOCKED`. The approved local CSV has no recognized official group column. Category/object names and image/file identity do not establish physical-object independence. No image-as-independent-group substitute was used. The frozen contract requires a separately documented official physical-group key or a researcher-approved exception; absent either, VisA online feedback and Static-36+32 eligibility remain blocked. External release provenance is also unauthenticated locally.

## Local source hashes

- Real-IAD 30-annotation artifact-set SHA-256: `8652d842d26785fff780fa3ac429592a39da72675c8dacb5fe56eabf512ce420`; the per-category JSON SHA-256 values are recorded in the private manifest.
- VisA split CSV plus dataset-license artifact-set SHA-256: `5612ff8ec0b9ca5c1545b9f08aaba71b89f3b36bcaf8dc2e9fee82589949d029`.
- Real-IAD local identity snapshot SHA-256: `d44a059bf998231fcdc2a296b97ca45544a2554075e1f92b4dfc15c89a5fe4e2`.
- VisA local identity snapshot SHA-256: `15535268a41c5c163b9906874fa148478f9da0bf9394659f448d320de778a7f2`.
- Real-IAD private manifest payload SHA-256: `ba6708ee33a08112c6b96b340234a90e46a914530ea4bcf796fea567f00a1c16`.
- VisA private manifest payload SHA-256: `7cb4f0a9bf6c036fb55ecfab56938f15da1b3897440045e311455935513be734`.
- Census receipt SHA-256: `68fa913129927d692074aca9251f2a905f14395dc771088d73f0fc04c8f4cbab`.

## Frozen role rules and unresolved dependencies

The role-builder implements the Gate0 NFC/UTF-8 `Gate0-role-v1`, `Gate0-image-v1`, `Gate0-stream-v1`, `Gate0-event-v1`, and `Gate0-static36-c64-v1` digest strings. It requires a caller-supplied 64-character official release digest. It does not substitute the local snapshot digest. P0/P1 share Support Seed 0 identities; P2 uses Support Seed 1; streams use Stream Seeds 0 and 11. Real-IAD stream ordering keeps the five cameras contiguous and orders cameras 1 through 5.

Static-36+32 schedule construction preserves the principal four-Support-plus-64-C64 manifest and separately ranks the same P0 C64 set into 32 reference and 32 C32 identities. The schedule is metadata-only and is not emitted for the real datasets while official release identity remains unresolved.

## Access record

- Real-IAD TEST outcome field: not exposed to callers or written to manifests.
- VisA TEST label cells: skipped by the projection.
- Masks: not opened.
- Image bytes: opened only as opaque streams for SHA-256; no image decoder or model was called.
- GPU/model: not accessed.
- Scientific Stage A or Stage B: not run.
