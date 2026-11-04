# E123 K1024 Mechanism Feasibility Note

## Scope

This note preserves feasibility limitations observed in the existing sealed
`audiojack_s0_json` Stage A condition. It does not change the E123 design,
promote this condition into another campaign, or use results to tune a method.
The condition is valid, while the original 16-condition campaign remains
incomplete and Stage B was not started. The aggregate-only verification receipt
is [audit_receipt.json](../../../../outputs/calibration/e123_readonly_audit_20261105_top8_r2/audit_receipt.json).

## K1024 replacement activation

The condition contains 74 queried normal feedback events. The frozen common
candidate contract supplies 8 patches for each eligible queried normal, giving
592 candidate patches. The four K1024 arms that accept every candidate finish
at 592/1024 occupied slots with zero replacements:

| K1024 arm | Occupied slots | Accepted | Rejected | Replaced |
|---|---:|---:|---:|---:|
| FIFO | 592 | 592 | 0 | 0 |
| Reservoir | 592 | 592 | 0 | 0 |
| Random Replacement | 592 | 592 | 0 | 0 |
| Greedy | 592 | 592 | 0 | 0 |
| Utility STM | 549 | 549 | 43 | 0 |

Utility STM has lower occupancy because its admission gate rejected 43
candidates. No K1024 arm activated replacement in this condition.

The largest registered query count across the campaign is 116. Even if all 116
were eligible normal queries, the eight-patch contract yields at most 928
candidates, below K1024. Thus K1024 is a valid nominal-capacity comparison, but
the registered stream cannot identify full-capacity replacement-policy
superiority. This limits that mechanism claim; it does not justify changing K,
the query budget, or candidate count after observing outcomes.

## Persistent-byte tolerance

For this condition, the registered 5% same-capacity tolerance is met at K256
and K512, but not K1024. Using the sealed resource summary's spread
`(maximum - minimum) / maximum`:

| Capacity | Sealed persistent-byte range | Spread | 5% tolerance |
|---|---:|---:|---|
| K256 | 2,128,135–2,139,421 bytes | 0.5275% | Pass |
| K512 | 4,225,627–4,248,929 bytes | 0.5484% | Pass |
| K1024 | 4,553,701–4,880,479 bytes | 6.6956% | Fail |

The K1024 comparison therefore also exceeds its registered resource-match
tolerance in this condition. No post-hoc capacity adjustment is proposed.

## Fixed Offset boundary result

The frozen leave-one-category-out Fixed Offset selection chose coefficient
`-1.0`, the lower boundary of the registered grid, for audiojack, regulator,
switch, and rolled strip base. This is recorded as a boundary-selection
limitation. The grid is not expanded and no retuning is performed here.

## Provenance and limits

- Frozen contract digest: `a76efe732276bf84cfbbc941593d59ef3cb221c64fbebb08b0d088b719b6b622`.
- Condition seal digest: `ded0e11abdd54619d16f5c06bb4c31dfb2745f38e6d0b2ee9950293182a04291`.
- The source-bound hashes, supervisor receipt, score/C64 state links, query
  positions, and private-feedback file hash are recorded in the separate audit
  receipt. The receipt contains aggregate feedback counts and hashes, not label
  values.
- Only `audiojack_s0_json` is sealed in this namespace. There is no global Stage
  A seal and no Stage B aggregate. The classification remains
  `CONDITION_VALID_BUT_ORIGINAL_CAMPAIGN_INCOMPLETE`.
- These are exposed development data and mechanism-feasibility observations,
  not independent confirmation or evidence of scientific superiority.
