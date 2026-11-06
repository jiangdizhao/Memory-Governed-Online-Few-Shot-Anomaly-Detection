# Experiment A integrated A2/A3 results

**Status:** COMPLETE. **Run:** `a23_20261106_full08`. **Decision candidate:** `HOLD_CALIBRATION`.

## Scope and integrity

This continuation used the registered four development categories, P0/P1/P2 event orders,
Q-A-TS-v1 queried feedback, native R9/R10 scorer and update functions, and the sealed A1
role memberships. Stage A completed all 12 category-condition streams before Stage B read
unqueried image labels. No confirmation category or pixel mask was accessed. Private score,
identity, and queried-feedback tapes remain under `E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection\outputs\calibration\a_calibration_scale_a23_20261106_full08`. The float32 TEST feature
cache is temporary scratch and is removed after each category completes.

Fresh disposable TRAIN-normal score/update parity checks:
- **audiojack:** overall `PASS`, native score `True`, native update `True`.
- **regulator:** overall `PASS`, native score `True`, native update `True`.
- **switch:** overall `PASS`, native score `True`, native update `True`.
- **rolled_strip_base:** overall `PASS`, native score `True`, native update `True`.

The contract fixes C64, C128, and Cmax-G as distinct roles; Cmax-G is never described as
strict four-shot. A3b compares F4 with an 8,192-prototype, fp16, 4,096D FEXP-KC bank at
equal deployed fixed-reference bytes, while reporting its larger training-data access.

## Core A2 results

Mean P0 G AUROC/AP/BER/FPR/FNR across the four categories:

| Calibration | AUROC | AP | BER | FPR | FNR | Mean threshold total variation |
|---|---:|---:|---:|---:|---:|---:|
| C64 | 0.8319932369243984 | 0.7638391626248721 | 0.2320049139913403 | 0.10843337579271627 | 0.35557645218996436 | 0.09020158648490906 |
| C128 | 0.8319932369243984 | 0.7638391626248721 | 0.23153907904605758 | 0.10038704654392708 | 0.3626911115481881 | 0.09068651497364044 |
| Cmax-G | 0.8319932369243984 | 0.7638391626248721 | 0.2319869794923526 | 0.10571116524043611 | 0.3582627937442691 | not pooled when a full trajectory was held |

Greedy+C1 P0 means by calibration (AUROC/AP/BER/FPR/FNR): C64 {'auroc': 0.8568682585950181, 'average_precision': 0.8133226728781693, 'ber': 0.21625991602627317, 'fpr': 0.11042583822348556, 'fnr': 0.32209399382906084}; C128 {'auroc': 0.8568682585950181, 'average_precision': 0.8133226728781693, 'ber': 0.2155995440487093, 'fpr': 0.09940761708862153, 'fnr': 0.33179147100879713}; Cmax-G {'auroc': 0.8568682585950181, 'average_precision': 0.8133226728781693, 'ber': 0.21410613720866167, 'fpr': 0.10606228093669809, 'fnr': 0.32214999348062523} where a full trajectory was available.

## Per-category outcomes

| Category | A2 G C64 BER | A2 G C128 BER | A2 GC C64 BER | A2 GC C128 BER | A3a G C64 BER | A3a G C128 BER | A3b F4 static BER | A3b FEXP static BER | A3b F4 prequential BER | A3b FEXP prequential BER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| audiojack | 0.32570366449551313 | 0.32509700964031385 | 0.31068799959811605 | 0.30996678331103367 | 0.30176155048789693 | 0.3145042124299766 | 0.4878750202167233 | 0.3296370424281633 | 0.31220057807576007 | 0.2471590909090909 |
| regulator | 0.28725225225225226 | 0.2865165165165165 | 0.25512512512512514 | 0.2529179179179179 | 0.27434684684684685 | 0.2716266266266266 | 0.4782957957957958 | 0.2762287287287287 | 0.23084584584584586 | 0.19523523523523523 |
| switch | 0.2017484545641305 | 0.20609446081947927 | 0.19371834012442585 | 0.19579054798568762 | 0.19816569093919217 | 0.20335623518968737 | 0.39534924598549204 | 0.20196212954306514 | 0.2302158391933742 | 0.16683868692605747 |
| rolled_strip_base | 0.11331528465346534 | 0.10844832920792079 | 0.10550819925742574 | 0.10372292698019803 | 0.11410736386138613 | 0.10842667079207921 | 0.23862778465346535 | 0.09395266089108911 | 0.11223004331683167 | 0.0686246905940594 |

Greedy+C1 P0 means by calibration are recorded in the aggregate. A2 uses a common
normal-memory trajectory and reports C64/C128 readout changes on matched score histories.
P1 and P2 evaluate C64/C128 only. Full-stream ranking scores are invariant to a threshold
readout; any ranking differences between G and GC arise from Correction scoring, not from
the choice of B3 calibration size.

## A3a coupled learning

Each P0 C64/C128/Cmax-G arm owns an independent native trajectory. The first state
divergence, differing writer actions, normal-memory changes, Correction updates, and
subsequent score changes are recorded by category in the aggregate. These results measure
the combined effect of calibration-dependent decisions and later native state updates;
they do not attribute all downstream differences to threshold quality alone.

Mean P0 A3a G BER is 0.22209536303383054 for C64 and 0.22447843625959246 for C128, compared
with A2 shared-trajectory G means 0.2320049139913403 and 0.23153907904605758. This comparison is
descriptive: A3a changes future learning through native decisions and state updates.

## A3b expanded equal-byte reference

The paired P0 comparison reports static C64-calibrated and prequential Greedy AUROC/AP,
BER/FPR/FNR, exact deployed bytes, image access, feature extraction, k-center, and online
costs. The expanded pool uses one deterministic normal view per eligible official Source
Group, balanced identity-hashed candidate selection, and blockwise native-feature-space
k-center. It receives more predeployment data and compute than F4; equal bytes refer only
to deployed fixed-reference storage.

Mean static AUROC/AP is 0.7053106697347098/0.6140228792934561
for F4 and 0.8541716155864585/0.8108724158869043
for FEXP-KC. With native Greedy adaptation, prequential AUROC/AP is
0.8358764884413885/0.7584908011706163
for F4 and 0.895438275939917/0.8588182307803598 for FEXP-KC.

## Cmax-G resource gate

- **audiojack:** `FULL_TRAJECTORY_ENABLED`; measured profile 7.0488781000021845 s; projected remaining 3375.5356813274425 s.
- **regulator:** `FULL_TRAJECTORY_ENABLED`; measured profile 7.1300133999902755 s; projected remaining 3398.2708575094584 s.
- **switch:** `FULL_TRAJECTORY_ENABLED`; measured profile 6.852925900369883 s; projected remaining 3250.8239538074054 s.
- **rolled_strip_base:** `FULL_TRAJECTORY_ENABLED`; measured profile 7.125166299985722 s; projected remaining 2228.416386867728 s.

Snapshot-only Cmax diagnostics are identified as such in each Stage A seal and are not
included as a full B3 trajectory. The sealed resource profiles use the native scorer after
a genuine Greedy write; full Cmax branches were scheduled only by the frozen gate.

## Interpretation and decision candidate

The observed C64-to-C128 mean G BER change is -0.0004658349452827115;
mean threshold total variation is 0.09020158648490906 for C64 and 0.09068651497364044 for C128. Per-category
results and normal-only A1 H diagnostics are preserved in the aggregate. A3b identifies
whether a larger legal predeployment reference changes the apparent need for online normal
adaptation. These are development-category outcomes and do not establish confirmation
performance or authorize Experiments B/C.

Mean TEST feature encoding took 393.5617196999956 seconds per category. Mean A3b
reference feature extraction, selection, native bank construction, and k-center took
91.51920914987568 seconds per category. The measured Cmax native rescore profile
averaged 7.039245925087016 seconds; projected cost and dynamic Greedy/C1
allocated bytes are reported by category in the aggregate.

The candidate is **HOLD_CALIBRATION** because no practical-equivalence margin or minimum
meaningful improvement was preregistered. The evidence is reported without creating a
post hoc threshold for declaring C64, C128, or Cmax-G the winner. A researcher can review
the category effects, normal-only H stability, A3a coupling, A3b alternative reference, and
resource burden before choosing a commissioning configuration.

## Reproducibility and files

- Parent commit: `897dabe704bbb6a14779d47f93e7e4c3f42f298a`; continuation source SHA-256: `fa0c60d477c2ecbf731ea7e5af643f2aef173c5c94f2db3afa8d5b0b24211b95`.
- Aggregate assembly source SHA-256: `8c0b758c9177a0bc7f731674583852004f243518d303650fe54c54f3bd2e28c9`; final report renderer SHA-256: `cf86cc6c8199c704866b45c8a9b0e9b9ad8f1884a7e548f57aed20822619e25a`. Report-only repairs followed the sealed run; experimental trajectories and Stage B metrics were not rerun or changed.
- Stage A seal: `c15ccbeff03cb1a9f0341c3e2bb3d63416c2a3096e7d985df98c134ec4789131`.
- Stage B seal: `a44099936c3386bb8e1ed861fcb27683542cc9319b533dbf01e4c8ef64921af8`.
- Run outputs: `E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection\outputs\calibration\a_calibration_scale_a23_20261106_full08`.
- Aggregate: `A2_A3_AGGREGATE_RESULTS.json`; concise CSV: `A2_A3_RESULTS_SUMMARY.csv`.
- Figures: seven validated PNGs under `figures/`.

The aggregate contains metrics and summarized costs only. Raw test scores, event identities,
queried feedback, feature caches, and masks are not included in these public audit files.
