# Actual B3 Replay — Stage 0.75-A

**Status:** `PASS` for the offline replay and raw-artifact checks. This is a development-set diagnostic, not a Stage 1 result or a population-risk claim.

## Inputs and scope

- Original trajectories: `outputs/calibration/stage075a_greedy_ranking_repaired3_20261029`; parent commit `4d715a9b27027ee0c52dab4552d7296153ca8633`.
- Four exposed development categories and seeds 0–4: exactly 20 complete trajectories.
- Stage 0.75-A protocol, trajectory files, Stage 0.5 score tapes, C64 transports and source hashes verified before replay. Each of the 20 Stage 0.5 C64 vectors exactly matches the Stage 0.75-A version-0 vector and recorded higher-q90 `theta0`.
- The historical runner left an `INCOMPLETE` marker despite its `online_complete.json`, category manifests and all phase statuses being PASS. It was not edited. The original checksum manifest excludes that marker; this report records it as stale historical runner behavior.
- No new VisionAD/DINO inference, Native Scorer calls, Greedy writes, image reads, feature extraction or mask access occurred. The B3 replay completed before evaluator labels were loaded.

## Replay definition and event order

The unchanged `experiments.calibration.controllers.StateConsistentQuantile` was imported and used (`controllers.py` SHA-256 `25fbdfc9a92e8bfb6fe983185dca3cdd5ed24d804c28838b4797e0ae8670cf7d`). Configuration: `delta=0.01`, `min_normals=16`, protected C64 anchors, higher order-statistic q90 at index `ceil(0.90*(64-1)) = 57`, and defect prediction `score >= theta`.

At event *t*, the replay uses the stored Greedy dynamic score and pre-event memory version/digest, predicts with the threshold already available, then checks the recorded writer transition. Only a genuine memory-content version increment delivers the matching post-event C64 `EvidenceSnapshot` and calls `finalize_step()` for event *t+1*. No feedback method is called and no label is part of the replay input. The no-write path causes no B3 update. This uses the actual class lifecycle; the independently coded recurrence is only a test oracle.

For an eligible memory change, the source behavior is:

`theta_(t+1) = min(max(q90_C64_post, theta_t - 0.01), theta_t + 0.01)`.

Initialization starts at historical `theta0`; version-0 evidence is delivered and finalized without an extra threshold movement. At a content change, actual and post-change memory versions, SHA-256 identities, C64 pair hashes and score-derived q90 are reconciled. A genuine write is retained even when the C64 q90 itself is unchanged.

## Full-stream results

Cells give `BA / FPR / FNR`. Metrics are per trajectory; no pooled-image BA is used.

| Class | Seed | N | M0 + frozen (BA/FPR/FNR) | Greedy + frozen | Greedy + read-only q90 | Greedy + actual B3 | B3-q90 delta BA | B3-frozen delta BA | B3 threshold TV | B3 moves | Memory writes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| audiojack | 0 | 3815 | 0.550/0.119/0.780 | 0.547/0.003/0.902 | 0.694/0.118/0.495 | 0.694/0.118/0.495 | 0.00000 | 0.14615 | 0.06681 | 31 | 73 |
| audiojack | 1 | 3815 | 0.510/0.063/0.917 | 0.515/0.004/0.967 | 0.670/0.140/0.520 | 0.672/0.136/0.520 | 0.00179 | 0.15708 | 0.08435 | 22 | 74 |
| audiojack | 2 | 3815 | 0.548/0.095/0.809 | 0.545/0.003/0.908 | 0.665/0.107/0.562 | 0.665/0.107/0.562 | 0.00000 | 0.12064 | 0.05728 | 21 | 74 |
| audiojack | 3 | 3815 | 0.521/0.070/0.887 | 0.520/0.005/0.956 | 0.674/0.135/0.516 | 0.676/0.133/0.516 | 0.00139 | 0.15599 | 0.08994 | 26 | 74 |
| audiojack | 4 | 3815 | 0.614/0.174/0.598 | 0.591/0.013/0.804 | 0.721/0.115/0.442 | 0.721/0.115/0.442 | 0.00000 | 0.12993 | 0.04788 | 24 | 73 |
| regulator | 0 | 3855 | 0.480/0.123/0.917 | 0.523/0.001/0.953 | 0.638/0.104/0.619 | 0.638/0.103/0.621 | -0.00021 | 0.11511 | 0.07325 | 20 | 80 |
| regulator | 1 | 3855 | 0.530/0.089/0.852 | 0.510/0.004/0.977 | 0.637/0.066/0.660 | 0.636/0.065/0.664 | -0.00149 | 0.12603 | 0.08258 | 22 | 78 |
| regulator | 2 | 3855 | 0.524/0.132/0.820 | 0.567/0.006/0.861 | 0.666/0.082/0.586 | 0.666/0.082/0.586 | 0.00000 | 0.09948 | 0.06991 | 15 | 77 |
| regulator | 3 | 3855 | 0.519/0.114/0.849 | 0.506/0.004/0.985 | 0.605/0.104/0.685 | 0.600/0.101/0.698 | -0.00504 | 0.09469 | 0.09250 | 22 | 78 |
| regulator | 4 | 3855 | 0.622/0.117/0.639 | 0.588/0.003/0.821 | 0.773/0.121/0.333 | 0.772/0.120/0.336 | -0.00121 | 0.18385 | 0.09748 | 31 | 76 |
| switch | 0 | 3820 | 0.692/0.064/0.553 | 0.684/0.016/0.616 | 0.745/0.100/0.409 | 0.744/0.100/0.412 | -0.00150 | 0.05989 | 0.05872 | 15 | 53 |
| switch | 1 | 3820 | 0.608/0.093/0.691 | 0.625/0.004/0.746 | 0.738/0.103/0.421 | 0.735/0.100/0.429 | -0.00289 | 0.11031 | 0.07681 | 18 | 54 |
| switch | 2 | 3820 | 0.627/0.085/0.660 | 0.642/0.005/0.712 | 0.725/0.078/0.473 | 0.723/0.076/0.478 | -0.00166 | 0.08128 | 0.05718 | 16 | 54 |
| switch | 3 | 3820 | 0.600/0.103/0.696 | 0.608/0.004/0.780 | 0.694/0.123/0.488 | 0.692/0.122/0.494 | -0.00246 | 0.08406 | 0.06725 | 16 | 54 |
| switch | 4 | 3820 | 0.709/0.128/0.454 | 0.714/0.032/0.541 | 0.776/0.097/0.352 | 0.775/0.096/0.355 | -0.00098 | 0.06099 | 0.04909 | 13 | 53 |
| rolled_strip_base | 0 | 3805 | 0.805/0.130/0.259 | 0.833/0.013/0.321 | 0.878/0.106/0.138 | 0.878/0.105/0.139 | -0.00020 | 0.04441 | 0.05386 | 11 | 33 |
| rolled_strip_base | 1 | 3805 | 0.830/0.144/0.196 | 0.838/0.024/0.299 | 0.853/0.109/0.185 | 0.852/0.108/0.189 | -0.00159 | 0.01345 | 0.04974 | 14 | 33 |
| rolled_strip_base | 2 | 3805 | 0.824/0.096/0.257 | 0.825/0.026/0.324 | 0.809/0.097/0.285 | 0.807/0.095/0.291 | -0.00259 | -0.01853 | 0.05923 | 12 | 33 |
| rolled_strip_base | 3 | 3805 | 0.803/0.091/0.303 | 0.802/0.009/0.386 | 0.885/0.097/0.133 | 0.884/0.095/0.136 | -0.00080 | 0.08218 | 0.05694 | 14 | 33 |
| rolled_strip_base | 4 | 3805 | 0.854/0.123/0.170 | 0.887/0.019/0.208 | 0.895/0.105/0.105 | 0.895/0.105/0.105 | 0.00000 | 0.00813 | 0.04709 | 12 | 33 |

## Equal-class macro summaries

Seed metrics are averaged within each class first, then the four class means are averaged equally. `Greedy + read-only q90` is an offline diagnostic threshold series and is distinct from actual B3.

| Scope | Method | Equal-class macro BA | Equal-class macro FPR | Equal-class macro FNR |
| --- | --- | --- | --- | --- |
| as_operated_full_stream | M0_Frozen_theta0 | 0.63861 | 0.10756 | 0.61523 |
| as_operated_full_stream | Greedy_Frozen_theta0 | 0.64339 | 0.00988 | 0.70333 |
| as_operated_full_stream | Greedy_ReadOnly_current_C64_q90 | 0.73712 | 0.10532 | 0.42043 |
| as_operated_full_stream | Greedy_Actual_B3 | 0.73615 | 0.10417 | 0.42353 |
| conservative_ever_group_clean | M0_Frozen_theta0 | 0.63832 | 0.10803 | 0.61532 |
| conservative_ever_group_clean | Greedy_Frozen_theta0 | 0.64328 | 0.01023 | 0.70320 |
| conservative_ever_group_clean | Greedy_ReadOnly_current_C64_q90 | 0.73676 | 0.10813 | 0.41836 |
| conservative_ever_group_clean | Greedy_Actual_B3 | 0.73578 | 0.10692 | 0.42152 |
| resident_source_clean_sensitivity | M0_Frozen_theta0 | 0.63832 | 0.10803 | 0.61532 |
| resident_source_clean_sensitivity | Greedy_Frozen_theta0 | 0.64328 | 0.01023 | 0.70320 |
| resident_source_clean_sensitivity | Greedy_ReadOnly_current_C64_q90 | 0.73676 | 0.10813 | 0.41836 |
| resident_source_clean_sensitivity | Greedy_Actual_B3 | 0.73578 | 0.10692 | 0.42152 |

## Lag and movement-cap evidence

Lag is `theta_B3_pre - q90_C64_pre`. Event and memory-update lag are summarized separately. Integrated lag is the sum of absolute event lag over a trajectory; class values below are the mean over five seeds. Bound episodes are contiguous runs of binding memory-change updates; spans are in stream events.

| Class | Mean signed event lag | Mean |lag|/event | Max |lag| | Mean integrated |lag| | Nonzero event fraction | Mean |lag| at updates | Lower-bound updates | Upper-bound updates | Binding episodes | Longest binding span (events) | Longest lag episode (events) | B3 BA - q90 BA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| audiojack | 0.000209 | 0.000209 | 0.032936 | 0.796 | 0.0154 | 0.000418 | 9 | 0 | 3 | 110 | 138 | 0.000635 |
| regulator | 0.000378 | 0.000378 | 0.049081 | 1.458 | 0.0380 | 0.000656 | 20 | 0 | 9 | 144 | 208 | -0.001589 |
| switch | 0.000394 | 0.000394 | 0.018179 | 1.505 | 0.0583 | 0.000294 | 11 | 0 | 8 | 191 | 315 | -0.001900 |
| rolled_strip_base | 0.000349 | 0.000349 | 0.008025 | 1.326 | 0.1215 | 0.000186 | 7 | 0 | 5 | 252 | 1023 | -0.001036 |

Threshold movement distribution over genuine memory-content updates:

| Class | Updates | Mean signed movement | Median |movement| | Max |movement| | Downward moves | Upward moves | Mean raw q90 delta | Max |raw q90 delta| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| audiojack | 368 | -0.000797 | 0.000000 | 0.010000 | 107 | 17 | -0.001217 | 0.042936 |
| regulator | 389 | -0.000918 | 0.000000 | 0.010000 | 88 | 22 | -0.001575 | 0.059081 |
| switch | 268 | -0.001013 | 0.000000 | 0.010000 | 72 | 6 | -0.001308 | 0.028179 |
| rolled_strip_base | 165 | -0.001454 | 0.000000 | 0.010000 | 57 | 6 | -0.001639 | 0.018025 |

Mean absolute lag by fixed stream interval:

| Class | Interval | Mean |lag|/event | Lower-bound writes | Upper-bound writes |
| --- | --- | --- | --- | --- |
| audiojack | [0,400) | 0.001990 | 9 | 0 |
| audiojack | [400,800) | 0.000000 | 0 | 0 |
| audiojack | [800,1600) | 0.000000 | 0 | 0 |
| audiojack | [1600,2400) | 0.000000 | 0 | 0 |
| audiojack | [2400,3200) | 0.000000 | 0 | 0 |
| audiojack | [3200,3815) | 0.000000 | 0 | 0 |
| regulator | [0,400) | 0.003469 | 18 | 0 |
| regulator | [400,800) | 0.000075 | 1 | 0 |
| regulator | [800,1600) | 0.000013 | 0 | 0 |
| regulator | [1600,2400) | 0.000000 | 0 | 0 |
| regulator | [2400,3200) | 0.000000 | 0 | 0 |
| regulator | [3200,3855) | 0.000046 | 1 | 0 |
| switch | [0,400) | 0.002836 | 7 | 0 |
| switch | [400,800) | 0.000699 | 2 | 0 |
| switch | [800,1600) | 0.000108 | 1 | 0 |
| switch | [1600,2400) | 0.000006 | 1 | 0 |
| switch | [2400,3200) | 0.000000 | 0 | 0 |
| switch | [3200,3820) | 0.000000 | 0 | 0 |
| rolled_strip_base | [0,400) | 0.000799 | 4 | 0 |
| rolled_strip_base | [400,800) | 0.000213 | 0 | 0 |
| rolled_strip_base | [800,1600) | 0.000090 | 0 | 0 |
| rolled_strip_base | [1600,2400) | 0.000342 | 1 | 0 |
| rolled_strip_base | [2400,3200) | 0.000719 | 2 | 0 |
| rolled_strip_base | [3200,3805) | 0.000000 | 0 | 0 |

The 1,190 real content writes produced lower-bound binding on 47 updates and upper-bound binding on 0. Rare large lag excursions occur, including a maximum absolute lag of 0.049081, but class-mean event absolute lag is at most 0.000394. The full-stream equal-class macro BA difference (B3 minus read-only q90) is -0.000973.

## Integrity and interpretation

- Historical frozen/static and current-q90 metrics reproduce from raw arrays in all 180 declared checks, tolerance `1e-12`.
- Original ever-admitted and resident-source masks match event ordering; fixed-snapshot memory version, digest and q90 match the exact pre-event boundary.
- Query labels were not supplied to B3. Labels were opened only for offline metric calculation after all 20 B3 replays.
- All AUROC/AUPRC values for the three threshold-only Greedy methods use the same score vector and are identical by construction; thresholds cannot change ranking.
- Full per-event replay and update records are in the local output at `outputs/calibration/stage075a_b3_headroom_v1_20261030_verified2/b3_replay.json`. Compact audited results are in `docs/calibration/audit/stage075a_b3_headroom_v1/results.json.gz`.

No empirical superiority or online population-risk guarantee is claimed.
