# Experiment A integrated interpretation addendum

This addendum answers the registered interpretation questions using the completed A2/A3
aggregate and the unchanged A1 report. The principal report and machine-readable results
remain [the integrated report](EXPERIMENT_A_A2_A3_INTEGRATED_REPORT.md) and
[the aggregate](A2_A3_AGGREGATE_RESULTS.json). A1 sampling and H-audit values below come
from [the original A1 report](REPORT.md); A1 was not rerun or modified.

## Calibration size and threshold stability

A1's three-draw resampling showed lower average threshold variation at C128 than C64:

| Condition | C64 mean threshold SD | C128 mean threshold SD | C64 mean range | C128 mean range |
|---|---:|---:|---:|---:|
| P0 | 0.00341 | 0.00155 | 0.00619 | 0.00280 |
| P2 | 0.00270 | 0.00099 | 0.00518 | 0.00184 |

This is a sampling-stability result, not a universal operating-point improvement. Regulator
P0 was the A1 category exception. The 32-group H audit has 1/32 = 0.03125 resolution; for
P0 C64 its exceedance fractions were audiojack 0.125, regulator 0.031, switch 0.063, and
rolled_strip_base 0.063. H is a normal-only development audit, not an independent production
sample.

In A2's shared current-state trajectory, mean B3 threshold total variation was 0.09020 for
C64 and 0.09069 for C128. Thus C128 did not reduce this online movement measure. P0 G BER
changed from 0.23200 to 0.23154, while FPR fell 0.10843 to 0.10039 and FNR rose 0.35558 to
0.36269. AUROC/AP were identical because both calibration readouts used the same G score
history. Cmax-G's P0 G BER was 0.23199, effectively the same observed operating result as
C64. No practical-equivalence margin was preregistered, so these small differences do not
establish practical equivalence or a winner.

P1/P2 equal-weight means also did not favor C128: G BER was 0.21647 vs 0.21800 in P1 and
0.24194 vs 0.24344 in P2 (C64 vs C128); GC BER was 0.19700 vs 0.19893 in P1 and 0.23497
vs 0.23591 in P2. Category-specific values remain in the aggregate. Across conditions,
C64 remains a useful limited-calibration benchmark. A1 supports retaining C128 as a
stability sensitivity, but the online results do not justify selecting it as the sole
commissioning size.

## A2, correction, and calibration-coupled learning

At P0, G mean AUROC/AP/BER were 0.83199/0.76384/0.23200 for C64 and
0.83199/0.76384/0.23154 for C128. GC values were 0.85687/0.81332/0.21626 and
0.85687/0.81332/0.21560. The G-to-GC ranking improvement is a Correction scoring effect;
the C64/C128 readout itself leaves ranking scores unchanged. Cmax-G GC BER was 0.21411,
but its small difference from C64 lacks a registered meaningful-improvement threshold.
The correction-related ranking conclusion therefore did not materially change under C64
versus C128 readout in A2.

A3a used independent calibration-coupled trajectories. Mean P0 G BER was 0.22210 for C64
and 0.22448 for C128, with mixed category directions. The first state-hash divergence
occurred at event positions 35 (audiojack), 268 (regulator), 55 (switch), and 35
(rolled_strip_base). Initial normal-memory parity and paired query identities were
preserved. The subsequent update and score histories are not a threshold-only effect.
Correction updates differed between the arms in all four categories; G scores differed on
53 audiojack, 11 switch, and 20 rolled_strip_base events, and on no regulator events.
Calibration can therefore alter later learning, but its effect was not uniformly beneficial.

## A3b alternative reference and online adaptation

FEXP-KC and F4 used equal deployed storage (8,192 fp16 prototypes at 4,096 dimensions,
67,108,864 bytes). FEXP-KC had a larger predeployment normal-data pool and construction
cost. At P0, static mean BER was 0.40004 for F4 and 0.22545 for FEXP-KC; prequential
Greedy mean BER was 0.22137 and 0.16946, respectively. The expanded reference therefore
substantially changes the fixed-reference baseline and challenges attributing prior online
gains solely to online adaptation. It does not remove the observed within-FEXP-KC
prequential improvement (0.22545 to 0.16946 BER). Both mechanisms contribute in this
comparison; they are not isolated by comparing different reference banks alone.

## Resources, status, and decision

All four P0 Cmax-G resource gates enabled full trajectories; projected remaining work was
within the registered category limit. Mean TEST feature encoding was 393.56 seconds per
category. Mean A3b reference construction was 91.52 seconds per category, and a measured
native full-reference Cmax rescore averaged 7.04 seconds. The retained per-category TEST
feature cache was about 49 GB of float32 scratch and was not retained in the public report.
Recorded dynamic state storage was about 16.84 MB for the C1 bank plus at most 7.03 MB for
the Greedy bank in the reported trajectories; the fixed reference bank was 64 MiB. Peak
CUDA allocation was not recorded in the sealed resource metrics and is therefore unverified.
The full campaign finished in about 4 hours 24 minutes on the RTX 3090 Ti.

All registered A2, A3a, and A3b branches completed on the four authorized development
categories. No scientific branch was held for resources. An earlier attempt stopped before
TEST score rows at an initialization error; the final run sealed all Stage A conditions and
Stage B metrics. A report-assembly defect was repaired from those checksum-verified seals;
experimental trajectories and Stage B metrics were not rerun. The current source therefore
does not match the runner SHA recorded in the immutable run manifest. The strict source-bound
`--verify-only` command was not rerun; the finalization manually verified the manifest,
contract, category-condition seals, Stage A/B linkage, complete seal, and metrics checksum.
The primary report records renderer SHA-256
`cf86cc6c8199c704866b45c8a9b0e9b9ad8f1884a7e548f57aed20822619e25a`; the committed
runner SHA-256 is `ed1c5908ea9e380dfbca6c4cb52ffe74de7901a883806fa5c020e96bd5e1b111` after
removing one trailing blank line. No executable logic changed after report rendering.

**Decision candidate: `HOLD_CALIBRATION`.** Retain C64 as the benchmark, keep C128 as a
registered stability sensitivity, and do not freeze a B/C calibration size from these
development categories alone. The evidence does not show a consistent C128 online benefit;
Cmax-G's full trajectories were feasible but had little direct A2 G BER change. The A3b
result makes fixed-reference construction a material alternative explanation. Independent
confirmation performance, physical object independence beyond the official source-key
proxy, and production risk remain unproven. This decision does not authorize Experiments
B/C.
