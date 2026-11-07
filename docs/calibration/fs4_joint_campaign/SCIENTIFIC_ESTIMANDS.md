# Scientific estimands

## H1: threshold-only effect

Within `S`, the native encoder, score sequences, Greedy/C1 trajectory, BEA reference, query decisions, and legal feedback are shared. L and E differ only in their threshold evidence update. The primary estimand is the equal-category-weighted mean of `BER(S/E/GC) - BER(S/L/GC)`, paired within category. Negative values favor E.

The `FrozenTheta` readout uses the frozen four-Support threshold on this same evolving memory trajectory. It is not a separate static VisionAD memory baseline.

## H2: closed-loop system effect

CL and CE are independent query-only trajectories. Their active GC controller threshold changes the BEA query reference. Each owns its own query policy, queried feedback, Greedy memory, C1 state, Coverage Memory and checkpoint lineage. The secondary estimand is the equal-category-weighted mean of `BER(CE/E/GC) - BER(CL/L/GC)`. This is a whole-system contrast; it does not isolate threshold estimation.

## H3: reliability and cost

Report query behavior, normal-evidence counts and ESS, threshold activity and holds, E cache occupancy, exact rescore count and measured rescore time, native memory state, event/run timing, host/GPU memory, and all suppressed or undefined metrics. An inactive or expensive controller and a negative or inconclusive estimate are valid outcomes.

The four categories are previously exposed development data and one query seed is used. Images within a five-view Source Group are correlated. No population guarantee, formal unbiasedness, conformal validity, or exact counterfactual-policy confidence interval is claimed. No resampling interval is registered.
