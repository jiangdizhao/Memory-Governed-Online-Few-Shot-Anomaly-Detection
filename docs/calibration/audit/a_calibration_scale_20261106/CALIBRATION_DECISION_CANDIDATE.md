# Calibration Decision Candidate

**Candidate: HOLD C* freeze.** A1 found somewhat lower q90 threshold draw variation at C128 than C64 on average (P0 mean sample SD 0.00155 vs 0.00341; P2 0.00099 vs 0.00270), but the improvement is not universal across categories and the H-tail and TEST operating outcomes vary by category. Fixed 0.5, empirical q90, Gaussian-Q90, and support-derived rules each have category-specific tradeoffs.

Retain historical C64 as the A1 comparator. Carry C128 as a candidate to a separately authorized A2 study because it tended to reduce threshold sampling variation with only 64 additional calibration images. Do not freeze C128 or C64 for Experiments B/C from A1 alone. A2 must test current-state recalibration on shared Greedy and Greedy+C1 trajectories; A3a must test calibration-coupled learning. A3b remains `HOLD_BASELINE_FIDELITY` until its equal-byte expanded coreset rule is preregistered and verified.

A1's TEST metrics are development evaluation, not a tuning criterion. No independent confirmation, formal risk guarantee, or evidence of superiority is claimed.
