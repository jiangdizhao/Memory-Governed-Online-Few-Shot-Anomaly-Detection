# FS4 R1 synthetic query stress

The [machine-readable stress record](HARDENING_R1_QUERY_STRESS.json) contains every scenario's candidate and execution counts, score-region coverage, eligible probability counts, and the price sampled every 100 events. Each scenario uses 400 synthetic groups, five views per group, seed 37, and no image or outcome data. Outcomes are supplied only after an executed synthetic query and never influence the generated score stream or policy. The low-feedback scenario supplies no normal labels.

| Scenario | Queries / 2,000 events | Rate | Exploration queries | First / final price | Longest query-free interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| High utility | 101 | 5.05% | 21 | 0.7993 / 1.7627 | 265 |
| Low utility | 25 | 1.25% | 25 | 0.7993 / 0.0500 | 272 |
| Stable scores | 99 | 4.95% | 24 | 0.7993 / 1.3744 | 74 |
| Abrupt upward shift | 78 | 3.90% | 24 | 0.7993 / 1.0133 | 272 |
| Abrupt downward shift | 78 | 3.90% | 24 | 0.7993 / 1.0133 | 272 |
| Gradual drift | 38 | 1.90% | 24 | 0.7993 / 0.1659 | 272 |
| Long no-exploitation interval | 25 | 1.25% | 25 | 0.7993 / 0.0500 | 272 |
| Interleaved views | 100 | 5.00% | 20 | 0.7993 / 1.4249 | 90 |
| Exploration/exploitation imbalance | 74 | 3.70% | 22 | 0.7993 / 1.2298 | 272 |
| Score and novelty ties | 74 | 3.70% | 21 | 0.7993 / 0.9924 | 272 |

Each run had exactly 400 eligible candidate opportunities. The realized group query fraction is five times the image-level rate because each group has five views and at most one candidate. Eligible opportunities recorded only 0.2 or 0.01 design probabilities; ineligible views recorded no gate or design probability. The action and candidate use separate SHA-256 domains, and the test checks each realized action against the independently recomputed hash and gate. This is a computational domain-separation audit, not proof of physical randomness or independence under a fixed seed. All score regions observed in these particular runs received at least one query; unobserved regions and per-region counts remain in the JSON record. The no-exploitation stream produces zero verified normals without inserting any labels or queries.

Historical deterministic BEA can have zero query propensity. These stochastic probabilities are the registered slot-and-gate factors at the realized opportunity, not an unconditional fixed-population inclusion probability when price, earlier feedback and scorer states adapt. Weight clipping and ESS gates remain unchanged. The stress suite does not tune the policy, impose a quota, or establish an FPR, coverage or conformal guarantee.
