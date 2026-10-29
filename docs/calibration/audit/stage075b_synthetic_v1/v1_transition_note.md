# Candidate v1 to v2 transition note

Candidate v1 limited B8's searched raw-threshold set to the immediately reachable `theta ± 0.01` interval. The rejoin fixture showed a mechanism trap: if each adjacent move has zero empirical gain but a farther score boundary improves the declared objective, B8 cannot start moving toward that boundary.

Candidate v2 therefore searches the full current evidence grid, selects the global desired threshold under the same objective, then projects only the executed action into the unchanged `0.01` movement interval. This is a methodological revision and has a new contract and method version. The primary 48-row matrix used v2 only; no v1 outcome was mixed into it.

One v1 development integration smoke ran `S1_monotone_coordinates`, seed `37`, and only printed the scenario key. It did not save metrics or a trajectory artifact, so it is not included as primary evidence and no result is reconstructed. The first rejoin fixture assertion failed before the fixture was corrected; this was an engineering test failure that exposed the candidate-set trap, not an empirical controller result. The original candidate v1 contract is preserved in the repository with a superseded status.
