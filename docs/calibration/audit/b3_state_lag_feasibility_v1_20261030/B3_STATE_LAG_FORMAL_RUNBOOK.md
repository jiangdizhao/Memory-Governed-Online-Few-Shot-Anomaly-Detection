# Later formal Experiment A runbook

This runbook is for a separately authorized performance evaluation. Do not run its outcome-scoring stage until the researcher explicitly authorizes it.

## Frozen inputs

- Supply the original checkout as the read-only frozen root.
- Verify all source and input digests before replay.
- Use only the four registered development classes and support seeds 0-4, fixed Q-A-TS tape, stored Native score/memory transitions, and fixed C64 identities.
- Never reconstruct missing C64 vectors or memory states.

## Matched controller replay

For each trajectory, construct its complete VersionedC64History and immutable TrajectoryEvent sequence, then call replay_matched_arms(events, history, delta=0.01). This returns A2/A3/A4 predictions, pre/post thresholds, and update ledgers. A2 uses q_v; A3 uses q_max(0,v-1); A4 uses q_max(0,v-5). All use the same q90 estimator, initial q0, Delta, event positions, scores, and memory transitions. Prediction uses score >= theta. Updates happen after prediction and affect only the next event.

Before offline metrics, assert identical positions, score inputs, memory digests, update positions/counts, C64 identity hashes, and initial thresholds across arms. Persist each arm under a new immutable, checksummed namespace. Bind source commit, inputs, query tape, arm configuration, and environment in its manifest. Do not write to the frozen root.

## Outcome stage

Only after separate authorization may an evaluator join prediction streams to permitted labels and calculate preregistered full-stream, common-horizon, and worst-window metrics. Keep labels outside controller state and replay checkpoints. Do not access confirmation classes or masks. Do not tune on these results.
