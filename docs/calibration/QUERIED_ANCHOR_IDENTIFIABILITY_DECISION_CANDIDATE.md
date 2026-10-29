# Queried-Anchor Identifiability Decision Candidate

**Candidate decision:** `LOCAL_FEEDBACK_UNRELIABLE`

**Review status:** requires explicit researcher approval. This report does
not authorize B8 v4, Stage 0.75-C, Stage 1/2, or confirmation-class access.

## Evidence

90/120 state estimates had complete two-class query evidence. Query actions: {'LOWER': 33, 'KEEP': 26, 'RAISE': 30, 'ABSTAIN': 31}; per-class mean selected-action future ΔBA across defined cohorts: {'audiojack': -0.006555810308456675, 'regulator': -0.0063731523189561965, 'switch': -0.0019785435416539965, 'rolled_strip_base': -0.001661986551452527}; per-class unique-best agreement: {'audiojack': [0.4, 0.2], 'regulator': [0.15, 0.28], 'switch': [0.4, 0.4], 'rolled_strip_base': [0.5, 0.5]}; identified future-defined states by defect bin and cohort: {'0': {'next256_group_clean': 0, 'all_remaining_group_clean': 0}, '1-2': {'next256_group_clean': 5, 'all_remaining_group_clean': 5}, '3-7': {'next256_group_clean': 10, 'all_remaining_group_clean': 10}, '8-13': {'next256_group_clean': 10, 'all_remaining_group_clean': 10}, '14+': {'next256_group_clean': 49, 'all_remaining_group_clean': 64}}. All figures describe temporally ordered, previously exposed development data; overlapping states/cohorts are not independent and no population guarantee follows.

Descriptive candidate rule: any blocked native snapshot yields NATIVE_EVIDENCE_BLOCKED; otherwise require two-class query estimates and defined future gains. Mixed signs across class means yield MIXED_BY_CATEGORY. A positive mean only in the frozen 8+ evidence bins yields IDENTIFIABLE_ONLY_WITH_SUFFICIENT_DEFECT_EVIDENCE. LOCAL_FEEDBACK_IDENTIFIABLE requires positive selected-action mean BA gain and >0.5 unique-best agreement in every category; other observed patterns are unreliable. These are report classification rules, not online activation thresholds or risk guarantees.

## Required interpretation

Query-derived gains describe the selected queried anchors, not an unbiased
population BA estimate. Evaluations are temporally separated within four
previously exposed development categories; snapshot windows overlap. No IID
coverage, deployment risk, or independent confirmation claim is made.

## Researcher decisions still required

- Whether the observed identification coverage and harm frequency justify any separately frozen follow-up.
- Whether B8 Recovery should be paused, simplified, or reconsidered under a new protocol.
- Whether any next-stage experiment should be authorized at all.
