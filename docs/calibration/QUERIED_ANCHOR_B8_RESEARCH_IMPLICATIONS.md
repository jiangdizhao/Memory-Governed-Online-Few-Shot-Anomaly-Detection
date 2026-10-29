# Queried-Anchor Identifiability: Research Implications

This is a development-set measurement, not a deployment or risk guarantee.

## Findings

1. Legal sparse feedback yielded a complete two-class estimate in 90/120 snapshot states (75.0%); the selected actions were `{"ABSTAIN": 31, "KEEP": 26, "LOWER": 33, "RAISE": 30}`. This is query-anchor coverage, not population coverage.
2. Future-selected-action mean ΔBA by frozen defect-count bin is `{"0": {"all_remaining_group_clean": {"active_action_count": 0, "false_activation_count": 0, "false_activation_rate": null, "future_defined_count": 30, "mean_selected_future_delta_ba": null, "query_action_counts": {"ABSTAIN": 30, "KEEP": 0, "LOWER": 0, "RAISE": 0}, "query_identifiable_future_defined_count": 0, "state_count": 30}, "next256_group_clean": {"active_action_count": 0, "false_activation_count": 0, "false_activation_rate": null, "future_defined_count": 15, "mean_selected_future_delta_ba": null, "query_action_counts": {"ABSTAIN": 30, "KEEP": 0, "LOWER": 0, "RAISE": 0}, "query_identifiable_future_defined_count": 0, "state_count": 30}}, "1-2": {"all_remaining_group_clean": {"active_action_count": 4, "false_activation_count": 4, "false_activation_rate": 1.0, "future_defined_count": 5, "mean_selected_future_delta_ba": -0.010430028110202928, "query_action_counts": {"ABSTAIN": 0, "KEEP": 1, "LOWER": 1, "RAISE": 3}, "query_identifiable_future_defined_count": 5, "state_count": 5}, "next256_group_clean": {"active_action_count": 4, "false_activation_count": 3, "false_activation_rate": 0.75, "future_defined_count": 5, "mean_selected_future_delta_ba": -0.003003003003003002, "query_action_counts": {"ABSTAIN": 0, "KEEP": 1, "LOWER": 1, "RAISE": 3}, "query_identifiable_future_defined_count": 5, "state_count": 5}}, "14+": {"all_remaining_group_clean": {"active_action_count": 46, "false_activation_count": 25, "false_activation_rate": 0.5434782608695652, "future_defined_count": 65, "mean_selected_future_delta_ba": -0.0037904054724333235, "query_action_counts": {"ABSTAIN": 1, "KEEP": 18, "LOWER": 26, "RAISE": 20}, "query_identifiable_future_defined_count": 64, "state_count": 65}, "next256_group_clean": {"active_action_count": 38, "false_activation_count": 20, "false_activation_rate": 0.5263157894736842, "future_defined_count": 50, "mean_selected_future_delta_ba": -0.0043625519443712445, "query_action_counts": {"ABSTAIN": 1, "KEEP": 18, "LOWER": 26, "RAISE": 20}, "query_identifiable_future_defined_count": 49, "state_count": 65}}, "3-7": {"all_remaining_group_clean": {"active_action_count": 6, "false_activation_count": 5, "false_activation_rate": 0.8333333333333334, "future_defined_count": 10, "mean_selected_future_delta_ba": -0.004395939734166382, "query_action_counts": {"ABSTAIN": 0, "KEEP": 4, "LOWER": 0, "RAISE": 6}, "query_identifiable_future_defined_count": 10, "state_count": 10}, "next256_group_clean": {"active_action_count": 6, "false_activation_count": 4, "false_activation_rate": 0.6666666666666666, "future_defined_count": 10, "mean_selected_future_delta_ba": -0.004364394822060724, "query_action_counts": {"ABSTAIN": 0, "KEEP": 4, "LOWER": 0, "RAISE": 6}, "query_identifiable_future_defined_count": 10, "state_count": 10}}, "8-13": {"all_remaining_group_clean": {"active_action_count": 7, "false_activation_count": 2, "false_activation_rate": 0.2857142857142857, "future_defined_count": 10, "mean_selected_future_delta_ba": 0.0012248865600002184, "query_action_counts": {"ABSTAIN": 0, "KEEP": 3, "LOWER": 6, "RAISE": 1}, "query_identifiable_future_defined_count": 10, "state_count": 10}, "next256_group_clean": {"active_action_count": 7, "false_activation_count": 5, "false_activation_rate": 0.7142857142857143, "future_defined_count": 10, "mean_selected_future_delta_ba": -0.0041659838406110285, "query_action_counts": {"ABSTAIN": 0, "KEEP": 3, "LOWER": 6, "RAISE": 1}, "query_identifiable_future_defined_count": 10, "state_count": 10}}}`. The 8+ bins are descriptive; no label count is certified as sufficient for population reliability.
3. LOWER was selected 33 times and RAISE 30 times. Their per-state queried gains are in Report B; future unique-best agreement and selected-action gains are reported separately with denominators.
4. Full-memory rather than LSO normal scoring changed the primary action in 36/90 comparable states; primary estimates use source-aware all-layer LSO.
5. Current-memory rescoring is the primary evidence. Source-policy-comparable arrival-score sensitivity changed the action in 31/90 states; source-policy-mismatched normal arrivals are excluded and reported.
6. Regulator and Rolled Strip Base class-first future selected ΔBA summaries are `{"regulator": {"all_remaining_group_clean": -0.005702723211323179, "next256_group_clean": -0.007043581426589213}, "rolled_strip_base": {"all_remaining_group_clean": -0.0025889735298500562, "next256_group_clean": -0.0007349995730549975}}`. Equal-class macros are provided below.
7. The measured candidate decision is `LOCAL_FEEDBACK_UNRELIABLE`. It summarizes temporally separated development outcomes; it does not implement an online threshold correction.
8. Additional percentile transport was not tested by this audit. C64 parity is a reconstruction check, so these measurements alone do not justify a new transport correction.
9. Sparse selective feedback, source/group correlation, temporal dependence, overlapping snapshots/cohorts, changing memory and prior exposure of all four categories remain the central uncertainties. No IID or population-risk claim is made.
10. B8 Recovery remains unimplemented. Any further method must receive a new scoped researcher decision; this report does not authorize B8 v4 or another stage.

## Class-specific summary

```json
{
  "regulator": {
    "cohorts": {
      "all_remaining_group_clean": {
        "abstention_count": 5,
        "direction_agreement": 0.28,
        "direction_agreement_denominator": 25,
        "direction_agreement_numerator": 7,
        "false_activation_denominator": 18,
        "false_activation_numerator": 12,
        "false_activation_rate": 0.6666666666666666,
        "future_blocked_native_rows": 0,
        "future_defined_rows": 30,
        "future_one_class_rows": 0,
        "harm_magnitude_conditional_mean_ba_loss": 0.017874464029069947,
        "mean_loss_to_best_registered_ba": 0.013843629373944313,
        "mean_selected_future_delta_ba": -0.005702723211323179,
        "query_action_counts": {
          "ABSTAIN": 5,
          "KEEP": 7,
          "LOWER": 5,
          "RAISE": 13
        },
        "query_identifiable_defined_rows": 25,
        "unique_future_best_rows": 30
      },
      "next256_group_clean": {
        "abstention_count": 5,
        "direction_agreement": 0.15,
        "direction_agreement_denominator": 20,
        "direction_agreement_numerator": 3,
        "false_activation_denominator": 14,
        "false_activation_numerator": 11,
        "false_activation_rate": 0.7857142857142857,
        "future_blocked_native_rows": 0,
        "future_defined_rows": 25,
        "future_one_class_rows": 5,
        "harm_magnitude_conditional_mean_ba_loss": 0.015080653059437434,
        "mean_loss_to_best_registered_ba": 0.01860520864004892,
        "mean_selected_future_delta_ba": -0.007043581426589213,
        "query_action_counts": {
          "ABSTAIN": 5,
          "KEEP": 7,
          "LOWER": 5,
          "RAISE": 13
        },
        "query_identifiable_defined_rows": 20,
        "unique_future_best_rows": 25
      }
    },
    "query_action_counts": {
      "ABSTAIN": 5,
      "KEEP": 7,
      "LOWER": 5,
      "RAISE": 13
    },
    "query_identifiable_coverage": 0.8333333333333334,
    "query_identifiable_state_count": 25,
    "snapshot_state_count": 30
  },
  "rolled_strip_base": {
    "cohorts": {
      "all_remaining_group_clean": {
        "abstention_count": 6,
        "direction_agreement": 0.5,
        "direction_agreement_denominator": 24,
        "direction_agreement_numerator": 12,
        "false_activation_denominator": 16,
        "false_activation_numerator": 8,
        "false_activation_rate": 0.5,
        "future_blocked_native_rows": 0,
        "future_defined_rows": 30,
        "future_one_class_rows": 0,
        "harm_magnitude_conditional_mean_ba_loss": 0.015410931773834756,
        "mean_loss_to_best_registered_ba": 0.010072271061015102,
        "mean_selected_future_delta_ba": -0.0025889735298500562,
        "query_action_counts": {
          "ABSTAIN": 6,
          "KEEP": 8,
          "LOWER": 13,
          "RAISE": 3
        },
        "query_identifiable_defined_rows": 24,
        "unique_future_best_rows": 30
      },
      "next256_group_clean": {
        "abstention_count": 6,
        "direction_agreement": 0.5,
        "direction_agreement_denominator": 18,
        "direction_agreement_numerator": 9,
        "false_activation_denominator": 16,
        "false_activation_numerator": 8,
        "false_activation_rate": 0.5,
        "future_blocked_native_rows": 0,
        "future_defined_rows": 25,
        "future_one_class_rows": 5,
        "harm_magnitude_conditional_mean_ba_loss": 0.018156223898068292,
        "mean_loss_to_best_registered_ba": 0.010423278429757167,
        "mean_selected_future_delta_ba": -0.0007349995730549975,
        "query_action_counts": {
          "ABSTAIN": 6,
          "KEEP": 8,
          "LOWER": 13,
          "RAISE": 3
        },
        "query_identifiable_defined_rows": 19,
        "unique_future_best_rows": 24
      }
    },
    "query_action_counts": {
      "ABSTAIN": 6,
      "KEEP": 8,
      "LOWER": 13,
      "RAISE": 3
    },
    "query_identifiable_coverage": 0.8333333333333334,
    "query_identifiable_state_count": 25,
    "snapshot_state_count": 30
  }
}
```

## Equal-class summaries

```json
{
  "all_remaining_group_clean": {
    "direction_agreement": {
      "categories": [
        "audiojack",
        "regulator",
        "switch",
        "rolled_strip_base"
      ],
      "contributing_class_count": 4,
      "equal_class_mean": 0.34500000000000003
    },
    "false_activation_rate": {
      "categories": [
        "audiojack",
        "regulator",
        "switch",
        "rolled_strip_base"
      ],
      "contributing_class_count": 4,
      "equal_class_mean": 0.5986111111111111
    },
    "harm_magnitude_conditional_mean_ba_loss": {
      "categories": [
        "audiojack",
        "regulator",
        "switch",
        "rolled_strip_base"
      ],
      "contributing_class_count": 4,
      "equal_class_mean": 0.01758187042315581
    },
    "mean_loss_to_best_registered_ba": {
      "categories": [
        "audiojack",
        "regulator",
        "switch",
        "rolled_strip_base"
      ],
      "contributing_class_count": 4,
      "equal_class_mean": 0.01294066301217537
    },
    "mean_selected_future_delta_ba": {
      "categories": [
        "audiojack",
        "regulator",
        "switch",
        "rolled_strip_base"
      ],
      "contributing_class_count": 4,
      "equal_class_mean": -0.003929498707270034
    }
  },
  "next256_group_clean": {
    "direction_agreement": {
      "categories": [
        "audiojack",
        "regulator",
        "switch",
        "rolled_strip_base"
      ],
      "contributing_class_count": 4,
      "equal_class_mean": 0.36250000000000004
    },
    "false_activation_rate": {
      "categories": [
        "audiojack",
        "regulator",
        "switch",
        "rolled_strip_base"
      ],
      "contributing_class_count": 4,
      "equal_class_mean": 0.5853174603174603
    },
    "harm_magnitude_conditional_mean_ba_loss": {
      "categories": [
        "audiojack",
        "regulator",
        "switch",
        "rolled_strip_base"
      ],
      "contributing_class_count": 4,
      "equal_class_mean": 0.022479517653621432
    },
    "mean_loss_to_best_registered_ba": {
      "categories": [
        "audiojack",
        "regulator",
        "switch",
        "rolled_strip_base"
      ],
      "contributing_class_count": 4,
      "equal_class_mean": 0.017747343539792956
    },
    "mean_selected_future_delta_ba": {
      "categories": [
        "audiojack",
        "regulator",
        "switch",
        "rolled_strip_base"
      ],
      "contributing_class_count": 4,
      "equal_class_mean": -0.004355247652989663
    }
  }
}
```

No confidence interval is interpreted as IID coverage. Future cohorts overlap
and all categories were previously exposed during development.
