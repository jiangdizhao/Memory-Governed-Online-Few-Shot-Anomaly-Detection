from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from experiments.calibration import nshadow_independent_b3_c64 as experiment


def _synthetic_row(position: int) -> dict[str, object]:
    return {
        "position": position,
        "event_id_sha256": f"{position:064x}",
        "image_identity_sha256": f"{position + 100:064x}",
        "source_group_sha256": "a" * 64,
        "camera_id": (position % 5) + 1,
        "query_actions": {"F-B": {"executed": position == 2}},
        "n_shadow_score": 0.1 * position,
    }


def test_five_view_predictions_share_one_pre_group_state_and_threshold() -> None:
    rows = experiment._atomic_unit_prediction_rows(
        [_synthetic_row(position) for position in range(5)],
        threshold=0.25, state_sha256="b" * 64,
        state_version=7, b3_evidence_version=4)

    assert len(rows) == 5
    assert {row["source_state_sha256"] for row in rows} == {"b" * 64}
    assert {row["source_state_version"] for row in rows} == {7}
    assert {row["b3_evidence_version"] for row in rows} == {4}
    assert {row["independent_b3_threshold"] for row in rows} == {0.25}
    assert [row["prediction"] for row in rows] == [0, 0, 0, 1, 1]
    assert "label" not in rows[0]
    assert [row["inherited_f_b_query_executed"] for row in rows] == [False, False, True, False, False]


def test_real_iad_and_visa_keep_their_registered_journal_envelope_schemas() -> None:
    assert experiment._original_unit_schema("realiad") == "final-benchmark.stage-a-group.v1"
    assert experiment._original_unit_schema("visa_1cls") == "final-benchmark.stage-a-atomic-unit.v1"
    with pytest.raises(experiment.SupplementaryError):
        experiment._original_unit_schema("unknown")


def test_replay_engine_exposes_the_historical_shadow_digest_interface() -> None:
    assert callable(getattr(experiment.ReplayEngine, "shadow_state_digest", None))


def test_query_receipt_matches_dataset_specific_historical_contract() -> None:
    row = {"event_id_sha256": "d" * 64, "source_group_sha256": "e" * 64,
           "atomic_unit_sha256": "f" * 64, "atomic_unit_kind": "VISA_SINGLE_IMAGE",
           "position": 12}
    real = experiment._query_receipt(SimpleNamespace(
        dataset_id="realiad", category="audiojack", condition="P0"), row)
    visa = experiment._query_receipt(SimpleNamespace(
        dataset_id="visa_1cls", category="cashew", condition="P0"), row)

    assert real == {"schema": "final-benchmark.query-receipt.v1", "arm_id": "F-B",
        "category": "audiojack", "condition_id": "P0", "event_id_sha256": "d" * 64,
        "position": 12, "source_group_sha256": "e" * 64}
    assert visa == {"schema": "final-benchmark.query-receipt.atomic-unit.v1",
        "arm_id": "F-B", "category": "cashew", "condition_id": "P0",
        "event_id_sha256": "d" * 64, "position": 12,
        "atomic_unit_sha256": "f" * 64, "atomic_unit_kind": "VISA_SINGLE_IMAGE"}


@pytest.mark.parametrize(
    ("dataset_id", "schema", "extra"),
    [
        ("realiad", "final-benchmark.private-feedback.v1", {}),
        ("visa_1cls", "final-benchmark.visa-private-feedback.v1",
         {"atomic_unit_sha256": "f" * 64, "image_identity_sha256": "d" * 64}),
    ],
)
def test_purchased_feedback_validation_uses_exact_dataset_identity_fields(
    dataset_id: str, schema: str, extra: dict[str, str],
) -> None:
    trajectory = SimpleNamespace(dataset_id=dataset_id)
    source_row = {"position": 12, "event_id_sha256": "a" * 64,
        "source_group_sha256": "e" * 64,
        "atomic_unit_sha256": "f" * 64, "image_identity_sha256": "d" * 64}
    action = {"query_receipt_sha256": "b" * 64}
    item = {"schema": schema, "arm_id": "F-B", "position": 12,
        "event_id_sha256": "a" * 64, "query_receipt_sha256": "b" * 64,
        "label": 0, **extra}

    assert experiment._purchased_feedback_matches(trajectory, "F-B", source_row, action, item)
    corrupted = dict(item, image_identity_sha256="0" * 64)
    if dataset_id == "visa_1cls":
        assert not experiment._purchased_feedback_matches(
            trajectory, "F-B", source_row, action, corrupted)


def test_atomic_prediction_rejects_nonfinite_score_and_invalid_state() -> None:
    bad_score = _synthetic_row(0)
    bad_score["n_shadow_score"] = float("nan")
    with pytest.raises(experiment.SupplementaryError):
        experiment._atomic_unit_prediction_rows(
            [bad_score], threshold=0.2, state_sha256="b" * 64,
            state_version=0, b3_evidence_version=0)
    with pytest.raises(experiment.SupplementaryError):
        experiment._atomic_unit_prediction_rows(
            [_synthetic_row(0)], threshold=0.2, state_sha256="not-a-state-digest",
            state_version=0, b3_evidence_version=0)


def test_fresh_campaign_root_never_reuses_existing_path(tmp_path) -> None:
    base = tmp_path / "campaign"
    base.mkdir()
    sentinel = base / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    actual = experiment._fresh_output_root(base)

    assert actual == tmp_path / "campaign_attempt_02"
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_category_bootstrap_is_reproducible_and_preserves_values() -> None:
    values = {"alpha": -0.2, "beta": 0.0, "gamma": 0.4}
    first = experiment._paired_bootstrap(values, seed=123)
    second = experiment._paired_bootstrap(values, seed=123)

    assert first == second
    assert first["status"] == "DEFINED"
    assert first["category_count"] == 3
    assert first["mean"] == pytest.approx(0.2 / 3)
    assert first["replicates"] == 10_000


def test_no_mask_or_label_fields_are_added_to_prediction_rows() -> None:
    rows = experiment._atomic_unit_prediction_rows(
        [_synthetic_row(0)], threshold=1.0, state_sha256="c" * 64,
        state_version=0, b3_evidence_version=0)
    prohibited = {"label", "mask", "outcome", "ground_truth"}
    assert not prohibited.intersection(rows[0])
    # Records are canonical-JSON serializable before their durable fsync.
    assert json.loads(experiment.canonical(rows).decode("utf-8"))[0]["prediction"] == 0
