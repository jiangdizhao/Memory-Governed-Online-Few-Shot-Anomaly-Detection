"""D1-A-R1 CPU-only tests: official Real-IAD identity, partitions, Q-A tapes.

No Real-IAD images, no DINOv2, no GPU.  Synthetic metadata lives in temporary
directories and, for most tests, in in-memory sample objects whose file names
are constructed independently of the implementation.
"""
from __future__ import annotations

from dataclasses import fields, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import experiments.risk_safe.preflight as pf          # noqa: E402
import experiments.risk_safe.query_protocol as qp    # noqa: E402

CLASS = "audiojack"
IMAGE_ROOT = "C:/synth/realiad/images/audiojack"
OK_TS = "20231021130235"
NG_TS = "20231023111538"


def ok_sample(obj: int, camera: int, ts: str = OK_TS) -> SimpleNamespace:
    path = (f"{IMAGE_ROOT}/OK/S{obj:04d}/"
            f"{CLASS}_{obj:04d}_OK_C{camera}_{ts}.jpg")
    return SimpleNamespace(path=path, is_good=True)


def ng_sample(obj: int, camera: int, defect: str = "BX", ts: str = NG_TS,
              good: bool = False) -> SimpleNamespace:
    path = (f"{IMAGE_ROOT}/NG/{defect}/S{obj:04d}/"
            f"{CLASS}__{obj:04d}_NG_{defect}_C{camera}_{ts}.jpg")
    return SimpleNamespace(path=path, is_good=good)


def build_samples(train_objects: int = 120, normal_test: int = 250,
                  defect_test: int = 450, test_base: int = 1000):
    """Normal train samples, normal test samples, and defective test samples.

    Defective samples are photographed in one NG session; cameras 1-2 carry a
    normal *image-level* label (the defect is not visible from those views) and
    cameras 3-5 carry a defect label.  That is the Real-IAD label structure which
    forbids sample-to-view label propagation.

    Default sizes are large enough for the historical partition
    (2 x 48 probes + 3 x 175 stream samples = 621 test samples).
    """
    train = [ok_sample(obj, camera)
             for obj in range(1, train_objects + 1)
             for camera in range(1, 6)]
    test = [ok_sample(test_base + i, camera)
            for i in range(normal_test)
            for camera in range(1, 6)]
    test += [ng_sample(test_base + normal_test + i, camera, good=(camera <= 2))
             for i in range(defect_test)
             for camera in range(1, 6)]
    return train, test


_CACHE = {}


def cached_records():
    if "records" not in _CACHE:
        train, test = build_samples()
        _CACHE["records"] = (
            pf.records_from_samples(train, CLASS, "train"),
            pf.records_from_samples(test, CLASS, "test"))
    return _CACHE["records"]


def cached_plan_and_stream():
    if "plan" not in _CACHE:
        train, test = cached_records()
        plan = pf.build_partitions(train, test)
        _CACHE["plan"] = (plan, qp.build_stream(plan.phases))
    return _CACHE["plan"]


def labels_of(records) -> dict:
    return {record.path: record.is_good for record in records}


class OfficialIdentityTests(unittest.TestCase):
    def test_ok_and_ng_sessions_are_distinct_official_samples(self):
        ok = pf.parse_image_name(
            f"{IMAGE_ROOT}/OK/S0004/{CLASS}_0004_OK_C3_{OK_TS}.jpg")
        ng = pf.parse_image_name(
            f"{IMAGE_ROOT}/NG/BX/S0004/{CLASS}__0004_NG_BX_C5_{NG_TS}.jpg")
        self.assertEqual(ok.official_sample_key, f"{CLASS}_0004_OK")
        self.assertEqual(ng.official_sample_key, f"{CLASS}__0004_NG_BX")
        self.assertNotEqual(ok.official_sample_key, ng.official_sample_key)
        self.assertEqual(ng.capture_session, "NG_BX")
        self.assertEqual(ok.class_name, CLASS)

    def test_two_views_of_one_official_sample_share_one_key(self):
        first = pf.parse_image_name(
            f"{IMAGE_ROOT}/OK/S0004/{CLASS}_0004_OK_C1_{OK_TS}.jpg")
        second = pf.parse_image_name(
            f"{IMAGE_ROOT}/OK/S0004/{CLASS}_0004_OK_C5_{OK_TS}.jpg")
        self.assertEqual(first.official_sample_key, second.official_sample_key)
        self.assertEqual(first.directory_serial_key, "S0004")
        self.assertEqual(second.directory_serial_key, "S0004")
        self.assertNotEqual(first.image_id, second.image_id)

    def test_different_defect_conditions_are_different_official_samples(self):
        bx = pf.parse_image_name(
            f"{IMAGE_ROOT}/NG/BX/S0007/{CLASS}__0007_NG_BX_C1_{NG_TS}.jpg")
        hs = pf.parse_image_name(
            f"{IMAGE_ROOT}/NG/HS/S0007/{CLASS}__0007_NG_HS_C1_20231023091121.jpg")
        self.assertEqual(bx.official_sample_key, f"{CLASS}__0007_NG_BX")
        self.assertEqual(hs.official_sample_key, f"{CLASS}__0007_NG_HS")
        self.assertNotEqual(bx.official_sample_key, hs.official_sample_key)
        self.assertEqual(bx.directory_serial_key, hs.directory_serial_key)

    def test_official_key_matches_official_expression_and_v02_rule(self):
        """The extracted key must equal the official pattern and V0.2's rule."""
        import final_decision_core as legacy
        names = [
            f"{CLASS}_0001_OK_C1_{OK_TS}.jpg",
            f"{CLASS}__0001_NG_ZW_C3_{NG_TS}.jpg",
            "bottle_cap_0042_NG_AK_C2_20231023090000.jpg",
            "rolled_strip_base_0100_OK_C5_20231021130000.jpg",
            "plastic_plug__0250_NG_HS_C4_20231023090500.jpg",
        ]
        expected = [
            f"{CLASS}_0001_OK",
            f"{CLASS}__0001_NG_ZW",
            "bottle_cap_0042_NG_AK",
            "rolled_strip_base_0100_OK",
            "plastic_plug__0250_NG_HS",
        ]
        for name, key in zip(names, expected):
            self.assertEqual(
                pf.OFFICIAL_SAMPLE_KEY_RE.search(name).group("sample_key"), key)
            self.assertEqual(legacy.object_id(name), key)

    def test_grouping_units_are_separate(self):
        train = pf.records_from_samples([ok_sample(1, camera)
                                         for camera in range(1, 6)], CLASS, "train")
        test = pf.records_from_samples([ng_sample(1, camera)
                                        for camera in range(1, 6)], CLASS, "test")
        records = train + test
        self.assertEqual(len(pf.group_by_sample(records)), 2)
        self.assertEqual(len(pf.group_by_serial(records)), 1)
        self.assertEqual(list(pf.group_by_sample(
            records, unit=pf.SENSITIVITY_GROUPING_UNIT)),
            list(pf.group_by_serial(records)))

    def test_camera_view_extraction(self):
        for camera in range(1, 6):
            parsed = pf.parse_image_name(
                f"{IMAGE_ROOT}/OK/S0007/{CLASS}_0007_OK_C{camera}_{OK_TS}.jpg")
            self.assertEqual(parsed.camera_id, camera)
        ng = pf.parse_image_name(
            f"{IMAGE_ROOT}/NG/ZW/S0009/{CLASS}__0009_NG_ZW_C4_{NG_TS}.png")
        self.assertEqual((ng.camera_id, ng.capture_session), (4, "NG_ZW"))
        self.assertTrue(ng.rel_path.startswith("NG/ZW/S0009/"))

    def test_camera_distribution_and_histograms(self):
        train, test = cached_records()
        cameras = pf._camera_distribution(test)
        self.assertEqual(set(cameras["counts"]), {"C1", "C2", "C3", "C4", "C5"})
        self.assertAlmostEqual(sum(cameras["fractions"].values()), 1.0, places=5)
        histogram = pf._histogram([len(v) for v in
                                   pf.group_by_sample(train).values()])
        self.assertEqual(histogram, {"5": 120})

    def test_five_view_structure_is_reported(self):
        train, test = cached_records()
        comparison = pf.identity_comparison(CLASS, train, test)
        self.assertEqual(comparison["five_view_structure"]
                         ["samples_without_exact_C1_to_C5"], 0)
        self.assertEqual(comparison["five_view_structure"]
                         ["samples_with_duplicate_camera_id"], 0)
        self.assertEqual(comparison["images_per_official_sample"], {"5": 820})

    def test_malformed_image_names_fail_closed(self):
        cases = [
            f"{IMAGE_ROOT}/OK/S0001/{CLASS}_0001_OK_20231021130235.jpg",
            f"{IMAGE_ROOT}/OK/S0001/{CLASS}_0001_OK_C6_{OK_TS}.jpg",
            f"{IMAGE_ROOT}/OK/S0002/{CLASS}_0001_OK_C1_{OK_TS}.jpg",
            f"{IMAGE_ROOT}/OK/S0001/{CLASS}_0001_OK_C1_20231321130235.jpg",
            f"{IMAGE_ROOT}/NG/S0001/{CLASS}__0001_NG_C1_{NG_TS}.jpg",
            f"{IMAGE_ROOT}/OK/S0001/{CLASS}_0001_OK_BX_C1_{OK_TS}.jpg",
            f"{IMAGE_ROOT}/NG/HS/S0001/{CLASS}__0001_NG_BX_C1_{NG_TS}.jpg",
            f"{IMAGE_ROOT}/OK/S0001/{CLASS}_0001_OK_C1_{OK_TS}.bmp",
            f"{CLASS}_0001_OK_C1_{OK_TS}.jpg",
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(pf.MalformedImageName):
                    pf.parse_image_name(case)
        with self.assertRaises(pf.MalformedImageName):
            pf.parse_image_name(
                f"{IMAGE_ROOT}/OK/S0001/bottle_cap_0001_OK_C1_{OK_TS}.jpg",
                expected_class=CLASS)


class RecordValidationTests(unittest.TestCase):
    def test_duplicate_paths_are_rejected(self):
        sample = ok_sample(1, 1)
        with self.assertRaises(pf.DatasetProtocolError):
            pf.records_from_samples([sample, sample], CLASS, "train")

    def test_duplicate_file_names_across_splits_are_rejected(self):
        train = pf.records_from_samples([ok_sample(1, camera)
                                         for camera in range(1, 6)], CLASS, "train")
        duplicated_samples = [SimpleNamespace(path=ok_sample(1, camera).path,
                                              is_good=True)
                              for camera in range(1, 6)]
        duplicated = pf.records_from_samples(duplicated_samples, CLASS, "test")
        with self.assertRaises(pf.DatasetProtocolError):
            pf.validate_class_records(CLASS, train, duplicated)

    def test_shared_serial_alone_is_not_sample_leakage(self):
        """Train holds audiojack_0001_OK, test holds audiojack__0001_NG_BX."""
        train = pf.records_from_samples([ok_sample(1, camera)
                                         for camera in range(1, 6)], CLASS, "train")
        test = pf.records_from_samples([ng_sample(1, camera)
                                        for camera in range(1, 6)], CLASS, "test")
        report = pf.validate_class_records(CLASS, train, test)
        self.assertEqual(
            report["official_sample_key"]["N_samples_in_both_splits"], 0)
        self.assertTrue(report["official_sample_key"]["train_test_disjoint"])
        self.assertEqual(
            report["directory_serial_key"]["N_serials_in_both_splits"], 1)
        self.assertFalse(report["directory_serial_key"]["is_a_sample_identity"])

    def test_actual_duplicate_official_sample_across_splits_is_reported(self):
        train = pf.records_from_samples([ok_sample(1, camera)
                                         for camera in range(1, 6)], CLASS, "train")
        test = pf.records_from_samples(
            [ok_sample(1, camera, ts="20231022101010")
             for camera in range(1, 6)], CLASS, "test")
        report = pf.validate_class_records(CLASS, train, test)
        self.assertEqual(
            report["official_sample_key"]["N_samples_in_both_splits"], 1)
        self.assertFalse(report["official_sample_key"]["train_test_disjoint"])

    def test_duplicate_official_sample_across_partitions_raises(self):
        train, test = cached_records()
        plan = pf.build_partitions(train, test)
        leaked = plan.support_samples_by_seed[0][0]
        tampered = replace(
            plan,
            phase_samples=(tuple(plan.phase_samples[0]) + (leaked,),
                           plan.phase_samples[1], plan.phase_samples[2]))
        with self.assertRaises(pf.DatasetProtocolError) as caught:
            pf.audit_sample_disjointness(tampered)
        message = str(caught.exception)
        self.assertIn("official-sample leakage", message)
        self.assertIn(leaked, message)


class PartitionTests(unittest.TestCase):
    def test_sample_disjoint_partitions(self):
        train, test = cached_records()
        plan = pf.build_partitions(train, test)
        sets = plan.sample_sets()
        names = sorted(sets)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                self.assertEqual(
                    sets[left] & sets[right], set(),
                    f"{left} and {right} share official samples")
        self.assertTrue(
            pf.audit_sample_disjointness(plan)["all_partitions_disjoint"])

    def test_partition_sizes_follow_the_historical_protocol(self):
        train, test = cached_records()
        plan = pf.build_partitions(train, test)
        self.assertEqual(sorted(plan.support_samples_by_seed), [0, 1, 2])
        for seed, samples in plan.support_samples_by_seed.items():
            self.assertEqual(len(samples), 4, f"seed {seed} support size")
        self.assertEqual(len(plan.support), 12)          # 4 shots x 3 seeds
        self.assertEqual(len(plan.history_samples), 64)
        self.assertEqual(len(set(plan.history_samples)), 64)
        self.assertEqual(len(plan.probe_a_samples), 48)
        self.assertEqual(len(plan.probe_b_samples), 48)
        self.assertEqual(len(plan.phases), 3)
        for phase in plan.phase_samples:
            self.assertEqual(len(phase), 175)
        # probe composition: 24 normal + 24 defect samples each
        probe_a_labels = {key: all(r.is_good for r in pf.group_of(plan.probe_a, key))
                          for key in plan.probe_a_samples}
        self.assertEqual(sum(probe_a_labels.values()), 24)
        self.assertEqual(len(probe_a_labels) - sum(probe_a_labels.values()), 24)
        self.assertEqual(sum(len(p) for p in plan.phases), 525 * 5)

    def test_support_and_history_samples_are_removed_from_the_test_pool(self):
        train, test = cached_records()
        plan = pf.build_partitions(train, test)
        reserved = set(plan.reserved_samples)
        for column in (plan.phase_samples, [plan.probe_a_samples],
                       [plan.probe_b_samples]):
            for keys in column:
                self.assertEqual(set(keys) & reserved, set())
        # synthetic train/test do not overlap, so nothing was excluded
        self.assertEqual(plan.excluded_shared_samples, tuple())

    def test_overlapping_train_test_samples_are_reserved(self):
        """A train sample that also appears in test must not enter any test part."""
        train = [ok_sample(obj, camera) for obj in range(1, 121)
                 for camera in range(1, 6)]
        # 101-120 are captured again in the test split (same official keys)
        test = [ok_sample(obj, camera, ts="20231022101010")
                for obj in range(101, 351) for camera in range(1, 6)]
        test += [ng_sample(obj, camera, good=(camera <= 2))
                 for obj in range(351, 801) for camera in range(1, 6)]
        train_records = pf.records_from_samples(train, CLASS, "train")
        test_records = pf.records_from_samples(test, CLASS, "test")
        plan = pf.build_partitions(train_records, test_records)
        reserved = set(plan.reserved_samples)
        self.assertTrue(plan.excluded_shared_samples)
        self.assertEqual(
            {k for keys in plan.phase_samples for k in keys} & reserved, set())
        self.assertTrue(plan.history_samples[0] in reserved)
        self.assertEqual(
            set(plan.excluded_shared_samples) & reserved,
            set(plan.excluded_shared_samples))

    def test_insufficient_samples_fail_explicitly(self):
        train = [ok_sample(obj, camera) for obj in range(1, 21)
                 for camera in range(1, 6)]
        test = [ok_sample(obj, camera) for obj in range(500, 560)
                for camera in range(1, 6)]
        train_records = pf.records_from_samples(train, CLASS, "train")
        test_records = pf.records_from_samples(test, CLASS, "test")
        with self.assertRaises(pf.InsufficientObjects) as caught:
            pf.build_partitions(train_records, test_records)
        self.assertIn("verified-normal train", str(caught.exception))

    def test_stream_requirement_is_verified_not_assumed(self):
        """700 test samples are enough for 621; fewer must fail explicitly."""
        feasible = [ok_sample(obj, camera) for obj in range(500, 750)
                    for camera in range(1, 6)]
        feasible += [ng_sample(obj, camera, good=(camera <= 2))
                     for obj in range(750, 1200) for camera in range(1, 6)]
        records = pf.records_from_samples(feasible, CLASS, "test")
        report = pf.partition_feasibility(
            CLASS, cached_records()[0], records, pf.SplitConfig())
        self.assertTrue(report["test_capacity_ok"])
        self.assertEqual(report["N_test_samples"], 700)
        self.assertEqual(report["unused_test_samples_if_built"], 700 - 621)

        too_small = [ok_sample(obj, camera) for obj in range(500, 750)
                     for camera in range(1, 6)]
        too_small += [ng_sample(obj, camera, good=(camera <= 2))
                      for obj in range(750, 1118) for camera in range(1, 6)]
        self.assertEqual(
            pf.partition_feasibility(CLASS, cached_records()[0],
                                     pf.records_from_samples(
                                         too_small, CLASS, "test"),
                                     pf.SplitConfig())["N_test_samples"], 618)
        with self.assertRaises(pf.InsufficientObjects) as caught:
            pf.build_partitions(
                cached_records()[0],
                pf.records_from_samples(too_small, CLASS, "test"))
        self.assertIn("525", str(caught.exception))

    def test_label_structure_refuses_sample_level_propagation(self):
        train, test = cached_records()
        structure = pf.label_structure(test)
        self.assertEqual(
            structure["N_defect_labelled_images_inside_normal_samples"], 0)
        # 450 defective samples x 2 normal-labelled views
        self.assertEqual(
            structure["N_normal_labelled_images_inside_defect_samples"], 900)
        self.assertEqual(structure["N_samples_all_images_normal"], 250)
        self.assertEqual(
            structure["N_samples_with_at_least_one_defect_image"], 450)


def cached_labels():
    if "labels" not in _CACHE:
        train, test = cached_records()
        _CACHE["labels"] = {**labels_of(train), **labels_of(test)}
    return _CACHE["labels"]


class QueryTapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.stream = cached_plan_and_stream()
        cls.labels = cached_labels()

    def test_query_determinism_under_a_fixed_seed(self):
        first = qp.build_query_tape(self.stream, 32, qp.QUERY_SEED, CLASS)
        second = qp.build_query_tape(self.stream, 32, qp.QUERY_SEED, CLASS)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.events, second.events)
        rebuilt = qp.build_stream(self.plan.phases)
        third = qp.build_query_tape(rebuilt, 32, qp.QUERY_SEED, CLASS)
        self.assertEqual(first.digest, third.digest)

    def test_query_changes_with_a_different_seed(self):
        base = qp.build_query_tape(self.stream, 64, qp.QUERY_SEED, CLASS)
        other = qp.build_query_tape(self.stream, 64, qp.QUERY_SEED + 1, CLASS)
        self.assertNotEqual(base.priority_order, other.priority_order)
        self.assertNotEqual(set(base.sample_keys), set(other.sample_keys))
        self.assertNotEqual(base.digest, other.digest)

    def test_query_selection_is_label_blind(self):
        stream_image_fields = {field.name for field in fields(qp.StreamImage)}
        self.assertEqual(
            stream_image_fields & {"is_good", "label", "anomaly_class",
                                   "defect", "score"}, set())
        baseline = qp.build_query_tape(self.stream, 48, qp.QUERY_SEED, CLASS)
        flipped_phases = tuple(
            tuple(replace(record, is_good=not record.is_good) for record in phase)
            for phase in self.plan.phases)
        flipped_stream = qp.build_stream(flipped_phases)
        self.assertEqual(qp.stream_digest(flipped_stream),
                         qp.stream_digest(self.stream))
        flipped_tape = qp.build_query_tape(flipped_stream, 48, qp.QUERY_SEED, CLASS)
        self.assertEqual(baseline.digest, flipped_tape.digest)
        self.assertEqual(baseline.sample_keys, flipped_tape.sample_keys)
        self.assertEqual(baseline.positions, flipped_tape.positions)
        flipped_labels = {path: not good for path, good in self.labels.items()}
        self.assertEqual(
            qp.label_diagnostics(self.stream, baseline, self.labels)["N_defect_labels"],
            qp.label_diagnostics(self.stream, flipped_tape,
                                 flipped_labels)["N_normal_labels"])

    def test_nested_budgets_are_prefixes(self):
        tapes = qp.build_nested_tapes(self.stream, qp.BUDGET_CANDIDATES,
                                      qp.QUERY_SEED, CLASS)
        self.assertEqual(sorted(tapes), list(qp.BUDGET_CANDIDATES))
        for small, large in zip(qp.BUDGET_CANDIDATES, qp.BUDGET_CANDIDATES[1:]):
            qp.assert_nested(tapes[small], tapes[large])
            large_events = {event.sample_key: event for event in tapes[large].events}
            for event in tapes[small].events:
                twin = large_events[event.sample_key]
                self.assertEqual(
                    (twin.position, twin.rel_path, twin.camera_id),
                    (event.position, event.rel_path, event.camera_id))
        self.assertTrue(qp.tape_nesting_report(tapes)["all_nested"])

    def test_nesting_check_rejects_a_redrawn_image(self):
        tapes = qp.build_nested_tapes(self.stream, (8, 16), qp.QUERY_SEED, CLASS)
        reference = qp.build_query_tape(self.stream, 8, qp.QUERY_SEED + 1, CLASS)
        with self.assertRaises(qp.QueryProtocolError):
            qp.assert_nested(reference, tapes[16])
        tampered_events = tuple(replace(event, camera_id=(event.camera_id % 5) + 1)
                                for event in tapes[8].events)
        with self.assertRaises(qp.QueryProtocolError):
            qp.assert_nested(replace(tapes[8], events=tampered_events), tapes[16])

    def test_exactly_one_queried_image_per_sample(self):
        tape = qp.build_query_tape(self.stream, 96, qp.QUERY_SEED, CLASS)
        self.assertEqual(len(tape.events), 96)
        self.assertEqual(len(set(tape.sample_keys)), 96)
        counts = {}
        for event in tape.events:
            counts[event.sample_key] = counts.get(event.sample_key, 0) + 1
        self.assertEqual(set(counts.values()), {1})

    def test_query_positions_are_chronological(self):
        tape = qp.build_query_tape(self.stream, 64, qp.QUERY_SEED, CLASS)
        positions = list(tape.positions)
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(len(set(positions)), len(positions))
        phases = [event.phase for event in tape.events]
        self.assertEqual(phases, sorted(phases))
        by_position = {image.position: image for image in self.stream}
        for event in tape.events:
            self.assertEqual(by_position[event.position].rel_path, event.rel_path)
            self.assertEqual(by_position[event.position].sample_key, event.sample_key)

    def test_budget_accounting_fields(self):
        tape = qp.build_query_tape(self.stream, 79, qp.QUERY_SEED, CLASS)
        manifest = tape.manifest()
        for key in ("N_images", "N_samples", "budget_requested",
                    "N_queries_selected", "N_unique_queried_samples",
                    "image_query_fraction", "unique_sample_audit_fraction",
                    "query_seed", "query_positions", "selected_image_paths",
                    "selected_sample_keys", "selected_camera_ids",
                    "query_manifest_sha256"):
            self.assertIn(key, manifest)
        self.assertEqual(manifest["budget_requested"], 79)
        self.assertEqual(manifest["N_queries_selected"], 79)
        self.assertEqual(manifest["N_unique_queried_samples"], 79)
        self.assertEqual(manifest["N_images"], len(self.stream))
        self.assertEqual(manifest["N_samples"],
                         len({image.sample_key for image in self.stream}))
        self.assertAlmostEqual(manifest["image_query_fraction"],
                               79 / len(self.stream), places=5)
        self.assertEqual(manifest["query_manifest_sha256"], tape.digest)

    def test_impossible_budget_fails_explicitly(self):
        n_samples = len({image.sample_key for image in self.stream})
        with self.assertRaises(qp.QueryBudgetInfeasible) as caught:
            qp.build_query_tape(self.stream, n_samples + 1, qp.QUERY_SEED, CLASS)
        self.assertIn("refuse to reduce", str(caught.exception))
        with self.assertRaises(qp.QueryBudgetInfeasible):
            qp.build_query_tape(self.stream, 0, qp.QUERY_SEED, CLASS)
        with self.assertRaises(qp.QueryBudgetInfeasible):
            qp.build_nested_tapes(self.stream, (8, n_samples + 1),
                                  qp.QUERY_SEED, CLASS)


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan, cls.stream = cached_plan_and_stream()
        cls.labels = cached_labels()
        cls.tape = qp.build_query_tape(cls.stream, 32, qp.QUERY_SEED, CLASS)

    def test_shared_tape_replay_parity(self):
        methods = qp.default_simulated_methods()
        result = qp.replay_shared_tape(self.stream, self.tape, self.labels, methods)
        report = result.parity_report()
        self.assertTrue(report["identical_feedback_events"])
        self.assertTrue(report["identical_query_positions"])
        self.assertTrue(report["identical_feedback_order"])
        self.assertEqual(report["N_disclosures"], 32)
        first, second = sorted(methods)
        self.assertEqual(result.deliveries[first], result.deliveries[second])
        self.assertEqual([row[0] for row in result.deliveries[first]],
                         list(self.tape.positions))
        for name in methods:
            self.assertEqual(len(result.predictions[name]), len(self.stream))

    def test_labels_are_disclosed_only_after_the_query_decision(self):
        methods = qp.default_simulated_methods()
        result = qp.replay_shared_tape(self.stream, self.tape, self.labels, methods)
        audit = pf.causal_visibility_audit(result, methods)
        self.assertTrue(audit["passed"], audit["examples"])
        for name, method in methods.items():
            delivered = [row[0] for row in result.deliveries[name]]
            for step, seen in method.labels_visible_at_prediction:
                self.assertEqual(seen, sum(1 for p in delivered if p < step),
                                 f"{name} saw a label too early at {step}")

    def test_current_label_cannot_influence_the_current_prediction(self):
        methods = qp.default_simulated_methods()
        baseline = qp.replay_shared_tape(self.stream, self.tape, self.labels,
                                         methods)
        first_query = self.tape.positions[0]
        flipped = dict(self.labels)
        flipped_path = self.tape.events[0].path
        flipped[flipped_path] = not flipped[flipped_path]
        other_methods = qp.default_simulated_methods()
        changed = qp.replay_shared_tape(self.stream, self.tape, flipped,
                                        other_methods)
        for name in methods:
            before = [row for row in baseline.predictions[name]
                      if row[0] <= first_query]
            after = [row for row in changed.predictions[name]
                     if row[0] <= first_query]
            self.assertEqual(before, after,
                             f"{name} prediction changed before the label was due")
        self.assertEqual(baseline.disclosures[0].position, first_query)
        self.assertNotEqual(baseline.disclosures[0].label,
                            changed.disclosures[0].label)

    def test_prediction_interface_hides_label_encoded_file_names(self):
        """A controller cannot infer OK/NG from the dataset path pre-query."""
        class InspectingAdmission(qp.PassiveAdmission):
            def predict(self, step, image):
                self_seen = tuple(f.name for f in fields(image))
                self.assert_no_label_tokens = self_seen
                forbidden = ("path", "rel_path", "sample_key", "label",
                             "is_good", "anomaly_class", "timestamp")
                if any(hasattr(image, name) for name in forbidden):
                    raise AssertionError("Pre-query observation exposed identity/label")
                if "OK" in image.observation_handle or "NG" in image.observation_handle:
                    raise AssertionError("Opaque handle encodes label")
                return super().predict(step, image)

        method = InspectingAdmission()
        qp.replay_shared_tape(self.stream, self.tape, self.labels,
                              {"inspector": method})
        self.assertEqual(len(method.predictions), len(self.stream))
        self.assertEqual(set(method.assert_no_label_tokens),
                         {"position", "observation_handle", "camera_id", "phase"})

    def test_public_observation_is_opaque_for_unqueried_images(self):
        for image in self.stream[:10]:
            public = qp.PublicObservation.from_stream(image)
            self.assertFalse(hasattr(public, "path"))
            self.assertFalse(hasattr(public, "sample_key"))
            self.assertFalse(hasattr(public, "rel_path"))
            self.assertEqual(public.observation_handle,
                             f"observation:{image.position}")

    def test_replay_fails_without_a_label(self):
        labels = dict(self.labels)
        labels.pop(self.tape.events[0].path)
        with self.assertRaises(qp.QueryProtocolError):
            qp.replay_shared_tape(self.stream, self.tape, labels,
                                  qp.default_simulated_methods())

    def test_replay_rejects_a_foreign_tape(self):
        other = qp.build_query_tape(self.stream, 8, qp.QUERY_SEED + 5, CLASS)
        with self.assertRaises(qp.QueryProtocolError):
            qp.replay_shared_tape(self.stream[:-1], other, self.labels,
                                  qp.default_simulated_methods())


class SidecarFeasibilityTests(unittest.TestCase):
    def test_documented_thresholds_are_read_from_the_real_source(self):
        config = pf.read_dataclass_defaults(pf.ROOT / pf.SIDECAR_SOURCE,
                                           pf.SIDECAR_CONFIG_CLASS)
        for key, expected in pf.DOCUMENTED_SIDECAR_VALUES.items():
            self.assertEqual(int(config[key]), expected, key)
        self.assertFalse(bool(config["enable_fast_upward_rescue"]))
        self.assertFalse(bool(config["enable_balanced_rescue"]))

    def test_budget_feasibility_classification(self):
        report = pf.sidecar_feasibility(pf.ROOT / pf.SIDECAR_SOURCE,
                                       (8, 16, 24, 32, 79))
        self.assertFalse(report["thresholds_modified"])
        self.assertTrue(report["documented_values_match_source"])
        verdict = {budget: {row["mechanism"]: row["verdict"]
                            for row in values["mechanisms"]}
                   for budget, values in report["per_budget"].items()}
        for impossible in ("8", "16"):
            self.assertTrue(verdict[impossible]["dual_anchor_full"].startswith(
                "mathematically_impossible"))
            self.assertTrue(verdict[impossible]["normal_dominant_fallback"]
                            .startswith("mathematically_impossible"))
        for feasible in ("24", "32", "79"):
            self.assertEqual(verdict[feasible]["dual_anchor_full"],
                             "potentially_feasible_depends_on_label_composition")
        for budget in verdict:
            self.assertEqual(verdict[budget]["fast_upward_rescue"],
                             "disabled_by_default_no_activation_expected")
            self.assertEqual(report["per_budget"][budget]["actual_activation"],
                             "not_established_without_online_controller_execution")
        self.assertFalse(report["per_budget"]["16"]
                         ["can_ever_reach_min_normal_anchor"])
        self.assertTrue(report["per_budget"]["32"]
                        ["can_ever_reach_min_normal_anchor"])
        self.assertFalse(report["per_budget"]["32"]
                         ["can_ever_satisfy_both_anchor_counts_simultaneously"])
        self.assertTrue(report["per_budget"]["79"]
                        ["can_ever_satisfy_both_anchor_counts_simultaneously"])
        self.assertEqual(report["per_budget"]["79"]
                         ["update_opportunities_upper_bound"], 79 // 8)

    def test_a_relaxed_threshold_breaks_the_preflight(self):
        config = dict(pf.read_dataclass_defaults(pf.ROOT / pf.SIDECAR_SOURCE,
                                                pf.SIDECAR_CONFIG_CLASS))
        config["min_normal_anchor"] = 16
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "fake_sidecar.py"
            body = [f"class {pf.SIDECAR_CONFIG_CLASS}:"]
            body += [f"    {key}: object = {value!r}"
                     for key, value in config.items()]
            fake.write_text("\n".join(body) + "\n", encoding="utf-8")
            with self.assertRaises(pf.DatasetProtocolError):
                pf.sidecar_feasibility(fake, (8, 16, 32))


class HistoricalReferenceTests(unittest.TestCase):
    def test_historical_525_samples_2625_images_79_queries(self):
        train, test = build_samples(train_objects=120, normal_test=250,
                                    defect_test=450, test_base=2000)
        train_records = pf.records_from_samples(train, CLASS, "train")
        test_records = pf.records_from_samples(test, CLASS, "test")
        audit = pf.historical_reference_audit(train_records, test_records, (0,))
        check = audit["arithmetic_check"]
        self.assertEqual(check["N_stream_groups"], [525])
        self.assertEqual(check["N_stream_images"], [2625])
        self.assertEqual(check["N_queries"], [79])
        self.assertTrue(check["queries_equal_round_3pct_of_images"])
        self.assertTrue(check["historical_3pct_means_79_queries_not_16"])
        self.assertAlmostEqual(check["image_query_fraction"], 0.030095, places=6)
        self.assertAlmostEqual(
            check["unique_sample_audit_fraction_vs_stream_groups"], 0.150476,
            places=6)
        self.assertIn("identity_sensitivity_audit", audit)
        self.assertEqual(audit["per_support_seed"][0]["N_probe_samples_A"], 48)
        self.assertEqual(audit["per_support_seed"][0]["N_probe_samples_B"], 48)

    def test_historical_reference_fails_loudly_when_it_cannot_be_reproduced(self):
        train, test = build_samples(train_objects=120, normal_test=60,
                                    defect_test=60, test_base=3000)
        train_records = pf.records_from_samples(train, CLASS, "train")
        test_records = pf.records_from_samples(test, CLASS, "test")
        with self.assertRaises(pf.DatasetProtocolError) as caught:
            pf.historical_reference_audit(train_records, test_records, (0,))
        self.assertIn("legacy", str(caught.exception).lower())


class OutputValidationTests(unittest.TestCase):
    def test_idempotent_write_validate_and_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preflight.json"
            payload = {"schema": "test", "values": [1, 2.5, {"a": "b"}]}
            self.assertEqual(pf.write_json_idempotent(path, payload), "created")
            self.assertEqual(pf.write_json_idempotent(path, payload), "reused")
            self.assertEqual(pf.verify_json_output(path, payload), "verified")
            with self.assertRaises(pf.OutputMismatch):
                pf.write_json_idempotent(path, {"schema": "different"})
            path.write_text("{not valid json", encoding="utf-8")
            with self.assertRaises(pf.OutputMismatch):
                pf.write_json_idempotent(path, payload)
            with self.assertRaises(pf.OutputMismatch):
                pf.verify_json_output(Path(tmp) / "absent.json", payload)

    def test_semantically_equal_but_byte_different_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            payload = {"a": 1}
            pf.write_json_idempotent(path, payload)
            path.write_text('{"a": 1}\n', encoding="utf-8")
            with self.assertRaises(pf.OutputMismatch):
                pf.write_json_idempotent(path, payload)

    def test_file_digest_equals_manifest_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {"text": "ünïcode", "numbers": [1, 2, 3]}
            path = Path(tmp) / "digest.json"
            pf.write_json_idempotent(path, payload)
            self.assertEqual(pf.sha256_text(pf.json_text(payload)),
                             pf.sha256_file(path))


class HistoricalDenominatorParityTests(unittest.TestCase):
    """The corrected grouping must restore the historical denominators."""

    def test_budget_79_on_525_samples_matches_history(self):
        plan, stream = cached_plan_and_stream()
        self.assertEqual(len(stream), 2625)
        self.assertEqual(len({image.sample_key for image in stream}), 525)
        tape = qp.build_query_tape(stream, 79, qp.QUERY_SEED, CLASS)
        manifest = tape.manifest()
        self.assertEqual(manifest["N_queries_selected"], 79)
        self.assertEqual(manifest["N_unique_queried_samples"], 79)
        self.assertAlmostEqual(manifest["image_query_fraction"], 79 / 2625,
                               places=6)
        self.assertAlmostEqual(manifest["unique_sample_audit_fraction"],
                               79 / 525, places=6)
        self.assertEqual(manifest["identity_version"], pf.IDENTITY_VERSION)
        self.assertEqual(manifest["grouping_unit"], pf.PRIMARY_GROUPING_UNIT)
        self.assertEqual(manifest["selected_sample_keys"],
                         list(tape.sample_keys))


class IdentityVersionSupersessionTests(unittest.TestCase):
    """A different identity version must never reuse prior evidence silently."""

    def test_tape_digest_binds_the_identity_version(self):
        _, stream = cached_plan_and_stream()
        official = qp.build_query_tape(stream, 16, qp.QUERY_SEED, CLASS)
        original = qp.IDENTITY_VERSION
        try:
            qp.IDENTITY_VERSION = "serial-directory-v1"
            serial = qp.build_query_tape(stream, 16, qp.QUERY_SEED, CLASS)
        finally:
            qp.IDENTITY_VERSION = original
        self.assertNotEqual(official.digest, serial.digest)
        self.assertEqual(official.events, serial.events)  # same RNG, new contract

    def test_preserved_manifest_with_another_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset_preflight.json"
            previous = {"schema": "risk_safe.d1a.dataset_preflight.v1",
                        "identity_version": "serial-directory-v1",
                        "stage": "D1-A"}
            self.assertEqual(pf.write_json_idempotent(path, previous), "created")
            current = {"schema": "risk_safe.d1a.dataset_preflight.v1",
                       "identity_version": pf.IDENTITY_VERSION,
                       "stage": "D1-A"}
            with self.assertRaises(pf.OutputMismatch):
                pf.write_json_idempotent(path, current)
            # the superseded evidence is preserved byte for byte
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             previous)

    def test_identity_rule_marks_the_serial_as_diagnostic_only(self):
        rule = pf.identity_rule_block()
        self.assertTrue(rule["directory_serial_key"]["not_a_sample_identity"])
        self.assertIn("Tencent", rule["official_sample_key"]["authority"])
        self.assertEqual(rule["physical_specimen_identity"]["status"],
                         "undocumented")


class SourceManifestTests(unittest.TestCase):
    def test_source_hashes_match_the_working_tree(self):
        hashes = pf.source_hashes()
        self.assertEqual(set(hashes), set(pf.SOURCE_FILES))
        for relative, digest in hashes.items():
            self.assertEqual(digest, pf.sha256_file(pf.ROOT / relative), relative)

    def test_missing_source_fails_closed(self):
        with self.assertRaises(pf.DatasetProtocolError):
            pf.source_hashes(("experiments/risk_safe/does_not_exist.py",))

    def test_manifest_records_git_environment_and_output_digests(self):
        payloads = {name: {"schema": name}
                    for name in (pf.OUTPUT_DATASET, pf.OUTPUT_BUDGET,
                                 pf.OUTPUT_TAPE)}
        configuration = pf.configuration_block(
            pf.SplitConfig(), qp.BUDGET_CANDIDATES, qp.QUERY_SEED)
        fingerprint = {"json_dir": "synthetic",
                       "json_files_sha256": {"audiojack.json": "0" * 64},
                       "semantic_index_sha256": "0" * 64,
                       "N_index_records": 0}
        manifest = pf.build_source_manifest(configuration, fingerprint, payloads)
        self.assertEqual(manifest["schema"], pf.SCHEMA_SOURCE)
        self.assertEqual(manifest["configuration_sha256"],
                         qp.sha256_of(configuration))
        self.assertEqual(manifest["git"]["head"],
                         pf._run_git("rev-parse", "HEAD"))
        self.assertEqual(manifest["environment"]["cuda_initialised"], False)
        self.assertEqual(manifest["environment"]["features_extracted"], False)
        for name, digest in manifest["outputs_sha256"].items():
            self.assertEqual(digest, pf.sha256_text(pf.json_text(payloads[name])))
        self.assertEqual(manifest["registered_defaults"]["query_seed"],
                         qp.QUERY_SEED)
        self.assertEqual(manifest["registered_defaults"]["budget_candidates"],
                         list(qp.BUDGET_CANDIDATES))


def _json_record(class_name: str, obj: int, camera: int, state: str,
                 defect: str = "BX", anomaly=None) -> dict:
    if state == "OK":
        image = f"OK/S{obj:04d}/{class_name}_{obj:04d}_OK_C{camera}_{OK_TS}.jpg"
        anomaly = anomaly or "OK"
    else:
        image = (f"NG/{defect}/S{obj:04d}/"
                 f"{class_name}__{obj:04d}_NG_{defect}_C{camera}_{NG_TS}.jpg")
        anomaly = anomaly or defect
    return {"category": class_name, "anomaly_class": anomaly,
            "image_path": image, "mask_path": None}


def write_synthetic_dataset(root: Path, train_objects: int = 70,
                            normal_test: int = 250, defect_test: int = 450,
                            test_base: int = 1000) -> Path:
    """Six-class synthetic Real-IAD JSON dataset using Real-IAD naming."""
    json_dir = root / "jsons" / "realiad_jsons"
    json_dir.mkdir(parents=True, exist_ok=True)
    for class_name in pf.CLASSES:
        payload = {
            "meta": {"prefix": f"{class_name}/", "normal_class": "OK",
                     "pre_transform": False},
            "train": [_json_record(class_name, obj, camera, "OK")
                      for obj in range(1, train_objects + 1)
                      for camera in range(1, 6)],
            "test": [],
        }
        payload["test"] += [_json_record(class_name, test_base + i, camera, "OK")
                            for i in range(normal_test)
                            for camera in range(1, 6)]
        payload["test"] += [
            _json_record(class_name, test_base + normal_test + i, camera, "NG",
                         anomaly=("OK" if camera <= 2 else "BX"))
            for i in range(defect_test)
            for camera in range(1, 6)]
        (json_dir / f"{class_name}.json").write_text(
            json.dumps(payload), encoding="utf-8")
    return json_dir


class EndToEndPreflightTests(unittest.TestCase):
    """Full metadata preflight on a synthetic six-class dataset directory."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "realiad"
        cls.json_dir = write_synthetic_dataset(cls.root)
        cls.out_dir = Path(cls._tmp.name) / "d1a_preflight"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _run(self, *extra):
        import contextlib
        import io
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = pf.main(["--dataset-root", str(self.root),
                            "--json-dir", str(self.json_dir),
                            "--out-dir", str(self.out_dir), *extra])
        return code, json.loads(stream.getvalue())

    def test_preflight_writes_and_reuses_all_four_manifests(self):
        code, first = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(set(first["output_status"]), set(pf.OUTPUT_FILES))
        self.assertEqual(set(first["output_status"].values()), {"created"})
        self.assertTrue(first["sample_disjoint"])
        self.assertTrue(first["nested_budgets"])
        self.assertTrue(first["query_parity"])
        self.assertTrue(first["causal_visibility"])
        self.assertTrue(first["historical_all_classes_525_2625_79"])

        code, second = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(set(second["output_status"].values()), {"reused"})

        code, verified = self._run("--verify-only")
        self.assertEqual(code, 0)
        self.assertEqual(set(verified["output_status"].values()), {"verified"})

        for name in pf.OUTPUT_FILES:
            self.assertTrue((self.out_dir / name).is_file())

        source = json.loads((self.out_dir / pf.OUTPUT_SOURCE).read_text("utf-8"))
        for name, digest in source["outputs_sha256"].items():
            self.assertEqual(digest, pf.sha256_file(self.out_dir / name), name)
        self.assertEqual(source["dataset"]["N_index_records"],
                         len(pf.CLASSES) * (70 + 250 + 450) * 5)
        for class_name in pf.CLASSES:
            self.assertIn(f"{class_name}.json",
                          source["dataset"]["json_files_sha256"])

        dataset = json.loads((self.out_dir / pf.OUTPUT_DATASET).read_text("utf-8"))
        budget = json.loads((self.out_dir / pf.OUTPUT_BUDGET).read_text("utf-8"))
        tape = json.loads((self.out_dir / pf.OUTPUT_TAPE).read_text("utf-8"))
        self.assertEqual(dataset["schema"], pf.SCHEMA_DATASET)
        self.assertEqual(budget["schema"], pf.SCHEMA_BUDGET)
        self.assertEqual(tape["schema"], pf.SCHEMA_TAPE)
        for class_name in pf.CLASSES:
            row = dataset["classes"][class_name]
            self.assertEqual(row["partitions"]["support"]["N_images"], 12)
            self.assertLessEqual(row["partitions"]["support"]["N_samples"], 12)
            for seed_samples in row["partitions"]["support"][
                    "samples_by_support_seed"].values():
                self.assertEqual(len(seed_samples), 4)
                self.assertEqual(len(set(seed_samples)), 4)
            self.assertEqual(row["partitions"]["mature_history"]["N_samples"], 64)
            self.assertEqual(row["partitions"]["probe_A"]["N_samples"], 48)
            self.assertEqual(row["partitions"]["probe_B"]["N_samples"], 48)
            tapes = tape["classes"][class_name]["tapes"]
            self.assertEqual(sorted(int(b) for b in tapes),
                             list(qp.BUDGET_CANDIDATES))
            for value in tapes.values():
                self.assertEqual(value["N_queries_selected"],
                                 value["budget_requested"])
                self.assertEqual(value["N_queries_selected"],
                                 value["N_unique_queried_samples"])

    def test_tampered_output_is_rejected_by_rerun(self):
        target = self.out_dir / pf.OUTPUT_DATASET
        if not target.exists():
            self._run()
        original = target.read_text(encoding="utf-8")
        self.assertIn('"D1-A"', original)
        target.write_text(original.replace('"D1-A"', '"D1-X"', 1),
                          encoding="utf-8")
        try:
            with self.assertRaises(pf.OutputMismatch):
                self._run()
        finally:
            target.write_text(original, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
