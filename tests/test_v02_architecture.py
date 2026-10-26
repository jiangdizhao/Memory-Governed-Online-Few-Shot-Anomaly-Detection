"""CPU-only V0.2 protocol invariants; no Real-IAD or GPU required."""
import sys
import tempfile
import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import v02_architecture_core as core
import v02_shift_guard as shift_guard
import run_v02_architecture as runner


def sample(label,idx,good=True):
    return SimpleNamespace(
        path=f"G:/dataset/realiad/{label}_{idx:05d}_C1_20261026000000.jpg",
        is_good=good,
    )


class V02ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.train=[sample("OK",i) for i in range(150)]
        self.test=[sample("OK",i+1000) for i in range(350)]+[
            sample("NG",i+2000,False) for i in range(350)]

    def test_case_matrix(self):
        self.assertEqual(len(core.CLASSES),6)
        self.assertEqual(len(core.SEEDS),3)
        self.assertEqual(len(core.CLASSES)*len(core.SEEDS)*len(core.ORDERS),36)
        self.assertEqual(core.ARMS,("greedy_cold","hierarchy_cold",
                                   "greedy_mature","hierarchy_mature"))

    def test_same_history_and_distinct_support(self):
        samples=[core.select_history_and_support(self.train,s) for s in core.SEEDS]
        self.assertEqual(len({core.digest([h.path for h in p[1]]) for p in samples}),1)
        for support,hist in samples:
            self.assertEqual(len(support),4)
            self.assertEqual(len(hist),64)
            self.assertEqual(len({core.object_id(x.path) for x in hist}),64)
            self.assertFalse(set(map(core.object_id,support)) &
                             {core.object_id(x.path) for x in hist})
        self.assertGreater(len({core.digest(p[0]) for p in samples}),1)

    def test_probe_grouping_and_tape(self):
        p,phases,groups=core.split_test(self.test)
        self.assertEqual(len(groups["A"]),48)
        self.assertEqual(len(groups["B"]),48)
        self.assertTrue(all(len(group)==175 for group in groups["phases"]))
        selected=groups["A"]+groups["B"]+sum(groups["phases"],[])
        self.assertEqual(len(selected),len(set(selected)))
        self.assertEqual(sum(not x.is_good for x in p["A"]),24)
        self.assertEqual(sum(not x.is_good for x in p["B"]),24)
        t=core.query_tape(phases)
        flat=sum(phases,[])
        self.assertLess(abs(len(t)/len(flat)-.03),.001)
        self.assertEqual(len(t),len({core.object_id(flat[i-1].path) for i in t}))

    def test_resource_ratio_by_real_width(self):
        mock=SimpleNamespace(normal_dyn={"p0":{0:{"id":[
            SimpleNamespace(feat_dim=4096)]}}})
        cfg=core.make_memory_settings(mock)
        self.assertEqual((cfg.stm,cfg.ltm,cfg.buffer),(1024,512,4096))
        self.assertGreater(cfg.flat,1024)
        self.assertGreater(cfg.cap_bytes,0)

    def test_small_real_shift_is_valid_not_excluded(self):
        # Photometric pixel intervention is real even if DINO is almost
        # invariant (the previous unregistered 0.01 rejection was incorrect).
        checked=shift_guard.validate_shift_observations(
            [0.00003,0.00012,0.003], [0.22,0.24,0.21])
        self.assertTrue(checked["passed_pixel_intervention_check"])
        self.assertTrue(checked["feature_shift_below_legacy_0p01_diagnostic_only"])
        with self.assertRaises(AssertionError):
            shift_guard.validate_shift_observations(
                [0.0,0.0],[0.0,0.0])
        with self.assertRaises(AssertionError):
            shift_guard.validate_shift_observations(
                [float("nan")],[0.1])

    def test_archive_only_known_failed_incomplete_case(self):
        from types import SimpleNamespace
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(out_root=tmp)
            failed=Path(tmp)/"eraser_s0_ABA"
            failed.mkdir()
            (failed/"incomplete.txt").write_text("preserve me")
            log=Path(tmp)/"eraser_s0_ABA.run.log"
            log.write_text(
                "AssertionError: Frozen B shift failed feature-space verification\\n"
                "Exit code: 1\\n",encoding="utf-8")
            self.assertTrue(runner.archive_known_failed_shift_gate(
                args,("eraser",0,"ABA")))
            self.assertFalse(failed.exists())
            archive=Path(tmp)/"_failed_attempts"/"eraser_s0_ABA_obsolete_shift_gate"
            self.assertEqual((archive/"partial_run"/"incomplete.txt").read_text(),
                             "preserve me")
            self.assertIn("Exit code: 1",
                          (archive/"original_runner_log.txt").read_text())
            self.assertFalse(runner.archive_known_failed_shift_gate(
                args,("eraser",0,"ABA")))

    def test_historical_source_lineage_allowlist(self):
        # This runs against the exact Git commit whose twelve complete
        # outputs were produced, never by trusting claimed SHA metadata.
        for path in sorted(runner.LEGACY_CODE_ONLY_ALLOWLIST):
            actual=hashlib.sha256(runner._read_pinned_historical_source(path)).hexdigest()
            checked=runner._verify_source_manifest({path:actual})
            self.assertEqual(checked,[path])
        original=hashlib.sha256(
            runner._read_pinned_historical_source(
                "experiments/v02_architecture_core.py")).hexdigest()
        self.assertEqual(
            runner._verify_source_manifest({
                "experiments/v02_architecture_core.py":original}),[])
        with self.assertRaises(RuntimeError):
            runner._verify_source_manifest({
                "experiments/v02_architecture_core.py":"deadbeef"})

    def test_decision_gate(self):
        def rows(ret,quality=0,source=.006,ltm=1):
            return [dict(class_name=cl,seed=s,
                retention_advantage_mature=ret,
                A_quality_mature=quality,B_quality_mature=quality,
                A_quality_cold=quality,source_effect_mature=source,
                ltm_mature=ltm)
                for cl in core.CLASSES for s in core.SEEDS]
        self.assertEqual(core.decision(rows(0))["decision"],
                         "USE_GREEDY_WITH_RISK_SAFE_ADMISSION")
        self.assertEqual(core.decision(rows(.02))["decision"],"RETAIN_HIERARCHY")
        self.assertEqual(core.decision(rows(.02,-.03))["decision"],
                         "USE_GREEDY_WITH_RISK_SAFE_ADMISSION")
        with self.assertRaises(ValueError):
            core.decision(rows(.02)[:-1])


if __name__=="__main__":
    unittest.main()
