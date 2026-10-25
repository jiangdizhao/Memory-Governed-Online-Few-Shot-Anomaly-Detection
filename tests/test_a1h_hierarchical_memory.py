"""A1-H no-dataset contract regression tests (GitHub Actions-compatible)."""
from __future__ import annotations
from pathlib import Path
import sys
import tempfile
import unittest
import json
import csv

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
import a1h_memory as mem
import a1h_hierarchical_matched as trial
import run_a1h_hierarchical_memory as runner


class ContractTests(unittest.TestCase):
    def test_pre_registered_storage_split(self):
        self.assertEqual((mem.Budget(128).stm,mem.Budget(128).ltm,
                          mem.Budget(128).buffer),(64,32,32))
        self.assertEqual((mem.Budget(256).stm,mem.Budget(256).ltm,
                          mem.Budget(256).buffer),(128,64,64))
        with self.assertRaises(ValueError):
            mem.Budget(129)

    def test_all_eight_arms_and_matrix(self):
        self.assertEqual(mem.ARMS,runner.ARMS)
        self.assertEqual(len(mem.ARMS),8)
        self.assertEqual(len(runner.conditions(
            runner.parser().parse_args(["--all"]))),12)
        self.assertEqual(len(runner.conditions(runner.parser().parse_args([
            "--all","--orders","native,normal_first,defect_first"]))),36)
        with self.assertRaises(ValueError):
            runner.conditions(runner.parser().parse_args([
                "--all","--budgets","128,128"]))
        with self.assertRaises(ValueError):
            runner.conditions(runner.parser().parse_args([
                "--all","--orders","native,oracle"]))

    def test_temporal_auc(self):
        pts=[dict(step=0,auroc=.5),dict(step=100,auroc=.7),
             dict(step=300,auroc=.8)]
        self.assertAlmostEqual(trial.aulc(pts),(.6*100+.75*200)/300)
        self.assertIsNone(trial.aulc([dict(step=0,auroc=.5)]))
        self.assertIsNone(trial.aulc([dict(step=0,auroc=.5),
                                      dict(step=1,auroc=None)]))

    def test_gpu_smoke_command_is_single_trial(self):
        a=runner.parser().parse_args(["--all","--math-sdpa"])
        c=runner.command(a,"audiojack",0,128,"native",smoke=True)
        self.assertIn("--math-sdpa",c)
        self.assertEqual(c[c.index("--max-samples")+1],"64")
        self.assertEqual(c[c.index("--budget")+1],"128")
        self.assertTrue(runner.output_dir(a,"switch",1,256,"native").name.endswith(
            "K256_native_full"))

    def test_known_failed_mechanism_log_is_archived_for_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "mechanism_test.run.log"
            log.write_text(
                "UnicodeEncodeError: cp1252 cannot encode arrow\n"
                "Exit code: 1\n", encoding="utf-8")
            archive = runner.archive_failed_mechanism_log(log)
            self.assertFalse(log.exists())
            self.assertTrue(archive.exists())
            self.assertEqual(
                archive.name, "mechanism_test.run.failed-001.log")
            self.assertIn("UnicodeEncodeError", archive.read_text(encoding="utf-8"))
            log.write_text("Other exception\nExit code: 2\n", encoding="utf-8")
            archive2 = runner.archive_failed_mechanism_log(log)
            self.assertEqual(archive2.name, "mechanism_test.run.failed-002.log")
            self.assertTrue(archive.exists())

    def test_incomplete_or_success_preflight_log_not_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "mechanism_test.run.log"
            for body in ("in progress\n", "Exit code: 0\n", "Exit code: ????\n"):
                log.write_text(body, encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    runner.archive_failed_mechanism_log(log)
                self.assertEqual(log.read_text(encoding="utf-8"), body)

    def test_a1h_validator_fails_closed_and_accepts_consistent_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=runner.parser().parse_args([
                "--validate","--classes","fake","--seeds","0",
                "--budgets","128","--probe-count","4","--out-root",tmp])
            path=runner.output_dir(a,"fake",0,128,"native")
            path.mkdir()
            supports=["N1","N2","N3","N4"]
            probe=["P1","P2","P3","P4"]
            all_names=[f"T{i}" for i in range(100)]
            qpos=[1,10,20]
            p=dict(
                strict_four_shot=True,support_access=4,
                support_paths=supports,support_hash=runner.sha(supports),
                loso_folds=[dict(heldout=x,training=[y for y in supports if y!=x])
                            for x in supports],
                loso_scores=[.1,.2,.3,.4],
                probes_never_used_for_training=True,
                labels_after_predictions=True,
                candidate_from_shared_static=True,
                uniform_retrieval="global_top5_fixed_plus_stm_plus_optional_ltm",
                evidence_mode="neutral",
                hierarchical_allocation=dict(stm=64,ltm=32,buffer=32),
                query_positions=qpos,query_hash=runner.sha(qpos),
                probe_paths=probe,probe_hash=runner.sha(probe),
                stream_hash=runner.sha(all_names))
            arms={}
            for name in mem.ARMS:
                arms[name]=dict(auroc=.7,balanced_accuracy=.5,
                    probe_aulc=.6,probe_final_auc=.7,probe_worst_auc=.5,
                    memory=dict(capacity_total=128,
                                occupied_total=0 if name=="static" else 80,
                                candidate_events=0 if name=="static" else 1,
                                candidate_patches=0 if name=="static" else 8))
            j=dict(schema=runner.SCHEMA,complete=True,
                   source_sha256={
                       "experiments/a1h_memory.py":
                       __import__("hashlib").sha256(
                           (ROOT/"experiments/a1h_memory.py").read_bytes()).hexdigest()},
                   ltm_activation={"H3_ltm_retrieval":{"final_ltm":0}},
                   class_name="fake",support_seed=0,budget=128,order="native",
                   n_queries=3,queried_normal=1,queried_defect=2,
                   n_stream=100,n_probe=4,protocol=p,arms=arms)
            (path/"summary.json").write_text(json.dumps(j),encoding="utf-8")
            rows=[dict(t=i,path=f"T{i}",label_eval_only=0 if i==1 else 1,
                       queried=int(i in qpos)) for i in range(100)]
            # query positions 1,10,20; T0 carries label 1.
            with (path/"prequential.csv").open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            for file in ("probe_scores.csv","probe_summary.csv","memory_snapshots.csv"):
                (path/file).write_text("dummy\n",encoding="utf-8")
            r=runner.check(a,"fake",0,128,"native")
            self.assertEqual(r["budget"],128)
            j["protocol"]["loso_folds"][0]["training"].append("N1")
            (path/"summary.json").write_text(json.dumps(j),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Self match"):
                runner.check(a,"fake",0,128,"native")


if __name__=="__main__":
    unittest.main()
