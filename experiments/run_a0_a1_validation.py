#!/usr/bin/env python3
"""ONE-COMMAND A0/A1 strict-four-shot matched-memory experiment orchestrator.

Example (from repository root in torch environment):
 python experiments/run_a0_a1_validation.py --all --dataset-root G:/dataset/realiad --math-sdpa

Default matrix: audiojack, rolled_strip_base, switch; seeds 0/1; native.
--with-stress adds normal_first/defect_first for each. A tiny 64-image
smoke is executed first. Runs automatically resume after completed,
validated conditions; invalid or partial previous outputs fail closed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SCHEMA = "a0a1.strict4.loso.matched.v1"
ARMS = ("static", "fifo", "reservoir", "online_greedy_coreset",
        "v22_normal_stm", "v22_utility_stm")
DEFAULT_CLASSES = ("audiojack", "rolled_strip_base", "switch")
DEFAULT_SEEDS = (0, 1)
DEFAULT_ORDERS = ("native",)


def stream_conditions(args):
    classes = tuple(x.strip() for x in args.classes.split(",") if x.strip())
    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Empty or duplicate class list")
    if not seeds or len(seeds) != len(set(seeds)) or any(n < 0 for n in seeds):
        raise ValueError("Invalid support seed list")
    orders = ("native", "normal_first", "defect_first") if args.with_stress else DEFAULT_ORDERS
    return [(cls, seed, order) for cls in classes
            for seed in seeds for order in orders]


def location(args, cls, seed, order, *, smoke=False):
    suffix = "smoke" if smoke else "full"
    return Path(args.out_root) / f"{cls}_s{seed}_{order}_{suffix}"


def argv(args, cls, seed, order, *, smoke=False):
    path = ROOT / "experiments" / "a0_a1_strict4_matched.py"
    cmd = [sys.executable, str(path), "--dataset-root", args.dataset_root,
           "--class-name", cls, "--support-seed", str(seed),
           "--stream-order", order, "--out-dir",
           str(location(args, cls, seed, order, smoke=smoke)),
           "--probe-count", str(args.probe_count),
           "--dynamic-capacity", str(args.dynamic_capacity),
           "--probe-every", str(args.probe_every),
           "--print-every", str(args.print_every),
           "--max-samples", "64" if smoke else "0"]
    if args.json_dir:
        cmd += ["--json-dir", args.json_dir]
    if args.math_sdpa:
        cmd.append("--math-sdpa")
    return cmd


def sha(items):
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def validate(path: Path, *, cls: str, seed: int, order: str, smoke: bool,
             args) -> dict:
    reportfile = path / "summary.json"
    if not reportfile.is_file():
        raise ValueError(f"Missing complete summary at {path}; inspect partial output")
    j = json.loads(reportfile.read_text(encoding="utf-8"))
    if (j.get("schema") != SCHEMA or j.get("class_name") != cls
            or j.get("support_seed") != seed or j.get("stream_order") != order):
        raise ValueError(f"Wrong schema/condition at {path}")
    spec = j["protocol"]
    if (spec["support_selection"] != "strict_random_four_shot"
            or spec["support_image_access_count"] != 4
            or len(spec["support_paths"]) != 4
            or len(set(spec["support_paths"])) != 4
            or sha(spec["support_paths"]) != spec["support_sha256"]):
        raise ValueError("Invalid strict-four-shot support lineage")
    if (spec["calibration"] !=
            "four_fold_leave_entire_support_image_and_all_augmentations_out"
            or len(spec["loso_fold_lineage"]) != 4
            or len(spec["loso_scores"]) != 4
            or spec["theta0_shared_across_all_methods"] !=
            spec["loso_stats"]["theta"]):
        raise ValueError("Invalid leave-one-out threshold contract")
    if any((f["heldout"] in f["support_paths"] or
            len(f["support_paths"]) != 3 or
            set(f["support_paths"]) != (set(spec["support_paths"]) - {f["heldout"]}))
           for f in spec["loso_fold_lineage"]):
        raise ValueError("LOSO self-match detected")
    if set(f["heldout"] for f in spec["loso_fold_lineage"]) != set(spec["support_paths"]):
        raise ValueError("LOSO fold coverage mismatch")
    if (not spec["probe_not_in_training_or_feedback"] or
            not spec["labels_after_predictions"] or
            not spec["no_adaptive_threshold_in_primary_experiment"]):
        raise ValueError("Invalid experiment privacy/causality controls")
    if tuple(j["arms"]) != ARMS:
        raise ValueError("Missing memory arm")
    if len(j["protocol"]["probe_paths"]) != j["n_probe"]:
        raise ValueError("Probe count mismatch")
    if sha(j["protocol"]["probe_paths"]) != spec["probe_paths_sha256"]:
        raise ValueError("Probe hash mismatch")
    if abs(j["protocol"]["query_rate_target"] - .03) > 1e-12:
        raise ValueError("Query budget not pre-registered 3%")
    if j["K"] != int(j["n_stream"] * .03 + .5):
        raise ValueError("Realized budget mismatch")
    if (spec["observed_query_normals"] + spec["observed_query_defects"]
            != j["K"] or len(spec["query_positions_1based"]) != j["K"]
            or sha(spec["query_positions_1based"]) != spec["query_tape_sha256"]):
        raise ValueError("Query feedback/tape mismatch")
    if (spec["query_positions_1based"] != sorted(set(spec["query_positions_1based"]))
            or any(not 1 <= t <= j["n_stream"]
                   for t in spec["query_positions_1based"])):
        raise ValueError("Duplicate or out-of-bounds queries")
    if (smoke and j["n_stream"] != 64) or (not smoke and j["n_stream"] <= 64):
        raise ValueError("Wrong smoke/full size")
    if j["n_probe"] != args.probe_count:
        raise ValueError("Probe count incompatible with requested run")
    if spec["dynamic_capacity_per_layer"] != args.dynamic_capacity:
        raise ValueError("Memory capacity incompatible with requested run")
    if (spec["feature_configuration"]["device"] != "cuda"
            and args.require_cuda):
        raise ValueError("Non-CUDA trial where CUDA required")
    capacities = {tuple(v["capacity_per_layer"]) for v in j["arms"].values()}
    if len(capacities) != 1:
        raise ValueError("Unequal memory capacities")
    if any(any(v != args.dynamic_capacity for v in caps) for caps in capacities):
        raise ValueError("Reported memory capacity not equal to requested bound")
    if j["arms"]["static"]["mutable_prototypes"] != 0:
        raise ValueError("Static baseline has mutable memory")
    for name, arm in j["arms"].items():
        if arm["mutable_prototypes"] > arm["mutable_capacity"]:
            raise ValueError(f"Overflow: {name}")
        if name != "static" and arm["candidate_events"] != spec["observed_query_normals"]:
            raise ValueError(f"Unmatched feedback count: {name}")
    csv_file = path / "prequential.csv"
    with csv_file.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != j["n_stream"]:
        raise ValueError("Prequential CSV count mismatch")
    names = [r["path"] for r in rows]
    if sha(names) != spec["stream_paths_sha256"]:
        raise ValueError("Stream path hash mismatch")
    if len(names) != len(set(names)) or set(names) & set(spec["probe_paths"]):
        raise ValueError("Probe leakage or repeated sample")
    if sum(int(r["queried"]) for r in rows) != j["K"]:
        raise ValueError("Query tape count vs CSV mismatch")
    actual_normals = sum(int(r["queried"]) for r in rows
                         if int(r["label_eval_only"]) == 0)
    actual_defects = sum(int(r["queried"]) for r in rows
                         if int(r["label_eval_only"]) == 1)
    if ((actual_normals, actual_defects) !=
            (spec["observed_query_normals"], spec["observed_query_defects"])):
        raise ValueError("Queried-label composition differs from CSV")
    candidate_counts = {v["candidate_patches"] for k, v in j["arms"].items()
                        if k != "static"}
    if len(candidate_counts) != 1:
        raise ValueError("Dynamic arms did not receive the same candidate patches")
    expected_query_indices = [int(r["t"]) for r in rows if int(r["queried"])]
    if expected_query_indices != spec["query_positions_1based"]:
        raise ValueError("Query tape indices mismatch")
    for r in rows:
        if abs(float(r["theta_loso_fixed"]) -
               spec["theta0_shared_across_all_methods"]) > 1e-12:
            raise ValueError("Shared threshold not fixed")
    for filename in ("probe_scores.csv", "probe_summary.csv"):
        if not (path / filename).is_file():
            raise ValueError(f"Missing fixed-probe artifact: {filename}")
    return {
        "class_name": cls, "support_seed": seed, "order": order,
        "n_stream": j["n_stream"], "K": j["K"],
        "normal_queries": spec["observed_query_normals"],
        "defect_queries": spec["observed_query_defects"],
        "theta0_loso": spec["theta0_shared_across_all_methods"],
        "theta0_selfmatch_diagnostic": spec["diagnostic_self_match_stats"]["theta"],
        "support_hash": spec["support_sha256"],
        "query_hash": spec["query_tape_sha256"],
        "fixed_bank_hash": spec["fixed_bank"]["sha256"],
        "scores": {k: {
            key: j["arms"][k][key]
            for key in ("auroc", "balanced_accuracy", "fpr", "fnr",
                        "mutable_prototypes", "accepted_patches")}
                   for k in ARMS},
        "output": str(path),
    }


def run_one(args, cls, seed, order, *, smoke=False):
    output = location(args, cls, seed, order, smoke=smoke)
    if output.exists():
        print(f"[A0A1] Existing condition found: {output}; validating", flush=True)
        return validate(output, cls=cls, seed=seed, order=order,
                        smoke=smoke, args=args)
    cmd = argv(args, cls, seed, order, smoke=smoke)
    print(f"[A0A1] RUN {cls} s{seed} {order}"
          f"{' SMOKE' if smoke else ''}", flush=True)
    started = time.time()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Each run owns its directory and is not retried over partial output.
    logfile = output.parent / (output.name + ".run.log")
    if logfile.exists():
        raise ValueError(f"Previous log exists without validated summary: {logfile}")
    with logfile.open("w", encoding="utf-8") as log:
        log.write("RUN: " + " ".join(cmd) + "\n")
        log.flush()
        child = subprocess.Popen(cmd, cwd=ROOT,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8",
                                 errors="replace", bufsize=1)
        for line in child.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        rc = child.wait()
        log.write(f"\nEXIT CODE {rc} elapsed_seconds={time.time()-started:.1f}\n")
    if rc:
        raise RuntimeError(f"GPU condition failed; inspect {logfile} and {output}")
    return validate(output, cls=cls, seed=seed, order=order,
                    smoke=smoke, args=args)


def write_aggregate(args, rows, *, complete):
    aggregate = {
        "schema": "a0a1.matched.aggregate.v1",
        "complete": complete,
        "expected_conditions": len(stream_conditions(args)),
        "validated_conditions": len(rows),
        "methods": list(ARMS),
        "conditions": rows,
        "fairness_notes": [
            "Strict 4 source-image access, LOSO normal-only threshold.",
            "Fixed query tape per condition and equal verified labels.",
            "One common static-reference patch-selection rule across arms.",
            "Fixed numeric theta0 across all arms, no adaptive threshold in A1.",
            "Held-out probe never joined stream, feedback or memory.",
            "v22_normal_stm isolates R6-style dynamic-normal update, while v22_utility_stm tests optional retention; neither is full STM/LTM.",
            "Online greedy coreset is approximate, not an optimized offline solver.",
            "Identical per-layer memory capacity; default 128 to create actual memory pressure; actual occupancy and compute differ.",
            "Two support seeds are screening evidence, not inferential significance.",
        ],
    }
    target = Path(args.out_root) / "aggregate.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Re-computing the aggregate from validated immutable summaries is okay;
    # individual GPU run directories are never overwritten.
    target.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False,
                                 allow_nan=False), encoding="utf-8")
    keys = ("class_name", "support_seed", "order", "method", "auroc",
            "balanced_accuracy", "fpr", "fnr", "mutable_prototypes")
    with (target.parent / "aggregate.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for c in rows:
            for method, score in c["scores"].items():
                w.writerow({**{k: c[k] for k in ("class_name",
                                                "support_seed", "order")},
                            "method": method,
                            **{k: score[k] for k in keys[4:]}})
    print(f"[A0A1] Aggregated {len(rows)}/{aggregate['expected_conditions']}"
          f" conditions: {target}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument("--all", action="store_true", help="Run smoke then whole matrix")
    m.add_argument("--validate", action="store_true", help="No GPU: aggregate existing")
    m.add_argument("--self-test", action="store_true", help="Pure Python offline test")
    p.add_argument("--dataset-root", default="G:/dataset/realiad")
    p.add_argument("--json-dir", default=None)
    p.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--with-stress", action="store_true")
    p.add_argument("--out-root", default="outputs/a0_a1_strict4_memory")
    p.add_argument("--probe-count", type=int, default=32)
    p.add_argument("--dynamic-capacity", type=int, default=128,
                   help="Per-layer normal STM prototype cap, matched across arms")
    p.add_argument("--probe-every", type=int, default=512)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--math-sdpa", action="store_true")
    p.add_argument("--require-cuda", action="store_true", default=True)
    p.add_argument("--no-smoke", action="store_true")
    return p


def self_test():
    args = parser().parse_args(["--all"])
    assert len(stream_conditions(args)) == 6
    assert location(args, "switch", 1, "native").name == "switch_s1_native_full"
    assert len(stream_conditions(parser().parse_args(
        ["--all", "--with-stress"]))) == 18
    assert sha(["a", "b"]) != sha(["b", "a"])
    a = argv(args, "audiojack", 0, "native", smoke=True)
    assert a[a.index("--max-samples") + 1] == "64"
    assert "--math-sdpa" not in a
    print("A0/A1 runner offline self-test: PASS")


if __name__ == "__main__":
    a = parser().parse_args()
    if a.self_test:
        self_test()
    else:
        todo = stream_conditions(a)
        if a.probe_count < 4 or a.probe_every < 0 or not 1 <= a.dynamic_capacity <= 1024:
            raise SystemExit("Invalid probe settings")
        if a.all and not a.no_smoke:
            run_one(a, todo[0][0], todo[0][1], "native", smoke=True)
        rows, failed = [], []
        for cls, seed, order in todo:
            if a.validate:
                path = location(a, cls, seed, order)
                if not path.is_dir():
                    failed.append(str(path))
                    continue
                rows.append(validate(path, cls=cls, seed=seed, order=order,
                                     smoke=False, args=a))
            else:
                rows.append(run_one(a, cls, seed, order))
                write_aggregate(a, rows, complete=False)
        if failed:
            raise SystemExit(f"Missing {len(failed)} conditions: {failed}")
        write_aggregate(a, rows, complete=True)
