#!/usr/bin/env python3
"""R0 causal opportunity preflight. Metadata diagnostics ONLY, not online control.

Run: python -m experiments.risk_safe.r0_opportunity --classes audiojack,bottle_cap --budgets 16,32,64,79,128
The offline reader can inspect ground truth ONLY to audit label availability
after the shared Q-A label-blind tape has been constructed and frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from . import preflight as pf
from . import d1b_protocol as d1b
from . import query_protocol as qp

SCHEMA = "risk_safe.r0.opportunity_preflight.v1"
DEFAULT_OUT = "outputs/r0_opportunity_20261028/metadata"


def integers(value: str) -> tuple[int, ...]:
    result = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not result or len(result) != len(set(result)) or any(x < 1 for x in result):
        raise ValueError("Expected unique positive integers")
    return result



def native_output_name(args):
    """Make smoke and full runs non-colliding without invoking the GPU."""
    if args.out_dir is not None:
        return Path(args.out_dir)
    caps = "-".join(str(x) for x in integers(args.capacities))
    suffix = f"smoke{args.max_query_events}" if args.max_query_events else "full"
    return Path("outputs/r0_opportunity_20261028/native") / (
        f"{args.class_name}_s{args.support_seed}_q{args.query_seed}_"
        f"b{args.budget}_cap{caps}_audit{args.audit_every}_{suffix}")


def probe_subset(plan, n):
    """Offline Probe-A subset; sample groups classified for evaluation only."""
    if n < 1 or n > 24:
        raise ValueError("Probe sample groups per type must be 1..24")
    normal_keys = sorted(k for k in plan.probe_a_samples if all(
        r.is_good for r in plan.probe_a if r.sample_key == k))
    defect_keys = sorted(k for k in plan.probe_a_samples if any(
        not r.is_good for r in plan.probe_a if r.sample_key == k))
    if len(normal_keys) < n or len(defect_keys) < n:
        raise RuntimeError("Insufficient evaluation-only sample groups")
    selected = set(normal_keys[:n] + defect_keys[:n])
    return tuple(r for r in plan.probe_a if r.sample_key in selected)


def offline_counts(events, label_by_path, *, anchors_needed=(1, 4, 8, 16)):
    """Audit true labels AFTER tape creation. Never export these to a policy."""
    normals = defects = 0
    eligible_by_k = {str(k): 0 for k in anchors_needed}
    normal_events = 0
    first_defect_step = None
    history = []
    for item in events:
        if item.path not in label_by_path:
            raise RuntimeError("Queried event has no authoritative image label")
        is_normal = bool(label_by_path[item.path])
        if is_normal:
            normal_events += 1
            for k in anchors_needed:
                eligible_by_k[str(k)] += int(defects >= k)
        else:
            if first_defect_step is None:
                first_defect_step = int(item.position)
        history.append(dict(step=int(item.position),
                            sample_key=item.sample_key,
                            camera_id=int(item.camera_id),
                            prior_normal=normals, prior_defect=defects,
                            label_after_query="normal" if is_normal else "defect"))
        normals += int(is_normal)
        defects += int(not is_normal)
    return dict(n_query_events=len(events), n_verified_normal=normals,
                n_verified_defect=defects, n_normal_candidate_upper_bound=normal_events,
                n_normal_with_prior_defect=eligible_by_k,
                first_verified_defect_step=first_defect_step,
                events=history)


def metadata_report(args):
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("R0 output directory exists and is nonempty; do not overwrite: " + str(out))
    budgets = integers(args.budgets)
    seeds = integers(args.support_seeds)
    query_seeds = integers(args.query_seeds)
    classes = tuple(c.strip() for c in args.classes.split(",") if c.strip())
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Provide unique classes")
    if any(c not in pf.CLASSES for c in classes):
        raise ValueError("R0 development classes must come from frozen D1-B classes")
    rows, detail = [], {}
    for name in classes:
        train, test = pf.load_class_records(Path(args.dataset_root),
                                            Path(args.json_dir), name)
        plan = pf.build_partitions(train, test, pf.SplitConfig())
        # All support seeds are predefined; reporting metadata across seeds
        # is intentional and does not grant extra supports to a detector.
        if any(s not in plan.support_samples_by_seed for s in seeds):
            raise RuntimeError("Support seed absent from frozen split")
        label_by_path = {r.path: r.is_good for r in test}
        stream = d1b.build_ordered_stream(plan.phases, "random", name)
        universe = tuple(sorted({x.sample_key for x in stream}))
        for qseed in query_seeds:
            priority = d1b.shared_query_priority(universe, name, qseed)
            tapes = d1b.build_paired_tapes(stream, budgets, qseed, name, priority)
            for budget, tape in sorted(tapes.items()):
                audit = offline_counts(tape.events, label_by_path)
                row = dict(class_name=name, query_seed=qseed,
                           budget=budget, stream_images=len(stream),
                           official_stream_samples=len(universe),
                           image_query_rate=len(tape.events)/len(stream),
                           official_sample_query_rate=len(tape.events)/len(universe),
                           n_verified_normal=audit["n_verified_normal"],
                           n_verified_defect=audit["n_verified_defect"],
                           normal_candidate_upper_bound=audit["n_normal_candidate_upper_bound"],
                           normal_with_prior_d1=audit["n_normal_with_prior_defect"]["1"],
                           normal_with_prior_d4=audit["n_normal_with_prior_defect"]["4"],
                           normal_with_prior_d8=audit["n_normal_with_prior_defect"]["8"],
                           first_defect_step=audit["first_verified_defect_step"],
                           tape_sha256=tape.digest)
                rows.append(row)
                detail[f"{name}/q{qseed}/B{budget}"] = dict(
                    tape_sha256=tape.digest, stream_sha256=tape.stream_digest,
                    audit=audit)
    payload = dict(schema=SCHEMA, scope="offline metadata; no GPU; no memory changes",
                   state="PRELIMINARY_UPPER_BOUNDS_NOT_MECHANISM_ACTIVATION",
                   base_branch="risk-safe-admission-sprint-20261027",
                   label_access="authoritative labels inspected only after Q-A tape fixed; not visible to any controller",
                   classes=classes, budgets=budgets, support_seeds=seeds,
                   query_seeds=query_seeds, summary_rows=rows,
                   per_tape=detail,
                   caveats=[
                       "A verified normal query is an UPPER BOUND on a v22 candidate, not an actual write.",
                       "Defect count is an image-label count; not a verified defect-patch count.",
                       "Official sample keys need not represent independent physical manufactured objects.",
                       "No oracle advantage, counterfactual defect risk, capacity replacement or probe accuracy is measured.",
                       "Metadata inspection never justifies adaptive selection of better-looking query tapes."])
    out.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    (out / "r0_metadata.json").write_text(content, encoding="utf-8")
    cols = [k for k in rows[0] if k != "tape_sha256"] + ["tape_sha256"]
    with (out / "r0_metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in out.iterdir() if p.is_file()}
    (out / "checksums.json").write_text(
        json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(stage="R0 metadata", complete=True,
                          combinations=len(rows), out_dir=str(out),
                          no_gpu=True, no_algorithmic_results=True), indent=2))
    return 0


def verify(args):
    out = Path(args.out_dir)
    sums = json.loads((out / "checksums.json").read_text(encoding="utf-8"))
    for name, digest in sums.items():
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("R0 checksum mismatch: " + name)
    result = json.loads((out / "r0_metadata.json").read_text(encoding="utf-8"))
    if result["schema"] != SCHEMA:
        raise RuntimeError("R0 schema mismatch")
    print(json.dumps(dict(verified=True, out_dir=str(out),
                          combinations=len(result["summary_rows"])), indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default=pf.DEFAULT_DATASET_ROOT)
    p.add_argument("--json-dir", default=None)
    p.add_argument("--classes", default=",".join(pf.CLASSES))
    p.add_argument("--budgets", default="16,32,64,79,128")
    p.add_argument("--support-seeds", default="0,1,2")
    p.add_argument("--query-seeds", default="37,101,211")
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args(argv)
    if args.json_dir is None:
        args.json_dir = str(Path(args.dataset_root) / "jsons" / "realiad_jsons")
    return verify(args) if args.verify_only else metadata_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
