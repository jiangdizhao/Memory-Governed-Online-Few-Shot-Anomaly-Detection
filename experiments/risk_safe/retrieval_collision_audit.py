#!/usr/bin/env python3
"""Offline D3 patch attribution report from an already frozen native GPU run.

This consumes ONLY the original read-only collision CSV and summary. It never
proposes/commits an update, changes a retrieval policy or views Probe B.
Top-1 dynamic provenance is diagnostic, NOT a complete top-5 causal assignment.
If the CSV was capped to top-N, resulting fractions are TOP-N ONLY; prefer
the uncapped totals in summary.json for denominator-aware interpretation.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def describe(rows, *, all_rows: int):
    output = {"saved": len(rows), "total_before_top_n_cap": all_rows,
              "top_n_censored": len(rows) != all_rows}
    for method in ("greedy", "causal_novelty"):
        subset = [r for r in rows if r["source"] == method]
        defects = [r for r in subset if int(r["label_offline_only"]) == 1]
        normals = [r for r in subset if int(r["label_offline_only"]) == 0]
        all_defect = [float(r["score_erosion"]) for r in defects]
        crossed = [r for r in defects
                   if str(r["cross_camera_top1_diagnostic"]).lower()
                   in ("true", "1")]
        output[method] = {
            "saved_collision_patches": len(subset),
            "saved_defect_image_patches": len(defects),
            "saved_normal_image_patches": len(normals),
            "saved_defect_erosion_sum": sum(all_defect),
            "saved_defect_erosion_max": max(all_defect, default=0),
            "top1_cross_camera_fraction_saved_defect_only":
                len(crossed) / len(defects) if defects else None,
            "n_distinct_defect_camera_views": len(set(
                (r["image"], r["camera_id"] if "camera_id" in r
                 else r["query_camera"]) for r in defects)),
            "interpretation": (
                "These are only strongest attribution rows due to top-N cap; "
                "do not interpret as full mask-based defect localization."
                if len(rows) != all_rows else
                "All scored global-vs-static collision rows are included; "
                "top1 source still diagnostic, not top5 causal share."),
        }
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir", required=True)
    args = p.parse_args(argv)
    root = Path(args.result_dir)
    main = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    if main.get("schema") != "risk_safe.d3.conditional_retrieval_mechanism_pilot.v2" \
            or not main.get("completed") or not main.get("global_native_parity_pass"):
        raise ValueError("Not a complete, native-parity-verified D3 mechanism result")
    with (root/"collision_patch_attribution.csv").open(
            "r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != main["collision_rows_saved_top"]:
        raise ValueError("Incomplete saved collision CSV")
    report = describe(rows, all_rows=main["collision_rows_total"])
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
