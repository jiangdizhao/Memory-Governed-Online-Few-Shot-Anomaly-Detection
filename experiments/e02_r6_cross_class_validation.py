#!/usr/bin/env python3
"""Pre-registered R6 cross-category stress validation (no algorithm changes).

Two Real-IAD categories were pre-selected on test prevalence only:
  defect_dominated = rolled_strip_base, near_balanced = switch.
Each uses support seeds 0 and 1, and native/normal_first/defect_first.
Run exactly ONE GPU condition per --run invocation. --summarize validates
all 12 runs, paired lineages and zero-oracle stream-membership controls.

Requires existing experiments/e02_r6_sidecar_stability.py; this script
does not import or change the original v22, R4, R5 or R6 model logic.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SELECTION = {
    "defect_dominated": "rolled_strip_base",
    "near_balanced": "switch",
}
CLASSES = ("rolled_strip_base", "switch")
SUPPORT_SEEDS = (0, 1)
ORDERS = ("native", "normal_first", "defect_first")
SCHEMA = "e02.r6.extreme_balanced_validation.v1"
R6_RUNNER = ROOT / "experiments" / "e02_r6_sidecar_stability.py"


def load_profile(path: Path) -> dict:
    """Refuse silent category re-selection, especially outcome-based choices."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "e02.r6.class_prevalence.v1":
        raise ValueError("Expected original R6 label-prevalence profile schema")
    selected = report.get("pre_registered_candidates", {})
    for key, value in EXPECTED_SELECTION.items():
        if selected.get(key) != value:
            raise ValueError(
                f"Pre-registration mismatch {key}: expected {value}, "
                f"found {selected.get(key)}")
    records = {c["class_name"]: c for c in report.get("classes", [])}
    if len(records) != len(report.get("classes", [])):
        raise ValueError("Duplicate class names in prevalence profile")
    for cls in CLASSES:
        if cls not in records:
            raise ValueError(f"Missing profile class {cls}")
        c = records[cls]
        if (int(c["normal"]) + int(c["defect"]) != int(c["total"])
                or int(c["normal"]) <= 0 or int(c["defect"]) <= 0):
            raise ValueError(f"Invalid class counts: {cls}")
        if abs(c["normal"] / c["total"] - c["normal_fraction"]) > 1e-9:
            raise ValueError(f"Invalid class prevalence: {cls}")
    return report


def run_dir(output_root: Path, cls: str, seed: int, order: str) -> Path:
    if cls not in CLASSES or seed not in SUPPORT_SEEDS or order not in ORDERS:
        raise ValueError("Condition is outside pre-registered R6 matrix")
    return output_root / f"e02_r6_{cls}_seed{seed}_{order}"


def build_argv(args, cls: str, seed: int, order: str) -> list[str]:
    """Fixed experiment flags: no user-selected performance hyperparameters."""
    folder = run_dir(args.output_root, cls, seed, order)
    cmd = [
        sys.executable, str(R6_RUNNER),
        "--dataset-root", str(args.dataset_root),
        "--json-dir", str(args.json_dir),
        "--class-name", cls,
        "--support-seed", str(seed),
        "--legacy-seed", "0",
        "--stream-seed", "0",
        "--query-seed", "37",
        "--query-rate", "0.03",
        "--max-samples", "0",
        "--device", "cuda",
        "--math-sdpa",
        "--local-saliency",
        "--print-every", "100",
        "--variants", "D",
        "--stream-order", order,
        "--out-dir", str(folder),
    ]
    if order != "native":
        native = run_dir(args.output_root, cls, seed, "native")
        cmd.extend([
            "--reference-native-summary", str(native / "r6_summary.json"),
            "--reference-native-csv", str(native / "r6_prequential.csv"),
        ])
    return cmd


def expected_budget(total: int) -> int:
    """R4 Budget's round-half-up query-cap rule; keep literal 3%."""
    return int(math.floor(total * 0.03 + 0.5))


def _check_report(report: dict, cls: str, seed: int, order: str,
                  counts: dict, *, native: dict | None = None) -> dict:
    """Read-only acceptance check on already-computed R6 outputs."""
    if report.get("schema") != "e02.r6.sidecar.audit.v1":
        raise AssertionError("Invalid R6 summary schema")
    if report.get("class_name") != cls or report.get("stream_order") != order:
        raise AssertionError(f"Wrong class/order in {cls}/seed{seed}/{order}")
    p = report["parameters"]
    required = {
        "class_name": cls, "support_seed": seed, "legacy_seed": 0,
        "stream_seed": 0, "query_seed": 37, "query_rate": 0.03,
        "max_samples": 0, "math_sdpa": True, "local_saliency": True,
        "variants": "D",
    }
    for key, value in required.items():
        if p.get(key) != value:
            raise AssertionError(
                f"{cls}/seed{seed}/{order}: settings {key} "
                f"expected {value!r}, got {p.get(key)!r}")
    total = int(counts["total"])
    k = expected_budget(total)
    result = report["schedules"]["D"]
    if report.get("n_stream") != total or report.get("total_budget_each") != k:
        raise AssertionError(f"{cls}/seed{seed}/{order}: dataset/budget mismatch")
    if result["queries"] != k or len(result["query_indices_1based"]) != k:
        raise AssertionError(f"{cls}/seed{seed}/{order}: under/over-spent queries")
    if result["queried_normals"] + result["queried_defects"] != k:
        raise AssertionError("Invalid verified label counts")
    if len(set(result["query_indices_1based"])) != k:
        raise AssertionError("Repeated query index")
    if any(t < 1 or t > total for t in result["query_indices_1based"]):
        raise AssertionError("Invalid query index")
    if len(report.get("label_windows", [])) == 0:
        raise AssertionError("Missing class-conditional evaluation windows")
    if sum(x["normal_count"] for x in report["label_windows"]) != counts["normal"]:
        raise AssertionError("Stream normal prevalence changed")
    if sum(x["defect_count"] for x in report["label_windows"]) != counts["defect"]:
        raise AssertionError("Stream defect prevalence changed")
    if not report["audit"]["n_update_attempts"] >= len(
            report["audit"]["accepted_steps"]):
        raise AssertionError("Malformed Sidecar audit")
    if order == "native":
        if report.get("prequential_parity", {}).get("passed"):
            # This class has no pre-existing R5 native trajectory: new R6
            # native is the baseline, NOT proof of R5 stepwise parity.
            pass
        if native is not None:
            raise AssertionError("Native cannot reference itself")
    else:
        if native is None:
            raise AssertionError("Missing matching native run")
        if not report.get("stress_membership_check", {}).get("passed"):
            raise AssertionError("Stress run did not verify exact image membership")
        if not report.get("reference_check", {}).get("passed"):
            raise AssertionError("Stress R6 native reference check failed")
        if report.get("original_stream_paths_sha256") != native.get(
                "stream_paths_sha256"):
            raise AssertionError("Stress source stream fingerprint mismatch")
        if report.get("canonical_fixed_sha256") != native.get(
                "canonical_fixed_sha256"):
            raise AssertionError("Stress fixed bank differs from native")
        if report.get("support_paths") != native.get("support_paths"):
            raise AssertionError("Stress support images differ from native")
        if abs(float(report["canonical_theta0"])
               - float(native["canonical_theta0"])) > 1e-12:
            raise AssertionError("Stress original threshold differs from native")
        if result["fixed_sha256"] != native["schedules"]["D"]["fixed_sha256"]:
            raise AssertionError("Stress immutable memory fingerprint mismatch")
    return {
        "class_name": cls, "support_seed": seed, "order": order,
        "n_stream": total, "query_budget": k,
        "queried_normal": int(result["queried_normals"]),
        "queried_defect": int(result["queried_defects"]),
        "queried_normal_fraction": result["queried_normals"] / k,
        "normal_population_fraction": counts["normal"] / total,
        "query_selection_odds_ratio_empirical": (
            (result["queried_defects"] / counts["defect"]) /
            (result["queried_normals"] / counts["normal"]))
            if result["queried_normals"] else None,
        "budget_forced_terminal": int(result["budget_forced_terminal"]),
        "longest_no_query_gap": int(result["longest_query_free_gap"]),
        "first_400_queries": int(result["first_400_queries"]),
        "stm_write_calls": int(result["stm_write_calls"]),
        "stm_prototypes": int(result["stm_prototypes"]),
        "balanced_updates": list(report["audit"]["accepted_steps"]),
        "sidecar_attempt_outcomes": dict(report["audit"]["outcome_counts"]),
        "balanced_ba": float(result["arm_metrics"]["balanced_rescue"]["balanced_accuracy"]),
        "balanced_fpr": float(result["arm_metrics"]["balanced_rescue"]["fpr"]),
        "balanced_fnr": float(result["arm_metrics"]["balanced_rescue"]["fnr"]),
        "balanced_theta_final": float(result["arm_metrics"]["balanced_rescue"]["theta_final"]),
        "mistake_ba": float(result["arm_metrics"]["mistake_feedback"]["balanced_accuracy"]),
        "stm_auroc": float(result["arm_metrics"]["frozen"]["auroc"]),
        "worst_defined_window_balanced_fnr": max(
            (w["balanced_rescue"]["fnr"] for w in report["label_windows"]
             if w["balanced_rescue"]["fnr"] is not None), default=None),
        "n_windows_without_normal": sum(
            w["normal_count"] == 0 for w in report["label_windows"]),
        "n_windows_without_defect": sum(
            w["defect_count"] == 0 for w in report["label_windows"]),
        "native_reference_validated": (
            order == "native" or
            bool(report.get("reference_check", {}).get("passed")
                 and report.get("stress_membership_check", {}).get("passed"))),
    }


def read_condition(args, cls, seed, order, profile, native=None):
    folder = run_dir(args.output_root, cls, seed, order)
    summary = folder / "r6_summary.json"
    csv = folder / "r6_prequential.csv"
    if not summary.is_file() or not csv.is_file():
        raise FileNotFoundError(f"Missing R6 condition outputs: {folder}")
    if csv.stat().st_size == 0:
        raise AssertionError(f"Empty prequential CSV: {csv}")
    data = json.loads(summary.read_text(encoding="utf-8"))
    class_counts = next(x for x in profile["classes"]
                        if x["class_name"] == cls)
    return data, _check_report(
        data, cls, seed, order, class_counts, native=native)


def _run(args, profile):
    cls, seed, order = args.class_name, args.support_seed, args.order
    out = run_dir(args.output_root, cls, seed, order)
    if out.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing run directory: {out}. "
            "Inspect it; move it aside explicitly if a clean rerun is required.")
    if order != "native":
        native, _ = read_condition(args, cls, seed, "native", profile)
        if native["stream_order"] != "native":
            raise AssertionError("Invalid native reference")
    cmd = build_argv(args, cls, seed, order)
    print("[R6 cross-class] Starting ONE GPU run:", cls, seed, order, flush=True)
    print("Command:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    native = None
    if order != "native":
        native, _ = read_condition(args, cls, seed, "native", profile)
    _, row = read_condition(args, cls, seed, order, profile, native=native)
    print("[R6 cross-class] Validated:", json.dumps(
        row, ensure_ascii=False, indent=2), flush=True)


def _plan(args):
    for cls in CLASSES:
        for seed in SUPPORT_SEEDS:
            for order in ORDERS:
                folder = run_dir(args.output_root, cls, seed, order)
                good = (folder / "r6_summary.json").is_file()
                print(f"{cls:19} seed={seed} {order:12} "
                      f"{'PRESENT' if good else 'MISSING':7} {folder}")
    print("Order: native BEFORE either stress mode; each condition is an "
          "independent GPU run. PRESENT means files exist, not validated.")


def summarize(args, profile):
    rows, pairs, missing = [], [], []
    for cls in CLASSES:
        for seed in SUPPORT_SEEDS:
            native_path = run_dir(args.output_root, cls, seed, "native")
            if not (native_path / "r6_summary.json").is_file():
                missing.extend(f"{cls}/seed{seed}/{mode}" for mode in ORDERS)
                continue
            native, row = read_condition(args, cls, seed, "native", profile)
            rows.append(row)
            for order in ORDERS[1:]:
                folder = run_dir(args.output_root, cls, seed, order)
                if not (folder / "r6_summary.json").is_file():
                    missing.append(f"{cls}/seed{seed}/{order}")
                    continue
                _, other = read_condition(args, cls, seed, order,
                                          profile, native=native)
                rows.append(other)
                pairs.append({
                    "class_name": cls, "support_seed": seed, "order": order,
                    "delta_balanced_ba": other["balanced_ba"] - row["balanced_ba"],
                    "delta_balanced_fnr": other["balanced_fnr"] - row["balanced_fnr"],
                    "delta_stm_auroc": other["stm_auroc"] - row["stm_auroc"],
                    "delta_stm_prototypes": (
                        other["stm_prototypes"] - row["stm_prototypes"]),
                    "delta_queried_normal": (
                        other["queried_normal"] - row["queried_normal"]),
                    "delta_terminal_catchup": (
                        other["budget_forced_terminal"]
                        - row["budget_forced_terminal"]),
                })
    # Independence is a DESIGN GOAL; a second seed yielding identical
    # supports must be disclosed, never quietly counted as independent.
    independence = {}
    for cls in CLASSES:
        folders = [
            run_dir(args.output_root, cls, seed, "native") / "r6_summary.json"
            for seed in SUPPORT_SEEDS]
        if all(x.is_file() for x in folders):
            a, b = [json.loads(x.read_text(encoding="utf-8")) for x in folders]
            different = (a["support_paths"] != b["support_paths"])
            independence[cls] = {
                "support_paths_differ": different,
                "fixed_bank_hashes_differ": (
                    a["canonical_fixed_sha256"] != b["canonical_fixed_sha256"]),
            }
            if not different:
                raise AssertionError(
                    f"{cls}: seeds 0 and 1 selected the SAME support files")
    full = len(rows) == len(CLASSES) * len(SUPPORT_SEEDS) * len(ORDERS)
    report = {
        "schema": SCHEMA,
        "complete": full and not missing,
        "expected_conditions": 12,
        "validated_conditions": len(rows),
        "missing_conditions": missing,
        "pre_registered_selection": EXPECTED_SELECTION,
        "fixed_support_seeds": list(SUPPORT_SEEDS),
        "fixed_orders": list(ORDERS),
        "run_metrics": rows,
        "paired_against_same_class_seed_native": pairs,
        "between_seed_support_distinctness": independence,
        "scope_notes": [
            "Diagnosis, not a claim of population/generalization robustness.",
            "Test-label-driven reorder performed ONLY before stream replay.",
            "Every stress/native pair has identical support, original image "
            "membership, initial theta0, bank fingerprint and 3% query budget.",
            "Query tapes and learned STM are intentionally endogenous and differ.",
            "Pooled AUROC compares scores formed under different memory states; "
            "it is potentially confounded with temporal class ordering.",
            "Undefined per-class 256-step risks remain null in R6 source files.",
            "Two support seeds are too few for inferential confidence intervals.",
            "This is simplified STM-only E02/R5-D, not the historical full v22.",
        ],
    }
    if not full and not args.allow_partial:
        raise RuntimeError(
            f"Matrix incomplete ({len(rows)}/12). Missing: {missing}. "
            "Use --allow-partial for a clearly labeled interim diagnostic.")
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_out.exists():
        raise FileExistsError(
            f"Refusing to replace previously committed summary: {args.summary_out}")
    args.summary_out.write_text(json.dumps(
        report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "complete": report["complete"], "validated": len(rows),
        "missing": missing, "summary": str(args.summary_out)}, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true",
                      help="Run exactly one of the 12 fixed GPU conditions")
    mode.add_argument("--summarize", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    p.add_argument("--dataset-root", type=Path, default=Path("G:/dataset/realiad"))
    p.add_argument("--json-dir", type=Path, default=None)
    p.add_argument("--profile", type=Path,
                   default=ROOT / "outputs" / "e02_r6_class_prevalence.json")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    p.add_argument("--summary-out", type=Path, default=ROOT / "outputs" /
                   "e02_r6_extreme_balanced_validation_summary.json")
    p.add_argument("--class-name", choices=CLASSES, default="rolled_strip_base")
    p.add_argument("--support-seed", type=int, choices=SUPPORT_SEEDS, default=0)
    p.add_argument("--order", choices=ORDERS, default="native")
    p.add_argument("--allow-partial", action="store_true")
    return p


def self_test():
    assert len(CLASSES) * len(SUPPORT_SEEDS) * len(ORDERS) == 12
    assert expected_budget(3805) == 114
    assert expected_budget(3820) == 115
    with_profile = load_profile(ROOT / "outputs" / "e02_r6_class_prevalence.json")
    assert with_profile["pre_registered_candidates"]["defect_dominated"] == CLASSES[0]
    assert with_profile["pre_registered_candidates"]["near_balanced"] == CLASSES[1]
    p = parser().parse_args(["--run"])
    p.json_dir = p.dataset_root / "jsons" / "realiad_jsons"
    argv = build_argv(p, CLASSES[0], 0, "defect_first")
    assert "--reference-native-summary" in argv
    assert "--reference-native-csv" in argv
    assert "--require-parity" not in argv
    assert "--reference-r4-summary" not in argv
    assert "--rescore-anchor-cap" not in argv
    print("R6 cross-category validation offline self-test: PASS")


if __name__ == "__main__":
    args = parser().parse_args()
    if args.self_test:
        self_test()
    else:
        profile = load_profile(args.profile)
        if args.json_dir is None:
            args.json_dir = args.dataset_root / "jsons" / "realiad_jsons"
        if args.plan:
            _plan(args)
        elif args.run:
            _run(args, profile)
        else:
            summarize(args, profile)
