"""Version-frozen CPU report renderer for A4_EB68_v1 sealed outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np

from experiments.calibration import a4_equal_budget_allocation as a4


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise a4.A4Error(f"Report output already exists with different content: {path}")
        return
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        import os
        os.fsync(f.fileno())


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False).encode("utf-8") + b"\n"


def _fmt(value: Any, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _save_figure(fig: Any, path: Path) -> None:
    from io import BytesIO
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    _write_once(path, buffer.getvalue())


def _classification(metrics: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    delta = metrics["category_equal_macro"]["contrasts_online"]["budget_D_minus_A"]["auroc"]
    per_category = {cat: metrics["categories"][cat]["contrasts_online"]["budget_D_minus_A"]["auroc"]
                    for cat in a4.CATEGORIES}
    d_positive = sum(x is not None and x > 0 for x in per_category.values())
    d_negative = sum(x is not None and x < 0 for x in per_category.values())
    if delta is not None and delta >= .01 and d_positive >= 3:
        classification = "A4_SUPPORTS_36_PLUS_32"
        short = "The 36+32 allocation is materially competitive under the registered criterion."
    elif delta is not None and delta <= -.01 and d_negative >= 3:
        classification = "A4_SUPPORTS_4_PLUS_64"
        short = "The 4+64 allocation is materially competitive under the registered criterion."
    else:
        classification = "A4_MIXED_OR_INCONCLUSIVE"
        short = "The registered macro margin and category consistency rule does not select an allocation."
    return classification, short, {"macro_delta_D_minus_A_auroc": delta,
        "category_delta_D_minus_A_auroc": per_category,
        "categories_D_positive": d_positive, "categories_D_negative": d_negative,
        "registered_margin": .01, "registered_category_consistency": "at least 3 of 4"}


def _table(metrics: dict[str, Any]) -> tuple[list[dict[str, Any]], bytes]:
    rows = []
    for category in a4.CATEGORIES:
        cat = metrics["categories"][category]
        for arm in a4.ARMS:
            for mode in ("static", "online"):
                value = cat["arm_metrics"][arm][mode]
                rows.append({"category": category, "condition": "P0", "arm": arm,
                    "detector": "static_M0" if mode == "static" else "prequential",
                    "auroc": value["auroc"], "average_precision": value["average_precision"],
                    "ber": value["ber"], "fpr": value["fpr"], "fnr": value["fnr"],
                    "balanced_accuracy": value["balanced_accuracy"],
                    "tn": value["confusion"]["tn"], "fp": value["confusion"]["fp"],
                    "fn": value["confusion"]["fn"], "tp": value["confusion"]["tp"],
                    "query_count": cat["query_count"], "query_rate": cat["query_rate"],
                    "queried_normal": cat["queried_normal"], "queried_defect": cat["queried_defect"],
                    "writer_calls": cat["writer_calls"],
                    "candidate_patches": cat["candidate_patches"],
                    "accepted_candidate_patches": cat["accepted_candidate_patches"],
                    "fixed_reference_bytes": a4.FIXED_BYTES})
    fields = list(rows[0])
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return rows, stream.getvalue().encode("utf-8")


def _resources(run_root: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    cats = {}
    for category in a4.CATEGORIES:
        result = _read(run_root / "stage_a" / category / "stage_a_result.json")
        cats[category] = {
            "event_count": result["event_count"], "query_count": result["query_count"],
            "normal_image_identities_authorized": 68,
            "support_images": 4, "additional_commissioning_C64": 64,
            "C32_calibration_images": 32, "R32_complement_images": 32,
            "fixed_reference_bytes_by_arm": {arm: a4.FIXED_BYTES for arm in a4.ARMS},
            "fixed_bank_sha256_by_arm": {arm: result["fixed_bank_receipts"][arm]["fixed_sha256"]
                                           for arm in a4.ARMS},
            "fixed_bank_profile_by_arm": {arm: result["fixed_bank_receipts"][arm]["profile"]
                                            for arm in a4.ARMS},
            "calibration_feature_bytes_by_arm": {
                "A": 64 * 784 * 4096 * 4,
                "B": 32 * 784 * 4096 * 4,
                "C": 32 * 784 * 4096 * 4,
                "D": 32 * 784 * 4096 * 4},
            "calibration_feature_bytes_by_arm_interpretation":
                "logical references; B/C/D point into the shared retained C64 feature pool",
            "unique_retained_calibration_feature_bytes": result["unique_calibration_feature_bytes"],
            "unique_retained_calibration_image_bytes": result["unique_calibration_image_bytes"],
            "support_augmentation_feature_bytes": result["support_augmentation_feature_bytes"],
            "fixed_memory_components": result["fixed_memory_components"],
            "identity_metadata_bytes_serialized": result["identity_metadata_bytes"],
            "initialization_metadata_json_bytes": result["initialization_metadata_json_bytes"],
            "test_query_feature_bytes_per_event": result["test_query_feature_bytes_per_event"],
            "temporary_test_feature_cache_bytes": 0,
            "temporary_test_feature_cache": "none; streamed one encoding per event shared across four arms",
            "kcenter_C_candidate_count": result["kcenter_receipts"]["C"]["candidate_count"],
            "kcenter_D_candidate_count": result["kcenter_receipts"]["D"]["candidate_count"],
            "kcenter_D_candidate_feature_bytes": result["kcenter_receipts"]["D"]["candidate_matrix_bytes"],
            "kcenter_D_cuda_fp32_candidate_copy_bytes": result["kcenter_receipts"]["D"]["cuda_fp32_candidate_copy_bytes"],
            "kcenter_C_selection_seconds": result["kcenter_receipts"]["C"]["selection_seconds"],
            "kcenter_D_selection_seconds": result["kcenter_receipts"]["D"]["selection_seconds"],
            "feature_access_roles": result["feature_access_roles"],
            "cost_seconds": result["cost_seconds"],
            "gpu_peak_allocated_bytes": result["gpu_peak_allocated_bytes"],
            "online_greedy_bytes_by_arm": result.get("online_memory_audit_by_arm", {}),
            "normal_feedback_and_writes": {"queried_normal": result["verified_normal_feedback"],
                "queried_defect": result["verified_defect_feedback"],
                "native_normal_write_actions": result.get("native_normal_write_actions", 0),
                "memory_state_changes": result["state_change_counts"]},
        }
    report_root = a4.REPORT_ROOT
    generated_names = set(a4.REPORT_OUTPUT_NAMES)
    metadata_bytes = 0
    if report_root.exists():
        metadata_bytes = sum(p.stat().st_size for p in report_root.rglob("*")
                             if p.is_file() and p.relative_to(report_root).as_posix() not in generated_names)
    retained = 0
    for p in run_root.rglob("*"):
        if p.is_file() and ".scratch" not in p.parts:
            retained += p.stat().st_size
    return {"schema": "A4_EB68_v1.resource_accounting.v1",
        "run_id": run_root.name, "category_equal_macro": True,
        "fixed_prototype_storage_bytes": a4.FIXED_BYTES,
        "fixed_prototype_storage_definition": "8192 x 4096 x fp16",
        "total_deployed_memory_is_not_equalized": True,
        "arms": {arm: {"fixed_reference_bytes": a4.FIXED_BYTES,
                       "logical_calibration_feature_reference_bytes": cats[a4.CATEGORIES[0]]["calibration_feature_bytes_by_arm"][arm],
                       "unique_shared_calibration_feature_pool_bytes": cats[a4.CATEGORIES[0]]["unique_retained_calibration_feature_bytes"],
                       "online_greedy_bytes_by_category": {
                           cat: cats[cat]["online_greedy_bytes_by_arm"].get(arm)
                           for cat in a4.CATEGORIES}}
                 for arm in a4.ARMS},
        "categories": cats,
        "additional_memory_components": {
            "maximum_D_candidate_matrix_bytes_per_category": max(
                cats[c]["kcenter_D_candidate_feature_bytes"] for c in a4.CATEGORIES),
            "maximum_D_cuda_fp32_candidate_copy_bytes_per_category": max(
                cats[c]["kcenter_D_cuda_fp32_candidate_copy_bytes"] for c in a4.CATEGORIES),
            "static_M0_shadow_bank_bytes_per_category": 3 * a4.FIXED_BYTES,
            "unique_online_fixed_bank_bytes_per_category": 3 * a4.FIXED_BYTES,
            "maximum_support_augmentation_feature_bytes_per_category": max(
                cats[c]["support_augmentation_feature_bytes"] for c in a4.CATEGORIES),
            "maximum_unique_calibration_image_bytes_per_category": max(
                cats[c]["unique_retained_calibration_image_bytes"] for c in a4.CATEGORIES),
            "maximum_test_query_feature_bytes_per_event": max(
                cats[c]["test_query_feature_bytes_per_event"] for c in a4.CATEGORIES),
            "temporary_test_feature_cache_bytes": 0,
            "report_namespace_metadata_bytes_before_final_artifacts": metadata_bytes,
            "retained_run_output_bytes": retained,
            "retained_output_limit_bytes": a4.MAX_RETAINED_BYTES},
        "gpu_encoding_coreset_and_scoring_costs": {
            c: cats[c]["cost_seconds"] for c in a4.CATEGORIES},
        "cost_means_seconds": {key: float(np.mean([cats[c]["cost_seconds"][key]
                                                    for c in a4.CATEGORIES]))
                               for key in ("test_feature_encoding", "support_augmentation_encoding",
                                           "C64_identity_feature_encoding",
                                           "D_R32_augmentation_encoding",
                                           "calibration_native_scoring",
                                           "Kcenter_C_candidate_staging",
                                           "Kcenter_D_candidate_staging",
                                           "Kcenter_C_selection_including_repeat",
                                           "Kcenter_D_selection_including_repeat",
                                           "static_native_scoring",
                                           "online_native_scoring", "writer_updates",
                                           "protected_reference_rescoring", "category_wall")},
        "note": "Per-arm calibration feature byte counts are logical references; the C32 arms share the same C64 tensors in memory. The unique C64 calibration pool, Support augmentation cache, fixed M0 scoring shadows, K-Center staging and one current TEST feature are listed separately. Greedy bytes are reported from the native writer audit. Equal fixed-bank bytes do not equalize total runtime memory."}


def _plot_figures(metrics: dict[str, Any], resources: dict[str, Any],
                  decision: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = a4.REPORT_ROOT / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300})
    colors = {"A": "#2f5597", "B": "#5b9bd5", "C": "#ed7d31", "D": "#548235"}
    cats = list(a4.CATEGORIES)
    arms = list(a4.ARMS)
    outputs = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(cats))
    width = .19
    for axis, mode, title in ((axes[0], "static", "Static M0"),
                              (axes[1], "online", "Prequential")):
        for i, arm in enumerate(arms):
            values = [metrics["categories"][c]["arm_metrics"][arm][mode]["auroc"] for c in cats]
            axis.bar(x + (i - 1.5) * width, values, width, color=colors[arm], label=arm)
        axis.set_title(title)
        axis.set_xticks(x, [c.replace("_", "\n") for c in cats])
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("AUROC")
    axes[1].legend(title="Arm", frameon=False, ncol=4, loc="lower right")
    fig.suptitle("A4 static and prequential image ranking")
    fig.tight_layout()
    path = outdir / "figure_1_static_prequential_auroc.png"
    _save_figure(fig, path); plt.close(fig); outputs.append(path.name)

    fig, ax = plt.subplots(figsize=(11, 5))
    metrics_order = ("ber", "fpr", "fnr")
    width = .075
    x = np.arange(len(cats))
    for i, arm in enumerate(arms):
        for j, metric in enumerate(metrics_order):
            vals = [metrics["categories"][c]["arm_metrics"][arm]["online"][metric] for c in cats]
            offsets = (i * 3 + j - 5.5) * width
            ax.bar(x + offsets, vals, width, color=colors[arm], alpha=(1 - .18 * j),
                   hatch=("" if j == 0 else ("//" if j == 1 else "xx")),
                   label=f"{arm} {metric.upper()}" if j == 0 else None)
    ax.set_xticks(x, cats)
    ax.set_ylabel("Error rate")
    ax.set_ylim(0, 1)
    ax.set_title("Prequential BER / FPR / FNR trade-offs (shade/hatch: BER, FPR, FNR)")
    ax.grid(axis="y", alpha=.2)
    ax.legend(title="Arm (BER bar)", frameon=False, ncol=4)
    fig.tight_layout()
    path = outdir / "figure_2_prequential_error_tradeoffs.png"
    _save_figure(fig, path); plt.close(fig); outputs.append(path.name)

    fig, ax = plt.subplots(figsize=(10, 5))
    contrast_names = [("B-A", "calibration_B_minus_A"), ("C-B", "selector_C_minus_B"),
                      ("D-C", "reference_D_minus_C"), ("D-A", "budget_D_minus_A")]
    width = .16
    x = np.arange(len(contrast_names))
    for j, cat in enumerate(cats):
        vals = [metrics["categories"][cat]["contrasts_online"][key]["auroc"]
                for _, key in contrast_names]
        ax.bar(x + (j - 1.5) * width, vals, width,
               color=["#2f5597", "#5b9bd5", "#ed7d31", "#548235"][j], label=cat)
    macro = metrics["category_equal_macro"]["contrasts_online"]
    ax.plot(x, [macro[k]["auroc"] for _, k in contrast_names], color="black",
            marker="D", linestyle="--", linewidth=1.5, label="Four-category macro")
    ax.axhline(0, color="#555555", linewidth=.8)
    ax.set_xticks(x, [label for label, _ in contrast_names])
    ax.set_ylabel("AUROC difference")
    ax.set_title("Registered online contrast decomposition")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    path = outdir / "figure_3_registered_contrast_decomposition.png"
    _save_figure(fig, path); plt.close(fig); outputs.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    ids = np.arange(4)
    fixed_images = np.asarray([4, 4, 4, 36], dtype=float)
    calibration_images = np.asarray([64, 32, 32, 32], dtype=float)
    axes[0].bar(ids, fixed_images, color="#2f5597", width=.68,
                label="Fixed-bank source images")
    axes[0].bar(ids, calibration_images, bottom=fixed_images, color="#ed7d31",
                width=.68, label="Calibration images")
    axes[0].set_xticks(ids, arms)
    axes[0].set_ylabel("Unique normal training images used")
    axes[0].set_title("Information consumed by arm")
    axes[0].axhline(68, color="#444444", linestyle="--", linewidth=1,
                   label="68-image study universe")
    axes[0].legend(frameon=False, fontsize=8)
    fixed_mib = 64.0
    cal_mib = np.asarray([64, 32, 32, 32]) * 784 * 4096 * 4 / (1024 * 1024)
    dyn_mib = np.asarray([np.mean([resources["categories"][c]["online_greedy_bytes_by_arm"].get(arm, {}).get("allocated_state_bytes", 0)
                                   for c in cats]) / (1024 * 1024) for arm in arms])
    axes[1].bar(ids, [fixed_mib] * 4, color="#548235", width=.68,
                label="Fixed reference (64 MiB)")
    axes[1].bar(ids, dyn_mib, bottom=[fixed_mib] * 4, color="#5b9bd5",
                width=.68, label="Online Greedy (mean)")
    axes[1].bar(ids, cal_mib, bottom=[fixed_mib + d for d in dyn_mib],
                color="#ed7d31", width=.68, label="Calibration features")
    axes[1].set_xticks(ids, arms)
    axes[1].set_ylabel("Runtime feature memory (MiB)")
    axes[1].set_title("Logical per-arm memory references")
    axes[1].text(.01, -.22,
        "C32 arms share the physical C64 pool. Runtime also holds 3 static M0 bank shadows; "
        "K-Center staging and Support augmentations are reported separately.",
        transform=axes[1].transAxes, fontsize=8, va="top")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle(f"Equal fixed storage; total persistent memory can differ — {decision}")
    fig.tight_layout()
    path = outdir / "figure_4_equal_budget_memory_accounting.png"
    _save_figure(fig, path); plt.close(fig); outputs.append(path.name)
    return outputs


def _report_text(run_id: str, metrics: dict[str, Any], resources: dict[str, Any],
                 decision: str, decision_summary: str, decision_evidence: dict[str, Any],
                 figures: list[str]) -> tuple[str, str, str, str]:
    macro = metrics["category_equal_macro"]
    bycat = metrics["categories"]
    delta = macro["contrasts_online"]["budget_D_minus_A"]["auroc"]
    delta_sel = macro["contrasts_online"]["selector_C_minus_B"]["auroc"]
    delta_ref = macro["contrasts_online"]["reference_D_minus_C"]["auroc"]
    delta_cal = macro["contrasts_online"]["calibration_B_minus_A"]["auroc"]
    rows = []
    for cat in a4.CATEGORIES:
        vals = bycat[cat]["arm_metrics"]
        contrasts = bycat[cat]["contrasts_online"]
        rows.append(f"| {cat} | " + " | ".join(
            _fmt(vals[a]["static"]["auroc"]) + " / " + _fmt(vals[a]["online"]["auroc"])
            for a in a4.ARMS) + f" | {_fmt(contrasts['budget_D_minus_A']['auroc'])} |")
    metric_table = "\n".join(rows)
    safety_rows = []
    for cat in a4.CATEGORIES:
        cells = []
        for arm in a4.ARMS:
            value = bycat[cat]["arm_metrics"][arm]["online"]
            cells.append(" / ".join(_fmt(value[key]) for key in ("ber", "fpr", "fnr")))
        safety_rows.append(f"| {cat} | " + " | ".join(cells) + " |")
    safety_table = "\n".join(safety_rows)
    contrast_lines = []
    for name, key in (("Calibration B−A", "calibration_B_minus_A"),
                      ("Selector C−B", "selector_C_minus_B"),
                      ("Reference D−C", "reference_D_minus_C"),
                      ("Total budget D−A", "budget_D_minus_A")):
        values = macro["contrasts_online"][key]
        contrast_lines.append(f"| {name} | {_fmt(values['auroc'])} | {_fmt(values['ber'])} | {_fmt(values['fpr'])} | {_fmt(values['fnr'])} |")
    contrast_table = "\n".join(contrast_lines)
    category_contrast_lines = []
    for cat in a4.CATEGORIES:
        for name, key in (("B−A", "calibration_B_minus_A"),
                          ("C−B", "selector_C_minus_B"),
                          ("D−C", "reference_D_minus_C"),
                          ("D−A", "budget_D_minus_A")):
            value = bycat[cat]["contrasts_online"][key]
            category_contrast_lines.append(
                f"| {cat} | {name} | {_fmt(value['auroc'])} | {_fmt(value['ber'])} | "
                f"{_fmt(value['fpr'])} | {_fmt(value['fnr'])} |")
    category_contrast_table = "\n".join(category_contrast_lines)
    query_lines = "\n".join(f"| {cat} | {bycat[cat]['query_count']} | {bycat[cat]['query_rate']:.4%} | {bycat[cat]['queried_normal']} | {bycat[cat]['queried_defect']} | {bycat[cat]['writer_calls']} | {bycat[cat]['accepted_candidate_patches']} |"
        for cat in a4.CATEGORIES)
    static_online_d = macro["D"]["online"]["auroc"] - macro["D"]["static"]["auroc"]
    support_deltas = [bycat[c]["arm_metrics"]["D"]["online"]["auroc"] -
                      bycat[c]["arm_metrics"]["D"]["static"]["auroc"] for c in a4.CATEGORIES]
    calibration_operating = macro["contrasts_online"]["calibration_B_minus_A"]
    reviewer = (
        "The original Experiment A allocation used four deployed normal Support images and 64 additional labelled normal commissioning images for empirical q90 calibration. In this four-category development sensitivity study, we compared that allocation with an equal 68-image alternative that placed the four Supports and the complementary 32 commissioning normals into an equal-storage K-Center reference and calibrated on the remaining 32. "
        f"Under the registered P0 streams, the D−A macro prequential AUROC difference was {delta:+.4f} (category differences: " + ", ".join(f"{c} {bycat[c]['contrasts_online']['budget_D_minus_A']['auroc']:+.4f}" for c in a4.CATEGORIES) + "). "
        + ("These results support the expanded-reference allocation under the registered development-set criterion." if decision == "A4_SUPPORTS_36_PLUS_32" else
           "These results support the 4+64 allocation under the registered development-set criterion." if decision == "A4_SUPPORTS_4_PLUS_64" else
           "The registered criterion does not select an allocation; the comparison establishes sensitivity to how the same normal-image budget is divided, not independent confirmation or universal superiority.")
    )
    category_delta_text = ", ".join(
        f"{c} {bycat[c]['contrasts_online']['budget_D_minus_A']['auroc']:+.4f}"
        for c in a4.CATEGORIES)
    d_online_text = ", ".join(f"{c} {x:+.4f}" for c, x in zip(a4.CATEGORIES, support_deltas))
    report = f"""# Experiment A4 integrated report

**Run:** `{run_id}`
**Protocol:** `A4_EB68_v1`; P0 only; four previously exposed development categories.
**Classification:** `{decision}` — {decision_summary}

## Question and allocation

The reviewer question was whether 64 additional labelled normal commissioning images should be reserved for calibration or used to strengthen the fixed normal reference. Every category used exactly the A1 seed-0 Support-4 plus the same historical additional C64 identities: 68 distinct normal TRAIN image identities from distinct official Source Groups, disjoint from TEST groups. A1 draw-0 C32 membership was preserved, and R32 was its exact complement. This is four Supports plus **additional** C64; it is not strict four-shot total.

Arm A used the historical native four-Support fixed bank and C64 B3. B shared A's exact fixed bank, writer, dynamic memory and score trajectory but used C32 B3. C used the registered A3b K-Center algorithm on the same Support augmentation descriptors and C32 calibration. D used that same K-Center algorithm on Support-4 plus the complementary R32 descriptors and C32 calibration. Each fixed bank held 8,192 × 4,096 fp16 prototypes (67,108,864 bytes). D selected from the exact 65,536-descriptor, group-balanced candidate cap. C32 never entered a fixed reference.

![Static and prequential AUROC](figures/{figures[0]})

## Results

| Category | A static / online | B static / online | C static / online | D static / online | D−A online AUROC |
|---|---:|---:|---:|---:|---:|
{metric_table}

Category online BER / FPR / FNR, in arm order A, B, C, D:

| Category | A BER/FPR/FNR | B BER/FPR/FNR | C BER/FPR/FNR | D BER/FPR/FNR |
|---|---:|---:|---:|---:|
{safety_table}

Four-category equal-weight online macro AUROC/AP: A {_fmt(macro['A']['online']['auroc'])}/{_fmt(macro['A']['online']['average_precision'])}; B {_fmt(macro['B']['online']['auroc'])}/{_fmt(macro['B']['online']['average_precision'])}; C {_fmt(macro['C']['online']['auroc'])}/{_fmt(macro['C']['online']['average_precision'])}; D {_fmt(macro['D']['online']['auroc'])}/{_fmt(macro['D']['online']['average_precision'])}. Static macro AUROC: A {_fmt(macro['A']['static']['auroc'])}, B {_fmt(macro['B']['static']['auroc'])}, C {_fmt(macro['C']['static']['auroc'])}, D {_fmt(macro['D']['static']['auroc'])}.

![Operating-point trade-offs](figures/{figures[1]})

## Registered contrasts

| Contrast | Macro AUROC Δ | Macro BER Δ | Macro FPR Δ | Macro FNR Δ |
|---|---:|---:|---:|---:|
{contrast_table}

Category-level differences:

| Category | Contrast | AUROC Δ | BER Δ | FPR Δ | FNR Δ |
|---|---|---:|---:|---:|---:|
{category_contrast_table}

`B−A` AUROC/AP are exactly zero by construction and test; the calibration allocation changes operating decisions, not scores. Its macro online BER/FPR/FNR differences were {_fmt(calibration_operating['ber'])}/{_fmt(calibration_operating['fpr'])}/{_fmt(calibration_operating['fnr'])}. `C−B` changes the Support-only fixed-reference constructor and includes any legitimate downstream Greedy effects. `D−C` adds R32 to that same K-Center constructor. `D−A` is the registered full allocation contrast: representation and calibration allocation both change, so it does not isolate B3.

![Registered contrast decomposition](figures/{figures[2]})

## Online memory and operation

| Category | Queries | Query rate | Queried normals | Queried defects | Native writer calls | Accepted candidate prototypes |
|---|---:|---:|---:|---:|---:|---:|
{query_lines}

The A4 writer reference was frozen at each bank's C32 M0 higher-q90 and sample standard deviation (floor 0.02), separately from live B3. This matched writer rule is A4-specific; historical R9/A2 used its registered C64 writer reference. A/B used one shared writer call per queried normal; C and D used independent calls. Unqueried outcomes never entered the controller, and queried defects had no normal-memory write.

For D, prequential minus static macro AUROC was {static_online_d:+.4f}; category differences were {d_online_text}. This is the observed within-D online change under its stronger fixed reference; it does not imply every category improved.

![Data access and memory](figures/{figures[3]})

## Storage and compute

Every logical arm used a 67,108,864-byte fixed bank. A/B shared one online bank; three distinct online fixed banks and three static M0 scoring shadows were resident per category. Dynamic Greedy memory, the unique C64 calibration tensor pool, its source images, Support augmentation features, identity metadata, and coreset staging are reported separately in `EXPERIMENT_A4_RESOURCE_ACCOUNTING.json`; equal fixed storage did not equalize total runtime memory. A retained 49 GB TEST feature cache was unnecessary: TEST images were encoded once in stream order and the current feature was shared across all arms. C32 logical references share tensors from that C64 pool. D's 65,536×4,096 fp16 candidate matrix and its temporary CUDA float32 expansion were not retained in the run output.

Mean category wall time was {resources['cost_means_seconds']['category_wall']:.1f}s; mean TEST feature encoding was {resources['cost_means_seconds']['test_feature_encoding']:.1f}s; mean Support/C64/D-R32 identity feature encoding was {resources['cost_means_seconds']['support_augmentation_encoding']:.1f}/{resources['cost_means_seconds']['C64_identity_feature_encoding']:.1f}/{resources['cost_means_seconds']['D_R32_augmentation_encoding']:.1f}s. Mean C and D K-Center candidate staging took {resources['cost_means_seconds']['Kcenter_C_candidate_staging']:.1f}/{resources['cost_means_seconds']['Kcenter_D_candidate_staging']:.1f}s; K-Center selection including deterministic repeats took {resources['cost_means_seconds']['Kcenter_C_selection_including_repeat']:.1f}/{resources['cost_means_seconds']['Kcenter_D_selection_including_repeat']:.1f}s. Mean static and online native scoring times were {resources['cost_means_seconds']['static_native_scoring']:.1f}s and {resources['cost_means_seconds']['online_native_scoring']:.1f}s; calibration native scoring took {resources['cost_means_seconds']['calibration_native_scoring']:.1f}s. Mean protected-reference rescoring after genuine Greedy state changes was {resources['cost_means_seconds']['protected_reference_rescoring']:.1f}s. Per-category costs are in the resource JSON.

## Answers to the registered questions

1. **Q1 — 4+64 versus 36+32:** D−A macro online AUROC was {delta:+.4f}; per-category differences are shown above. Registered classification: `{decision}`.
2. **Q2 — Support-only K-Center effect:** C−B macro AUROC was {delta_sel:+.4f}. This includes legitimate downstream state differences.
3. **Q3 — Add 32 normals to K-Center reference:** D−C macro AUROC was {delta_ref:+.4f}.
4. **Q4 — C64 versus C32 calibration:** B−A score ranking was identical; macro BER/FPR/FNR differences were {_fmt(calibration_operating['ber'])}/{_fmt(calibration_operating['fpr'])}/{_fmt(calibration_operating['fnr'])} on the shared score trajectory.
5. **Q5 — Does Greedy help under D?** D prequential minus static macro AUROC was {static_online_d:+.4f}; category-specific changes are stated above.
6. **Q6 — Retain C64+B3?** {decision_summary} The result supports retaining C64 as the baseline only if the registered criterion favors A; otherwise treat the result as mixed or retain the competitive alternative for this exposed development scope. Safety metrics must be inspected separately.
7. **Q7 — Defensible manuscript statement:** The same 68 normal training identities were allocated between calibration and fixed-reference construction in a registered development comparison. The observed category and macro results quantify sensitivity to this allocation for P0 on four exposed categories; they do not establish strict few-shot access, independent confirmation, or population-level superiority.

## Interpretation and limits

Native Support construction and K-Center differ in algorithm and prototype geometry; it is a deliberate comparison factor. C−B estimates changing that constructor on the same four Support images, with subsequent online state differences included. D−C estimates the registered expanded K-Center reference. The full A/D comparison changes both the fixed bank and B3 calibration allocation. AUROC ranking does not guarantee acceptable FPR/FNR; the operating-point table and window diagnostics must be considered separately. Official Source Groups are the operational identity unit and do not prove distinct physical objects. These categories were previously exposed development data, not confirmation data.

No masks, other categories, confirmation outcomes, new query policy, Correction, BEA, or TEST-driven parameter selection were used. The complete run, source and Stage A/B seals remain in `{run_root}`. The source-freeze commit is recorded in the run manifest; the report commit does not alter executed source.
"""
    decision_doc = f"""# Experiment A4 final decision

**Classification:** `{decision}`
**Protocol:** `A4_EB68_v1`
**Run:** `{run_id}`

## Decision

{decision_summary} The registered D-A macro prequential AUROC difference was {delta:+.4f}; category differences were {category_delta_text}. The criterion was fixed at >=0.01 macro improvement and the same direction in at least three of four categories.

## Required answers

- **Q1:** Under the same 68 normal identities, D−A = {delta:+.4f} macro prequential AUROC. See category outcomes and BER/FPR/FNR in the integrated report.
- **Q2:** Support-only K-Center effect C−B = {delta_sel:+.4f} macro AUROC.
- **Q3:** Adding R32 to K-Center D−C = {delta_ref:+.4f} macro AUROC.
- **Q4:** C64 versus C32 changes only B3 readout on the shared A/B score tape: ΔAUROC = 0 by design; ΔBER/FPR/FNR = {_fmt(calibration_operating['ber'])}/{_fmt(calibration_operating['fpr'])}/{_fmt(calibration_operating['fnr'])}.
- **Q5:** D prequential minus static macro AUROC = {static_online_d:+.4f}; the per-category changes are in the report.
- **Q6:** {decision_summary} The exposed development results do not establish a universal resource allocation.
- **Q7:** The defensible reviewer statement is limited to a registered P0 comparison on four previously exposed development categories with the same 68 normal identities and equal fixed-reference storage; it measures allocation sensitivity and makes no independent-confirmation, strict-four-shot, or population-superiority claim.

Safety metrics (BER/FPR/FNR) are reported separately from ranking and may conflict with it. This classification is the preregistered decision rule, not a significance claim.
"""
    resource_summary = resources
    reviewer = reviewer
    return report, decision_doc, reviewer, contrast_table


def render_run(run_id: str) -> dict[str, Any]:
    run_root = a4.RUN_ROOT / run_id
    if not (run_root / "stage_b_seal.json").is_file():
        raise a4.A4Error("Cannot render A4 reports before Stage B seal")
    metrics = _read(run_root / "stage_b_metrics_private.json")
    resource = _resources(run_root, metrics)
    decision, short, evidence = _classification(metrics)
    figures = _plot_figures(metrics, resource, decision)
    resource = _resources(run_root, metrics)
    report, decision_doc, reviewer, _ = _report_text(
        run_id, metrics, resource, decision, short, evidence, figures)
    report_dir = a4.REPORT_ROOT
    report_dir.mkdir(parents=True, exist_ok=True)
    _, csv_bytes = _table(metrics)
    public_aggregate = {
        "schema": "A4_EB68_v1.aggregate_public.v1", "run_id": run_id,
        "protocol_id": a4.PROTOCOL_ID, "categories": list(a4.CATEGORIES),
        "arms": list(a4.ARMS), "category_equal_macro": metrics["category_equal_macro"],
        "categories_detail": {cat: {"arm_metrics": metrics["categories"][cat]["arm_metrics"],
            "contrasts_online": metrics["categories"][cat]["contrasts_online"],
            "query_count": metrics["categories"][cat]["query_count"],
            "query_rate": metrics["categories"][cat]["query_rate"],
            "queried_normal": metrics["categories"][cat]["queried_normal"],
            "queried_defect": metrics["categories"][cat]["queried_defect"],
            "writer_calls": metrics["categories"][cat]["writer_calls"],
            "candidate_patches": metrics["categories"][cat]["candidate_patches"],
            "accepted_candidate_patches": metrics["categories"][cat]["accepted_candidate_patches"],
            "windows": metrics["categories"][cat]["windows"]}
            for cat in a4.CATEGORIES},
        "scientific_decision": {"classification": decision, "summary": short,
                                "evidence": evidence},
        "stage_a_global_seal_sha256": _sha_file(run_root / "stage_a_global_seal.json"),
        "stage_b_seal_sha256": _sha_file(run_root / "stage_b_seal.json"),
        "test_labels_or_identities_included": False,
    }
    provenance = _read(run_root / "run_manifest.json")
    audit_lines = ["# Experiment A4 provenance audit", "",
        f"Run `{run_id}`; protocol `{a4.PROTOCOL_ID}`.", "",
        f"- Branch: `{provenance['git']['branch']}`.",
        f"- Reference parent: `{provenance['git']['expected_parent']}`.",
        f"- Source-freeze commit: `{provenance['git']['source_freeze_commit']}`.",
        f"- Execution source-map SHA-256: `{provenance['execution_source_hashes_sha256']}`.",
        f"- Global Stage A seal SHA-256: `{provenance.get('stage_a_global_seal_sha256', _sha_file(run_root / 'stage_a_global_seal.json'))}`.",
        f"- Stage B seal SHA-256: `{_sha_file(run_root / 'stage_b_seal.json')}`.",
        "- A1 seed-0 Supports and historical C64 identities were checked against A1 receipts.",
        "- A1 draw-0 C32 and its exact R32 complement were checked before scoring.",
        "- P0 query identities and group counts were compared to the A2 sealed event/query receipt.",
        "- Stage A label-blind score rows and private queried feedback have per-file SHA-256 seals.",
        "- Stage B labels were read only after all four complete Stage A category seals.",
        "- Stage B metric vectors were reconstructed from exact arm keys; A/B score equality passed.",
        "- Native direct-score, untouched-shadow score, and V0.2 Greedy update parity passed for each distinct fixed bank.",
        "- A1/A2/A3/R9/R10 protected output trees were hashed in preflight and rechecked by the independent verifier.",
        "- No TEST masks, confirmation outcomes, or other categories were accessed.", ""]
    results, csv_data = _table(metrics)
    resources_data = _json(resource)
    aggregate_data = _json(public_aggregate)
    decision_data = decision_doc.encode("utf-8")
    writes = {
        "EXPERIMENT_A4_INTEGRATED_REPORT.md": report.encode("utf-8"),
        "EXPERIMENT_A4_RESULTS.csv": csv_data,
        "EXPERIMENT_A4_AGGREGATE.json": aggregate_data,
        "EXPERIMENT_A4_RESOURCE_ACCOUNTING.json": resources_data,
        "EXPERIMENT_A4_PROVENANCE_AUDIT.md": "\n".join(audit_lines).encode("utf-8"),
        "EXPERIMENT_A4_REVIEWER_RESPONSE_CANDIDATE.md": ("# Reviewer response candidate\n\n" + reviewer + "\n").encode("utf-8"),
        "EXPERIMENT_A4_FINAL_DECISION.md": decision_data,
    }
    for name, data in writes.items():
        _write_once(report_dir / name, data)
    # Figure outputs were written by Matplotlib and must be immutable/repeatable on report-only reruns.
    for name in figures:
        path = report_dir / "figures" / name
        if not path.is_file() or path.stat().st_size < 10_000:
            raise a4.A4Error(f"Publication figure is missing or implausibly small: {path}")
    manifest = {"schema": "A4_EB68_v1.report_manifest.v1", "run_id": run_id,
        "decision": decision, "files": {name: _sha_file(report_dir / name) for name in writes},
        "figures": {name: _sha_file(report_dir / "figures" / name) for name in figures},
        "created_utc": metrics.get("created_utc")}
    _write_once(report_dir / "EXPERIMENT_A4_REPORT_MANIFEST.json", _json(manifest))
    retained = sum(p.stat().st_size for p in run_root.rglob("*") if p.is_file())
    report_retained = sum(p.stat().st_size for p in report_dir.rglob("*") if p.is_file())
    if retained + report_retained > a4.MAX_RETAINED_BYTES:
        raise a4.A4Error(f"A4 retained run and report outputs exceed 4 GiB: {retained + report_retained}")
    return {"decision": decision, "summary": short,
            "files": sorted(list(writes) + ["EXPERIMENT_A4_REPORT_MANIFEST.json"]),
            "figures": figures, "retained_run_bytes": retained,
            "retained_report_bytes": report_retained,
            "retained_total_bytes": retained + report_retained}
