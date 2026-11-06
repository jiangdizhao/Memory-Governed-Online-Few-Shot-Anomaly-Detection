"""Integrity-checked A1 reporting and figures for Experiment A.

This reporting utility consumes only a completed protected run directory.  It
verifies the category seals and score-artifact digests before emitting compact,
label-free public summaries and A1 figures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.calibration import a_calibration_scale as a


def _digest(value: Any) -> str:
    return hashlib.sha256(a.canonical_json(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = __import__("os").open(str(path), __import__("os").O_WRONLY |
                                __import__("os").O_CREAT |
                                __import__("os").O_EXCL, 0o644)
    with __import__("os").fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        __import__("os").fsync(stream.fileno())


def _verified_payload(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    preflight = _read_json(run_dir / "preflight_private.json")
    contract = _read_json(run_dir / "experiment_contract.json")
    summary = _read_json(run_dir / "stage_b_summary_private.json")
    if summary.get("status") != "STAGE_B_COMPLETE":
        raise a.ExperimentAError("Report rebuild requires complete Stage B metrics")
    if summary.get("test_masks_read") is not False:
        raise a.ExperimentAError("Reported run accessed masks")
    for category in a.CLASSES:
        category_dir = run_dir / "a1" / category
        seal = a.require_stage_a_seal(category_dir / "stage_a_seal.json")
        score_path = category_dir / "stage_a_scores_private.json"
        if a.sha256_file(score_path) != seal["score_digests"]["stage_a_scores_private.json"]:
            raise a.ExperimentAError(f"{category}: score tape no longer matches its Stage A seal")
        metrics = _read_json(category_dir / "stage_b_metrics_private.json")
        if metrics.get("stage_a_seal_sha256") != seal["seal_sha256"]:
            raise a.ExperimentAError(f"{category}: Stage B is bound to another Stage A seal")
        if metrics.get("masks_read") is not False:
            raise a.ExperimentAError(f"{category}: Stage B reports mask access")
    if contract["git_parent_sha"] != preflight["git"]["head"]:
        raise a.ExperimentAError("Run contract and preflight Git parent differ")
    return preflight, contract, summary


def build_public_contract(run_dir: Path, report_dir: Path) -> dict[str, Any]:
    preflight, contract, summary = _verified_payload(run_dir)
    categories = {}
    for category, data in preflight["classes"].items():
        roles = data["role_manifest"]
        support = data["support_manifest"]
        c64 = data["historical_C64_manifest"]
        categories[category] = {
            "category_json_sha256": data["json_sha256"],
            "test_identity_sha256": data["test_identity_sha256"],
            "test_image_count": data["test_image_count"],
            "test_group_count": data["test_group_count"],
            "support": {
                seed: {"image_count": len(receipt["images"]),
                       "group_count": len(receipt["groups"]),
                       "image_membership_sha256": receipt["image_sha256"],
                       "group_membership_sha256": receipt["group_sha256"]}
                for seed, receipt in support.items() if seed in {"0", "1"}},
            "historical_C64": {
                "image_count": len(c64["images"]),
                "group_count": len(c64["groups"]),
                "image_membership_sha256": c64["image_sha256"],
                "group_membership_sha256": c64["group_sha256"]},
            "H": {"group_count": len(roles["H_groups"]),
                  "image_count": len(roles["H_images"]),
                  "group_membership_sha256": _digest(sorted(roles["H_groups"])),
                  "image_membership_sha256": _digest(sorted(roles["H_images"]))},
            "calibration_draws": {
                draw: {name: {"group_count": len(groups),
                              "group_membership_sha256": _digest(sorted(groups))}
                       for name, groups in sizes.items()}
                for draw, sizes in roles["draws"].items()},
            "Cmax_G": {"group_count": len(roles["eligible_calibration_groups"]),
                       "group_membership_sha256": _digest(
                           sorted(roles["eligible_calibration_groups"]))},
            "Cmax_V": {"group_count": roles["Cmax_V_group_count"],
                       "image_count": roles["Cmax_V_image_count"],
                       "group_membership_sha256": _digest(
                           sorted(roles["eligible_calibration_groups"])),
                       "image_membership_sha256": _digest(
                           sorted(roles["Cmax_V_images"]))},
            "A3b_reference_pool": {
                "group_count": len(roles["A3b_reference_building_groups"]),
                "group_membership_sha256": _digest(
                    sorted(roles["A3b_reference_building_groups"]))},
            "registered_stream_orders": {
                key: {"seed": value["group_shuffle_seed"],
                      "event_count": value["event_count"],
                      "identity_map_sha256": value["identity_map_sha256"]}
                for key, value in data["registered_stream_orders"].items()},
            "all_disjointness_checks_passed": all(roles["disjointness"].values()),
        }
    result = {
        "schema": "a_calibration_scale.public_contract_a1.v1",
        "protocol_id": contract["protocol_id"],
        "run_id": contract["run_id"],
        "git_parent_sha": contract["git_parent_sha"],
        "source_hashes_at_contract": contract["source_hashes_at_contract"],
        "reporting_source_sha256": a.sha256_file(Path(__file__).resolve()),
        "model": contract["model"],
        "preprocessing": contract["preprocessing"],
        "conditions": contract["condition_profiles"],
        "categories": categories,
        "thresholds": contract["thresholds"],
        "statistical_units": {"independent_unit": "official_sample_key group",
                              "view_unit": "individual recorded image/view",
                              "H": "32 additional normal groups; one view per group"},
        "allowed_data_access": {"categories": list(a.CLASSES),
                                "train_normal": True,
                                "test_pixels": True,
                                "queried_labels": "registered query gate only",
                                "unqueried_labels": "Stage B only after Stage A seal",
                                "masks": False},
        "access_audit": {"test_labels_read_in_stage_a": False,
                         "stage_b_started_after_all_seals": True,
                         "masks_read": False,
                         "confirmation_categories_accessed": False},
        "stage_a_seals": {category: _read_json(
            run_dir / "a1" / category / "stage_a_seal.json")["seal_sha256"]
            for category in a.CLASSES},
        "stage_b_status": summary["status"],
        "component_status": {"A1": "COMPLETE_VALID",
                             "A2": "NOT_EXECUTED",
                             "A3a": "NOT_EXECUTED",
                             "A3b": "HOLD_BASELINE_FIDELITY"},
    }
    return result


def build_public_results(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = []
    for category, data in summary["categories"].items():
        for condition, condition_data in data["metrics"].items():
            for arm, row in condition_data["arms"].items():
                met = row["metrics"]
                results.append({
                    "category": category, "condition": condition, "arm": arm,
                    "threshold": row["threshold"],
                    "normal_H_exceedance_fraction": row["H_exceedance_fraction"],
                    "n": met["n"], "n_normal": met["n_normal"],
                    "n_defect": met["n_defect"], "fpr": met["fpr"],
                    "fnr": met["fnr"], "ber": met["ber"], "ba": met["ba"],
                    "auroc": condition_data["auc_parity"]["auroc_by_arm"][0],
                    "score_sha256": condition_data["score_sha256"],
                    "score_only_auroc_identical": condition_data["auc_parity"]["identical"],
                })
    return results


def write_results_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise a.ExperimentAError("No A1 metric rows")
    fields = list(rows[0])
    import io
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    _write_new(path, out.getvalue().encode("utf-8"))


def make_figures(run_dir: Path, report_dir: Path, summary: Mapping[str, Any]) -> list[Path]:
    figure_dir = report_dir / "figures_a1"
    figure_dir.mkdir(parents=True, exist_ok=False)
    paths = []

    # Figure 1: draw-to-draw empirical q90 variation; full pools are fixed sets,
    # so they are intentionally not plotted as having zero sampling variance.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sizes = [32, 64, 128]
    for ax, condition in zip(axes, ("P0", "P2")):
        for category, result in summary["categories"].items():
            stability = result["metrics"][condition]["threshold_stability"]
            y = [stability[f"Q90:C{size}"]["sample_std"] for size in sizes]
            ax.plot(sizes, y, marker="o", linewidth=1.4, label=category)
        ax.set_title(f"{condition}: support seed {0 if condition == 'P0' else 1}")
        ax.set_xlabel("Independent TRAIN-normal groups")
        ax.set_xticks(sizes)
        ax.set_xlim(24, 136)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Sample SD of q90 threshold across 3 draws")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("A1 threshold sampling variation at M0")
    fig.tight_layout()
    path = figure_dir / "figure_1_calibration_size_threshold_variation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    # Figure 2: threshold-only operating points on the primary P0 score vectors.
    fig, ax = plt.subplots(figsize=(8, 7))
    methods = ["FIXED05", "Q90:C32", "Q90:C64", "Q90:C128", "Q90:Cmax-G",
               "Q90:Cmax-V", "GAUSSIAN-Q90:C64", "GAUSSIAN-2.5:C64", "SELF", "LOSO"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    markers = {"audiojack": "o", "regulator": "s", "switch": "^",
               "rolled_strip_base": "D"}
    for method, color in zip(methods, colors):
        for category, result in summary["categories"].items():
            row = result["metrics"]["P0"]["arms"][method]["metrics"]
            ax.scatter(row["fpr"], row["fnr"], color=color,
                       marker=markers[category], s=62,
                       label=f"{method} / {category}")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate",
           ylabel="False negative rate",
           title="A1 P0 operating points (same frozen score vector per category)")
    ax.grid(True, alpha=0.25)
    method_handles = [plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=color, label=method, markersize=7)
                      for method, color in zip(methods, colors)]
    category_handles = [plt.Line2D([0], [0], marker=marker, color="#444444",
                                   linestyle="None", label=category, markersize=7)
                        for category, marker in markers.items()]
    first = ax.legend(handles=method_handles, loc="upper center", ncol=2,
                      fontsize=7, bbox_to_anchor=(0.5, -0.13))
    ax.add_artist(first)
    ax.legend(handles=category_handles, loc="upper right", fontsize=8,
              title="Category marker")
    fig.tight_layout()
    path = figure_dir / "figure_2_threshold_fpr_fnr_tradeoff.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    # Figure 5: registered initial reference sizes, float32 feature footprint,
    # and one-pass native scoring estimate from measured A1 throughput.
    preflight = _read_json(run_dir / "preflight_private.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    styles = {"audiojack": "o-", "regulator": "s-", "switch": "^-",
              "rolled_strip_base": "D-"}
    for category, values in preflight["classes"].items():
        role = values["role_manifest"]
        class_result = _read_json(run_dir / "a1" / category / "stage_a_result.json")
        sizes_x = [32, 64, 128, len(role["eligible_calibration_groups"]),
                   role["Cmax_V_image_count"]]
        raw_bytes = [n * 784 * 4096 * 4 for n in sizes_x]
        eligible_plus_h = role["Cmax_V_image_count"] + len(role["H_images"])
        seconds_per_call = class_result["timing_seconds"]["reference_scoring"] / (2 * eligible_plus_h)
        one_pass_s = [n * seconds_per_call for n in sizes_x]
        axes[0].plot(sizes_x, [v / (1024**3) for v in raw_bytes], styles[category], label=category)
        axes[1].plot(sizes_x, one_pass_s, styles[category], label=category)
    axes[0].set(xlabel="Calibration images/views", ylabel="Raw float32 feature bytes (GiB)",
                title="Temporary feature footprint")
    axes[1].set(xlabel="Calibration images/views", ylabel="Native score time (seconds)",
                title="One-pass C0 scoring cost estimated from A1")
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("A1 calibration size and measured single-pass cost")
    fig.tight_layout()
    path = figure_dir / "figure_5_calibration_size_storage_scoring_cost.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    status = (
        "Figure 3 (state-consistent B3 trajectories): NOT GENERATED; A2 was not run.\n"
        "Figure 4 (C64 versus expanded-reference prequential outcomes): NOT GENERATED; A2/A3b were not run.\n"
        "Figure 6 (A3a divergence and A3b comparison): NOT GENERATED; A3a/A3b were not run.\n")
    _write_new(figure_dir / "FIGURES_3_4_6_STATUS.txt", status.encode("utf-8"))
    return paths


def _num(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def build_report(run_dir: Path, report_dir: Path,
                 preflight: Mapping[str, Any], summary: Mapping[str, Any],
                 figure_paths: list[Path], contract_path: Path,
                 results_path: Path) -> str:
    run_id = run_dir.name.removeprefix("a_calibration_scale_")
    lines = [
        "# Experiment A — Calibration Scale, Initialization and State-Dependence Audit",
        "",
        f"**Run:** `{run_id}`  \n**Scope completed:** A1 only  \n**Study type:** four-category development study; not independent confirmation.",
        "",
        "## Questions and hypotheses",
        "",
        "A1 tested whether the historical additional C64 calibration set yields a stable initial operating threshold; whether nested C32/C128/Cmax pools change threshold stability and the normal-tail behavior; and how SELF, LOSO, fixed raw score 0.5, empirical q90, and Gaussian moment comparators move the operating point. A1 does not answer whether calibration changes future learning; that requires A2/A3 trajectories.",
        "",
        "## Protocol and implementation identity",
        "",
        f"Branch `{preflight['git']['branch']}`; parent `{preflight['git']['head']}`. The verified DINOv2 ViT-L/14 register checkpoint SHA-256 is `{preflight['model']['checkpoint_sha256']}`; local source-tree SHA-256 is `{preflight['model']['hub_source_tree_sha256']}`; loaded model-state SHA-256 is `{preflight['model']['loaded_model_state_sha256']}`. The scorer used resize 448, center crop 392, layers 5/11/17/23, concatenated normalized 4096D descriptors, identity view, native Top-32 scoring, and spatial maximum aggregation.",
        "",
        "The fixed bank was configured at nominal capacity 25,000 and per-image cap 5,000. Its verified realized bank was 8,192 float16 prototypes × 4,096 dimensions, 67,108,864 bytes per category and Support seed. This is the realized count, distinct from nominal capacity.",
        "",
        "The Stage05 score-tape manifest has stale hashes for `stage05.py` and `stage05_runner.py`. Experiment A did not use those vectors as primary scores: new A1 scores were produced through the current native scorer and compared to the frozen tapes. All 15,295 image scores for both Support seeds and all eight 64-image C64 vectors matched exactly (maximum absolute difference 0.0; tolerance 1e-5 + 1e-4 relative). Fixed-bank structured digests matched the registered receipts.",
        "",
        "The machine-readable public contract and role-membership digests are in [EXPERIMENT_A_CONTRACT_A1_ADDENDUM.json](EXPERIMENT_A_CONTRACT_A1_ADDENDUM.json). Per-image raw identities, score vectors, and outcome-derived metrics remain in the protected local run output.",
        "",
        "## Data roles and units",
        "",
        "| Category | TRAIN-normal groups | S0/S1 groups | Historical additional C64 groups/images | H groups/images | Cmax-G groups | Cmax-V groups/images | TEST events/groups |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, data in preflight["classes"].items():
        roles = data["role_manifest"]
        lines.append(
            f"| {category} | {data['train_group_count']} | 8 | 64 / 64 | 32 / 32 | "
            f"{len(roles['eligible_calibration_groups'])} | "
            f"{roles['Cmax_V_group_count']} / {roles['Cmax_V_image_count']} | "
            f"{data['test_image_count']} / {data['test_group_count']} |")
    lines += [
        "",
        "The recorded `official_sample_key` is the operational group unit; it does not independently prove physical-object identity. H was 32 additional TRAIN-normal groups, one deterministic view per group, and was excluded from calibration and A3b reference construction. C64 is additional to four Supports; it is not strict four-shot total. Cmax-G and Cmax-V are different estimands; Cmax-V views are correlated within group.",
        "",
        "Condition order identity was checked against completed label-blind R9 maps: P0 = Support 0 / group-shuffle seed 0, P1 = Support 0 / seed 11, and P2 = Support 1 / seed 0. Each was an exact event-map match. P0 and P1 share the same full-stream score vector; their order-sensitive prefixes and 512-event windows differ.",
        "",
        "## A1 equations and causal boundary",
        "",
        "`theta_SELF = mean(support_self) + 2.5 × sample_std(ddof=1)`. `theta_LOSO` uses four held-out Support scores, each scored against a three-Support fixed bank with the held-out image and all augmentation descendants excluded. `FIXED05 = 0.5` is a raw anomaly-score threshold. Empirical q90 uses the higher order statistic at zero-based index `ceil(0.90 × (n−1))`. Gaussian comparators use `mean + 1.2815515655 × sample_std` and `mean + 2.5 × sample_std`. All thresholds share the same frozen score vector within each Support condition.",
        "",
        "Stage A scored TEST pixels, sealed score vectors and identities, and read no TEST labels. Four category seals were validated before a separate Stage B process read image labels for offline evaluation. No masks, other categories, or confirmation outcomes were accessed.",
        "",
        "## A1 results by category and condition",
        "",
        "The table reports AUROC for the shared score vector and full-stream FPR/FNR/BER for four registered operating points. P1 has identical full-stream metrics to P0 because it permutes the same score-label pairs; P2 uses Support seed 1 and its own score vector. These are development-set evaluations, not threshold-selection targets.",
        "",
        "| Category | Cond. | AUROC | FIXED05 θ / FPR / FNR / BER | Q90 C64 θ / FPR / FNR / BER | Q90 C128 θ / FPR / FNR / BER | Q90 Cmax-G θ / FPR / FNR / BER |",
        "|---|---|---:|---|---|---|---|",
    ]
    for category, data in summary["categories"].items():
        for condition in ("P0", "P2"):
            values = data["metrics"][condition]
            auroc = values["auc_parity"]["auroc_by_arm"][0]
            cells = []
            for arm in ("FIXED05", "Q90:C64", "Q90:C128", "Q90:Cmax-G"):
                result = values["arms"][arm]
                metrics = result["metrics"]
                cells.append(f"{result['threshold']:.4f} / {_num(metrics['fpr'])} / {_num(metrics['fnr'])} / {_num(metrics['ber'])}")
            lines.append(f"| {category} | {condition} | {auroc:.3f} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Across P0 categories, equal-weight mean BER was 0.349 for FIXED05, 0.368 for Q90-C64, 0.365 for Q90-C128, 0.366 for Q90-Cmax-G, 0.366 for Q90-Cmax-V, 0.367 for Gaussian-Q90 C64, 0.397 for Gaussian-2.5 C64, 0.374 for SELF, and 0.421 for LOSO. These descriptive aggregates conceal substantial category variation and do not define a winner.",
        "",
        "For P0 q90-C64, H exceedance fractions were 0.125 (audiojack), 0.031 (regulator), 0.063 (switch), and 0.063 (rolled_strip_base). H has only 32 group-level observations, so its empirical resolution is 1/32 = 0.03125. P2 q90-C64 values were 0.000, 0.125, 0.063, and 0.063. H is a normal-only development audit, not an independent production sample.",
        "",
        "## Sampling stability and negative findings",
        "",
        "| Condition | q90 size | Mean across-category SD across 3 draws | Mean across-category range |",
        "|---|---|---:|---:|",
    ]
    for condition in ("P0", "P2"):
        for size in ("C32", "C64", "C128"):
            vals = [data["metrics"][condition]["threshold_stability"][f"Q90:{size}"]
                    for data in summary["categories"].values()]
            mean_sd = statistics.mean(row["sample_std"] for row in vals)
            mean_range = statistics.mean(row["range"] for row in vals)
            lines.append(f"| {condition} | {size} | {mean_sd:.5f} | {mean_range:.5f} |")
    lines += [
        "",
        "C128 had lower average draw-to-draw threshold variation than C64 in both Support conditions, but not in every category (regulator P0 is one exception). The largest P0 C64 draw range was 0.0119; the largest C128 range was 0.0065. Cmax-G/Cmax-V values are identical across the three draws because those arms reuse the complete fixed population; their zero draw range is not an uncertainty estimate.",
        "",
        "There is no category-independent operating-point winner. Fixed 0.5 has high FPR in audiojack and switch but lower BER than q90 in regulator and switch; q90 variants behave differently by category. Gaussian-2.5 often drives FPR close to zero while FNR approaches one. SELF/LOSO depend strongly on the Support seed and are diagnostic only. Empirical q90 and Gaussian-Q90 are close in some categories and differ in others.",
        "",
        "All threshold-only arms had exactly identical AUROC within each Support condition. A1 does not test state dependence or learning effects. It therefore cannot establish that C64 is sufficient for stable online recalibration or that an expanded fixed bank explains online adaptation.",
        "",
        "## Runtime, storage and figures",
        "",
        "A1 processed 15,295 TEST images, encoding each once and scoring both frozen Support banks. It also encoded 4,268 Cmax-V/H reference images, 32 raw Support images, and 128 allowed Support augmentation views: 19,723 frozen-backbone forward passes total. The A1 campaign took 2,033 seconds; per-category Stage A wall times were about 479–501 seconds. Peak CUDA allocated memory was about 1.94 GiB and reserved memory about 14.46 GiB. Retained run output was 6.12 MB, below the 4 GiB cap. No full TEST feature cache was retained.",
        "",
        "Figures 1, 2, and 5 are included below. Figures 3, 4, and 6 are not generated because A2/A3a/A3b did not run; no trajectory or reference-comparison values are fabricated.",
        "",
    ]
    for path in figure_paths:
        lines += [f"![{path.stem}](figures_a1/{path.name})", ""]
    lines += [
        "## Component status and interpretation",
        "",
        "- A1: `COMPLETE_VALID` for P0/P1/P2 on the four authorized development categories.",
        "- A2: `INCONCLUSIVE` / not executed. A1 created no persistent test feature cache; the R9 artifacts lack state snapshots required to score C32/C128 under current states. A new shared-trajectory implementation and parity audit are required. Full Cmax rescore remains subject to the registered resource gate.",
        "- A3a: `INCONCLUSIVE` / not executed. No independent C64-versus-Cmax-G calibration-coupled trajectories were generated.",
        "- A3b: `HOLD_BASELINE_FIDELITY`. The native fixed-bank builder realizes 8,192 prototypes, but the registered expanded-reference coreset selection rule and exact equal-byte parity receipt were not frozen before this run. No substitute was introduced.",
        "",
        "A1 does not select C*. The small but generally lower C128 draw variation warrants a registered A2 comparison, while its absence of a consistent H or category-level operating advantage prevents a freeze. Keep the historical C64 condition as the comparator; consider C128 only as a candidate for the next authorized experiment. The researcher retains the final C* decision.",
        "",
        "A1 is a separate initial calibration study. It does not overwrite, invalidate, or extend historical R9/R10/D2 empirical claims. Those results remain conditional on their recorded C64 protocols; A2/A3 evidence would be needed to assess sensitivity of their learned states to calibration choice.",
        "",
        "## Reproducibility and output paths",
        "",
        f"Run command: `D:\\anaconda3\\envs\\torch\\python.exe -m experiments.calibration.a_calibration_scale --run --run-id {run_id}`",
        f"Protected local run: `{run_dir}`",
        f"Integrity audit: `{run_dir / 'integrity_audit.json'}`",
        f"Public results CSV: `{results_path.name}`",
        f"Public contract addendum: `{contract_path.name}`",
        "",
        "The source manifest and all event-level score/identity records remain in the protected local output and are not committed. No Stage 1/2 program, Experiment B/C, confirmation outcome, or mask was accessed.",
        "",
    ]
    return "\n".join(lines)


def generate(run_dir: Path, report_dir: Path) -> None:
    preflight, contract, summary = _verified_payload(run_dir)
    public_contract = build_public_contract(run_dir, report_dir)
    contract_path = report_dir / "EXPERIMENT_A_CONTRACT_A1_ADDENDUM.json"
    results_path = report_dir / "A1_RESULTS_SUMMARY.csv"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    _write_new(contract_path, a.canonical_json(public_contract) + b"\n")
    rows = build_public_results(summary)
    write_results_csv(rows, results_path)
    figure_paths = make_figures(run_dir, report_dir, summary)
    report = build_report(run_dir, report_dir, preflight, summary,
                          figure_paths, contract_path, results_path)
    report_path = report_dir / "REPORT.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    decision_path = report_dir / "CALIBRATION_DECISION_CANDIDATE.md"
    decision = (
        "# Calibration Decision Candidate\n\n"
        "**Candidate: HOLD C* freeze.** A1 found somewhat lower q90 threshold draw variation at C128 than C64 on average (P0 mean sample SD 0.00155 vs 0.00341; P2 0.00099 vs 0.00270), but the improvement is not universal across categories and the H-tail and TEST operating outcomes vary by category. Fixed 0.5, empirical q90, Gaussian-Q90, and support-derived rules each have category-specific tradeoffs.\n\n"
        "Retain historical C64 as the A1 comparator. Carry C128 as a candidate to a separately authorized A2 study because it tended to reduce threshold sampling variation with only 64 additional calibration images. Do not freeze C128 or C64 for Experiments B/C from A1 alone. A2 must test current-state recalibration on shared Greedy and Greedy+C1 trajectories; A3a must test calibration-coupled learning. A3b remains `HOLD_BASELINE_FIDELITY` until its equal-byte expanded coreset rule is preregistered and verified.\n\n"
        "A1's TEST metrics are development evaluation, not a tuning criterion. No independent confirmation, formal risk guarantee, or evidence of superiority is claimed.\n")
    decision_path.write_text(decision, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--report-dir", default=str(a.REPORT_ROOT))
    args = parser.parse_args()
    generate(Path(args.run_dir).resolve(strict=True), Path(args.report_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
