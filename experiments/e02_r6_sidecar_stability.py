#!/usr/bin/env python3
"""R6 audit-only Sidecar failure attribution and extreme-stream stress tests.

Reuses the exact R5-D initialization, query, STM, pacing and Sidecar paths.
No change to v22, R3, R4 or R5. Offline label-aware stream permutations are
EXPLICIT STRESS TESTS, not a deployable policy or naturally occurring stream.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import e02_r5_d_temporal_pacing as r5
r4 = r5.r4


def order_stream(stream, order="native", block_size=256):
    """Offline-only stream construction; never exposed to a query decision."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if order not in {"native", "normal_first", "defect_first", "alternating_blocks"}:
        raise ValueError(f"Unknown stress order: {order}")
    stream = list(stream)
    if order == "native":
        return stream
    normals = [x for x in stream if x.is_good]
    defects = [x for x in stream if not x.is_good]
    if order == "normal_first":
        result = normals + defects
    elif order == "defect_first":
        result = defects + normals
    else:
        result = []
        while normals or defects:
            result.extend(normals[:block_size])
            del normals[:block_size]
            result.extend(defects[:block_size])
            del defects[:block_size]
    if Counter(x.path for x in result) != Counter(x.path for x in stream):
        raise AssertionError("Offline ordering changed stream membership")
    return result


def class_windows(rows, width=256):
    """Independent offline evaluation; undefined class-conditional risks are null."""
    ans = []
    for start in range(0, len(rows), width):
        part = rows[start:start + width]
        n0 = sum(int(r["y_eval_only"]) == 0 for r in part)
        n1 = len(part) - n0
        q0 = sum(int(r["queried"]) and int(r["y_eval_only"]) == 0 for r in part)
        q1 = sum(int(r["queried"]) and int(r["y_eval_only"]) == 1 for r in part)
        entry = {"first_step": start + 1, "last_step": start + len(part),
                 "normal_count": n0, "defect_count": n1,
                 "queried_normals": q0, "queried_defects": q1,
                 "true_label_diversity": int(n0 > 0) + int(n1 > 0)}
        for arm in ("frozen", "guarded", "balanced_rescue", "mistake_feedback"):
            fp = sum(int(r["y_eval_only"]) == 0 and int(r[f"{arm}_pred"]) == 1
                     for r in part)
            fn = sum(int(r["y_eval_only"]) == 1 and int(r[f"{arm}_pred"]) == 0
                     for r in part)
            fpr = fp / n0 if n0 else None
            fnr = fn / n1 if n1 else None
            entry[arm] = {"fp": fp, "fn": fn, "fpr": fpr, "fnr": fnr,
                          "balanced_accuracy": (1 - (fpr + fnr) / 2)
                          if fpr is not None and fnr is not None else None}
        ans.append(entry)
    return ans


def _jsonable(obj):
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return _jsonable(obj.item())
    return str(obj)


class SidecarAudit:
    """Observe the actual controller methods without re-evaluating or editing them.

    A balanced update may be triggered by either a forced FP/FN rescue or
    update_every_labeled. Capture actual candidate/rate/objective calls and
    the exact before/after state. It is NOT an alternative threshold policy.
    """
    def __init__(self, controller):
        self.controller = controller
        self.events = []
        self.step = 0
        self.label = None
        self.score = None
        self._install()

    def _install(self):
        ctrl = self.controller
        old_observe = ctrl.observe_labeled
        old_update = ctrl._maybe_balanced_update_theta

        def observe(score, label):
            self.label = int(label)
            self.score = float(score)
            try:
                return old_observe(score, label)
            finally:
                self.label = None
                self.score = None

        def update(force_hint=None):
            before = {"theta": float(ctrl.theta()),
                      "cooldown": int(ctrl.balanced_cooldown_remaining),
                      "labeled_since_update": int(ctrl.labeled_since_update),
                      "normal_buf": len(ctrl.bufN),
                      "defect_buf": len(ctrl.bufA),
                      "recent_labels": len(ctrl.recent_labeled),
                      "recent_normal": sum(int(y) == 0 for _, y in ctrl.recent_labeled),
                      "recent_defect": sum(int(y) == 1 for _, y in ctrl.recent_labeled)}
            original_candidates = ctrl._candidate_thetas_balanced
            original_rates = ctrl._balanced_rates
            original_objective = ctrl._balanced_objective
            sampled = {"candidates": [], "rates": [], "objectives": []}

            def candidates(mode, anchors):
                result = original_candidates(mode, anchors)
                sampled["candidates"].append({
                    "mode": str(mode), "theta": list(map(float, result))})
                return result

            def rates(theta):
                result = original_rates(theta)
                sampled["rates"].append({"theta": float(theta),
                                         "risk": _jsonable(result)})
                return result

            def objective(theta, risk):
                value = original_objective(theta, risk)
                sampled["objectives"].append({"theta": float(theta),
                                              "value": float(value)})
                return value

            ctrl._candidate_thetas_balanced = candidates
            ctrl._balanced_rates = rates
            ctrl._balanced_objective = objective
            error = None
            try:
                return old_update(force_hint=force_hint)
            except Exception as exc:
                error = repr(exc)
                raise
            finally:
                ctrl._candidate_thetas_balanced = original_candidates
                ctrl._balanced_rates = original_rates
                ctrl._balanced_objective = original_objective
                after_theta = float(ctrl.theta())
                detail = _jsonable(getattr(ctrl, "last_eval", {}))
                phase = detail.get("status", "unknown") if isinstance(detail, dict) else "unknown"
                if error is not None:
                    outcome = "exception"
                elif abs(after_theta - before["theta"]) > 1e-12:
                    outcome = "accepted"
                elif phase == "balanced_rescue_cooldown":
                    outcome = "cooldown_blocked"
                elif phase == "balanced_insufficient_buffers":
                    outcome = "insufficient_buffers"
                elif sampled["candidates"]:
                    outcome = "evaluated_no_move"
                else:
                    outcome = "no_candidates"
                self.events.append({
                    "event_index": len(self.events) + 1, "step": self.step,
                    "feedback_label": self.label, "feedback_score": self.score,
                    "force_hint": force_hint or "periodic",
                    "before": before,
                    "after": {"theta": after_theta,
                              "cooldown": int(ctrl.balanced_cooldown_remaining),
                              "last_mode": str(getattr(ctrl, "last_mode", "")),
                              "last_accept": bool(getattr(ctrl, "last_accept", False))},
                    "outcome": outcome, "last_eval": detail,
                    "actual_calls": sampled, "exception": error})

        ctrl.observe_labeled = observe
        ctrl._maybe_balanced_update_theta = update


def summarize_events(events):
    counts = Counter(e["outcome"] for e in events)
    triggers = Counter(e["force_hint"] for e in events)
    return {"n_update_attempts": len(events),
            "outcome_counts": dict(counts), "trigger_counts": dict(triggers),
            "accepted_steps": [e["step"] for e in events if e["outcome"] == "accepted"],
            "last_accepted_step": max((e["step"] for e in events
                                       if e["outcome"] == "accepted"), default=None),
            "cooldown_blocked": counts["cooldown_blocked"],
            "candidate_evaluations": sum(len(e["actual_calls"]["objectives"])
                                         for e in events)}


PARITY_INTS = ("t", "queried", "y_eval_only", "native_proposal",
               "bootstrap_active", "bootstrap_coverage_normal_count",
               "static_pred", "frozen_pred", "guarded_pred",
               "balanced_rescue_pred", "mistake_feedback_pred")
PARITY_FLOATS = ("query_cost_pre", "stm_score", "static_score",
                 "static_theta", "frozen_theta", "guarded_theta",
                 "balanced_rescue_theta", "mistake_feedback_theta")
PARITY_STRS = ("path", "query_reason", "budget_reason", "score_region",
               "region_selection", "policy")


def compare_reference_csv(rows, path, tolerance=1e-7):
    """Full stepwise R5 parity, not a final aggregate-only comparison."""
    count = 0
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for old in csv.DictReader(fh):
            if count >= len(rows):
                raise AssertionError("R5 CSV longer than R6 trajectory")
            row = rows[count]
            for k in PARITY_INTS:
                if int(old[k]) != int(row[k]):
                    raise AssertionError(f"R5 parity mismatch t={count+1}: {k}")
            for k in PARITY_STRS:
                if str(old[k]) != str(row[k]):
                    raise AssertionError(f"R5 parity mismatch t={count+1}: {k}")
            for k in PARITY_FLOATS:
                if abs(float(old[k]) - float(row[k])) > tolerance:
                    raise AssertionError(f"R5 parity mismatch t={count+1}: {k}")
            count += 1
    if count != len(rows):
        raise AssertionError(f"R5 CSV {count} rows != R6 {len(rows)}")
    return {"passed": True, "n_compared": count, "tolerance": tolerance,
            "fields_compared": list(PARITY_INTS + PARITY_STRS + PARITY_FLOATS),
            "reference": str(path)}


def check_summary_reference(old, current, *, native):
    if old["schema"] not in ("e02.r5.d.time_only.v1", "e02.r6.sidecar.audit.v1"):
        raise AssertionError("Expected a native R5-D or R6-D summary")
    if old["schema"] == "e02.r6.sidecar.audit.v1" and old["stream_order"] != "native":
        raise AssertionError("Stress trajectory cannot serve as native reference")
    for field in ("class_name", "total_budget_each", "support_paths",
                  "canonical_fixed_sha256", "canonical_theta0"):
        if old[field] != current[field]:
            raise AssertionError(f"R5/R6 initial state mismatch: {field}")
    if old["n_stream"] != current["n_stream"]:
        raise AssertionError("Different stream length")
    if native and old["stream_paths_sha256"] != current["stream_paths_sha256"]:
        raise AssertionError("Native stream path ordering changed")
    for key in r5.REFERENCE_KEYS:
        if old["parameters"][key] != current["parameters"][key]:
            raise AssertionError(f"R5/R6 configuration mismatch: {key}")
    if native:
        prev = old["schedules"]["D"]
        now = current["schedules"]["D"]
        for field in ("query_indices_1based", "queried_normals", "queried_defects",
                      "stm_prototypes", "bootstrap_steps", "budget_forced_terminal"):
            if prev[field] != now[field]:
                raise AssertionError(f"R5/R6 native schedule mismatch: {field}")
        for arm in prev["arm_metrics"]:
            for metric in ("auroc", "fpr", "fnr", "balanced_accuracy",
                           "theta_final", "theta_changes"):
                if abs(prev["arm_metrics"][arm][metric] -
                       now["arm_metrics"][arm][metric]) > 1e-9:
                    raise AssertionError(f"R5/R6 metric mismatch: {arm}.{metric}")
        if prev["sidecar_moves"] != now["sidecar_moves"]:
            raise AssertionError("R5/R6 threshold move trace mismatch")
    return {"passed": True, "reference": "R5-D", "native_full_parity": bool(native)}


class ShadowRescore:
    """Optional evaluation-only re-scoring of a small number of queried anchors.

    Cache frozen patch descriptors on CPU; never add the anchor to memory
    except through the original R4 queried-normal writer. No feedback to policy.
    """
    def __init__(self, cap, every):
        self.cap = cap
        self.every = every
        self.anchors = []
        self.snapshots = []

    def add(self, row, feats, image, stm_version):
        if not int(row["queried"]) or len(self.anchors) >= self.cap:
            return
        self.anchors.append({
            "origin_step": int(row["t"]), "label": int(row["y_eval_only"]),
            "score_then": float(row["stm_score"]), "memory_version": stm_version,
            "features": [f.detach().to("cpu").clone() for f in feats],
            "image": image.copy()})

    def checkpoint(self, t, total, e02, v22, state, args):
        if not self.anchors or (t % self.every and t != total):
            return
        import torch
        for anchor in self.anchors:
            feats = [z.to(args.device) for z in anchor["features"]]
            with torch.no_grad():
                score_now, _ = e02.frozen_score(
                    v22, state["arm"], args.class_name,
                    feats, anchor["image"], args)
            self.snapshots.append({
                "checkpoint": t, "origin_step": anchor["origin_step"],
                "label": anchor["label"],
                "score_then": anchor["score_then"], "score_now": float(score_now),
                "delta": float(score_now) - anchor["score_then"],
                "memory_version_then": anchor["memory_version"],
                "memory_version_now": state["arm"].verified_normal_update_calls,
                "offline_only": True})


def _save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main_run(args):
    import torch
    v22, e02, ext, e01 = r4.dependencies()
    if args.math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    # R4 reference is optional for other categories: canonical v22 + R5 D
    # primitives are still used, with native R6 parity as the local control.
    if args.reference_r4_summary:
        r5.read_r4_reference(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    canonical, backbone, bank, theta = e02.init_canonical(args, v22, e01, out)
    sha = e01.memory_profile(bank, args.class_name)["sha256"]
    native_stream = list(e02.select_stream(v22, args))
    stream = order_stream(native_stream, args.stream_order, args.block_size)
    args.max_steps_effective = len(stream)
    arm = r5.make_d(v22, e02, ext, e01, bank, canonical, backbone, args, sha, theta)
    audit = SidecarAudit(arm["controllers"]["balanced_rescue"])
    shadow = (ShadowRescore(args.rescore_anchor_cap, args.rescore_every)
              if args.rescore_every and args.rescore_anchor_cap else None)
    rows, truth = [], []
    for t, sample in enumerate(stream, 1):
        audit.step = t
        image = v22.load_rgb(sample.path)
        patches, global_feat = v22.encode_pil(backbone, image, args.feature_fuse)
        features = [v22.l2_normalize(z) for z in patches]
        # Reuse untouched R4 state step: query precedes label disclosure.
        r4.step_state(v22, e02, arm, t=t, sample=sample, image=image,
                      features=features, global_feat=global_feat,
                      args=args, evaluation_rows=rows)
        truth.append(0 if sample.is_good else 1)
        if shadow is not None:
            shadow.add(rows[-1], features, image,
                       arm["arm"].verified_normal_update_calls)
            shadow.checkpoint(t, len(stream), e02, v22, arm, args)
        if args.print_every and t % args.print_every == 0:
            print(f"[R6] {t}/{len(stream)}, queried={arm['budget'].used}/"
                  f"{arm['budget'].K}, sidecar_attempts={len(audit.events)}",
                  flush=True)
    if len(rows) != len(stream):
        raise AssertionError("Missing R6 prequential records")
    result = r4.report_state(e02, ext, e01, arm, truth, args)
    result["stability_windows"] = r4.stability_windows(
        rows, stride=args.stability_window)
    report = {
        "schema": "e02.r6.sidecar.audit.v1",
        "class_name": args.class_name, "n_stream": len(stream),
        "total_budget_each": r4.validated_equal_budget({"D": arm}),
        "stream_order": args.stream_order,
        "block_size": args.block_size,
        "stream_paths_sha256": r4.sha([x.path for x in stream]),
        "original_stream_paths_sha256": r4.sha([x.path for x in native_stream]),
        "support_paths": canonical["support_paths_per_mode"]["0"],
        "canonical_fixed_sha256": sha, "canonical_theta0": float(theta.theta()),
        "schedules": {"D": result},
        "parameters": {k: v for k, v in vars(args).items()
                       if k != "max_steps_effective"},
        "audit": summarize_events(audit.events),
        "label_windows": class_windows(rows, args.stability_window),
        "scope": [
            "Audit-only wrappers; same original R5 D policy, memory and Sidecars.",
            "Extreme orders use ground truth OFFLINE to build a fixed test stream.",
            "All online query actions occur before current-step label disclosure.",
            "R3-R5 histories and original v22 are unchanged.",
            "STM-only, not historical correction/LTM/maturation full v22.",
            "R6 risk windows use evaluation labels OFFLINE and never influence updates.",
            "Any single-class window risk and balanced accuracy are undefined (null).",
        ]
    }
    if args.reference_native_summary:
        old = json.loads(Path(args.reference_native_summary).read_text(encoding="utf-8"))
        report["reference_check"] = check_summary_reference(
            old, report, native=args.stream_order == "native")
    if args.reference_native_csv and args.stream_order == "native":
        report["prequential_parity"] = compare_reference_csv(
            rows, args.reference_native_csv, args.parity_tolerance)
    if args.require_parity:
        if args.stream_order != "native":
            raise AssertionError("R5 parity only meaningful with native order")
        if not report.get("reference_check", {}).get("native_full_parity"):
            raise AssertionError("Missing R5 summary parity")
        if not report.get("prequential_parity", {}).get("passed"):
            raise AssertionError("Missing full R5 CSV parity")
    if args.stream_order != "native":
        if not args.reference_native_summary or not args.reference_native_csv:
            raise AssertionError("Stress ordering needs R5 summary and native CSV")
        # Ensure exact membership. Order is intentionally different.
        with Path(args.reference_native_csv).open(encoding="utf-8", newline="") as fh:
            previous = Counter(row["path"] for row in csv.DictReader(fh))
        if previous != Counter(sample.path for sample in stream):
            raise AssertionError("Extreme stream differs in membership from R5")
        report["stress_membership_check"] = {"passed": True,
                                              "same_samples_as_native": True}
    # Only write outputs once all invariants, parity and membership checks pass.
    _save_csv(out / "r6_prequential.csv", rows)
    with (out / "r6_sidecar_events.jsonl").open(
            "w", encoding="utf-8") as fh:
        for event in audit.events:
            fh.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    if shadow is not None:
        _save_csv(out / "r6_shadow_rescore.csv", shadow.snapshots)
    (out / "r6_summary.json").write_text(
        json.dumps(_jsonable(report), indent=2, ensure_ascii=False,
                   allow_nan=False), encoding="utf-8")
    print(json.dumps({"result": "PASS", "class": args.class_name,
                      "order": args.stream_order, "queries": result["queries"],
                      "balanced_steps": report["audit"]["accepted_steps"],
                      "outcomes": report["audit"]["outcome_counts"],
                      "parity": report.get("prequential_parity"),
                      "output": str(out)}, indent=2), flush=True)


def parser():
    p = r5.parser()
    p.description = __doc__
    p.set_defaults(out_dir="outputs/e02_r6_audiojack_native",
                   variants="D")
    p.add_argument("--stream-order", default="native",
                   choices=["native", "normal_first", "defect_first",
                            "alternating_blocks"])
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--reference-native-summary", "--reference-r5-summary",
                   dest="reference_native_summary", default=None)
    p.add_argument("--reference-native-csv", "--reference-r5-csv",
                   dest="reference_native_csv", default=None)
    p.add_argument("--require-parity", action="store_true")
    p.add_argument("--parity-tolerance", type=float, default=1e-7)
    p.add_argument("--rescore-every", type=int, default=0,
                   help="Optional evaluation-only re-score interval; 0 disables")
    p.add_argument("--rescore-anchor-cap", type=int, default=0,
                   help="Number of queried anchors held on CPU; 0 disables")
    return p


def self_test():
    from types import SimpleNamespace
    stream = [SimpleNamespace(path=str(i), is_good=(i % 3 != 0))
              for i in range(30)]
    for mode in ("native", "normal_first", "defect_first", "alternating_blocks"):
        ordered = order_stream(stream, mode, block_size=4)
        assert Counter(x.path for x in stream) == Counter(x.path for x in ordered)
        if mode == "native":
            assert [x.path for x in ordered] == [x.path for x in stream]
        if mode == "normal_first":
            assert all(x.is_good for x in ordered[:20])
        if mode == "defect_first":
            assert all(not x.is_good for x in ordered[:10])
    rows = [
        {"y_eval_only": 0, "queried": 0, **{f"{a}_pred": 0 for a in
         ("frozen", "guarded", "balanced_rescue", "mistake_feedback")}},
        {"y_eval_only": 0, "queried": 1, **{f"{a}_pred": 0 for a in
         ("frozen", "guarded", "balanced_rescue", "mistake_feedback")}},
    ]
    assert class_windows(rows)[0]["balanced_rescue"]["fnr"] is None
    assert class_windows(rows)[0]["balanced_rescue"]["balanced_accuracy"] is None
    print("R6 offline self-test: PASS")


if __name__ == "__main__":
    args = parser().parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.dataset_root:
            raise SystemExit("--dataset-root required")
        if not args.json_dir:
            args.json_dir = str(Path(args.dataset_root)/"jsons"/"realiad_jsons")
        if args.variants != "D" or args.max_samples < 0:
            raise SystemExit("R6 only supports R5-D, max_samples >= 0")
        if not 0 < args.query_rate < 1 or args.block_size < 1:
            raise SystemExit("Invalid budget or stress block size")
        if args.rescore_every < 0 or args.rescore_anchor_cap < 0:
            raise SystemExit("Invalid shadow rescore settings")
        if args.rescore_every == 0 and args.rescore_anchor_cap != 0:
            raise SystemExit("Set both shadow flags, or neither")
        if args.rescore_every > 0 and args.rescore_anchor_cap == 0:
            raise SystemExit("Set both shadow flags, or neither")
        if args.require_parity and args.stream_order != "native":
            raise SystemExit("Require-parity only applies to native order")
        main_run(args)
