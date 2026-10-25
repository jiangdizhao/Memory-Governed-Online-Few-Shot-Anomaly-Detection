#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E02-R1 held-out probes, E02-R2 legal online threshold calibration.

Original v22, original E01 and E02 are NEVER modified.
R1: reserve label-blind test images before the online stream begins; NEVER
    update memory/controller with probes. Evaluate identical probe images at
    predeclared checkpoints, using common query encodings for both banks.
R2: run matched E02 stream/query tape; one common verified-normal STM path;
    compare frozen, original guarded, normal-quantile and mistake-feedback
    thresholds, using verified labels ONLY after all predictions at t.

These are exploratory protocol-locked pilots, NOT the full v22 BEA system.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from collections import deque
from typing import Any

import numpy as np


def load_deps():
    root = Path(__file__).resolve().parents[1]
    for d in (root, root / "experiments"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    import test_phase3_AIF_simplified_v22 as impl
    import e02_matched_feedback as e02
    import p0_v22_equivalence as e01
    return impl, e02, e01


def fingerprint(items: Any) -> str:
    return hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def split_heldout(stream: list, count: int, seed: int) -> tuple[list, list]:
    """Random split depends ONLY on list size + seed; never inspects labels."""
    import random
    if not 0 < count < len(stream):
        raise ValueError("Probe count must be between 1 and stream_size-1")
    probe_idx = set(random.Random(seed).sample(range(len(stream)), count))
    probe = [item for i, item in enumerate(stream) if i in probe_idx]
    train_stream = [item for i, item in enumerate(stream) if i not in probe_idx]
    probe_paths = {obj.path for obj in probe}
    remaining_paths = {obj.path for obj in train_stream}
    if len(probe_paths) != len(probe) or len(remaining_paths) != len(train_stream):
        raise AssertionError("Duplicate test paths; probe isolation not guaranteed")
    if probe_paths & remaining_paths:
        raise AssertionError("Probe stream overlap")
    return train_stream, probe


def checkpoint_steps(n: int, fractions: tuple[float, ...] = (0., .25, .50, .75, 1.)) -> list[int]:
    if n <= 0:
        raise ValueError("Empty training stream")
    if any(not 0 <= x <= 1 for x in fractions):
        raise ValueError("Invalid checkpoint fraction")
    return sorted(set(min(n, int(round(n * f))) for f in fractions))


def disabled_sidecars(bank):
    bank.enable_ltm_retrieval = False
    bank.enable_ltm_promotion = False
    bank.enable_maturation_buffer_retrieval = False
    bank.enable_maturation_buffer = False
    bank.enable_utility_aware_stm_retention = False
    bank.use_defect_bank = False
    bank.corr_cfg.enabled = False


def startup(args, impl, e02, e01, out):
    init, backbone, bank, controller = e02.init_canonical(args, impl, e01, out)
    canonical_sha = e01.memory_profile(bank, args.class_name)["sha256"]
    static = copy.deepcopy(bank)
    dynamic = copy.deepcopy(bank)
    for b in (static, dynamic):
        disabled_sidecars(b)
        if e01.memory_profile(b, args.class_name)["sha256"] != canonical_sha:
            raise AssertionError("Nonidentical initial Fixed Banks")
    ctrl = copy.deepcopy(controller)
    fixed_theta = float(controller.theta())
    if not math.isfinite(fixed_theta):
        raise AssertionError("Nonfinite canonical initial theta")
    stm_arm = e02.Arm(
        "shared_stm", True, False, dynamic, ctrl,
        scores=[], thresholds=[], predictions=[]
    )
    return init, backbone, static, stm_arm, controller, fixed_theta, canonical_sha


def feature_encode(impl, backbone, sample, args):
    image = impl.load_rgb(sample.path)
    feats, _ = impl.encode_pil(backbone, image, args.feature_fuse)
    return image, [impl.l2_normalize(x) for x in feats]


def score_two(impl, e02, static_bank, arm, image, features, args):
    holder = type("_Holder", (), {"bank": static_bank})()
    s_static, _ = e02.frozen_score(impl, holder, args.class_name, features, image, args)
    s_stm, amap = e02.frozen_score(impl, arm, args.class_name, features, image, args)
    if not math.isfinite(s_static) or not math.isfinite(s_stm):
        raise AssertionError("Non-finite score")
    return s_static, s_stm, amap


def dynamic_prototype_fingerprint(bank, cls: str) -> str:
    """Hash actual STM feature bytes, not only bank size, to exclude probe writes."""
    h = hashlib.sha256()
    for cb in bank.normal_dyn[cls][0]["id"]:
        h.update(cb.protos.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def score_probe_set(impl, e02, args, backbone, static_bank, arm,
                    probes, t: int, fixed_theta: float) -> tuple[dict, list[dict]]:
    y, score_static, score_stm, rows = [], [], [], []
    stm_size_before = stm_total(arm.bank, args.class_name)
    stm_digest_before = dynamic_prototype_fingerprint(arm.bank, args.class_name)
    for sample in probes:
        image, feats = feature_encode(impl, backbone, sample, args)
        sa, sb, _ = score_two(impl, e02, static_bank, arm, image, feats, args)
        # Evaluation labels are consulted after both scores; never for update.
        label = 0 if sample.is_good else 1
        y.append(label)
        score_static.append(sa)
        score_stm.append(sb)
        rows.append({
            "step": t, "path": sample.path, "y_eval_only": label,
            "static_score": sa, "stm_score": sb,
            "score_delta": sb - sa,
        })
    if stm_total(arm.bank, args.class_name) != stm_size_before or \
       dynamic_prototype_fingerprint(arm.bank, args.class_name) != stm_digest_before:
        raise AssertionError("Probe scoring altered STM contents; held-out evaluation leaked state")
    a = e02.metrics(y, score_static, [int(x >= fixed_theta) for x in score_static])
    b = e02.metrics(y, score_stm, [int(x >= fixed_theta) for x in score_stm])
    return {
        "t": t, "fixed_theta": fixed_theta, "stm_prototypes": stm_size_before,
        "normal_probes": int(y.count(0)), "defect_probes": int(y.count(1)),
        "static": a, "stm": b,
        "delta_auroc": (b["auroc"] - a["auroc"]
                        if a["auroc"] is not None and b["auroc"] is not None else None),
        "mean_score_change_normals": float(np.mean([b-a for a,b,y0 in
            zip(score_static,score_stm,y) if y0==0])) if 0 in y else None,
        "mean_score_change_defects": float(np.mean([b-a for a,b,y0 in
            zip(score_static,score_stm,y) if y0==1])) if 1 in y else None
    }, rows


def stm_total(bank, cls) -> int:
    return sum(int(cb.size()) for cb in bank.normal_dyn[cls][0]["id"])


def save_rows(out: Path, filename: str, rows: list[dict]) -> None:
    if not rows:
        raise AssertionError("Empty output CSV")
    with (out / filename).open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


class RollingNormalQuantile:
    """Pre-registered label-only rolling normal quantile with bounded EMA steps.

    Only *verified normal* scores from the query tape enter the window. No
    labels of non-queried items nor held-out probe scores are used. A target
    upper normal quantile is an FPR-control heuristic, NOT guaranteed FPR.
    """
    def __init__(self, theta0: float, quantile=.95, window=64,
                 min_normals=8, alpha=.25, max_step=.01,
                 lower_range=.15, upper_range=.10):
        if not 0 < quantile < 1 or window < min_normals or min_normals < 2:
            raise ValueError("Invalid normal quantile calibration settings")
        self.theta0 = float(theta0)
        self.value = float(theta0)
        self.quantile = float(quantile)
        self.history = deque(maxlen=int(window))
        self.min_normals = int(min_normals)
        self.alpha = float(alpha)
        self.max_step = float(max_step)
        self.lower = float(theta0 - lower_range)
        self.upper = float(theta0 + upper_range)
        self.observed = 0
        self.updates = 0

    def theta(self):
        return float(self.value)

    def observe_labeled(self, score, y):
        if not math.isfinite(score) or y not in (0,1):
            raise ValueError("Untrusted labeled observation")
        self.observed += 1
        if int(y) != 0:
            return
        self.history.append(float(score))
        if len(self.history) < self.min_normals:
            return
        target = float(np.quantile(np.asarray(self.history, dtype=np.float64), self.quantile))
        movement = float(np.clip(self.alpha*(target - self.value), -self.max_step, self.max_step))
        self.value = float(np.clip(self.value + movement, self.lower, self.upper))
        self.updates += 1


class MistakeFeedback:
    """Pre-specified bounded threshold correction for queried classification errors.

    If a queried verified defect is missed, decrease theta; if a queried
    verified normal is falsely flagged, increase theta. Correct decisions
    do not change theta. No access to unqueried labels or test-level ROC.
    """
    def __init__(self, theta0: float, step=.002, lower_range=.15, upper_range=.10):
        if step <= 0:
            raise ValueError("step must be positive")
        self.theta0 = float(theta0)
        self.value = float(theta0)
        self.step = float(step)
        self.lower = float(theta0 - lower_range)
        self.upper = float(theta0 + upper_range)
        self.observed = 0
        self.updates = 0

    def theta(self):
        return float(self.value)

    def observe_labeled(self, score, y):
        if not math.isfinite(score) or y not in (0,1):
            raise ValueError("Untrusted labeled observation")
        self.observed += 1
        predicted_defect = float(score) >= self.value
        delta = 0.
        if y == 0 and predicted_defect:
            delta = self.step
        elif y == 1 and not predicted_defect:
            delta = -self.step
        if delta:
            self.value = float(np.clip(self.value + delta, self.lower, self.upper))
            self.updates += 1


def init_settings(args):
    if args.query_rate < 0 or args.query_rate > 1:
        raise ValueError("Query budget must be in [0,1]")
    if args.probe_count < 1 or args.max_samples < 0:
        raise ValueError("Invalid probe count / stream cap")
    if not 0 < args.rolling_quantile < 1:
        raise ValueError("Invalid rolling quantile")
    if args.rolling_window < args.rolling_min_normals or args.rolling_min_normals < 2:
        raise ValueError("Invalid rolling normal calibration buffer")
    if args.rolling_alpha <= 0 or args.rolling_max_step <= 0 or args.mistake_step <= 0:
        raise ValueError("Invalid calibration step")


def run_r1(args, impl, e02, e01, out):
    print("[R1] Genuine v22 initial state; reserve label-blind held-out probe set.", flush=True)
    init, backbone, static, arm, ctrl, theta0, sha = startup(
        args, impl, e02, e01, out
    )
    ordered = e02.select_stream(impl, args)
    stream, probes = split_heldout(ordered, args.probe_count, args.probe_seed)
    assert not set(x.path for x in stream) & set(x.path for x in probes)
    tape = e02.make_query_tape(len(stream), args.query_rate, args.query_seed)
    steps = checkpoint_steps(len(stream))
    report_rows, probe_rows = [], []
    labels_seen, normal_writes = 0, 0
    probe_manifest = {
        "probe_count": len(probes), "stream_count": len(stream),
        "probe_seed": args.probe_seed,
        "probe_paths_sha256": fingerprint([s.path for s in probes]),
        "stream_paths_sha256": fingerprint([s.path for s in stream]),
        "probe_paths": [s.path for s in probes],
        "stream_probe_disjoint": True, "query_tape_positions_1based": sorted(tape),
        "query_tape_sha256": fingerprint(sorted(tape)),
        "query_count": len(tape), "query_rate_actual": len(tape)/len(stream),
        "label_blind_selection": True, "checkpoint_steps": steps,
        "probe_used_for_updates": False,
    }
    (out / "e02_r1_probe_manifest.json").write_text(
        json.dumps(probe_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for t in [0] + list(range(1,len(stream)+1)):
        if t > 0:
            sample = stream[t-1]
            image, feats = feature_encode(impl, backbone, sample, args)
            s_static, s_stm, amap = score_two(impl, e02, static, arm, image, feats, args)
            if t in tape:
                # Label is read ONLY after pre-update scores.
                y = 0 if sample.is_good else 1
                labels_seen += 1
                if y == 0:
                    e02.write_verified_normal(
                        impl, arm, args.class_name, feats, amap, s_stm, theta0, t
                    )
                    normal_writes += 1
            if args.print_every and t % args.print_every == 0:
                print(f"[R1] online={t}/{len(stream)}, query={labels_seen}", flush=True)
        if t in steps:
            item, rows = score_probe_set(
                impl, e02, args, backbone, static, arm, probes, t, theta0
            )
            report_rows.append(item)
            probe_rows.extend(rows)
            print(f"[R1] probe checkpoint t={t}, STM={item['stm_prototypes']}, "
                  f"AUROC static={item['static']['auroc']} STM={item['stm']['auroc']}", flush=True)
    if len(report_rows) != len(steps) or labels_seen != len(tape):
        raise AssertionError("Probe checkpoints / query tape mismatch")
    baseline = report_rows[0]
    if baseline["t"] != 0 or baseline["static"]["auroc"] != baseline["stm"]["auroc"]:
        raise AssertionError("At t=0 both branches must have the same probe AUROC")
    if e01.memory_profile(static, args.class_name)["sha256"] != sha:
        raise AssertionError("Static initial Fixed Bank mutated")
    if e01.memory_profile(arm.bank, args.class_name)["sha256"] != sha:
        raise AssertionError("STM changed immutable Fixed Bank")
    if ctrl.theta() != theta0 or arm.controller.theta() != theta0:
        raise AssertionError("R1 must never update thresholds")
    initial = [row for row in probe_rows if row["step"] == 0]
    last = [row for row in probe_rows if row["step"] == len(stream)]
    if len(initial) != len(last):
        raise AssertionError("Different held-out probes at t0 and final")
    drift = max(abs(x["static_score"]-y["static_score"]) for x,y in zip(initial,last))
    if drift > args.score_tolerance:
        raise AssertionError(f"Static probe scores drifted {drift:.3g} across checkpoints")
    report = {
        "schema": "e02.r1.heldout.probes.v1", "class": args.class_name,
        "initial": {"support_paths": init["support_paths_per_mode"]["0"],
                    "fixed_bank_sha256": sha, "theta0": theta0},
        "probe_protocol": probe_manifest, "checkpoints": report_rows,
        "normal_stm_update_calls": normal_writes,
        "final_stm_prototypes": stm_total(arm.bank, args.class_name),
        "static_probe_replay_max_delta": drift,
        "source_protocol": "canonical v22 + E02 verified-normal STM, original theta frozen",
        "limitations": [
            "Held-out subset is a label-blind partition of Real-IAD test samples, not an external dataset.",
            "Probe labels used only for offline evaluation; probes never enter queried feedback or memory.",
            "Same probes tested repeatedly as snapshots; do not tune controller based on probe outcomes.",
            "Queries drawn label-blind from remaining stream; STM only receives queried verified normals.",
        ],
        "settings": vars(args)
    }
    save_rows(out, "e02_r1_probe_scores.csv", probe_rows)
    (out/"e02_r1_probe_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[R1] Report:", out/"e02_r1_probe_report.json", flush=True)
    return report


def run_r2(args, impl, e02, e01, out):
    print("[R2] Canonical v22 + ONE shared verified-normal STM; controller alternatives.", flush=True)
    init, backbone, static, arm, guard, theta0, sha = startup(
        args, impl, e02, e01, out
    )
    controllers = {
        "stm_frozen": None,
        "stm_guarded": copy.deepcopy(guard),
        "stm_normal_quantile": RollingNormalQuantile(
            theta0, quantile=args.rolling_quantile, window=args.rolling_window,
            min_normals=args.rolling_min_normals, alpha=args.rolling_alpha,
            max_step=args.rolling_max_step),
        "stm_mistake_feedback": MistakeFeedback(theta0, step=args.mistake_step),
    }
    stream = e02.select_stream(impl, args)
    tape = e02.make_query_tape(len(stream), args.query_rate, args.query_seed)
    tape_manifest = {
        "n": len(stream), "query_positions_1based": sorted(tape),
        "query_seed": args.query_seed, "query_rate": len(tape)/len(stream),
        "tape_sha256": fingerprint(sorted(tape)),
        "stream_paths_sha256": fingerprint([s.path for s in stream]),
        "label_blind": True,
    }
    (out/"e02_r2_query_tape.json").write_text(
        json.dumps(tape_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = ["static", *controllers]
    predictions = {k:[] for k in columns}
    scores = {k:[] for k in columns}
    theta_history = {k:[] for k in columns}
    rows, labels, queried_normals, queried_defects = [], [], 0, 0
    for t, sample in enumerate(stream, start=1):
        image, feats = feature_encode(impl, backbone, sample, args)
        s_static, s_stm, amap = score_two(impl, e02, static, arm, image, feats, args)
        current = {}
        # Snapshot every controller theta BEFORE label reveal/update.
        for name in columns:
            score = s_static if name == "static" else s_stm
            controller = controllers.get(name)
            theta = theta0 if controller is None else float(controller.theta())
            prediction = int(score >= theta)
            if not math.isfinite(theta):
                raise AssertionError("Nonfinite online threshold")
            current[name] = (score, theta, prediction)
            predictions[name].append(prediction)
            scores[name].append(score)
            theta_history[name].append(theta)
        # Identical anomaly-score trajectory across ALL adaptive threshold arms.
        if max(abs(current[k][0]-s_stm) for k in controllers) > 0:
            raise AssertionError("Threshold controller affected shared anomaly scores")
        asked = t in tape
        # y is evaluation-only except at predetermined query positions.
        y = 0 if sample.is_good else 1
        labels.append(y)
        rows.append({
            "t": t, "path": sample.path, "y_eval_only": y, "queried": int(asked),
            **{f"{k}_{suffix}": current[k][j]
               for k in columns for suffix,j in [("score",0),("theta",1),("pred",2)]}
        })
        if asked:
            if y == 0:
                queried_normals += 1
                # STM update always uses the FROZEN reference theta0 and
                # shared score/map. This prevents each adaptive threshold
                # from changing which memory prototypes get written.
                e02.write_verified_normal(
                    impl, arm, args.class_name, feats, amap, s_stm, theta0, t
                )
            else:
                queried_defects += 1
            for name, controller in controllers.items():
                if controller is not None:
                    controller.observe_labeled(s_stm, y)
        if args.print_every and t % args.print_every == 0:
            print(f"[R2] online={t}/{len(stream)} query={queried_normals+queried_defects}",
                  flush=True)
    if queried_normals + queried_defects != len(tape):
        raise AssertionError("Wrong queried label count")
    for name, controller in controllers.items():
        if controller is None:
            continue
        received = (controller.observed if hasattr(controller, "observed")
                    else controller.n_normals_labeled + controller.n_defects_labeled)
        if received != len(tape):
            raise AssertionError(f"Controller {name} received {received} instead of {len(tape)} labels")
    if e01.memory_profile(static,args.class_name)["sha256"] != sha or \
       e01.memory_profile(arm.bank,args.class_name)["sha256"] != sha:
        raise AssertionError("Fixed Bank altered during calibration experiment")
    if stm_total(static,args.class_name) != 0:
        raise AssertionError("Static normal memory grew")
    if any(len(predictions[k]) != len(stream) for k in columns):
        raise AssertionError("Missing predictions")
    summary = {}
    for name in columns:
        ctrl = controllers.get(name)
        summary[name] = {
            **e02.metrics(labels, scores[name], predictions[name]),
            "initial_theta": theta0,
            "final_theta": theta0 if ctrl is None else float(ctrl.theta()),
            "theta_min_prequential": float(min(theta_history[name])),
            "theta_max_prequential": float(max(theta_history[name])),
            "theta_unique_values": len(set(theta_history[name])),
            "labeled_observations": (0 if ctrl is None else
                                    (ctrl.observed if hasattr(ctrl,"observed")
                                     else ctrl.n_normals_labeled+ctrl.n_defects_labeled)),
            "controller_updates": getattr(ctrl,"updates",None) if ctrl is not None else 0,
        }
    if summary["static"]["auroc"] != summary["stm_frozen"]["auroc"]:
        pass  # Different bank scores are expected.
    if len({summary[k]["auroc"] for k in controllers}) != 1:
        raise AssertionError("A threshold-only strategy changed AUROC")
    report = {
        "schema": "e02.r2.verified.threshold.calibration.v1", "class": args.class_name,
        "initial": {"support_paths": init["support_paths_per_mode"]["0"],
                    "fixed_bank_sha256":sha, "theta0":theta0},
        "query_tape": tape_manifest,
        "observed_feedback": {"normals":queried_normals, "defects":queried_defects,
                              "total":len(tape)},
        "shared_stm": {"write_calls":arm.verified_normal_update_calls,
                       "final_prototypes":stm_total(arm.bank,args.class_name)},
        "arms":summary,
        "invariants": {"single_shared_STM_trajectory_for_calibrators":True,
                       "predictions_before_feedback":True,
                       "equal_label_positions":True, "no_probe_or_evaluation_label_tuning":True,
                       "fixed_bank_equal_and_unchanged":True},
        "limitations": [
            "Online calibrators are pre-specified exploratory heuristics, not hyperparameter-tuned on test outcomes.",
            "Compared methods share a single E02 verified-normal STM score stream; neither BEA nor original full v22.",
            "Rolling normal q is a target percentile of labeled-normal scores, not a guaranteed realized FPR.",
            "Mistake feedback reacts only to queried supervised classification errors; class imbalance can cause instability.",
            "Calibration strategies do not modify anomaly scores, hence STM-arm AUROC/AP must match.",
            "Always report FPR and FNR jointly; heldout R1 probes must not be used to tune these parameters.",
        ],
        "settings":vars(args)
    }
    save_rows(out, "e02_r2_prequential_scores.csv", rows)
    (out/"e02_r2_calibration_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[R2] FINAL",json.dumps(summary,indent=2),flush=True)
    print("[R2] Report:",out/"e02_r2_calibration_report.json",flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=["r1","r2"], required=False)
    p.add_argument("--dataset-root")
    p.add_argument("--json-dir")
    p.add_argument("--class-name",default="audiojack")
    p.add_argument("--device",default="cuda")
    p.add_argument("--support-seed",type=int,default=0)
    p.add_argument("--legacy-seed",type=int,default=0)
    p.add_argument("--stream-seed",type=int,default=0)
    p.add_argument("--query-seed",type=int,default=37)
    p.add_argument("--query-rate",type=float,default=.03)
    p.add_argument("--max-samples",type=int,default=100,
                   help="0 means full class evaluation stream")
    p.add_argument("--probe-count",type=int,default=16,
                   help="R1: held-out test images never included in online stream")
    p.add_argument("--probe-seed",type=int,default=97)
    p.add_argument("--fixed-capacity",type=int,default=25000)
    p.add_argument("--fixed-per-image",type=int,default=5000)
    p.add_argument("--augmentations",type=int,default=4)
    p.add_argument("--include-identity",action="store_true")
    p.add_argument("--feature-fuse",default="concat",choices=["concat","none","sum"])
    p.add_argument("--image-aggregation",default="max",
                   choices=["max","topk_mean","spiky_quantile"])
    p.add_argument("--topk-patches",type=int,default=32)
    p.add_argument("--knn",type=int,default=5)
    p.add_argument("--math-sdpa",action="store_true")
    p.add_argument("--score-tolerance",type=float,default=1e-4)
    p.add_argument("--rolling-quantile",type=float,default=.95)
    p.add_argument("--rolling-window",type=int,default=64)
    p.add_argument("--rolling-min-normals",type=int,default=8)
    p.add_argument("--rolling-alpha",type=float,default=.25)
    p.add_argument("--rolling-max-step",type=float,default=.01)
    p.add_argument("--mistake-step",type=float,default=.002)
    p.add_argument("--print-every",type=int,default=100)
    p.add_argument("--out-dir",default="outputs/e02_r1_r2")
    p.add_argument("--self-test",action="store_true")
    return p


def main():
    args = parser().parse_args()
    if args.self_test:
        fake = [type("Item",(),{"path":str(i),"is_good":bool(i%2)})() for i in range(100)]
        stream,probe=split_heldout(fake,16,97)
        assert len(stream)==84 and len(probe)==16
        assert checkpoint_steps(84)==[0,21,42,63,84]
        q=RollingNormalQuantile(.55,min_normals=2)
        q.observe_labeled(.40,0); assert q.theta()==.55
        q.observe_labeled(.41,0); assert q.theta()<.55
        m=MistakeFeedback(.55,step=.002)
        m.observe_labeled(.50,1); assert m.theta()<.55
        m.observe_labeled(.70,0); assert m.theta()==.55
        print("E02-R1/R2 offline self-test: PASS")
        return
    if not args.experiment or not args.dataset_root:
        raise SystemExit("Specify --experiment r1|r2 and --dataset-root")
    if not args.json_dir:
        args.json_dir=str(Path(args.dataset_root)/"jsons"/"realiad_jsons")
    init_settings(args)
    import torch
    if args.math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    impl,e02,e01=load_deps()
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    if args.experiment=="r1":
        run_r1(args,impl,e02,e01,out)
    else:
        run_r2(args,impl,e02,e01,out)


if __name__=="__main__":
    main()
