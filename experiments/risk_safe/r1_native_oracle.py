#!/usr/bin/env python3
"""R1 retrospective admission-oracle diagnostic on Real-IAD native v22.

A query event may add one predeclared fixed-source normal-patch proposal to
the state-dependent Greedy bank, or reject it. EXACT enumerates every subset;
BEAM finds a heuristic feasible sequence without global-optimality claims.

Probe-A search labels are knowingly accessed by the OFFLINE ORACLE ONLY.
A disjoint Probe-A audit split is evaluation-only; Probe B is untouched.
This is NOT a realistic online query controller or deployable policy.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

from . import preflight as pf
from . import d1b_protocol as d1b
from . import d1d_native_smoke as d1d
from .d2_fixed_threshold import fit_fixed_normal_thresholds
from .native_bridge import (dynamic_codebooks, extract_v22_candidates,
                            normalized_native_candidate_rows, native_score,
                            snapshot, NativeParityError)
from .r0_opportunity import frozen_r0_tape
from .r1_oracle_core import (DecisionState, R1ProtocolError, evaluate_scores,
                            run_exact, run_beam, frontier_by_admission_count)

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
import p0_protocol_audit as p0
import v02_architecture_core as v02
import v02_memory as memory

SCHEMA = "risk_safe.r1.native_admission_oracle.v1"
DEFAULT_ROOT = "outputs/r1_oracle_20261028"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default=pf.DEFAULT_DATASET_ROOT)
    p.add_argument("--json-dir")
    p.add_argument("--class-name", default="audiojack", choices=d1b.DEV_CLASSES)
    p.add_argument("--support-seed", type=int, default=0, choices=d1b.SUPPORT_SEEDS)
    p.add_argument("--query-seed", type=int, default=37, choices=d1b.QUERY_SEEDS)
    p.add_argument("--budget", type=int, default=16,
                   choices=tuple(sorted(d1b.MAIN_BUDGETS +
                                        d1b.SENSITIVITY_BUDGETS)))
    p.add_argument("--capacity", type=int, default=128)
    p.add_argument("--mode", choices=("exact", "beam"), default="exact")
    p.add_argument("--beam-width", type=int, default=4)
    p.add_argument("--max-leaves", type=int, default=4096)
    p.add_argument("--max-normal-events", type=int, default=0,
                   help="0 full B. Other values are explicitly incomplete prefix SMOKES.")
    p.add_argument("--search-groups-per-type", type=int, default=4)
    p.add_argument("--lambda-defect", type=float, default=1.)
    p.add_argument("--theta-mode", choices=("loso", "train64-q90"),
                   default="train64-q90")
    p.add_argument("--probe-cache-device", choices=("cpu", "cuda"), default="cuda",
                   help="Only caches search Probe-A image feature tensors; large GPUs recommended")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--math-sdpa", action="store_true")
    p.add_argument("--d1b-manifest-dir", default=d1d.DEFAULT_D1B)
    p.add_argument("--out-dir")
    p.add_argument("--verify-only", action="store_true")
    return p


def output_directory(args):
    if args.out_dir is not None:
        return Path(args.out_dir)
    prefix = f"prefix{args.max_normal_events}" if args.max_normal_events else "full"
    width = f"w{args.beam_width}" if args.mode == "beam" else "all"
    lam = format(args.lambda_defect, ".4g")
    return Path(DEFAULT_ROOT) / (f"{args.class_name}_s{args.support_seed}_q{args.query_seed}"
        f"_b{args.budget}_cap{args.capacity}_{args.mode}_{width}"
        f"_theta{args.theta_mode}_P{args.search_groups_per_type}_lambda{lam}_{prefix}")


def probe_a_partition(plan, search_groups_per_type):
    """Stratified fixed Probe-A split. No model score is used for selection."""
    if not 1 <= search_groups_per_type < 24:
        raise R1ProtocolError("Search Probe A requires 1..23 groups per type; audit remainder")
    groups = tuple(plan.probe_a_samples)
    normals = sorted(k for k in groups if all(
        r.is_good for r in plan.probe_a if r.sample_key == k))
    defects = sorted(k for k in groups if any(
        not r.is_good for r in plan.probe_a if r.sample_key == k))
    if len(normals) != 24 or len(defects) != 24:
        raise R1ProtocolError("Probe A official group counts not 24 normal + 24 defective")
    search_keys = set(normals[:search_groups_per_type] +
                      defects[:search_groups_per_type])
    audit_keys = set(normals[search_groups_per_type:] +
                     defects[search_groups_per_type:])
    if search_keys & audit_keys or search_keys | audit_keys != set(groups):
        raise R1ProtocolError("Probe-A selection overlap or missing groups")
    search = tuple(r for r in plan.probe_a if r.sample_key in search_keys)
    audit = tuple(r for r in plan.probe_a if r.sample_key in audit_keys)
    if not search or not audit:
        raise R1ProtocolError("Empty search/audit partition")
    return search, audit


class NativeStateEvaluator:
    """Reuses ONE native bank for scorer. Swaps dynamic tensors in try/finally.

    Static four-shot codebooks remain frozen. Each state uses its own immutable
    GreedyBank snapshot; evaluation only touches a separate scoring bank.
    """
    def __init__(self, impl, frozen, cfg, initial_state, bundles, cache_limit=2048):
        self.impl, self.cfg = impl, cfg
        self.native = copy.deepcopy(frozen)
        self.codebooks = dynamic_codebooks(self.native)
        self.caps = tuple(initial_state.bank.capacities)
        for cb, cap in zip(self.codebooks, self.caps):
            cb.K = cap
        self.bundles = tuple(bundles)
        self.cache = {}
        self.cache_limit = cache_limit
        self.score_calls = 0

    @torch.no_grad()
    def score_state(self, state):
        if state.bank.capacities != self.caps:
            raise R1ProtocolError("State bank capacity mismatch")
        digest = state.bank.sha256()
        if digest in self.cache:
            return self.cache[digest]
        before = [(cb.protos, cb.K) for cb in self.codebooks]
        try:
            for cb, features, cap in zip(self.codebooks, state.bank.layers, self.caps):
                if len(features) > cap or cb.feat_dim != features.shape[1]:
                    raise R1ProtocolError("Native state layer invalid")
                if cb.protos.device != features.device or cb.protos.dtype != features.dtype:
                    raise R1ProtocolError("Native state device/dtype mismatch")
                cb.protos = features
                cb.K = cap
            result = []
            for img, feats, _label, _rec in self.bundles:
                f = tuple(z.to(self.cfg.device) if z.device.type != self.cfg.device
                          else z for z in feats)
                result.append(native_score(self.impl, self.native, img, f, self.cfg)[0])
                self.score_calls += 1
        finally:
            for cb, (tensor, cap) in zip(self.codebooks, before):
                cb.protos = tensor
                cb.K = cap
        if self.cache_limit > 0:
            if len(self.cache) >= self.cache_limit:
                self.cache.clear()
            self.cache[digest] = tuple(result)
        return tuple(result)


def encode_search_probe(impl, backbone, cfg, records, cache_device):
    cache = []
    for rec in records:
        img, feature = p0.encode_query(impl, backbone, rec.path, cfg)
        if cache_device == "cpu":
            feature = tuple(layer.detach().cpu() for layer in feature)
        else:
            feature = tuple(layer.detach() for layer in feature)
        cache.append((img, feature, int(not rec.is_good), rec))
    return cache


def score_audit_probes(impl, backbone, frozen, cfg, states, records, theta, lam):
    # One encoded image at a time, no image/probe feature leakage into search.
    native_banks = {}
    for name, state in states.items():
        bank = copy.deepcopy(frozen)
        cbs = dynamic_codebooks(bank)
        for cb, feat, cap in zip(cbs, state.bank.layers, state.bank.capacities):
            cb.K = cap
            cb.protos = feat
        native_banks[name] = bank
    labels, scores = [], {k:[] for k in states}
    rows = []
    for rec in records:
        img, feat = p0.encode_query(impl, backbone, rec.path, cfg)
        label = int(not rec.is_good)
        labels.append(label)
        row = dict(image_id=rec.rel_path, official_sample_key=rec.sample_key,
                   camera_id=rec.camera_id, label_offline=label)
        for name, bank in native_banks.items():
            value = native_score(impl, bank, img, feat, cfg)[0]
            scores[name].append(value)
            row["score_"+name] = value
        rows.append(row)
        del img, feat
    metrics = {name: evaluate_scores(labels, scores["static"], values, theta,
                                     lambda_defect=lam) for name, values in scores.items()}
    return metrics, rows


def csv_file(path, rows):
    if not rows:
        return False
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return True


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
                               ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


@torch.no_grad()
def run(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise R1ProtocolError("CUDA not available")
    if args.probe_cache_device == "cuda" and args.device != "cuda":
        raise R1ProtocolError("CPU execution requires --probe-cache-device cpu")
    if not (1 <= args.capacity <= 8192):
        raise R1ProtocolError("Invalid capacity")
    if args.max_normal_events < 0 or args.beam_width < 1 or args.max_leaves < 1:
        raise R1ProtocolError("Invalid search bounds")
    if args.mode == "exact" and args.budget > 32 and not args.max_normal_events:
        raise R1ProtocolError("Full exact mode B>32 is not tractable; use beam")
    if not math.isfinite(args.lambda_defect) or args.lambda_defect < 0:
        raise R1ProtocolError("Invalid lambda")
    out = output_directory(args)
    if out.exists():
        raise FileExistsError("Never overwrite a prior R1 result: " + str(out))
    impl = p0.require_sources()
    impl.set_seed(0)
    if args.math_sdpa and args.device == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    train, test = pf.load_class_records(Path(args.dataset_root), Path(args.json_dir),
                                        args.class_name)
    plan = pf.build_partitions(train, test, pf.SplitConfig())
    supports_ids = set(plan.support_samples_by_seed[args.support_seed])
    supports = sorted(r.path for r in plan.support if r.sample_key in supports_ids)
    if len(supports) != 4 or len(supports_ids) != 4:
        raise R1ProtocolError("Strict four support identities violated")
    tape, n_stream, tape_manifest = frozen_r0_tape(
        plan, class_name=args.class_name, query_seed=args.query_seed,
        budget=args.budget, d1b_manifest_dir=args.d1b_manifest_dir)
    if n_stream != 2625 or len(tape.events) != args.budget:
        raise R1ProtocolError("R1 tape length/stream mismatch")
    search_recs, audit_recs = probe_a_partition(plan, args.search_groups_per_type)
    cfg = p0.Config(device=args.device, augmentation_seed=0, bank_seed=0,
                    fixed_capacity=25000, fixed_per_image=5000, augmentations=4,
                    include_identity=False, multiplier=2.5, feature_fuse="concat",
                    topk_patches=32, image_aggregation="max", image_topk=64,
                    image_quantile=.95, knn=5)
    start_time = time.perf_counter()
    backbone = impl.DINOv2MultiLayerBackbone(device=args.device)
    refs = p0.encode_reference_features(impl, backbone, supports, cfg)
    fixed = p0.build_vmb(impl, refs, supports, cfg)
    fixed_profile = p0.fixed_bank_profile(fixed)
    loso = []
    for support in supports:
        fold = p0.build_vmb(impl, refs, [r for r in supports if r != support], cfg)
        loso.append(float(p0.score_image(impl, backbone, fold, support, cfg)))
        del fold
    info = p0.threshold_from_scores(loso, 2.5)
    theta, sigma = float(info["theta"]), float(info["sigma"])
    calibration = dict(mode="loso", extra_normal_images=0,
                       sigma_loso=float(info["sigma"]))
    if args.theta_mode == "train64-q90":
        train_scores = [float(p0.score_image(impl, backbone, fixed, r.path, cfg))
                        for r in plan.history]
        if len(train_scores) != 64:
            raise R1ProtocolError("Expected exactly 64 reserved normal training views")
        fitted = fit_fixed_normal_thresholds(train_scores)
        theta = float(fitted["0.90"]["theta"])
        sigma = float(__import__("statistics").stdev(train_scores))
        calibration = dict(mode="train64-q90", extra_normal_images=64,
                           sigma_loso=float(info["sigma"]),
                           sigma_calibration=sigma, q90=theta,
                           no_probe_or_stream_in_calibration=True)
    del refs
    settings = v02.make_memory_settings(fixed)
    initial_arm = memory.MemoryArm("greedy_coreset", copy.deepcopy(fixed),
                                   settings, seed=args.support_seed + 37)
    for cb in dynamic_codebooks(initial_arm.bank):
        if len(cb.protos):
            raise R1ProtocolError("Starting dynamic bank nonempty")
        cb.K = args.capacity
    initial = DecisionState(snapshot(initial_arm.bank))
    candidates, queried = [], []
    images = {rec.path:rec for rec in test}
    for event in tape.events:
        rec = images[event.path]
        if rec.sample_key != event.sample_key:
            raise R1ProtocolError("Queried sample key mismatch")
        img, features = p0.encode_query(impl, backbone, event.path, cfg)
        score, amap = native_score(impl, fixed, img, features, cfg)
        item = dict(stream_position=event.position,
                    image_id=rec.rel_path, official_sample_key=rec.sample_key,
                    camera_id=rec.camera_id, label_revealed="normal" if rec.is_good else "defect",
                    static_score=score, eligible_normal=bool(rec.is_good))
        if rec.is_good:
            allow, keep, reason, extra = impl.select_informative_normal_write(
                score=score, theta=theta, sigmaN=max(sigma,.02),
                pred_defect=bool(score >= theta), novelty=0.0)
            if not allow or keep not in (8,16,24):
                raise R1ProtocolError("Native candidate rule unexpectedly refused verified normal")
            _raw, chosen = extract_v22_candidates(impl, amap, features, keep=keep)
            patches = normalized_native_candidate_rows(
                impl, initial_arm.bank, features, chosen, keep=keep)
            candidates.append(tuple(layer.detach().clone() for layer in patches))
            item["normal_candidate_index"] = len(candidates)-1
            item["keep"] = keep
            item["write_reason"] = reason
        queried.append(item)
        del img, features
    n_total_candidates = len(candidates)
    if args.max_normal_events:
        if args.max_normal_events > n_total_candidates:
            raise R1ProtocolError("Prefix requests more candidates than tape contains")
        candidates = candidates[:args.max_normal_events]
    if args.mode == "exact" and (len(candidates)>20 or (1<<len(candidates))>args.max_leaves):
        raise R1ProtocolError(
            f"Exact 2^{len(candidates)} exceeds hard leaf limit; no partial search allowed")
    search_bundles = encode_search_probe(
        impl, backbone, cfg, search_recs, args.probe_cache_device)
    evaluator = NativeStateEvaluator(impl, fixed, cfg, initial, search_bundles)
    labels = tuple(bundle[2] for bundle in search_bundles)
    static_scores = evaluator.score_state(initial)
    def evaluate(state):
        return evaluate_scores(labels, static_scores,
                               evaluator.score_state(state), theta,
                               lambda_defect=args.lambda_defect)
    greedy_all = initial
    for cand in candidates:
        from .r1_oracle_core import transition
        greedy_all = transition(greedy_all, cand, True)
    baseline_static = evaluate(initial)
    baseline_greedy = evaluate(greedy_all)
    search = (run_exact(candidates, initial, evaluate, max_leaves=args.max_leaves)
              if args.mode == "exact" else
              run_beam(candidates, initial, evaluate, beam_width=args.beam_width))
    oracle = search["best_state"]
    if len(oracle.decisions) != len(candidates):
        raise R1ProtocolError("Oracle decisions must cover every searched normal event")
    if search["globally_exhaustive"] != (args.mode == "exact"):
        raise R1ProtocolError("Exhaustivity flag mismatch")
    # Diagnostic-only selected sequence tested on untouched disjoint Probe-A
    # audit groups; this remains development work, NOT confirmatory Probe B.
    states = dict(static=initial, greedy_all=greedy_all, oracle_found=oracle)
    audit_metrics, audit_rows = score_audit_probes(
        impl, backbone, fixed, cfg, states, audit_recs, theta,
        args.lambda_defect)
    search_rows = []
    state_search_scores = {name: evaluator.score_state(state)
                           for name,state in states.items()}
    for i, (_img, _features, label, rec) in enumerate(search_bundles):
        search_rows.append(dict(image_id=rec.rel_path,
            official_sample_key=rec.sample_key, camera_id=rec.camera_id,
            label_offline=label, **{
                "score_"+name: state_search_scores[name][i] for name in states}))
    searched = {name: evaluate(state) for name,state in states.items()}
    full_labels = [int(r["label_offline"]) for r in search_rows+audit_rows]
    full_metrics = {}
    for name in states:
        full_scores = [float(r["score_"+name]) for r in search_rows+audit_rows]
        static_full = [float(r["score_static"]) for r in search_rows+audit_rows]
        full_metrics[name] = evaluate_scores(full_labels, static_full,
                                             full_scores, theta,
                                             lambda_defect=args.lambda_defect)
    frontier = frontier_by_admission_count(search["leaf_rows"])
    summary = dict(schema=SCHEMA,
        status=("COMPLETE" if not args.max_normal_events else "PREFIX_SMOKE_INCOMPLETE"),
        scope="OFFLINE ORACLE DIAGNOSTIC, NOT A DEPLOYABLE CAUSAL CONTROLLER",
        exact_global_only_within_fixed_offline_probe_a_objective=True,
        no_probe_b_used=True, no_test_label_available_to_online_policy=True,
        no_full_stream_prequential_claim=True,
        search_probe_a_is_optimistically_selected=True,
        audit_probe_a_disjoint_but_development_only=True,
        oracle_provenance=tape_manifest,
        source_fixed_bank_sha256=fixed_profile["sha256"],
        threshold=theta, calibration=calibration,
        n_query_labels=len(tape.events), n_verified_normal=n_total_candidates,
        n_verified_defect=len(tape.events)-n_total_candidates,
        n_candidate_events_searched=len(candidates),
        n_search_images=len(search_recs), n_audit_images=len(audit_recs),
        n_search_groups=2*args.search_groups_per_type,
        n_audit_groups=48-2*args.search_groups_per_type,
        config=dict(class_name=args.class_name, support_seed=args.support_seed,
            query_seed=args.query_seed, budget=args.budget, capacity=args.capacity,
            mode=args.mode, beam_width=args.beam_width, max_leaves=args.max_leaves,
            max_normal_events=args.max_normal_events,
            search_groups_per_type=args.search_groups_per_type,
            lambda_defect=args.lambda_defect, theta_mode=args.theta_mode,
            probe_cache_device=args.probe_cache_device, device=args.device),
        search_status=search["status"], globally_exhaustive=search["globally_exhaustive"],
        evaluated_terminal_states=search["evaluated_leaves"],
        theoretical_decision_sequences=str(search["total_possible_leaves"]),
        search_scoring_evaluations=search["search_evaluations"],
        native_score_calls=evaluator.score_calls,
        oracle_decision_bits=oracle.decisions,
        oracle_admitted_event_indices=[i for i,b in enumerate(oracle.decisions)
                                       if b=="1"],
        oracle_n_admissions=oracle.admissions,
        oracle_n_effective_updates=oracle.effective_updates,
        oracle_memory_sha256=oracle.bank.sha256(),
        greedy_memory_sha256=greedy_all.bank.sha256(),
        score_search_subset=searched,
        score_disjoint_probe_a_audit=audit_metrics,
        score_all_probe_a_development=full_metrics,
        elapsed_seconds=time.perf_counter()-start_time,
        claims=[
            "Exact oracle is optimal only within this class, seed, budget, "
            "capacity, fixed candidate rules and its searched Probe A subset.",
            "Beam best-found is never a global upper bound.",
            "Probe A audit split was not used for Oracle decisions; neither Probe A split is confirmatory.",
            "Only frozen Q-A queried events generate writes, not full-stream prequential scoring.",
            "No adaptive Query controller, Sidecar or Signed Correction is active.",
            "Greedy-all baseline matches oracle bank capacity but NOT its accepted event count.",
            "No claim of deployment attainability or ability to predict Oracle admission labels."])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"r1_summary.json",summary)
    csv_file(out/"r1_queries.csv",queried)
    csv_file(out/"r1_search_states.csv",search["leaf_rows"])
    csv_file(out/"r1_frontier_by_count.csv",frontier)
    csv_file(out/"r1_search_probe_scores.csv",search_rows)
    csv_file(out/"r1_audit_probe_scores.csv",audit_rows)
    if args.mode == "beam":
        csv_file(out/"r1_beam_trace.csv",search["stage_rows"])
    source_files=("experiments/risk_safe/r1_oracle_core.py",
                  "experiments/risk_safe/r1_native_oracle.py",
                  "experiments/risk_safe/r0_opportunity.py",
                  "experiments/risk_safe/native_bridge.py",
                  "experiments/risk_safe/shadow_memory.py",
                  "experiments/risk_safe/d1b_protocol.py",
                  "experiments/v02_memory.py",
                  "test_phase3_AIF_simplified_v22.py")
    provenance = dict(schema="risk_safe.r1.provenance.v1",
        git=pf.git_state(), source_sha256=pf.source_hashes(source_files),
        dataset_json_sha256=pf.sha256_file(
            Path(args.json_dir)/(args.class_name+".json")),
        d1b_archived_query_manifest_sha256=pf.sha256_file(
            Path(args.d1b_manifest_dir)/d1b.OUTPUT_QUERIES),
        query_tape=tape_manifest,
        search_probe_sample_keys=sorted(set(x.sample_key for x in search_recs)),
        audit_probe_sample_keys=sorted(set(x.sample_key for x in audit_recs)))
    write_json(out/"r1_provenance.json",provenance)
    checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in out.iterdir() if p.is_file()}
    write_json(out/"checksums.json",checksums)
    print(json.dumps(dict(status=summary["status"], path=str(out),
        search_status=search["status"], n_candidates=len(candidates),
        examined_terminal_states=search["evaluated_leaves"],
        oracle_n_admissions=oracle.admissions, greedy_n_admissions=greedy_all.admissions,
        scores_audit=audit_metrics,
        elapsed_seconds=summary["elapsed_seconds"]),indent=2))
    return 0


def verify(args):
    out = output_directory(args)
    checks = json.loads((out/"checksums.json").read_text(encoding="utf-8"))
    for name, sha in checks.items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest() != sha:
            raise R1ProtocolError("Result SHA-256 mismatch: "+name)
    summary = json.loads((out/"r1_summary.json").read_text(encoding="utf-8"))
    if summary["schema"] != SCHEMA or summary["config"]["mode"] != args.mode:
        raise R1ProtocolError("Summary schema/mode mismatch")
    if summary["config"]["capacity"] != args.capacity or summary["config"]["budget"] != args.budget:
        raise R1ProtocolError("Summary capacity/budget mismatch")
    if summary["globally_exhaustive"] and summary["evaluated_terminal_states"] != (
            1 << summary["n_candidate_events_searched"]):
        raise R1ProtocolError("Exact search missing leaf outcomes")
    print(json.dumps(dict(verified=True, no_gpu_work=True,
                          out_dir=str(out), search_status=summary["search_status"]),
                     indent=2))
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    if args.json_dir is None:
        args.json_dir = str(Path(args.dataset_root)/"jsons"/"realiad_jsons")
    return verify(args) if args.verify_only else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
