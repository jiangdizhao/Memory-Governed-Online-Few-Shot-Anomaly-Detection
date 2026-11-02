"""Synthetic CPU contracts for the isolated R6 capacity-1024 experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import test_phase3_AIF_simplified_v22 as native
from experiments.calibration import r3_fullstream_2x2 as r3
from experiments.calibration import r6_capacity1024_audiojack_runner as r6
from experiments.risk_safe.native_bridge import dynamic_codebooks
from experiments.v02_memory import MemoryArm


def _bank(*, corr_cfg=None):
    bank = native.PatchVMB(
        device="cpu", K_normal_fixed=2, K_normal_dyn=r6.REFERENCE_CAPACITY,
        K_normal_ltm=2, K_defect=2, K_concept=2,
        corr_cfg=corr_cfg or native.CorrectionConfig(enabled=True, K_correction=256),
    )
    bank._ensure_mode_view_normals("p0", 0, "id", [torch.zeros((1, 8), dtype=torch.float32)])
    return bank


def _writer(bank):
    settings = SimpleNamespace(
        flat=r6.MEMORY_CAPACITY, stm=r6.MEMORY_CAPACITY, ltm=1, buffer=1,
        min_events=2, promote_every=16, cap_bytes=2 * 1024**3,
    )
    return MemoryArm("greedy_coreset", bank, settings, seed=0)


def test_r6_capacity_is_1024_and_historical_reference_remains_256():
    assert r6.MEMORY_CAPACITY == 1024
    assert r6.REFERENCE_CAPACITY == 256
    assert r6.EVENTS == 3815
    assert r6.CATEGORY == "audiojack"
    assert r6.RUN_ID.startswith("r6_capacity1024_audiojack_")


def test_frozen_r3_recipe_is_checked_before_fresh_native_correction_allocation():
    bank = _bank()
    old = bank.corr_cb["p0"]["id"][0]
    assert old.K == 256
    cfg, codebooks = r6._configure_r6_correction_capacity(bank)
    assert cfg.K_correction == 1024
    assert len(codebooks) == 1
    new = codebooks[0]
    assert new is bank.corr_cb["p0"]["id"][0]
    assert new is not old
    assert new.K == 1024 and new.size() == 0
    for name in ("protos", "beta", "radius", "kappa", "count", "outcome_fp",
                 "outcome_fn", "outcome_tn", "outcome_tp", "last_step"):
        assert getattr(old, name).shape[0] == 256
        assert getattr(new, name).shape[0] == 1024


def test_frozen_correction_recipe_mutation_fails_before_capacity_resize():
    bank = _bank()
    bank.corr_cfg.radius0 += 0.001
    with pytest.raises(r3.R3Error, match="frozen profile"):
        r6._configure_r6_correction_capacity(bank)
    assert bank.corr_cfg.K_correction == 256


def test_shared_native_greedy_memoryarm_has_1024_capacity_and_metadata():
    bank = _bank()
    r6._configure_r6_correction_capacity(bank)
    writer = _writer(bank)
    profile = r6._validate_greedy_capacity(bank, writer)
    codebooks = dynamic_codebooks(bank)
    assert len(codebooks) == 1
    assert profile[0]["capacity"] == 1024
    assert profile[0]["occupancy"] == 0
    assert writer.settings.flat == bank.K_normal_dyn == codebooks[0].K == 1024

    synthetic = torch.arange(1, 1024 * 8 + 1, dtype=torch.float32).reshape(1024, 8)
    codebooks[0].set_prototypes(synthetic)
    assert codebooks[0].size() == 1024
    for name in ("meta_count", "meta_utility", "meta_boundary", "meta_fp_rescue",
                 "meta_coverage", "meta_first_step", "meta_last_step",
                 "meta_last_event", "meta_distinct_events"):
        assert getattr(codebooks[0], name).shape == (1024,)
    assert bank.device.type == "cpu"


def test_gc_and_fpsa_are_distinct_native_capacity_1024_stores():
    from experiments.calibration.r4_fpsa_codebook import FPSASignedCorrectionCodebook

    bank = _bank()
    gc_cfg, gc_books = r6._configure_r6_correction_capacity(bank)
    gc = gc_books[0]
    fpsa_cfg = type(gc_cfg)(**vars(gc_cfg))
    fpsa_cfg.enabled = False
    fpsa = FPSASignedCorrectionCodebook(1024, 8, torch.device("cpu"), fpsa_cfg)
    fpsa_cfg.K_correction = 1024
    assert gc.K == fpsa.K == gc_cfg.K_correction == fpsa_cfg.K_correction == 1024
    assert gc is not fpsa
    assert gc.protos.data_ptr() != fpsa.protos.data_ptr()
    assert all(getattr(gc, name).shape[0] == 1024 for name in (
        "protos", "beta", "radius", "kappa", "count", "outcome_fp",
        "outcome_fn", "outcome_tn", "outcome_tp", "last_step"))
    assert all(getattr(fpsa, name).shape[0] == 1024 for name in (
        "protos", "beta", "radius", "kappa", "count", "outcome_fp",
        "outcome_fn", "outcome_tn", "outcome_tp", "last_step"))
    frozen = Path(r6.ROOT) / "experiments/calibration/r4_fpsa_codebook.py"
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == r6.FPSA_SOURCE_SHA256


def test_identical_empty_gc_and_fpsa_stores_have_identical_synthetic_readout():
    from experiments.calibration.r4_fpsa_codebook import FPSASignedCorrectionCodebook

    cfg_native = native.CorrectionConfig(enabled=True, K_correction=1024)
    cfg_fpsa = native.CorrectionConfig(enabled=True, K_correction=1024)
    gc = native.SignedCorrectionCodebook(1024, 8, torch.device("cpu"), cfg_native)
    fpsa = FPSASignedCorrectionCodebook(1024, 8, torch.device("cpu"), cfg_fpsa)
    query = torch.arange(1, 65, dtype=torch.float32).reshape(8, 8)
    torch.testing.assert_close(gc.apply(query), fpsa.apply(query), rtol=0, atol=0)
    assert gc.size() == fpsa.size() == 0


def test_r5_runtime_and_model_identity_must_match_exactly():
    receipt = {"loaded_model_state_sha256": "a" * 64,
               "hub_source_tree_sha256": "b" * 64,
               "checkpoint_sha256": "c" * 64,
               "runtime_profile": "stage05_set_seed_0_math_sdp_v1"}
    runtime = {"hub_source_tree_sha256": "b" * 64,
               "checkpoint_sha256": "c" * 64,
               "runtime_profile": "stage05_set_seed_0_math_sdp_v1"}
    r6._assert_r5_runtime_compatible(runtime, "a" * 64, receipt)
    for key, value in (("hub_source_tree_sha256", "x" * 64),
                       ("checkpoint_sha256", "x" * 64),
                       ("runtime_profile", "incompatible")):
        changed = dict(runtime, **{key: value})
        with pytest.raises(r6.R6AudiojackError, match="sealed R5 baseline"):
            r6._assert_r5_runtime_compatible(changed, "a" * 64, receipt)
    with pytest.raises(r6.R6AudiojackError, match="sealed R5 baseline"):
        r6._assert_r5_runtime_compatible(runtime, "x" * 64, receipt)


def test_r5_c64_cache_image_and_feature_receipt_are_verified_without_encoding():
    rgb = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(8, 6, 3)
    feature = np.arange(15, dtype=np.float32).reshape(3, 5)
    item = {"rgb": rgb.copy(), "features": [feature.copy()]}
    receipt = {"feature_receipts": [{"index": 0, "shape": [3, 5], "dtype": "float32"}]}
    r6._validate_r5_c64_cache_item(item, receipt, rgb, index=0)
    wrong_rgb = rgb.copy()
    wrong_rgb[0, 0, 0] ^= 1
    with pytest.raises(r6.R6AudiojackError, match="image/feature receipt"):
        r6._validate_r5_c64_cache_item(item, receipt, wrong_rgb, index=0)
    wrong_receipt = {"feature_receipts": [{"index": 0, "shape": [5, 3], "dtype": "float32"}]}
    with pytest.raises(r6.R6AudiojackError, match="shape/dtype receipt"):
        r6._validate_r5_c64_cache_item(item, wrong_receipt, rgb, index=0)


def test_report_divergence_identifies_correction_and_shared_greedy_occupancies_separately():
    line = r6._capacity_divergence_report_line("GC", {
        "capacity_source": "GC Correction bank",
        "first_divergent_update_position": 859,
        "first_score_position_after_update": 860,
        "correction_occupancy_before": 225,
        "correction_occupancy_after": 257,
        "greedy_occupancy_before_by_layer": [224],
        "greedy_occupancy_after_by_layer": [232],
        "reason": "Native update admitted a prototype beyond the historical full-bank boundary.",
    })
    assert "GC Correction bank occupancy 225 -> 257" in line
    assert "shared Greedy occupancy [224] -> [232]" in line


@pytest.mark.parametrize("branch,head", [
    ("wrong-branch", r6.EXPECTED_PARENT),
    (r6.EXPECTED_BRANCH, "0" * 40),
])
def test_repository_identity_rejects_incorrect_branch_or_parent(monkeypatch, branch, head):
    def fake_git(*args):
        if args == ("branch", "--show-current"):
            return branch
        if args == ("rev-parse", "HEAD"):
            return head
        if args == ("status", "--porcelain", "--", *r6.SOURCE_PATHS):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(r6, "_git", fake_git)
    with pytest.raises(r6.R6AudiojackError, match="Source identity mismatch"):
        r6._repository_identity()


def test_score_disclosure_gate_and_b3_transition_are_causal():
    gate = r6._ScoreDisclosureGate(11)
    with pytest.raises(r6.R6AudiojackError, match="before score seal"):
        gate.authorize_disclosure(11)
    with pytest.raises(r6.R6AudiojackError, match="Invalid or repeated"):
        gate.mark_score_flushed(10)
    gate.mark_score_flushed(11)
    gate.authorize_disclosure(11)
    with pytest.raises(r6.R6AudiojackError, match="before score seal"):
        gate.authorize_disclosure(11)

    transition = r6._defer_b3_transition(
        11, "GC-FPSA", "d" * 64,
        {"new_theta": 0.51, "evidence_fingerprint": "e" * 64})
    assert transition["apply_from_position"] == 12
    assert transition["scorer_state_sha256"] == "d" * 64
    with pytest.raises(r6.R6AudiojackError, match="Invalid R6 B3 transition identity"):
        r6._defer_b3_transition(11, "unknown", "d" * 64,
                                {"new_theta": 0.51, "evidence_fingerprint": "e" * 64})


def test_writer_gate_depends_only_on_frozen_c0_score_and_threshold():
    # C1/C2 are not accepted by the writer gate; equal C0 inputs are invariant.
    assert r6._frozen_writer_gate(0.7, 0.6) == 1
    assert r6._frozen_writer_gate(0.7, 0.6) == 1
    assert r6._frozen_writer_gate(0.5, 0.6) == 0


def test_output_budget_refuses_overrun_before_write_and_preserves_failure_reserve(tmp_path):
    budget = r6.R6OutputBudget(tmp_path / "full", create=True)
    budget.max_bytes = 256
    budget.reserve = 32
    with pytest.raises(r6.R6AudiojackError, match="byte budget exhausted before write"):
        budget.write_once("stage_a_seal.json", b"x" * 224)
    assert not (tmp_path / "full" / "stage_a_seal.json").exists()

    budget.write_once("stage_a_seal.json", b"x" * 223)
    assert budget.audit()["used_bytes"] == 224
    budget.write_json("failure.json", {"status": "INCOMPLETE"}, failure_reserved=True)
    assert budget.audit()["used_bytes"] <= 256
    with pytest.raises(r6.R6AudiojackError, match="overwrite is forbidden"):
        budget.write_once("stage_a_seal.json", b"replacement")


def test_r6_namespace_is_unique_and_never_resumes_existing_output(tmp_path, monkeypatch):
    import experiments.calibration.r2_feedback_repair as repair

    output_base = tmp_path / "outputs" / "calibration"
    output_base.mkdir(parents=True)
    namespace = output_base / r6.RUN_ID
    monkeypatch.setattr(r6, "OUTPUT_BASE", output_base)
    monkeypatch.setattr(r6, "RUN_NAMESPACE", namespace)
    full = r6._output_path(r6.RUN_ID)
    assert full == namespace / "full"
    r6._prepare_output_namespace()
    full.mkdir()
    with pytest.raises(FileExistsError, match="restart/resume is forbidden"):
        r6._output_path(r6.RUN_ID)
    with pytest.raises(r6.R6AudiojackError, match="Only the fixed R6"):
        r6._output_path("r5_audiojack_b3_fpsa_20261102_a1")
    repair._reject_reparse_chain(output_base)


def test_confirmed_failed_smokes_are_adopted_only_as_immutable_prefix(tmp_path, monkeypatch):
    output_base = tmp_path / "outputs" / "calibration"
    namespace = output_base / r6.RUN_ID
    leaves = ("smoke", "smoke_retry1")
    smoke_dirs = []
    for run_id, leaf in zip(r6.SMOKE_ATTEMPT_IDS[:2], leaves):
        smoke = namespace / leaf
        smoke.mkdir(parents=True)
        smoke_dirs.append(smoke)
    monkeypatch.setattr(r6, "OUTPUT_BASE", output_base)
    monkeypatch.setattr(r6, "RUN_NAMESPACE", namespace)
    snapshots = []
    for run_id, smoke, index in zip(r6.SMOKE_ATTEMPT_IDS[:2], smoke_dirs, range(2)):
        manifest = {
            "schema": r6.RUN_SCHEMA, "run_id": run_id, "mode": "SMOKE_ONLY",
            "authorization": r6.AUTHORIZATION, "branch": r6.EXPECTED_BRANCH,
            "source_parent_commit": r6.EXPECTED_PARENT, "category": r6.CATEGORY,
            "source_fingerprints": r6._source_hashes(),
        }
        failure = {"schema": "r6.audiojack.capacity1024.failure.v1",
                   "status": "INCOMPLETE", "run_id": run_id,
                   "failure_message": f"synthetic failure {index}",
                   "automatic_restart": False}
        history = {"record_type": "supervisor_result", "stage": "STAGE_A_SMOKE",
                   "result": {"status": "FAILED_NONZERO_EXIT", "child_exit_code": index + 1,
                              "termination_confirmed": True, "active_job_processes": 0}}
        files = {
            "budget.lock": b"\0",
            "run_manifest.json": json.dumps(manifest).encode(),
            "failure.json": json.dumps(failure).encode(),
            "attempt_history.jsonl": json.dumps(history).encode() + b"\n",
        }
        for name, payload in files.items():
            (smoke / name).write_bytes(payload)
        snapshots.append((smoke, {name: r6._sha256_file(smoke / name) for name in files}))
    receipts = r6._preserved_failed_smoke_receipts()
    assert [item["run_id"] for item in receipts] == list(r6.SMOKE_ATTEMPT_IDS[:2])
    assert all(item["status"] == "FAILED_NONZERO_EXIT" and
               item["termination_confirmed"] for item in receipts)
    assert r6.SMOKE_ATTEMPT_IDS[len(receipts)] == r6.FINAL_SMOKE_ID
    for smoke, before in snapshots:
        assert {name: r6._sha256_file(smoke / name) for name in before} == before


def test_bootstrap_and_binary_metrics_are_deterministic_on_synthetic_groups():
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    pred_a = np.asarray([0, 1, 1, 0], dtype=np.int8)
    pred_b = np.asarray([0, 0, 0, 1], dtype=np.int8)
    groups = np.asarray(["g1", "g1", "g2", "g2"])
    metric = r6._binary_metrics(labels, pred_a)
    assert metric["balanced_accuracy"] == pytest.approx(0.5)
    assert metric["confusion"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    comparisons = {"a_minus_b": ("a", "b")}
    one = r6._bootstrap_contrasts({"a": pred_a, "b": pred_b}, labels, groups, comparisons)
    two = r6._bootstrap_contrasts({"a": pred_a, "b": pred_b}, labels, groups, comparisons)
    assert one == two
    assert one["replicates"] == r6.BOOTSTRAP_REPLICATES
    assert one["independent_confirmation"] is False
