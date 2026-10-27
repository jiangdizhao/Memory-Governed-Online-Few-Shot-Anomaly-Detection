# D1-C CPU mechanism implementation and acceptance gates

**Status:** newly implemented and committed as code; **Windows CPU execution is not yet verified**. This is a synthetic *causal* sanity exercise, not a Real-IAD anomaly-detection performance run, not a calibrated CDARG policy, and not a native v22 image-scoring equivalence test.

## Source authority

Primary: project PDF "Risk-Safe Memory Admission for v22: A Deployment-Oriented Research Strategy", Priority A — Counterfactual Dual-Anchor Risk Gate. Also follow D1_C_CDARG_DESIGN_GUARDRAILS_2026-10-28.md. D1-B registered data stream, labels, support, query seeds and budget are **unchanged**.

## Implemented now

1. shadow_memory.py: separate live/shadow layer tensors, native **unchanged** V0.2 greedy_slot() for capacity-full decisions, exact V0.2 append/skip/replace branch semantics, version-and-digest commit guard, non-aliasing clones. This is a tensor-level Greedy adapter, not the full native VisionAD scorer.
2. anchor_store.py: human-queried image-view Normal/Defect evidence with query decision and disclosure steps, effective no earlier than next step. For defect-labeled images, top anomaly-score patches are chosen from model-prelabel scores **without using ground-truth segmentation masks**; do not assume every patch/camera view in an NG session is defective.
3. admission.py: true counterfactual shadow score comparisons, separate normal false-positive hinge utility U_N, weighted/normalised defect erosion risk R_D, hard protected-defect margin crossing H_D. Evaluate never mutates live memory; commit only if all gates pass. Missing defect anchors mean risk **unavailable**, conservatively reject; not R_D=0. Missing normal anchors also reject.
4. d1c_cpu_smoke.py: controlled synthetic cases — harmful verified-normal append; benign verified-normal append; lack of defect risk anchors; harmful full-capacity replacement. Produces immutable JSON evidence with SHA-256 source hashes.
5. tests/test_risk_safe_d1c.py: tests of causal evidence access, native greedy slot parity, rejection isolation, positive/negative cases, heldout probe separation and idempotent behavior.

## Mathematics (from project strategy PDF)

Let s_t be the score with live normal memory, s_shadow with exact Greedy-proposed shadow, both under the SAME fixed pre-event threshold theta_t and correction state, computed on strictly prior verified anchors:

    U_N = mean_{n in A_N} ([s_t(n)-theta_t]_+ - [s_shadow(n)-theta_t]_+)

    R_D = mean_{d in A_D} w_d [s_t(d)-s_shadow(d)]_+ / max(sigma_N,sigma_0)

    H_D = any_{d in A_D} (s_t(d) >= theta_t + m_D AND s_shadow(d) < theta_t + m_D)

    Admit iff NOT H_D AND R_D <= epsilon_D AND U_N - lambda_D * R_D >= tau_U.

The smoke's config (epsilon_D=0.02, lambda_D=1, tau_U=0, m_D=0.05, sigma_N=1, sigma_0=0.01) is a purely **synthetic demonstration**. It is not supported by Real-IAD calibration and must NEVER be silently reused as a performance parameter. Actual calibration must be development-only.

The primary causal benchmark in subsequent GPU stages uses Q-A shared random tape, Random stream order, original four-shot support, original LOSO theta0, and Sidecar + Signed Correction OFF. Normal-first/Defect-first/Bursty remain later sensitivity studies; no class-prior controller is being developed here.

## Required Windows PowerShell commands (one command per line)

Run from E:\Memory-Governed-Online-Few-Shot-Anomaly-Detection after a clean fast-forward pull.

Compile:
    & "D:\anaconda3\envs\torch\python.exe" -m py_compile experiments/risk_safe/shadow_memory.py experiments/risk_safe/anchor_store.py experiments/risk_safe/admission.py experiments/risk_safe/d1c_cpu_smoke.py tests/test_risk_safe_d1c.py

CPU unit tests:
    & "D:\anaconda3\envs\torch\python.exe" -m unittest discover -s tests -p "test_risk_safe_d1*.py" -v

First synthetic run:
    & "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.d1c_cpu_smoke --out-dir "outputs/risk_safe_sprint_20261027/d1c_cpu_smoke_v1"

Idempotent read-only check:
    & "D:\anaconda3\envs\torch\python.exe" -m experiments.risk_safe.d1c_cpu_smoke --out-dir "outputs/risk_safe_sprint_20261027/d1c_cpu_smoke_v1" --verify-only

Only after **all** tests pass and both JSON files report verified, commit/push ONLY the output folder; otherwise preserve traceback and report the error. The output source manifest pins generating code SHA-256 and Git provenance.

## What remains BEFORE any real GPU inference or FNR claim

- Native scorer parity: a0_a1_strict4_matched.score_bank live versus untouched shadow-clone image scores/maps (within numerical tolerance), including static support, multi-layer fused features, per-camera view and image aggregation. Synthetic CosinePatchScorer is NOT this scorer.
- Exact image candidate parity: import same v22 8/16/24 top-k patch extractor and normal-write selector with layer indices, and verify Greedy layer updates under actual feature shapes and forced memory saturation.
- Native verified-label replay: D1-B Q-A tape plus trusted image/path adapter, no pre-query label-bearing metadata, only queried view label, then normal candidate gate/feedback ledger.
- Development-only calibration of epsilon_D, lambda_D, tau_U, margin and score normalisation; document support LOSO bias separately; do not tune against confirmatory categories/probes.
- Heldout probe audit: compare greedy/append/causal novelty/SAEG/CDARG under identical query and candidate features, bank budget and fixed theta0; log defect score erosion, FNR degradation, normal FPR benefit and admission counts. A reject-all method is not a successful adaptive detector.

**STOP:** CPU-only validation is prerequisite; do not start large GPU experiment until native scorer and native Greedy operator parity passes.
