# Synthetic acceptance matrix

All fixtures use generated identifiers, labels, scalar scores or tensors. None requires Real-IAD paths, images, masks, model weights or outcomes. The campaign source must pass compilation and the focused contract/evaluator tests before source freeze. Existing R4 tests cover synthetic ledger, broker, resume, mapping, pixel-resource, native scorer and checkpoint behavior; the R2 CUDA suite is the synthetic CUDA gate. The preserved adversarial encoder-substitution regression remains a separately reported known failure.

| Gate | Synthetic assertion | Expected result |
|---|---|---|
| Contract | Exact spec digest, four categories/counts, 4 Supports, three arm identities and frozen query/window settings | Pass; mutation rejected |
| Population | Contiguous groups of five views, category order and opaque-ID uniqueness | Pass; missing/duplicated/reordered view rejected |
| Support receipt | Exact support content/fixed bank, four finite self-scores, NumPy ddof=1 theta and Writer sigma, empty dynamic banks | Pass; wrong path, digest, score, theta, extra normal or nonempty memory rejected |
| Registration | One common registration digest and common event/group/view stream across S/CL/CE | Pass; arm-specific mapping/order mutation rejected |
| Authorization/default | No-argument mode resolves to `freeze-only`; each data or execution operation requires exact scope and identity binding | Pass; absent, malformed or mismatched record rejected |
| Resource | 784×4096 FP32 cache and immutable checkpoint sums; missing/invalid benchmark yields blocked status | Pass; no silent checkpoint/cache reduction |
| Query/feedback | Existing query receipt replay, legal intent, duplicate and unqueried broker requests | Pass for trusted-local research tests; no cross-arm disclosure |
| Recovery | Exact checkpoint restore and deterministic ledger replay; ambiguous intent fails closed | Pass in existing synthetic recovery tests |
| Stage A projection | All twelve seals, event counts, support bootstrap, pre-event threshold/prediction and query bindings | Pass; subset or mutation rejected |
| Metrics | BER/BA consistency, average-rank AUROC ties, stepwise AP, category equal weighting, shared L/E ranking equality | Pass on synthetic labels/scores |
| Temporal | 50-group windows, stride 10, deterministic thirds and one-class BER as NA | Pass; no zero-fill |
| Stage B seal | Mapping/outcome/metric input checksums and complete result receipt | Pass; altered mapping, outcome feedback mismatch or result bytes rejected |

No real Stage A/Stage B run is an acceptance test. A skipped mandatory synthetic CUDA test is not a PASS and must be reported as SKIP with the hardware reason.
