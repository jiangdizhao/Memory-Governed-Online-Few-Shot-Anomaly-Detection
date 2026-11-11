# FOADS/K-NG Paper-Consistent Protocol Adaptation

## Identity

Baseline name: `FOADS-KNG-PAPER-CONSISTENT-PROTOCOL-ADAPTED`

The immutable upstream reference is [Whishing/K-NG](https://github.com/Whishing/K-NG), commit `63dd8e7c9ade5619771d4b5544f032206818bb22`. The adapter verifies SHA-256 identities for `Konline.py`, `Klink.py`, and `datasets/mvtec.py` before loading upstream functions or the upstream `Klinks` class. It does not import the K-online experiment driver, whose top-level code changes CUDA visibility and whose test loop reads labels and masks.

## Registered P0 inputs

The adapter accepts only the exact identity manifest hash, role manifest hash, and dataset-release hash supplied by the certified Gate 1 registration. It selects that category's P0 role (`support_seed=0`) and P0 stream (`stream_seed=0`), verifies the complete ordered TEST identity list against the identity records, and requires each Source Group to contain cameras 1 through 5 exactly once. It rejects label, outcome, and mask fields before building the scoring schedule.

K-NG initialization uses the same registered 68 TRAIN-normal image identities selected for the principal four-Support/64-C64 allocation. K-NG consumes the 68 images together as its native initialization set. This is a matched-information protocol adaptation; the upstream script's training loop stops after its first DataLoader batch of size 10. The adapter records this difference and does not describe the initialization as an exact reproduction of the published run.

## Preserved upstream components

- ImageNet-pretrained ResNet18 and the official resize-to-256, center-crop-to-224, ImageNet normalization convention.
- Hooks at the end of ResNet18 layer1, layer2, and layer3; upstream `embedding_concat`; random selection of 100 dimensions from the 448-dimensional concatenated representation using the upstream seed-1024 selection.
- K-NG random-node initialization, target size 3,136, `age_max=25`, ten `fit_network` epochs, native threshold and covariance initialization.
- The pinned `Klinks.online_fit()` body, including pseudo-normal screening, graph topology changes, edge aging and pruning, node thresholds, regularized covariance, and incremental mean/covariance updates.
- Nearest Neural Gas node selection, 224-pixel bilinear interpolation with `align_corners=False`, and Gaussian smoothing with sigma 4.

## Compatibility restoration and explicit protocol adaptations

1. **PyTorch 1.8 distance compatibility restoration.** The pinned Klinks methods call `torch.pairwise_distance()` on `(B,C,1)` and `(1,C,N)` tensors. PyTorch 1.8 reduced dimension 1 and returned `(B,N)`; the installed PyTorch 2.0.1 reduces the innermost dimension and returns `(B,C)`. The local adapter binds the explicit PyTorch 1.8 operation `torch.linalg.vector_norm(test_batch.unsqueeze(-1) - centeroid + 1e-6, ord=2, dim=1)` to each verified Klinks instance. It retains `sqrt(dist_matrix)` in `gpu_cal_dist()` and top-2 selection before `sqrt(topk_values)` in `online_gpu_cal_dist()`. This restores upstream runtime semantics; it does not introduce a new distance metric. `Klink.py` remains byte-for-byte pinned.
2. **Paper-consistent Mahalanobis computation.** The pinned `Konline.py::Matrix_Mahalanobis` has an `einsum` discrepancy: its first contraction sums inverse-covariance rows before multiplying by the feature offset, so it does not compute Equation (11). With researcher approval, the adapter uses `einsum("bi,bij,bj->b", delta, inverse_covariance, delta)`, checks finiteness, clips only roundoff-sized negative values, and rejects materially negative quadratic forms. The upstream function and source bytes remain unchanged as reference evidence.
3. **Batch-independent image scores.** Each image score is the maximum of its raw Mahalanobis map after the upstream interpolation and smoothing. The upstream batch-global min-max normalization is omitted. The normalized image scores have the same within-batch ordering whenever the batch range is nonzero, while each raw score is independent of sibling images.
4. **Registered Real-IAD ordering and Batch-10 timing.** The upstream shuffled TEST loader is replaced by the registered P0 stream order. The runner scores each complete five-view group in order; it calls native `online_fit()` once after it has scored two consecutive groups (ten images). A final unpaired group is scored and remains unupdated; it is not sent as a partial update batch.
5. **Matched 68-normal initialization.** The 68 P0 TRAIN-normal identities are passed to native K-NG initialization, rather than the upstream script's first ten-image DataLoader batch. No other TRAIN identities enter initialization.

The adapter makes no paid queries and has no label or mask input to inference or updates. Its output is image-level raw anomaly scores. Image-level AUROC and AP may be computed only in a separately authorized post-seal evaluation. BER/FPR/FNR remain unavailable unless a deployment threshold is registered in advance. No Stage B result, Real-IAD image score, or TEST outcome was produced while implementing this adapter. Results must not be presented as an exact reproduction of the original published experiment.

## Synthetic CPU verification

`tests/calibration/final_benchmark/test_foads_kng.py` verifies the Equation (11) outputs against independent linear algebra and the pinned upstream helper; ResNet feature concatenation and dimension selection; singleton and sibling-independent raw scores; exact registered 68-image/stream binding; and absence of label/mask fields. It executes the actual pinned `fit_network()`, `cluster_label()`, and `online_fit()` methods on a deterministic four-node CPU synthetic model with the local distance compatibility methods. The runner's two-group schedule is compared with a direct upstream `online_fit()` call, including node means, covariances, thresholds, accepted/rejected pseudo-normal observations, and graph edges. No Real-IAD scoring or TEST outcome access occurs.
