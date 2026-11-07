"""Bounded synthetic-only CUDA checks for the R2 identity boundary."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def test_cuda_full_feature_and_cls_swap_are_rejected():
    from experiments.calibration.fs4_errors import FS4FatalIntegrityError
    from experiments.calibration.fs4_source_excluded_scorer import _feature_digest
    from experiments.calibration.fs4_threshold_integration import _validate_bound_inputs
    image = SimpleNamespace(width=28, height=28)
    original = torch.zeros((784, 4096), dtype=torch.float32, device="cuda"); original[:, 0] = 1
    swapped = torch.zeros_like(original); swapped[:, 1] = 1
    cls = np.asarray([1., 0.], dtype=np.float32)
    provenance = {"feature_sha256": _feature_digest(original), "cls_sha256": hashlib.sha256(cls.tobytes()).hexdigest(),
                  "geometry": [28, 28]}
    _validate_bound_inputs(provenance, image, original, cls)
    with pytest.raises(FS4FatalIntegrityError): _validate_bound_inputs(provenance, image, swapped, cls)
    with pytest.raises(FS4FatalIntegrityError): _validate_bound_inputs(provenance, image, original, np.asarray([0., 1.], dtype=np.float32))


def test_cuda_live_encoder_state_identity_rejects_mutation():
    from experiments.calibration.fs4_r2_context import FS4BoundaryError, verify_model_identity
    from experiments.calibration import r1_matched_correction_pilot as r1
    model = torch.nn.Linear(2, 2).cuda().eval()
    backbone = SimpleNamespace(model=model)
    encode = lambda *_: None
    impl = SimpleNamespace(encode_pil=encode)
    expected = r1._model_state_sha256(model)
    verify_model_identity(backbone, impl, encode, expected)
    with torch.no_grad(): model.weight[0, 0] += 1
    with pytest.raises(FS4BoundaryError, match="model state"):
        verify_model_identity(backbone, impl, encode, expected)
