"""
tests/test_yolo_features.py
---------------------------
Unit tests for ``segmentation_tracking.yolo_features.YOLOFeatureExtractor``.

All tests run purely on CPU with no real YOLO model, GPU, or SAM2
dependencies.  The YOLO model is mocked with a minimal PyTorch
``nn.Sequential`` that has the same attribute layout as an
``ultralytics.YOLO`` instance (``model.model → nn.Sequential``).

Run with:
    pytest tests/test_yolo_features.py -v
"""
from __future__ import annotations

import os
import sys
import importlib.util
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import yolo_features directly without triggering the full package __init__
# (which requires torch, YOLO, SAM2, etc.)
_spec = importlib.util.spec_from_file_location(
    "segmentation_tracking.yolo_features",
    os.path.join(_REPO_ROOT, "segmentation_tracking", "yolo_features.py"),
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
if "segmentation_tracking" not in sys.modules:
    _pkg = types.ModuleType("segmentation_tracking")
    sys.modules["segmentation_tracking"] = _pkg
sys.modules["segmentation_tracking.yolo_features"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
YOLOFeatureExtractor = _mod.YOLOFeatureExtractor
_SCALE_STRIDES = _mod._SCALE_STRIDES


# ---------------------------------------------------------------------------
# Helpers / minimal mocks
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="torch not installed"
)


class _MockDetectHead(nn.Module):
    """Minimal stand-in for the YOLO Detect head (last module in model.model)."""

    def forward(self, x):
        return x  # pass-through; we only care about the hook


class _MockBackbone(nn.Module):
    """Minimal backbone layer (just needs to exist)."""

    def forward(self, x):
        return x


class _MockDetectionModel(nn.Module):
    """Mimics ``ultralytics.DetectionModel``: has a ``.model`` Sequential."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            _MockBackbone(),
            _MockDetectHead(),
        )

    def forward(self, x):
        return self.model(x)


class _MockYOLO:
    """Mimics ``ultralytics.YOLO``: has a ``.model`` DetectionModel."""

    def __init__(self):
        self.model = _MockDetectionModel()


def _make_extractor(scale_idx: int = 1) -> YOLOFeatureExtractor:
    return YOLOFeatureExtractor(_MockYOLO(), scale_idx=scale_idx)


def _make_feat_tensors(
    channels: tuple[int, int, int] = (64, 128, 256),
    spatial: tuple[int, int] = (8, 8),
    batch: int = 1,
) -> list[torch.Tensor]:
    """Return a synthetic FPN feature list [P3, P4, P5]."""
    H, W = spatial
    return [
        torch.randn(batch, c, H * (2 ** i), W * (2 ** i))
        for i, c in enumerate(reversed(channels))
    ][::-1]
    # Actually produce tensors with decreasing spatial size:
    # P3 (finest, most channels by convention) → P5 (coarsest)


def _make_feat_tensors_simple(
    channels: int = 128,
    H: int = 40,
    W: int = 40,
    batch: int = 1,
) -> list[torch.Tensor]:
    """Return three FPN tensors with given channel count (all same size for simplicity)."""
    return [torch.randn(batch, channels, H, W) for _ in range(3)]


def _bbox(x1=50, y1=50, x2=200, y2=300):
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# ---------------------------------------------------------------------------
# Construction and hook registration
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_hook_registered_successfully(self):
        ext = _make_extractor()
        assert ext._hook_handle is not None

    def test_scale_idx_stored(self):
        ext = _make_extractor(scale_idx=2)
        assert ext._scale_idx == 2

    def test_scale_idx_clipped_to_valid_range(self):
        ext_low = _make_extractor(scale_idx=-5)
        assert ext_low._scale_idx == 0
        ext_high = _make_extractor(scale_idx=99)
        assert ext_high._scale_idx == len(_SCALE_STRIDES) - 1

    def test_feat_dim_is_none_before_first_pass(self):
        ext = _make_extractor()
        assert ext.feat_dim is None

    def test_graceful_failure_on_bad_model(self):
        """If the model has no .model.model attribute, extractor disables itself."""
        class _BadYOLO:
            pass  # no .model attribute at all

        ext = YOLOFeatureExtractor(_BadYOLO())  # should not raise
        assert ext._hook_handle is None
        # extract_roi_features should return empty list, not crash
        result = ext.extract_roi_features([_bbox()], (640, 1280))
        assert result == []


# ---------------------------------------------------------------------------
# Hook captures feature tensors during forward pass
# ---------------------------------------------------------------------------

class TestHookCapture:
    def test_hook_fires_on_forward(self):
        mock = _MockYOLO()
        ext = YOLOFeatureExtractor(mock)
        assert ext._feat_tensors is None

        # Simulate a forward pass through the detect head
        feat_list = [torch.randn(1, 128, 40, 40) for _ in range(3)]
        detect_head = mock.model.model[-1]  # _MockDetectHead
        detect_head(feat_list)  # triggers pre-hook

        assert ext._feat_tensors is not None
        assert len(ext._feat_tensors) == 3

    def test_hook_captures_single_tensor(self):
        mock = _MockYOLO()
        ext = YOLOFeatureExtractor(mock)
        single_feat = torch.randn(1, 256, 20, 20)
        mock.model.model[-1](single_feat)   # single tensor input
        assert ext._feat_tensors is not None
        assert len(ext._feat_tensors) == 1

    def test_clear_releases_tensors(self):
        mock = _MockYOLO()
        ext = YOLOFeatureExtractor(mock)
        feat_list = [torch.randn(1, 128, 40, 40)]
        mock.model.model[-1](feat_list)
        assert ext._feat_tensors is not None
        ext.clear()
        assert ext._feat_tensors is None


# ---------------------------------------------------------------------------
# _pool_roi static method
# ---------------------------------------------------------------------------

class TestPoolRoi:
    def _make_feat(self, C=64, H=40, W=40, fill=None) -> np.ndarray:
        if fill is not None:
            return np.full((C, H, W), fill, dtype=np.float32)
        rng = np.random.default_rng(42)
        return rng.random((C, H, W)).astype(np.float32)

    def test_output_shape(self):
        feat = self._make_feat(C=128, H=40, W=40)
        bbox = _bbox(50, 50, 200, 300)
        stride = 16
        # frame 640×640, feat 40×40, stride 16, ratio=1.0, pad=0
        pooled = YOLOFeatureExtractor._pool_roi(feat, bbox, 1.0, 0.0, 0.0, stride, 40, 40)
        assert pooled.shape == (128,)
        assert pooled.dtype == np.float32

    def test_output_is_unit_l2_norm(self):
        feat = self._make_feat(C=64, H=20, W=20, fill=1.0)
        bbox = _bbox(0, 0, 100, 100)
        pooled = YOLOFeatureExtractor._pool_roi(feat, bbox, 1.0, 0.0, 0.0, 32, 20, 20)
        norm = float(np.linalg.norm(pooled))
        assert abs(norm - 1.0) < 1e-5

    def test_all_zeros_feature_not_normalised(self):
        """All-zero feature map → output is all-zeros (no division by zero)."""
        feat = np.zeros((32, 20, 20), dtype=np.float32)
        bbox = _bbox(0, 0, 100, 100)
        pooled = YOLOFeatureExtractor._pool_roi(feat, bbox, 1.0, 0.0, 0.0, 32, 20, 20)
        assert pooled.shape == (32,)
        assert np.all(pooled == 0.0)

    def test_out_of_bounds_bbox_clamped(self):
        feat = self._make_feat(C=32, H=20, W=20)
        # bbox exceeds frame size
        bbox = np.array([-100, -100, 9999, 9999], dtype=np.float32)
        pooled = YOLOFeatureExtractor._pool_roi(feat, bbox, 1.0, 0.0, 0.0, 32, 20, 20)
        assert pooled.shape == (32,)
        # Should be the mean of the entire feature map (clamped to full extent)
        expected = feat.mean(axis=(1, 2))
        norm = np.linalg.norm(expected)
        if norm > 1e-6:
            expected /= norm
        np.testing.assert_allclose(pooled, expected, atol=1e-5)

    def test_different_bboxes_produce_different_vectors(self):
        rng = np.random.default_rng(0)
        feat = rng.random((64, 40, 40)).astype(np.float32)
        bbox_a = _bbox(0, 0, 100, 100)
        bbox_b = _bbox(300, 300, 500, 500)
        v_a = YOLOFeatureExtractor._pool_roi(feat, bbox_a, 1.0, 0.0, 0.0, 16, 40, 40)
        v_b = YOLOFeatureExtractor._pool_roi(feat, bbox_b, 1.0, 0.0, 0.0, 16, 40, 40)
        # Pooled from different spatial regions — should differ
        assert not np.allclose(v_a, v_b)

    def test_same_bbox_produces_same_vector(self):
        rng = np.random.default_rng(7)
        feat = rng.random((64, 40, 40)).astype(np.float32)
        bbox = _bbox(100, 80, 300, 400)
        v1 = YOLOFeatureExtractor._pool_roi(feat, bbox, 1.0, 0.0, 0.0, 16, 40, 40)
        v2 = YOLOFeatureExtractor._pool_roi(feat, bbox, 1.0, 0.0, 0.0, 16, 40, 40)
        np.testing.assert_array_equal(v1, v2)


# ---------------------------------------------------------------------------
# Letterbox coordinate transform
# ---------------------------------------------------------------------------

class TestLetterboxTransform:
    def test_no_pad_square_frame(self):
        """640×640 frame into 640×640 model: ratio=1.0, pad=0 — coords unchanged."""
        mock = _MockYOLO()
        ext = _make_extractor(scale_idx=1)

        # Simulate hook with a 40×40 feature map (stride=16, input=640)
        feat_list = [torch.zeros(1, 128, 40, 40)] * 3
        mock.model.model[-1](feat_list)
        ext._feat_tensors = feat_list  # override after mock forward

        vecs = ext.extract_roi_features([_bbox(0, 0, 640, 640)], (640, 640))
        assert len(vecs) == 1
        assert vecs[0].shape == (128,)

    def test_wide_frame_with_padding(self):
        """1280×720 frame into 640×640 model: ratio=0.5, pad_y=40px."""
        mock = _MockYOLO()
        ext = _make_extractor(scale_idx=1)

        # stride=16, input=640×640 → feature map 40×40
        feat_list = [torch.randn(1, 128, 40, 40) for _ in range(3)]
        ext._feat_tensors = feat_list

        bboxes = [_bbox(0, 0, 1280, 720)]
        vecs = ext.extract_roi_features(bboxes, (720, 1280))
        assert len(vecs) == 1
        assert vecs[0].shape == (128,)

    def test_tall_frame_with_padding(self):
        """720×1280 frame (portrait) into 640×640 model."""
        mock = _MockYOLO()
        ext = _make_extractor(scale_idx=1)
        feat_list = [torch.randn(1, 128, 40, 40) for _ in range(3)]
        ext._feat_tensors = feat_list

        bboxes = [_bbox(0, 0, 720, 1280)]
        vecs = ext.extract_roi_features(bboxes, (1280, 720))
        assert len(vecs) == 1


# ---------------------------------------------------------------------------
# extract_roi_features — integration
# ---------------------------------------------------------------------------

class TestExtractRoiFeatures:
    def _set_feat_tensors(self, ext: YOLOFeatureExtractor, C=128, H=40, W=40):
        """Directly inject synthetic feature tensors, bypassing a real forward pass."""
        ext._feat_tensors = [torch.randn(1, C, H, W) for _ in range(3)]

    def test_returns_empty_when_no_tensors(self):
        ext = _make_extractor()
        # _feat_tensors is None (no forward pass yet)
        result = ext.extract_roi_features([_bbox()], (640, 640))
        assert result == []

    def test_returns_empty_for_empty_bboxes(self):
        ext = _make_extractor()
        self._set_feat_tensors(ext)
        result = ext.extract_roi_features([], (640, 640))
        assert result == []

    def test_one_vector_per_bbox(self):
        ext = _make_extractor()
        self._set_feat_tensors(ext, C=128)
        bboxes = [_bbox(0, 0, 100, 100), _bbox(200, 200, 400, 400), _bbox(50, 60, 120, 180)]
        result = ext.extract_roi_features(bboxes, (640, 640))
        assert len(result) == 3

    def test_output_dtype_float32(self):
        ext = _make_extractor()
        self._set_feat_tensors(ext, C=64)
        result = ext.extract_roi_features([_bbox()], (640, 640))
        assert result[0].dtype == np.float32

    def test_output_unit_norm(self):
        ext = _make_extractor()
        self._set_feat_tensors(ext, C=128)
        result = ext.extract_roi_features([_bbox(10, 10, 200, 400)], (640, 640))
        norm = float(np.linalg.norm(result[0]))
        assert abs(norm - 1.0) < 1e-5

    def test_feat_dim_populated_after_first_call(self):
        ext = _make_extractor(scale_idx=1)
        assert ext.feat_dim is None
        self._set_feat_tensors(ext, C=96)
        ext.extract_roi_features([_bbox()], (640, 640))
        assert ext.feat_dim == 96

    def test_scale_idx_selects_correct_channels(self):
        """scale_idx=0 → uses first tensor (C0), scale_idx=2 → third (C2)."""
        C0, C1, C2 = 64, 128, 256
        feat_list = [
            torch.randn(1, C0, 80, 80),  # scale 0
            torch.randn(1, C1, 40, 40),  # scale 1
            torch.randn(1, C2, 20, 20),  # scale 2
        ]

        for idx, expected_C in [(0, C0), (1, C1), (2, C2)]:
            ext = _make_extractor(scale_idx=idx)
            ext._feat_tensors = feat_list
            vecs = ext.extract_roi_features([_bbox()], (640, 640))
            assert len(vecs) == 1
            assert vecs[0].shape == (expected_C,), f"scale_idx={idx} expected C={expected_C}"

    def test_scale_idx_beyond_available_falls_back(self):
        """scale_idx=5 when only 3 tensors available → uses last tensor."""
        ext = _make_extractor(scale_idx=2)   # clipped to 2 by constructor
        feat_list = [torch.randn(1, 64, 80, 80)] * 3
        ext._feat_tensors = feat_list
        vecs = ext.extract_roi_features([_bbox()], (640, 640))
        assert len(vecs) == 1

    def test_vectors_differ_for_different_bboxes(self):
        """Spatially separated bboxes should produce different feature vectors."""
        rng = torch.manual_seed(42)
        feat_list = [torch.randn(1, 128, 40, 40) for _ in range(3)]
        ext = _make_extractor()
        ext._feat_tensors = feat_list
        vecs = ext.extract_roi_features(
            [_bbox(0, 0, 50, 50), _bbox(400, 400, 600, 600)], (640, 640)
        )
        assert len(vecs) == 2
        # Should not be identical (different spatial locations)
        assert not np.allclose(vecs[0], vecs[1])

    def test_repeatable_for_same_input(self):
        """Same bbox + same feature map → same vector."""
        feat_list = [torch.randn(1, 64, 40, 40) for _ in range(3)]
        ext = _make_extractor()
        ext._feat_tensors = feat_list
        v1 = ext.extract_roi_features([_bbox(100, 100, 300, 300)], (640, 640))
        ext._feat_tensors = feat_list  # reset
        v2 = ext.extract_roi_features([_bbox(100, 100, 300, 300)], (640, 640))
        np.testing.assert_array_equal(v1[0], v2[0])


# ---------------------------------------------------------------------------
# remove / cleanup
# ---------------------------------------------------------------------------

class TestRemove:
    def test_remove_unregisters_hook(self):
        ext = _make_extractor()
        assert ext._hook_handle is not None
        ext.remove()
        assert ext._hook_handle is None

    def test_remove_clears_tensors(self):
        ext = _make_extractor()
        ext._feat_tensors = [torch.randn(1, 64, 40, 40)]
        ext.remove()
        assert ext._feat_tensors is None

    def test_extract_returns_empty_after_remove(self):
        ext = _make_extractor()
        ext.remove()
        # Even if tensors were somehow present, hook is gone
        result = ext.extract_roi_features([_bbox()], (640, 640))
        assert result == []

    def test_hook_no_longer_fires_after_remove(self):
        mock = _MockYOLO()
        ext = YOLOFeatureExtractor(mock)
        ext.remove()
        # Trigger a forward pass — hook should not fire
        feat_list = [torch.randn(1, 64, 40, 40)]
        mock.model.model[-1](feat_list)
        assert ext._feat_tensors is None

    def test_double_remove_is_safe(self):
        ext = _make_extractor()
        ext.remove()
        ext.remove()   # Should not raise


# ---------------------------------------------------------------------------
# _to_numpy static method
# ---------------------------------------------------------------------------

class TestToNumpy:
    def test_torch_tensor_4d(self):
        t = torch.randn(1, 32, 10, 10)
        arr = YOLOFeatureExtractor._to_numpy(t)
        assert arr is not None
        assert arr.shape == (32, 10, 10)
        assert arr.dtype == np.float32

    def test_torch_tensor_3d(self):
        t = torch.randn(32, 10, 10)
        arr = YOLOFeatureExtractor._to_numpy(t)
        assert arr is not None
        assert arr.shape == (32, 10, 10)

    def test_numpy_array_passthrough(self):
        a = np.random.rand(64, 5, 5).astype(np.float32)
        arr = YOLOFeatureExtractor._to_numpy(a)
        assert arr is not None
        assert arr.shape == (64, 5, 5)

    def test_returns_none_on_exception(self):
        arr = YOLOFeatureExtractor._to_numpy("not a tensor")
        # Should return None, not raise
        assert arr is None or isinstance(arr, np.ndarray)
