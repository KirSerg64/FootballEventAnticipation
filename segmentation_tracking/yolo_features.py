"""
yolo_features.py
----------------
Extract YOLO backbone/neck intermediate feature vectors for each detection
without running a second forward pass.

A PyTorch ``forward_pre_hook`` is registered on the YOLO Detect head.
Every time ``YOLO.track()`` or ``YOLO.predict()`` is called, the hook
automatically captures the multi-scale FPN feature tensors that are fed
into the Detect head.  Calling
:meth:`YOLOFeatureExtractor.extract_roi_features` then performs
per-bounding-box RoI average pooling on those cached tensors to produce
fixed-length, L2-normalised feature vectors.

The entire feature extraction adds **zero** extra GPU compute: all tensors
are already computed during the normal detection forward pass.

Architecture details
~~~~~~~~~~~~~~~~~~~~
YOLO11/v8 produces three FPN feature maps at strides 8, 16, 32 px relative
to the model's letterboxed input size (``imgsz``).  The Detect head receives
them as ``args[0] = [P3_feat, P4_feat, P5_feat]`` where:

* ``P3_feat``  — stride  8  (finest spatial resolution; large feature maps)
* ``P4_feat``  — stride 16  (medium; best for person-sized objects)
* ``P5_feat``  — stride 32  (coarsest; most semantic)

The hook fires on ``forward_pre_hook``, i.e. **before** the Detect module
applies its convolutions, so the captured tensors are the raw FPN neck
outputs.

Letterbox correction
~~~~~~~~~~~~~~~~~~~~
YOLO letterboxes (scales + pads) the input frame before inference.
``extract_roi_features`` accepts the original frame dimensions and
automatically corrects for the letterbox transform so that bounding boxes
in **original-frame pixel coordinates** are correctly projected onto the
feature map grid.

Public API
~~~~~~~~~~
``YOLOFeatureExtractor(yolo_model, scale_idx=1)``
    Hook the model.  ``scale_idx`` selects which FPN output scale:
    0 = stride-8, 1 = stride-16 (default), 2 = stride-32.

``YOLOFeatureExtractor.extract_roi_features(bboxes, frame_hw) -> list[np.ndarray]``
    Return one L2-normalised float32 feature vector per bbox.
    Must be called **after** a ``YOLO.track()`` / ``YOLO.predict()`` call.

``YOLOFeatureExtractor.clear()``
    Release stored feature tensors (call at the end of each frame).

``YOLOFeatureExtractor.remove()``
    Unregister the hook and release all state.

``YOLOFeatureExtractor.feat_dim``
    Number of dimensions in the output vectors (read-only property).
    Populated after the first forward pass; *None* until then.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# FPN output strides indexed by scale_idx.
# These are fixed in YOLOv8/YOLO11 architecture regardless of imgsz.
_SCALE_STRIDES: list[int] = [8, 16, 32]


class YOLOFeatureExtractor:
    """Extract per-detection FPN features from a YOLO model without extra cost.

    A single :meth:`torch.nn.Module.register_forward_pre_hook` is placed on
    the YOLO Detect head.  It fires automatically during every call to
    ``YOLO.track()`` or ``YOLO.predict()``, capturing the FPN feature tensors
    with zero additional GPU compute.

    Parameters
    ----------
    yolo_model:
        A loaded ``ultralytics.YOLO`` instance
        (e.g. ``YOLO("yolo11x.pt")``).  The model must be loaded before
        passing it here so that ``yolo_model.model.model`` is populated.
    scale_idx:
        Which FPN scale to pool features from:

        * ``0`` — stride  8  (finest; high spatial resolution)
        * ``1`` — stride 16  (medium; **default**; best for persons)
        * ``2`` — stride 32  (coarsest; most semantic)

    Raises
    ------
    RuntimeError
        If the YOLO Detect head cannot be found in
        ``yolo_model.model.model``.  When this happens, the extractor logs a
        warning and disables itself (all ``extract_roi_features`` calls return
        empty lists) rather than crashing the main tracking pipeline.
    """

    def __init__(self, yolo_model, scale_idx: int = 1) -> None:
        self._scale_idx: int = max(0, min(scale_idx, len(_SCALE_STRIDES) - 1))
        self._feat_tensors: list | None = None   # set by hook each forward pass
        self._feat_dim: int | None = None         # populated after first pass
        self._hook_handle = None

        self._register_hook(yolo_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def feat_dim(self) -> int | None:
        """Channel count of the extracted feature vectors.

        *None* until the first ``YOLO.track()`` / ``YOLO.predict()`` call
        after which the hook has fired and the tensor shape is known.
        """
        return self._feat_dim

    def extract_roi_features(
        self,
        bboxes: list[np.ndarray],
        frame_hw: tuple[int, int],
    ) -> list[np.ndarray]:
        """Return one L2-normalised feature vector per bounding box.

        Must be called **after** a ``YOLO.track()`` / ``YOLO.predict()``
        invocation so that the hook has captured the current frame's FPN
        tensors.  Call :meth:`clear` at the end of each frame to release
        the tensors.

        Parameters
        ----------
        bboxes:
            List of ``[x1, y1, x2, y2]`` bounding boxes in **original-frame
            pixel coordinates** (the same coordinate system as
            ``YOLO.track()`` results).
        frame_hw:
            ``(height, width)`` of the frame that was passed to YOLO.

        Returns
        -------
        list[np.ndarray]
            One L2-normalised ``float32`` array of shape ``(C,)`` per bbox,
            where ``C`` is the channel count of the selected FPN scale.
            Returns an **empty list** when no feature tensors were captured
            (e.g. hook not registered) or *bboxes* is empty.
        """
        if self._feat_tensors is None or not bboxes:
            return []

        tensors = self._feat_tensors
        idx = min(self._scale_idx, len(tensors) - 1)
        feat = tensors[idx]

        # Convert to a CPU numpy array (C, H_f, W_f)
        feat_np = self._to_numpy(feat)
        if feat_np is None or feat_np.ndim != 3:
            logger.debug("YOLOFeatureExtractor: unexpected feature tensor shape %s", feat_np.shape if feat_np is not None else "None")
            return []

        C, H_f, W_f = feat_np.shape

        # Update feat_dim after seeing the first real tensor
        if self._feat_dim is None:
            self._feat_dim = C

        fh, fw = frame_hw

        # Letterbox transform: YOLO scales the frame to the model's square
        # input (imgsz × imgsz) while preserving aspect ratio, then pads
        # gray borders.  We infer the model input size from the feature map
        # dimensions and the known stride for this scale.
        stride = _SCALE_STRIDES[self._scale_idx]
        input_h = H_f * stride
        input_w = W_f * stride

        # Scale ratio and padding offsets (pixels in model-input space)
        ratio = min(input_w / fw, input_h / fh)
        pad_x = (input_w - fw * ratio) / 2.0
        pad_y = (input_h - fh * ratio) / 2.0

        results: list[np.ndarray] = []
        for bbox in bboxes:
            vec = self._pool_roi(feat_np, bbox, ratio, pad_x, pad_y, stride, W_f, H_f)
            results.append(vec)

        return results

    def clear(self) -> None:
        """Release stored feature tensors from the previous forward pass.

        Call this at the end of each frame to avoid holding GPU tensors in
        memory longer than necessary.
        """
        self._feat_tensors = None

    def remove(self) -> None:
        """Unregister the forward hook and release all state.

        After calling this method the extractor is inert; create a new
        instance if feature extraction is needed again.
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._feat_tensors = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _register_hook(self, yolo_model) -> None:
        """Register a ``forward_pre_hook`` on the YOLO Detect head.

        The hook captures ``args[0]``, which is the list of FPN feature
        tensors passed to the Detect head just before its convolutions run.
        """
        try:
            # Ultralytics model hierarchy:
            #   yolo_model        → ultralytics.YOLO wrapper
            #   yolo_model.model  → DetectionModel (nn.Module)
            #   yolo_model.model.model  → nn.Sequential of all layer modules
            nn_seq = yolo_model.model.model   # nn.Sequential
            detect_module = nn_seq[-1]         # Detect head (always last)

            def _pre_hook(module, args):
                # args[0] is the FPN feature list: [P3, P4, P5]
                raw = args[0]
                if isinstance(raw, (list, tuple)):
                    self._feat_tensors = list(raw)
                else:
                    # Some custom heads pass a single tensor
                    self._feat_tensors = [raw]

            self._hook_handle = detect_module.register_forward_pre_hook(_pre_hook)
            logger.debug(
                "YOLOFeatureExtractor: hook registered on %s (scale_idx=%d, stride=%d)",
                type(detect_module).__name__,
                self._scale_idx,
                _SCALE_STRIDES[self._scale_idx],
            )

        except Exception as exc:            # Non-fatal: log and disable extraction gracefully so the
            # main tracker still works without YOLO features.
            logger.warning(
                "YOLOFeatureExtractor: could not register hook (%s). "
                "YOLO feature extraction will be disabled.",
                exc,
            )
            self._hook_handle = None

    @staticmethod
    def _to_numpy(feat) -> np.ndarray | None:
        """Convert a torch Tensor (or numpy array) to a 3-D numpy array.

        Removes a leading batch dimension if present.
        """
        try:
            import torch
            if isinstance(feat, torch.Tensor):
                arr = feat.float().detach().cpu().numpy()
            else:
                arr = np.asarray(feat, dtype=np.float32)
        except Exception:
            return None

        # Remove batch dimension: (B, C, H, W) → (C, H, W)
        if arr.ndim == 4:
            arr = arr[0]
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _pool_roi(
        feat_np: np.ndarray,
        bbox: np.ndarray,
        ratio: float,
        pad_x: float,
        pad_y: float,
        stride: int,
        W_f: int,
        H_f: int,
    ) -> np.ndarray:
        """Average-pool ``feat_np`` over the region corresponding to ``bbox``.

        Applies the letterbox transform (scale + pad) and then divides by
        ``stride`` to obtain feature-map coordinates.

        Parameters
        ----------
        feat_np:
            Feature map, shape ``(C, H_f, W_f)``.
        bbox:
            ``[x1, y1, x2, y2]`` in **original-frame** pixel coordinates.
        ratio:
            Letterbox scale ratio (original → model input).
        pad_x, pad_y:
            Letterbox padding in model-input-space pixels.
        stride:
            FPN stride for the selected scale.
        W_f, H_f:
            Feature map spatial dimensions.

        Returns
        -------
        L2-normalised ``float32`` array of shape ``(C,)``.
        """
        # Map original-frame bbox → model-input-space bbox → feature-map bbox
        x1_f = (float(bbox[0]) * ratio + pad_x) / stride
        y1_f = (float(bbox[1]) * ratio + pad_y) / stride
        x2_f = (float(bbox[2]) * ratio + pad_x) / stride
        y2_f = (float(bbox[3]) * ratio + pad_y) / stride

        ix1 = max(0, min(int(x1_f), W_f - 1))
        iy1 = max(0, min(int(y1_f), H_f - 1))
        ix2 = max(ix1 + 1, min(int(np.ceil(x2_f)), W_f))
        iy2 = max(iy1 + 1, min(int(np.ceil(y2_f)), H_f))

        region = feat_np[:, iy1:iy2, ix1:ix2]   # (C, h_roi, w_roi)
        pooled = region.mean(axis=(1, 2)).astype(np.float32)  # (C,)

        norm = float(np.linalg.norm(pooled))
        if norm > 1e-6:
            pooled = pooled / norm
        return pooled
