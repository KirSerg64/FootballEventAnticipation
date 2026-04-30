"""
pose_model.py - Pose estimation model loading and inference.

Supports two backends:
1. torchvision KeypointRCNN (default) — no extra dependencies required.
2. BBoxMaskPose (https://github.com/MiraPurkrabek/BBoxMaskPose) — optional,
   higher accuracy for occluded football players.

The BBoxMaskPose backend uses a top-down approach:
  - a person detector feeds bounding boxes to a ViTPose-based model
  - masking outside the bbox improves accuracy for occluded players

The KeypointRCNN backend performs joint detection + pose estimation
with a ResNet-50 FPN backbone, using COCO-pretrained weights.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# COCO 17-keypoint skeleton definition
COCO_KEYPOINT_NAMES: List[str] = [
    "nose",          # 0
    "left_eye",      # 1
    "right_eye",     # 2
    "left_ear",      # 3
    "right_ear",     # 4
    "left_shoulder", # 5
    "right_shoulder",# 6
    "left_elbow",    # 7
    "right_elbow",   # 8
    "left_wrist",    # 9
    "right_wrist",   # 10
    "left_hip",      # 11
    "right_hip",     # 12
    "left_knee",     # 13
    "right_knee",    # 14
    "left_ankle",    # 15
    "right_ankle",   # 16
]

# Skeleton limb connections (pairs of keypoint indices)
COCO_SKELETON: List[Tuple[int, int]] = [
    (0, 1), (0, 2),           # nose -> eyes
    (1, 3), (2, 4),           # eyes -> ears
    (5, 6),                   # shoulders
    (5, 7), (7, 9),           # left arm
    (6, 8), (8, 10),          # right arm
    (5, 11), (6, 12),         # shoulders -> hips
    (11, 12),                 # hips
    (11, 13), (13, 15),       # left leg
    (12, 14), (14, 16),       # right leg
]

# BGR color palette per limb group for visualization
LIMB_COLORS: List[Tuple[int, int, int]] = [
    (255, 0, 0),    # nose-eye
    (255, 0, 0),    # nose-eye
    (255, 128, 0),  # eye-ear
    (255, 128, 0),  # eye-ear
    (0, 255, 0),    # shoulders
    (0, 200, 255),  # left arm
    (0, 200, 255),
    (255, 0, 200),  # right arm
    (255, 0, 200),
    (0, 255, 100),  # torso
    (0, 255, 100),
    (100, 255, 0),  # hips
    (255, 220, 0),  # left leg
    (255, 220, 0),
    (200, 100, 255),# right leg
    (200, 100, 255),
]

KEYPOINT_COLOR: Tuple[int, int, int] = (0, 255, 255)


class PoseResult:
    """Container for pose estimation results for a single frame."""

    def __init__(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        bboxes: np.ndarray,
    ) -> None:
        """
        Args:
            keypoints: shape (N, 17, 2) — (x, y) per person per joint
            scores:    shape (N, 17)    — confidence per joint
            bboxes:    shape (N, 4)     — [x1, y1, x2, y2] per person
        """
        self.keypoints = keypoints  # (N, 17, 2)
        self.scores = scores        # (N, 17)
        self.bboxes = bboxes        # (N, 4)

    @property
    def num_persons(self) -> int:
        return self.keypoints.shape[0]


class BasePoseEstimator(ABC):
    """Abstract base class for pose estimators."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    @abstractmethod
    def predict(self, frame_bgr: np.ndarray) -> PoseResult:
        """Run pose estimation on a single BGR frame.

        Args:
            frame_bgr: HxWx3 uint8 numpy array (OpenCV format)

        Returns:
            PoseResult with detected poses
        """

    def to(self, device: torch.device) -> "BasePoseEstimator":
        self.device = device
        return self


class KeypointRCNNEstimator(BasePoseEstimator):
    """
    Pose estimator backed by torchvision KeypointRCNN (ResNet-50 FPN).

    This model jointly detects persons and estimates 17 COCO keypoints.
    Pretrained weights are downloaded automatically on first use.
    """

    def __init__(
        self,
        device: torch.device,
        score_threshold: float = 0.5,
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__(device)
        self.score_threshold = score_threshold
        self._load_model(weights_path)

    def _load_model(self, weights_path: Optional[str]) -> None:
        from torchvision.models.detection import (
            KeypointRCNN_ResNet50_FPN_Weights,
            keypointrcnn_resnet50_fpn,
        )

        if weights_path is not None:
            logger.info("Loading KeypointRCNN weights from %s", weights_path)
            self.model = keypointrcnn_resnet50_fpn(weights=None)
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
        else:
            logger.info(
                "Loading pretrained KeypointRCNN (COCO) weights from torchvision"
            )
            weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = keypointrcnn_resnet50_fpn(weights=weights)

        self.model.to(self.device)
        self.model.eval()
        logger.info("KeypointRCNN loaded on %s", self.device)

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> PoseResult:
        import torchvision.transforms.functional as TF

        # Convert BGR (H, W, 3) uint8 → RGB tensor (3, H, W) float [0, 1]
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        tensor = TF.to_tensor(frame_rgb).unsqueeze(0).to(self.device)

        outputs = self.model(tensor)
        out = outputs[0]

        # Filter by person detection score
        keep = out["scores"] > self.score_threshold
        if keep.sum() == 0:
            empty = np.zeros((0, 17, 2), dtype=np.float32)
            return PoseResult(
                keypoints=empty,
                scores=np.zeros((0, 17), dtype=np.float32),
                bboxes=np.zeros((0, 4), dtype=np.float32),
            )

        keypoints = out["keypoints"][keep].cpu().numpy()     # (N, 17, 3): x, y, vis
        kp_scores = out["keypoints_scores"][keep].cpu().numpy()  # (N, 17)
        bboxes = out["boxes"][keep].cpu().numpy()             # (N, 4)

        return PoseResult(
            keypoints=keypoints[:, :, :2],  # drop visibility flag
            scores=kp_scores,
            bboxes=bboxes,
        )


class BBoxMaskPoseEstimator(BasePoseEstimator):
    """
    Pose estimator backed by BBoxMaskPose (https://github.com/MiraPurkrabek/BBoxMaskPose).

    BBoxMaskPose extends ViTPose with bounding-box masking to improve
    accuracy for occluded football players.  This class wraps the
    MMPose-based inference API provided by BBoxMaskPose.

    Installation (see pose_estimation/README.md):
        git clone https://github.com/MiraPurkrabek/BBoxMaskPose.git
        cd BBoxMaskPose && pip install -e .
        pip install mmcv mmdet mmpose
    """

    def __init__(
        self,
        device: torch.device,
        config_file: str,
        checkpoint_file: str,
        det_config_file: str,
        det_checkpoint_file: str,
        score_threshold: float = 0.3,
    ) -> None:
        super().__init__(device)
        self.score_threshold = score_threshold
        self._load_models(
            config_file, checkpoint_file, det_config_file, det_checkpoint_file
        )

    def _load_models(
        self,
        config_file: str,
        checkpoint_file: str,
        det_config_file: str,
        det_checkpoint_file: str,
    ) -> None:
        try:
            from mmpose.apis import init_model as init_pose_model
            from mmdet.apis import init_detector
        except ImportError as exc:
            raise ImportError(
                "BBoxMaskPose requires mmpose and mmdet.\n"
                "Install with: pip install mmcv mmdet mmpose\n"
                "And clone: https://github.com/MiraPurkrabek/BBoxMaskPose"
            ) from exc

        device_str = str(self.device)
        logger.info("Loading BBoxMaskPose detector from %s", det_config_file)
        self._detector = init_detector(
            det_config_file, det_checkpoint_file, device=device_str
        )

        logger.info("Loading BBoxMaskPose pose model from %s", config_file)
        self._pose_model = init_pose_model(
            config_file, checkpoint_file, device=device_str
        )
        logger.info("BBoxMaskPose loaded on %s", self.device)

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> PoseResult:
        from mmdet.apis import inference_detector
        from mmpose.apis import inference_topdown
        from mmpose.structures import merge_data_samples

        # Step 1: person detection
        det_result = inference_detector(self._detector, frame_bgr)
        pred_instances = det_result.pred_instances.cpu().numpy()

        # Filter to person class (class index 0 in COCO)
        person_mask = pred_instances.labels == 0
        scores_det = pred_instances.scores[person_mask]
        bboxes_det = pred_instances.bboxes[person_mask]

        conf_mask = scores_det > self.score_threshold
        bboxes_det = bboxes_det[conf_mask]

        if len(bboxes_det) == 0:
            empty = np.zeros((0, 17, 2), dtype=np.float32)
            return PoseResult(
                keypoints=empty,
                scores=np.zeros((0, 17), dtype=np.float32),
                bboxes=np.zeros((0, 4), dtype=np.float32),
            )

        # Step 2: top-down pose estimation with BBoxMask
        pose_results = inference_topdown(self._pose_model, frame_bgr, bboxes_det)
        data_samples = merge_data_samples(pose_results)
        instances = data_samples.pred_instances.cpu().numpy()

        keypoints = instances.keypoints                 # (N, 17, 2)
        kp_scores = instances.keypoint_scores           # (N, 17)
        bboxes = instances.bboxes                       # (N, 4)

        return PoseResult(keypoints=keypoints, scores=kp_scores, bboxes=bboxes)


def create_pose_estimator(
    device: Optional[torch.device] = None,
    backend: str = "auto",
    score_threshold: float = 0.5,
    weights_path: Optional[str] = None,
    # BBoxMaskPose-specific arguments
    bbox_config: Optional[str] = None,
    bbox_checkpoint: Optional[str] = None,
    det_config: Optional[str] = None,
    det_checkpoint: Optional[str] = None,
) -> BasePoseEstimator:
    """
    Factory that creates the best available pose estimator.

    Args:
        device:           torch.device to run inference on.
                          Defaults to CUDA if available, else CPU.
        backend:          "auto" | "bboxmaskpose" | "keypointrcnn"
                          "auto" tries BBoxMaskPose first, then falls back.
        score_threshold:  Minimum detection confidence to keep a person.
        weights_path:     Optional path to local weights (KeypointRCNN only).
        bbox_config:      BBoxMaskPose pose model config path.
        bbox_checkpoint:  BBoxMaskPose pose model checkpoint path.
        det_config:       Person detector config path (BBoxMaskPose).
        det_checkpoint:   Person detector checkpoint path (BBoxMaskPose).

    Returns:
        A BasePoseEstimator instance ready for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Creating pose estimator | backend=%s | device=%s", backend, device)

    if backend in ("bboxmaskpose", "auto"):
        if bbox_config and bbox_checkpoint and det_config and det_checkpoint:
            try:
                return BBoxMaskPoseEstimator(
                    device=device,
                    config_file=bbox_config,
                    checkpoint_file=bbox_checkpoint,
                    det_config_file=det_config,
                    det_checkpoint_file=det_checkpoint,
                    score_threshold=score_threshold,
                )
            except ImportError as exc:
                if backend == "bboxmaskpose":
                    raise
                logger.warning(
                    "BBoxMaskPose not available (%s); falling back to KeypointRCNN",
                    exc,
                )
        elif backend == "bboxmaskpose":
            raise ValueError(
                "BBoxMaskPose backend requires --bbox_config, --bbox_checkpoint, "
                "--det_config, --det_checkpoint arguments."
            )

    # Default / fallback: torchvision KeypointRCNN
    return KeypointRCNNEstimator(
        device=device,
        score_threshold=score_threshold,
        weights_path=weights_path,
    )
