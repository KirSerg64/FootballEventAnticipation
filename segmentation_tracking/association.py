"""
association.py
--------------
Matches YOLO pose estimation keypoints to segmentation-tracked players.

The association is performed by computing the Intersection-over-Union (IoU)
between each pose bounding box and the per-player segmentation mask.  When two
detections have the same best IoU, the one with higher overlap wins.  If no
pose detection overlaps with a given tracked player (e.g. because the player is
partially occluded), the track is still returned but without keypoints.

Data structures
~~~~~~~~~~~~~~~
``PlayerTrack``
    Combined result for a single player in a single frame: segmentation mask,
    bounding box, player ID, and (optionally) pose keypoints / confidence
    scores.

``BallTrack``
    Ball position and mask for a single frame.

Public API
~~~~~~~~~~
``associate_poses_with_tracks(seg_result, pose_results, frame_shape)``
    Main entry point.  Returns a list of :class:`PlayerTrack` and a
    :class:`BallTrack` (or *None* if no ball detected).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# COCO 17-keypoint names (for documentation / downstream use)
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# COCO skeleton connectivity (pairs of keypoint indices, 0-based)
COCO_SKELETON = [
    (0, 1), (0, 2),          # nose → eyes
    (1, 3), (2, 4),          # eyes → ears
    (5, 6),                  # shoulders
    (5, 7), (7, 9),          # left arm
    (6, 8), (8, 10),         # right arm
    (5, 11), (6, 12),        # torso
    (11, 12),                # hips
    (11, 13), (13, 15),      # left leg
    (12, 14), (14, 16),      # right leg
]

# Small value subtracted from iou_threshold to initialise the best-match
# comparison so that a match at exactly the threshold is accepted.
_IOU_THRESHOLD_EPS = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlayerTrack:
    """Combined tracking + pose result for one player in one frame.

    Attributes
    ----------
    id:
        Persistent player identifier across frames.
    mask:
        Binary mask, shape ``(H, W)``, dtype ``uint8`` (0 / 255).
    bbox:
        Tight bounding box derived from the mask: ``[x1, y1, x2, y2]``.
    keypoints:
        Array of shape ``(17, 2)`` with (x, y) pixel coordinates for COCO
        keypoints, or *None* if no pose was associated.
    keypoint_scores:
        Array of shape ``(17,)`` with confidence scores for each keypoint,
        or *None* if no pose was associated.
    """

    id: int
    mask: np.ndarray
    bbox: np.ndarray
    keypoints: np.ndarray | None = None
    keypoint_scores: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "player_id": self.id,
            "bbox": self.bbox.tolist(),
            "mask": self.mask.tolist(),
            "keypoints": self.keypoints.tolist() if self.keypoints is not None else None,
            "keypoint_scores": (
                self.keypoint_scores.tolist()
                if self.keypoint_scores is not None
                else None
            ),
        }


@dataclass
class BallTrack:
    """Ball information for one frame.

    Attributes
    ----------
    center:
        ``(cx, cy)`` pixel coordinates of the ball centre.
    bbox:
        Bounding box ``[x1, y1, x2, y2]``, or *None*.
    mask:
        Binary mask for the ball, shape ``(H, W)``, or *None*.
    """

    center: tuple[float, float]
    bbox: np.ndarray | None = None
    mask: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "ball_center": list(self.center),
            "bbox": self.bbox.tolist() if self.bbox is not None else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# IoU helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union for two ``[x1, y1, x2, y2]`` boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-6))


def _center_distance(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    """Euclidean distance between the centres of two bounding boxes."""
    ca = np.array([(bbox_a[0] + bbox_a[2]) / 2, (bbox_a[1] + bbox_a[3]) / 2])
    cb = np.array([(bbox_b[0] + bbox_b[2]) / 2, (bbox_b[1] + bbox_b[3]) / 2])
    return float(np.linalg.norm(ca - cb))


def _pose_bbox(pose_result_box: np.ndarray) -> np.ndarray:
    """Extract ``[x1, y1, x2, y2]`` from a single YOLO pose box tensor."""
    return pose_result_box.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def associate_poses_with_tracks(
    seg_result: Any,
    pose_result: Any,
    frame_shape: tuple[int, int],
    iou_threshold: float = 0.1,
    max_center_dist: float | None = None,
) -> tuple[list[PlayerTrack], BallTrack | None]:
    """
    Associate pose estimations with segmentation-tracked players.

    Parameters
    ----------
    seg_result:
        A :class:`~segmentation_tracking.segmentation_model.SegmentationResult`
        for the current frame.
    pose_result:
        A single ultralytics YOLO pose result object (from
        ``YOLO("yolo11x-pose.pt").predict(frame)``).  May be *None* if pose
        estimation was skipped.
    frame_shape:
        ``(height, width)`` of the frame (used for centre-distance
        normalisation when ``max_center_dist`` is provided).
    iou_threshold:
        Minimum IoU required to associate a pose detection with a track.
        Lowering this helps when pose bboxes are slightly offset from masks.
    max_center_dist:
        Optional fallback maximum centre distance (pixels) used when no pose
        detection exceeds *iou_threshold*.

    Returns
    -------
    player_tracks:
        List of :class:`PlayerTrack`, one per tracked player in the frame.
        Players with no matched pose have ``keypoints = None``.
    ball_track:
        :class:`BallTrack` if a ball was detected, otherwise *None*.
    """
    player_tracks: list[PlayerTrack] = []
    ball_track: BallTrack | None = None

    # ── Build ball track ──────────────────────────────────────────────────────
    if seg_result.ball_center is not None:
        ball_track = BallTrack(
            center=seg_result.ball_center,
            bbox=seg_result.ball_bbox,
            mask=seg_result.ball_mask,
        )

    # ── Extract pose detections ───────────────────────────────────────────────
    pose_bboxes: list[np.ndarray] = []
    pose_keypoints: list[np.ndarray] = []
    pose_scores: list[np.ndarray] = []

    if pose_result is not None and pose_result.keypoints is not None:
        kps = pose_result.keypoints
        boxes = pose_result.boxes

        for i in range(len(kps)):
            if boxes is not None and i < len(boxes):
                bbox_i = boxes.xyxy[i].cpu().numpy().astype(np.float32)
            else:
                # Derive bbox from keypoints
                xy = kps.xy[i].cpu().numpy()
                valid = xy[(xy[:, 0] > 0) | (xy[:, 1] > 0)]
                if len(valid) == 0:
                    continue
                bbox_i = np.array(
                    [valid[:, 0].min(), valid[:, 1].min(),
                     valid[:, 0].max(), valid[:, 1].max()],
                    dtype=np.float32,
                )

            kp_xy = kps.xy[i].cpu().numpy()          # (17, 2)
            kp_conf = (
                kps.conf[i].cpu().numpy()             # (17,)
                if kps.conf is not None
                else np.ones(len(kp_xy), dtype=np.float32)
            )

            pose_bboxes.append(bbox_i)
            pose_keypoints.append(kp_xy)
            pose_scores.append(kp_conf)

    # ── Match pose → track via IoU ────────────────────────────────────────────
    n_poses = len(pose_bboxes)
    pose_used = [False] * n_poses

    for pid, mask, seg_bbox in zip(
        seg_result.player_ids,
        seg_result.player_masks,
        seg_result.player_bboxes,
    ):
        best_iou = iou_threshold - _IOU_THRESHOLD_EPS  # start below threshold
        best_pose_idx = -1

        for pi in range(n_poses):
            if pose_used[pi]:
                continue
            iou = _bbox_iou(seg_bbox, pose_bboxes[pi])
            if iou > best_iou:
                best_iou = iou
                best_pose_idx = pi

        # Fallback: centre-distance matching
        if best_pose_idx == -1 and max_center_dist is not None:
            min_dist = max_center_dist
            for pi in range(n_poses):
                if pose_used[pi]:
                    continue
                d = _center_distance(seg_bbox, pose_bboxes[pi])
                if d < min_dist:
                    min_dist = d
                    best_pose_idx = pi

        kp_arr: np.ndarray | None = None
        kp_scores: np.ndarray | None = None

        if best_pose_idx >= 0:
            kp_arr = pose_keypoints[best_pose_idx]
            kp_scores = pose_scores[best_pose_idx]
            pose_used[best_pose_idx] = True

        player_tracks.append(
            PlayerTrack(
                id=pid,
                mask=mask,
                bbox=seg_bbox,
                keypoints=kp_arr,
                keypoint_scores=kp_scores,
            )
        )

    logger.debug(
        "Frame association: %d tracks, %d pose detections, %d matched",
        len(player_tracks),
        n_poses,
        sum(1 for t in player_tracks if t.keypoints is not None),
    )

    return player_tracks, ball_track
