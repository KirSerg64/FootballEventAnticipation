"""
association.py
--------------
Matches pose estimation keypoints to segmentation-tracked players.

The association is performed by computing the Intersection-over-Union (IoU)
between each pose bounding box and the per-player segmentation mask bbox.
Instead of the earlier greedy matching, a globally-optimal assignment is
computed via the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``).
A centre-distance fallback runs a second Hungarian pass for tracks that
received no IoU match above the threshold.

If no pose detection overlaps with a given tracked player the track is still
returned, but with ``keypoints = None``.

Supported pose result formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``associate_poses_with_tracks`` accepts **two different** pose result objects:

1. **YOLO pose result** (ultralytics API):
   ``pose_result.keypoints`` (YOLO tensor), ``pose_result.boxes`` (YOLO tensor).

2. **pose_estimation.pose_model.PoseResult** (new, higher-performance backend):
   Plain numpy arrays — ``pose_result.keypoints`` ``(N, 17, 2)``,
   ``pose_result.scores`` ``(N, 17)``, ``pose_result.bboxes`` ``(N, 4)``.
   Detected by the presence of a ``num_persons`` attribute.

Data structures
~~~~~~~~~~~~~~~
``PlayerTrack``
    Combined result for a single player in a single frame: segmentation mask,
    bounding box, player ID, optional pose keypoints / confidence scores, and
    an optional team label (set externally by :class:`TeamClassifier`).

``BallTrack``
    Ball position and mask for a single frame.

Public API
~~~~~~~~~~
``associate_poses_with_tracks(seg_result, pose_result, frame_shape, ...)``
    Main entry point.  Returns a list of :class:`PlayerTrack` and a
    :class:`BallTrack` (or *None* if no ball detected).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from segmentation_tracking.segmentation_model import TrackerState

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
    (0, 1), (0, 2),          # nose -> eyes
    (1, 3), (2, 4),          # eyes -> ears
    (5, 6),                  # shoulders
    (5, 7), (7, 9),          # left arm
    (6, 8), (8, 10),         # right arm
    (5, 11), (6, 12),        # torso
    (11, 12),                # hips
    (11, 13), (13, 15),      # left leg
    (12, 14), (14, 16),      # right leg
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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
    team_label:
        Team index (0, 1, …) assigned by :class:`~segmentation_tracking.\
team_classifier.TeamClassifier`, or *None* if not yet classified.
    """

    id: int
    mask: np.ndarray
    bbox: np.ndarray
    keypoints: np.ndarray | None = None
    keypoint_scores: np.ndarray | None = None
    team_label: int | None = None

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
            "team_label": self.team_label,
        }


@dataclass
class BallTrack:
    """Ball information for one frame.

    Attributes
    ----------
    center:
        ``(cx, cy)`` pixel coordinates of the ball centre.
    bbox:
        Bounding box ``[x1, y1, x2, y2]``, or *None* when the position is
        a Kalman-filter prediction without a corresponding raw detection.
    mask:
        Binary mask for the ball, shape ``(H, W)``, or *None*.
    source:
        Source of the ball position.  One of:

        * ``"detected"`` — stage-1 global YOLO detection.
        * ``"roi"``      — stage-2 ROI YOLO detection (FRoG-MOT).
        * ``"mosse"``    — MOSSE correlation filter gap-fill.
        * ``"predicted"``— velocity extrapolation.
        * ``"none"``     — tracker not yet initialised (should not appear).
    """

    center: tuple[float, float]
    bbox: np.ndarray | None = None
    mask: np.ndarray | None = None
    source: str = "none"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "ball_center": list(self.center),
            "bbox": self.bbox.tolist() if self.bbox is not None else None,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def associate_poses_with_tracks(
    seg_result: TrackerState,
    pose_result: Any,
    frame_shape: tuple[int, int],
    iou_threshold: float = 0.1,
    max_center_dist: float | None = None,
) -> tuple[list[PlayerTrack], BallTrack | None]:
    """
    Associate pose estimations with segmentation-tracked players.

    Uses the Hungarian algorithm (globally-optimal 1-to-1 assignment) on the
    IoU cost matrix between pose bboxes and track bboxes.  A second Hungarian
    pass using centre distance handles tracks that received no IoU match when
    *max_center_dist* is provided.

    Parameters
    ----------
    seg_result:
        A :class:`~segmentation_tracking.segmentation_model.TrackerState`
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
    max_center_dist:
        Optional fallback maximum centre distance (pixels) used when no pose
        detection exceeds *iou_threshold* for a given track.

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

    # -- Build ball track -----------------------------------------------------
    if seg_result.ball_center is not None:
        ball_track = BallTrack(
            center=seg_result.ball_center,
            bbox=seg_result.xyxy[-1],
            mask=seg_result.mask[-1],
            source=getattr(seg_result, "ball_source", "none"),
        )

    # -- Extract pose detections ----------------------------------------------
    pose_bboxes: list[np.ndarray] = []
    pose_keypoints: list[np.ndarray] = []
    pose_scores: list[np.ndarray] = []

    if pose_result is not None:
        if hasattr(pose_result, "num_persons"):
            # ── New pose_estimation.pose_model.PoseResult format ──────────────
            # Plain numpy arrays: keypoints (N,17,2), scores (N,17), bboxes (N,4)
            for i in range(pose_result.num_persons):
                pose_bboxes.append(pose_result.bboxes[i].astype(np.float32))
                pose_keypoints.append(pose_result.keypoints[i].astype(np.float32))
                pose_scores.append(pose_result.scores[i].astype(np.float32))
        elif hasattr(pose_result, "keypoints") and pose_result.keypoints is not None:
            # ── Legacy YOLO ultralytics pose result format ────────────────────
            kps = pose_result.keypoints
            boxes = pose_result.boxes

            for i in range(len(kps)):
                if boxes is not None and i < len(boxes):
                    bbox_i = boxes.xyxy[i].cpu().numpy().astype(np.float32)
                else:
                    xy = kps.xy[i].cpu().numpy()
                    valid = xy[(xy[:, 0] > 0) | (xy[:, 1] > 0)]
                    if len(valid) == 0:
                        continue
                    bbox_i = np.array(
                        [valid[:, 0].min(), valid[:, 1].min(),
                         valid[:, 0].max(), valid[:, 1].max()],
                        dtype=np.float32,
                    )

                kp_xy = kps.xy[i].cpu().numpy()      # (17, 2)
                kp_conf = (
                    kps.conf[i].cpu().numpy()         # (17,)
                    if kps.conf is not None
                    else np.ones(len(kp_xy), dtype=np.float32)
                )

                pose_bboxes.append(bbox_i)
                pose_keypoints.append(kp_xy)
                pose_scores.append(kp_conf)

    # -- Hungarian IoU matching -----------------------------------------------
    n_tracks = len(seg_result.tracks.tracker_id)
    n_poses = len(pose_bboxes)

    # kp_map[track_idx] = pose_idx  (populated by matching passes)
    kp_map: dict[int, int] = {}
    matched_poses: set[int] = set()

    if n_tracks > 0 and n_poses > 0:
        # Build IoU cost matrix
        cost = np.ones((n_tracks, n_poses), dtype=np.float64)
        for ti, seg_bbox in enumerate(seg_result.tracks.xyxy):
            for pi, pose_bbox in enumerate(pose_bboxes):
                cost[ti, pi] = 1.0 - _bbox_iou(seg_bbox, pose_bbox)

        row_ind, col_ind = linear_sum_assignment(cost)
        for ti, pi in zip(row_ind, col_ind):
            if (1.0 - cost[ti, pi]) >= iou_threshold:
                kp_map[ti] = pi
                matched_poses.add(pi)

        # Fallback: centre-distance pass for unmatched tracks
        if max_center_dist is not None:
            unmatched_tracks = [ti for ti in range(n_tracks) if ti not in kp_map]
            unused_poses = [pi for pi in range(n_poses) if pi not in matched_poses]

            if unmatched_tracks and unused_poses:
                _large = max_center_dist + 1.0
                dist_cost = np.full(
                    (len(unmatched_tracks), len(unused_poses)), _large, dtype=np.float64
                )
                for i, ti in enumerate(unmatched_tracks):
                    for j, pi in enumerate(unused_poses):
                        d = _center_distance(
                            seg_result.tracks.xyxy[ti], pose_bboxes[pi]
                        )
                        if d < max_center_dist:
                            dist_cost[i, j] = d

                row_ind2, col_ind2 = linear_sum_assignment(dist_cost)
                for r, c in zip(row_ind2, col_ind2):
                    if dist_cost[r, c] < max_center_dist:
                        ti = unmatched_tracks[r]
                        pi = unused_poses[c]
                        kp_map[ti] = pi
                        matched_poses.add(pi)

    # -- Build PlayerTrack objects --------------------------------------------
    for ti, (pid, mask, seg_bbox) in enumerate(
        zip(seg_result.tracks.tracker_id, seg_result.tracks.mask, seg_result.tracks.xyxy)
    ):
        pi = kp_map.get(ti)
        player_tracks.append(
            PlayerTrack(
                id=pid,
                mask=mask,
                bbox=seg_bbox,
                keypoints=pose_keypoints[pi] if pi is not None else None,
                keypoint_scores=pose_scores[pi] if pi is not None else None,
            )
        )

    logger.debug(
        "Frame association: %d tracks, %d pose detections, %d matched",
        len(player_tracks),
        n_poses,
        sum(1 for t in player_tracks if t.keypoints is not None),
    )

    return player_tracks, ball_track
