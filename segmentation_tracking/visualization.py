"""
visualization.py
----------------
Frame annotation: semi-transparent segmentation masks, player IDs, pose
skeletons, and ball overlay.

Public API
~~~~~~~~~~
``Visualizer``
    Stateless helper class.  Call ``Visualizer.draw_frame(frame, player_tracks,
    ball_track)`` to get an annotated copy of a frame.
"""

from __future__ import annotations

import colorsys
import logging
from typing import Sequence

import cv2
import numpy as np

from segmentation_tracking.association import (
    BallTrack,
    COCO_SKELETON,
    PlayerTrack,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_MASK_ALPHA = 0.40          # transparency of the segmentation overlay
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.6
_LABEL_THICKNESS = 2
_SKEL_THICKNESS = 2
_KP_RADIUS = 4
_KP_CONF_THRESHOLD = 0.3    # draw keypoints only above this confidence
_BALL_COLOR = (0, 255, 255)  # cyan for ball
_BALL_MASK_ALPHA = 0.50

# Golden ratio conjugate – spreads player IDs to perceptually distinct hues
_GOLDEN_RATIO = 0.6180339887


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def _id_to_color(player_id: int) -> tuple[int, int, int]:
    """Map a player ID to a distinct BGR colour using the golden-ratio hue."""
    hue = (player_id * _GOLDEN_RATIO) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR


# ─────────────────────────────────────────────────────────────────────────────
# Visualizer
# ─────────────────────────────────────────────────────────────────────────────

class Visualizer:
    """Annotates video frames with masks, player IDs, and pose skeletons.

    Parameters
    ----------
    mask_alpha:
        Opacity of the segmentation colour overlay (0 = transparent,
        1 = opaque).  Defaults to 0.40.
    show_bbox:
        Draw the bounding box rectangle around each player.
    show_id:
        Render the ``[Player ID: N]`` label above the bounding box.
    show_skeleton:
        Render the COCO pose skeleton.
    show_ball:
        Render the ball mask overlay and centre dot.
    keypoint_conf_threshold:
        Only draw keypoints whose confidence exceeds this value.
    """

    def __init__(
        self,
        mask_alpha: float = _MASK_ALPHA,
        show_bbox: bool = True,
        show_id: bool = True,
        show_skeleton: bool = True,
        show_ball: bool = True,
        keypoint_conf_threshold: float = _KP_CONF_THRESHOLD,
    ) -> None:
        self.mask_alpha = mask_alpha
        self.show_bbox = show_bbox
        self.show_id = show_id
        self.show_skeleton = show_skeleton
        self.show_ball = show_ball
        self.keypoint_conf_threshold = keypoint_conf_threshold

    # ── Public ────────────────────────────────────────────────────────────────

    def draw_frame(
        self,
        frame: np.ndarray,
        player_tracks: Sequence[PlayerTrack],
        ball_track: BallTrack | None = None,
    ) -> np.ndarray:
        """
        Annotate a single frame and return the result.

        The original *frame* is not modified.

        Parameters
        ----------
        frame:
            BGR image array, shape ``(H, W, 3)``.
        player_tracks:
            Iterable of :class:`~segmentation_tracking.association.PlayerTrack`
            for the current frame.
        ball_track:
            Optional :class:`~segmentation_tracking.association.BallTrack`.

        Returns
        -------
        np.ndarray
            Annotated BGR image of the same shape as *frame*.
        """
        canvas = frame.copy()

        # Draw player segmentation masks (colour overlay)
        for track in player_tracks:
            color = _id_to_color(track.id)
            canvas = self._draw_mask(canvas, track.mask, color)

        # Draw player bounding boxes, IDs, and skeletons on top of masks
        for track in player_tracks:
            color = _id_to_color(track.id)
            if self.show_bbox:
                self._draw_bbox(canvas, track.bbox, color)
            if self.show_id:
                self._draw_label(canvas, track.id, track.bbox, color)
            if self.show_skeleton and track.keypoints is not None:
                self._draw_skeleton(canvas, track.keypoints, track.keypoint_scores, color)

        # Draw ball
        if self.show_ball and ball_track is not None:
            canvas = self._draw_ball(canvas, ball_track)

        return canvas

    # ── Private drawing helpers ───────────────────────────────────────────────

    def _draw_mask(
        self,
        canvas: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int],
    ) -> np.ndarray:
        """Blend a semi-transparent coloured mask onto *canvas*."""
        if mask is None or not mask.any():
            return canvas

        h, w = canvas.shape[:2]
        mh, mw = mask.shape
        if mh != h or mw != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        color_layer = np.zeros_like(canvas, dtype=np.uint8)
        color_layer[mask > 0] = color

        # alpha-blend only in the mask region
        mask_bool = mask > 0
        canvas[mask_bool] = cv2.addWeighted(
            canvas, 1.0 - self.mask_alpha,
            color_layer, self.mask_alpha,
            0,
        )[mask_bool]

        return canvas

    @staticmethod
    def _draw_bbox(
        canvas: np.ndarray,
        bbox: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    @staticmethod
    def _draw_label(
        canvas: np.ndarray,
        player_id: int,
        bbox: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        label = f"Player {player_id}"
        x1, y1 = int(bbox[0]), int(bbox[1])

        (tw, th), baseline = cv2.getTextSize(
            label, _LABEL_FONT, _LABEL_SCALE, _LABEL_THICKNESS
        )
        # Position label above the bbox; clamp to frame top
        lx, ly = x1, max(y1 - baseline - 4, th + baseline)

        # Background rectangle
        cv2.rectangle(
            canvas,
            (lx - 2, ly - th - baseline - 2),
            (lx + tw + 2, ly + baseline + 2),
            color,
            cv2.FILLED,
        )
        # Text in white for contrast
        cv2.putText(
            canvas, label, (lx, ly),
            _LABEL_FONT, _LABEL_SCALE, (255, 255, 255), _LABEL_THICKNESS,
            cv2.LINE_AA,
        )

    def _draw_skeleton(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        keypoint_scores: np.ndarray | None,
        color: tuple[int, int, int],
    ) -> None:
        """Draw COCO skeleton lines and keypoint circles."""
        scores = (
            keypoint_scores
            if keypoint_scores is not None
            else np.ones(len(keypoints), dtype=np.float32)
        )
        # Draw bones
        for i, j in COCO_SKELETON:
            if i >= len(keypoints) or j >= len(keypoints):
                continue
            if scores[i] < self.keypoint_conf_threshold:
                continue
            if scores[j] < self.keypoint_conf_threshold:
                continue
            xi, yi = int(keypoints[i][0]), int(keypoints[i][1])
            xj, yj = int(keypoints[j][0]), int(keypoints[j][1])
            if xi == 0 and yi == 0:
                continue
            if xj == 0 and yj == 0:
                continue
            cv2.line(canvas, (xi, yi), (xj, yj), color, _SKEL_THICKNESS, cv2.LINE_AA)

        # Draw joints
        for idx, (kp, sc) in enumerate(zip(keypoints, scores)):
            if sc < self.keypoint_conf_threshold:
                continue
            x, y = int(kp[0]), int(kp[1])
            if x == 0 and y == 0:
                continue
            cv2.circle(canvas, (x, y), _KP_RADIUS, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), _KP_RADIUS, color, 1, cv2.LINE_AA)

    def _draw_ball(
        self,
        canvas: np.ndarray,
        ball_track: BallTrack,
    ) -> np.ndarray:
        """Draw ball mask overlay and centre marker."""
        if ball_track.mask is not None and ball_track.mask.any():
            h, w = canvas.shape[:2]
            bm = ball_track.mask
            if bm.shape[0] != h or bm.shape[1] != w:
                bm = cv2.resize(bm, (w, h), interpolation=cv2.INTER_NEAREST)

            color_layer = np.zeros_like(canvas, dtype=np.uint8)
            color_layer[bm > 0] = _BALL_COLOR

            mask_bool = bm > 0
            canvas[mask_bool] = cv2.addWeighted(
                canvas, 1.0 - _BALL_MASK_ALPHA,
                color_layer, _BALL_MASK_ALPHA,
                0,
            )[mask_bool]

        if ball_track.bbox is not None:
            x1, y1, x2, y2 = ball_track.bbox.astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), _BALL_COLOR, 2)

        # Centre dot
        cx, cy = int(ball_track.center[0]), int(ball_track.center[1])
        cv2.circle(canvas, (cx, cy), 6, _BALL_COLOR, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 6, (0, 0, 0), 1, cv2.LINE_AA)

        # Label
        cv2.putText(
            canvas, "Ball",
            (cx + 8, cy - 8),
            _LABEL_FONT, _LABEL_SCALE, _BALL_COLOR, _LABEL_THICKNESS, cv2.LINE_AA,
        )

        return canvas
