"""
visualization.py - Skeleton drawing and output video generation.

Renders detected poses (keypoints + limbs) over each frame and
writes the result to an .mp4 file using OpenCV.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from pose_estimation.pose_model import (
    COCO_SKELETON,
    KEYPOINT_COLOR,
    LIMB_COLORS,
    PoseResult,
)

logger = logging.getLogger(__name__)

# Visual style constants
KEYPOINT_RADIUS = 4
KEYPOINT_THICKNESS = -1        # filled circle
LIMB_THICKNESS = 2
BBOX_COLOR = (255, 165, 0)     # orange
BBOX_THICKNESS = 1
CONF_FONT_SCALE = 0.35
CONF_FONT = cv2.FONT_HERSHEY_SIMPLEX
CONF_COLOR = (255, 255, 255)


def draw_poses(
    frame: np.ndarray,
    pose_result: PoseResult,
    confidence_threshold: float = 0.3,
    show_bbox: bool = True,
    show_confidence: bool = False,
) -> np.ndarray:
    """
    Draw skeleton keypoints and limbs on a copy of *frame*.

    Args:
        frame:                HxWx3 BGR uint8 image.
        pose_result:          PoseResult from the pose estimator.
        confidence_threshold: Minimum keypoint score to draw.
        show_bbox:            Whether to draw person bounding boxes.
        show_confidence:      Whether to display keypoint score labels.

    Returns:
        Annotated BGR image (same size as input).
    """
    canvas = frame.copy()

    for person_idx in range(pose_result.num_persons):
        kps = pose_result.keypoints[person_idx]    # (17, 2)
        scores = pose_result.scores[person_idx]    # (17,)

        # --- Bounding box ---
        if show_bbox and len(pose_result.bboxes) > person_idx:
            x1, y1, x2, y2 = pose_result.bboxes[person_idx].astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), BBOX_COLOR, BBOX_THICKNESS)

        # --- Limbs ---
        for limb_idx, (start_idx, end_idx) in enumerate(COCO_SKELETON):
            if scores[start_idx] < confidence_threshold:
                continue
            if scores[end_idx] < confidence_threshold:
                continue

            x1_kp, y1_kp = kps[start_idx].astype(int)
            x2_kp, y2_kp = kps[end_idx].astype(int)
            color = LIMB_COLORS[limb_idx % len(LIMB_COLORS)]
            cv2.line(canvas, (x1_kp, y1_kp), (x2_kp, y2_kp), color, LIMB_THICKNESS)

        # --- Keypoints ---
        for kp_idx, (kp, score) in enumerate(zip(kps, scores)):
            if score < confidence_threshold:
                continue
            cx, cy = int(kp[0]), int(kp[1])
            cv2.circle(
                canvas, (cx, cy), KEYPOINT_RADIUS, KEYPOINT_COLOR, KEYPOINT_THICKNESS
            )
            if show_confidence:
                cv2.putText(
                    canvas,
                    f"{score:.2f}",
                    (cx + 4, cy - 4),
                    CONF_FONT,
                    CONF_FONT_SCALE,
                    CONF_COLOR,
                    1,
                    cv2.LINE_AA,
                )

    return canvas


class VideoWriter:
    """Context manager that writes annotated frames to an MP4 file."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float = 25.0,
        codec: str = "mp4v",
    ) -> None:
        """
        Args:
            output_path: Destination .mp4 file path.
            width:       Frame width in pixels.
            height:      Frame height in pixels.
            fps:         Frames per second.
            codec:       FourCC codec string (default mp4v).
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self._writer: Optional[cv2.VideoWriter] = None

    def __enter__(self) -> "VideoWriter":
        os.makedirs(Path(self.output_path).parent, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height)
        )
        if not self._writer.isOpened():
            raise IOError(
                f"Cannot open VideoWriter for {self.output_path}. "
                "Ensure OpenCV was built with video codec support."
            )
        logger.info(
            "VideoWriter opened: %s  (%dx%d @ %.1f fps)",
            self.output_path,
            self.width,
            self.height,
            self.fps,
        )
        return self

    def write(self, frame: np.ndarray) -> None:
        """Write a single BGR frame."""
        if self._writer is None:
            raise RuntimeError("VideoWriter is not open. Use as a context manager.")
        self._writer.write(frame)

    def __exit__(self, *_) -> None:
        if self._writer is not None:
            self._writer.release()
            logger.info("VideoWriter closed: %s", self.output_path)
            self._writer = None


class PoseVisualizer:
    """
    High-level helper that combines pose drawing with video writing.

    Usage::

        vis = PoseVisualizer(output_path="output/pose_visualization.mp4")
        vis.open(width=1920, height=1080, fps=25.0)
        for frame, pose_result in ...:
            vis.process_frame(frame, pose_result)
        vis.close()

    Or as a context manager::

        with PoseVisualizer(...) as vis:
            vis.open(...)
            vis.process_frame(...)
    """

    def __init__(
        self,
        output_path: str = "output/pose_visualization.mp4",
        confidence_threshold: float = 0.3,
        show_bbox: bool = True,
        show_confidence: bool = False,
    ) -> None:
        self.output_path = output_path
        self.confidence_threshold = confidence_threshold
        self.show_bbox = show_bbox
        self.show_confidence = show_confidence
        self._video_writer: Optional[VideoWriter] = None

    def open(self, width: int, height: int, fps: float = 25.0) -> None:
        """Open the underlying VideoWriter."""
        self._video_writer = VideoWriter(self.output_path, width, height, fps)
        self._video_writer.__enter__()

    def process_frame(self, frame: np.ndarray, pose_result: PoseResult) -> np.ndarray:
        """Draw poses on *frame*, write to output, and return the annotated image."""
        annotated = draw_poses(
            frame,
            pose_result,
            confidence_threshold=self.confidence_threshold,
            show_bbox=self.show_bbox,
            show_confidence=self.show_confidence,
        )
        if self._video_writer is not None:
            self._video_writer.write(annotated)
        return annotated

    def close(self) -> None:
        """Release the VideoWriter."""
        if self._video_writer is not None:
            self._video_writer.__exit__(None, None, None)
            self._video_writer = None

    def __enter__(self) -> "PoseVisualizer":
        return self

    def __exit__(self, *_) -> None:
        self.close()
