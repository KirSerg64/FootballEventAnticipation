"""
pose_inference.py - Frame-level and video-level pose inference pipeline.

Supports two input modes:
  - Video file  (.mp4, .avi, etc.)
  - Directory of image frames (.jpg, .png, etc.)

Output is always a list of (frame_bgr, PoseResult) tuples, which are
passed to the visualization module to produce the output video.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

from pose_estimation.pose_model import BasePoseEstimator, PoseResult

logger = logging.getLogger(__name__)

# Image extensions treated as frames when the input is a directory
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _iter_video_frames(
    video_path: str,
    max_frames: Optional[int] = None,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Yield (frame_index, bgr_frame) tuples from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1
            if max_frames is not None and idx >= max_frames:
                break
    finally:
        cap.release()


def _iter_directory_frames(
    frames_dir: str,
    max_frames: Optional[int] = None,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Yield (frame_index, bgr_frame) tuples from sorted image files."""
    paths = sorted(
        p
        for p in Path(frames_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(
            f"No image files found in directory: {frames_dir}"
        )

    for idx, path in enumerate(paths):
        if max_frames is not None and idx >= max_frames:
            break
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("Could not read image: %s — skipping", path)
            continue
        yield idx, frame


def get_video_properties(input_path: str) -> Tuple[int, int, float]:
    """
    Return (width, height, fps) for a video file or frames directory.

    For a directory, fps defaults to 25.
    """
    if os.path.isfile(input_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return w, h, fps
    else:
        # Probe the first image in the directory
        paths = sorted(
            p
            for p in Path(input_path).iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise FileNotFoundError(f"No images in {input_path}")
        frame = cv2.imread(str(paths[0]))
        if frame is None:
            raise IOError(f"Cannot read {paths[0]}")
        h, w = frame.shape[:2]
        return w, h, 25.0


class PoseInferencePipeline:
    """
    Orchestrates frame reading and pose estimation.

    Usage::

        pipeline = PoseInferencePipeline(estimator, max_frames=300)
        for frame_idx, frame_bgr, pose_result in pipeline.run("video.mp4"):
            ...  # e.g. draw and write
    """

    def __init__(
        self,
        estimator: BasePoseEstimator,
        max_frames: Optional[int] = None,
    ) -> None:
        self.estimator = estimator
        self.max_frames = max_frames

    def run(
        self,
        input_path: str,
    ) -> Generator[Tuple[int, np.ndarray, PoseResult], None, None]:
        """
        Iterate over frames and yield (frame_idx, bgr_frame, pose_result).

        Args:
            input_path: Path to a video file or a directory of images.
        """
        if os.path.isfile(input_path):
            frame_iter = _iter_video_frames(input_path, self.max_frames)
        elif os.path.isdir(input_path):
            frame_iter = _iter_directory_frames(input_path, self.max_frames)
        else:
            raise FileNotFoundError(f"Input not found: {input_path}")

        for frame_idx, frame_bgr in frame_iter:
            logger.debug("Processing frame %d", frame_idx)
            pose_result = self.estimator.predict(frame_bgr)
            logger.debug(
                "Frame %d: %d person(s) detected",
                frame_idx,
                pose_result.num_persons,
            )
            yield frame_idx, frame_bgr, pose_result
