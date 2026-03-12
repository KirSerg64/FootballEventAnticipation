#!/usr/bin/env python3
"""
run_segmentation_pose_tracking.py
----------------------------------
CLI entry-point for the football player segmentation, tracking, and pose
estimation pipeline.

Usage
~~~~~
.. code-block:: bash

    python scripts/run_segmentation_pose_tracking.py \\
        --input path/to/video.mp4 \\
        --output output/segmentation_pose_tracking.mp4 \\
        --device cuda

Optional flags::

    --sam_model       sam2.1_b.pt          SAM2 model weights (downloaded if absent)
    --det_model       yolo11x.pt           YOLO detection model for initial prompts
    --pose_model      yolo11x-pose.pt      YOLO pose model for keypoints
    --max_frames      N                    Process only the first N frames
    --conf            0.25                 Detection confidence threshold
    --iou             0.5                  IoU threshold for ID matching
    --mask_alpha      0.40                 Segmentation overlay opacity
    --no_skeleton                          Disable skeleton rendering
    --no_ball                             Disable ball overlay
    --export_json                          Export player_tracks.json + ball_track.json
    --redetect_interval  30               Re-run YOLO every N frames for new players
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import cv2

# Allow the script to be run from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation_tracking import (
    SegmentationTracker,
    Visualizer,
    associate_poses_with_tracks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Football player segmentation + tracking + pose estimation"
    )
    parser.add_argument("--input", required=True, help="Path to the input video file")
    parser.add_argument(
        "--output",
        default="output/segmentation_pose_tracking.mp4",
        help="Path to the output annotated video (default: output/segmentation_pose_tracking.mp4)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Torch device: 'cuda' or 'cpu' (default: cuda)",
    )
    parser.add_argument(
        "--sam_model", default="sam2.1_b.pt",
        help="SAM2 model name / path (default: sam2.1_b.pt)",
    )
    parser.add_argument(
        "--det_model", default="yolo11x.pt",
        help="YOLO detection model (default: yolo11x.pt)",
    )
    parser.add_argument(
        "--pose_model", default="yolo11x-pose.pt",
        help="YOLO pose model (default: yolo11x-pose.pt)",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Maximum number of frames to process (default: all)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="IoU threshold for ID matching (default: 0.5)",
    )
    parser.add_argument(
        "--mask_alpha", type=float, default=0.40,
        help="Segmentation overlay opacity 0–1 (default: 0.40)",
    )
    parser.add_argument(
        "--no_skeleton", action="store_true",
        help="Disable pose skeleton rendering",
    )
    parser.add_argument(
        "--no_ball", action="store_true",
        help="Disable ball overlay",
    )
    parser.add_argument(
        "--export_json", action="store_true",
        help="Export player_tracks.json and ball_track.json to the output directory",
    )
    parser.add_argument(
        "--redetect_interval", type=int, default=30,
        help="Re-run YOLO detection every N frames to catch new players (default: 30; 0 disables)",
    )
    parser.add_argument(
        "--codec", default="mp4v",
        help=(
            "FourCC video codec for the output file (default: mp4v). "
            "Use 'avc1' for H.264 if supported by your OpenCV build."
        ),
    )
    return parser.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _load_pose_model(pose_model_path: str, device: str):
    """Lazy-load the YOLO pose model."""
    from ultralytics import YOLO
    logger.info("Loading YOLO pose model: %s", pose_model_path)
    return YOLO(pose_model_path)


def _open_video_writer(
    output_path: str,
    cap: cv2.VideoCapture,
    codec: str = "mp4v",
) -> cv2.VideoWriter:
    """Create an OpenCV VideoWriter compatible with the source video."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for path: {output_path}")
    return writer


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full segmentation + tracking + pose estimation pipeline."""

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.isfile(args.input):
        logger.error("Input video not found: %s", args.input)
        sys.exit(1)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Segmentation + tracking ──────────────────────────────────────
    logger.info("=== Step 1: Player segmentation and tracking (SAM2VideoPredictor) ===")
    tracker = SegmentationTracker(
        sam_model_path=args.sam_model,
        det_model_path=args.det_model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        redetect_interval=args.redetect_interval,
    )
    seg_results = tracker.process_video(args.input, max_frames=args.max_frames)
    logger.info("Segmentation complete: %d frames", len(seg_results))

    # ── Step 2: Pose estimation ───────────────────────────────────────────────
    logger.info("=== Step 2: Pose estimation ===")
    pose_model = _load_pose_model(args.pose_model, args.device)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        logger.error("Cannot re-open input video for pose estimation.")
        sys.exit(1)

    writer = _open_video_writer(args.output, cap, codec=args.codec)
    visualizer = Visualizer(
        mask_alpha=args.mask_alpha,
        show_skeleton=not args.no_skeleton,
        show_ball=not args.no_ball,
    )

    # JSON export accumulators
    player_tracks_export: list[dict] = []
    ball_track_export: list[dict] = []

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    logger.info("=== Step 3–6: Association, visualization, output ===")
    for frame_idx, seg_result in enumerate(seg_results):
        ok, frame = cap.read()
        if not ok:
            break

        # ── Pose estimation for this frame ────────────────────────────────────
        try:
            pose_result_list = pose_model.predict(
                frame,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )
            pose_result = pose_result_list[0] if pose_result_list else None
        except Exception as exc:
            logger.warning("Pose estimation failed on frame %d: %s", frame_idx, exc)
            pose_result = None

        # ── Association: mask ↔ keypoints ─────────────────────────────────────
        player_tracks, ball_track = associate_poses_with_tracks(
            seg_result=seg_result,
            pose_result=pose_result,
            frame_shape=(frame_h, frame_w),
            iou_threshold=args.iou,
        )

        # ── Visualization ─────────────────────────────────────────────────────
        annotated = visualizer.draw_frame(frame, player_tracks, ball_track)
        writer.write(annotated)

        # ── JSON export ───────────────────────────────────────────────────────
        if args.export_json:
            for pt in player_tracks:
                entry = pt.to_dict()
                entry["frame_index"] = frame_idx
                player_tracks_export.append(entry)

            if ball_track is not None:
                ball_entry = ball_track.to_dict()
                ball_entry["frame_index"] = frame_idx
                ball_track_export.append(ball_entry)

        if (frame_idx + 1) % 50 == 0:
            logger.info("Annotated %d / %d frames", frame_idx + 1, len(seg_results))

    cap.release()
    writer.release()
    logger.info("Output video saved to: %s", args.output)

    # ── Export JSON data ──────────────────────────────────────────────────────
    if args.export_json:
        player_json_path = output_dir / "player_tracks.json"
        ball_json_path = output_dir / "ball_track.json"

        with open(player_json_path, "w", encoding="utf-8") as f:
            json.dump(player_tracks_export, f, indent=2)
        logger.info("Player tracks exported to: %s", player_json_path)

        with open(ball_json_path, "w", encoding="utf-8") as f:
            json.dump(ball_track_export, f, indent=2)
        logger.info("Ball track exported to: %s", ball_json_path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(args)
