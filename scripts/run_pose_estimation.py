#!/usr/bin/env python3
"""
run_pose_estimation.py - CLI script for the pose estimation pipeline.

Runs player pose estimation on a football video or a directory of frames
and produces an annotated output video.

Example
-------
Basic usage (uses torchvision KeypointRCNN by default)::

    python scripts/run_pose_estimation.py \\
        --input path/to/video.mp4 \\
        --output output/pose_visualization.mp4 \\
        --device cuda

With BBoxMaskPose (requires separate installation)::

    python scripts/run_pose_estimation.py \\
        --input path/to/video.mp4 \\
        --output output/pose_visualization.mp4 \\
        --device cuda \\
        --backend bboxmaskpose \\
        --bbox_config BBoxMaskPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py \\
        --bbox_checkpoint BBoxMaskPose/weights/vitpose_base.pth \\
        --det_config BBoxMaskPose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \\
        --det_checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pose_estimation.pose_inference import PoseInferencePipeline, get_video_properties
from pose_estimation.pose_model import create_pose_estimator
from pose_estimation.visualization import PoseVisualizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Football player pose estimation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Required ----
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input video file or directory of frames.",
    )
    parser.add_argument(
        "--output",
        default="output/pose_visualization.mp4",
        help="Path to output annotated video file (.mp4).",
    )

    # ---- Model / backend ----
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Inference device: 'auto' (use CUDA if available), "
            "'cuda', 'cuda:0', 'cpu', etc."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "keypointrcnn", "bboxmaskpose"],
        default="auto",
        help="Pose estimation backend to use.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to local model weights (KeypointRCNN only).",
    )

    # ---- BBoxMaskPose-specific ----
    bbox_group = parser.add_argument_group("BBoxMaskPose options")
    bbox_group.add_argument(
        "--bbox_config",
        default=None,
        help="BBoxMaskPose pose model config file.",
    )
    bbox_group.add_argument(
        "--bbox_checkpoint",
        default=None,
        help="BBoxMaskPose pose model checkpoint.",
    )
    bbox_group.add_argument(
        "--det_config",
        default=None,
        help="Person detector config file (BBoxMaskPose).",
    )
    bbox_group.add_argument(
        "--det_checkpoint",
        default=None,
        help="Person detector checkpoint (BBoxMaskPose).",
    )

    # ---- Processing ----
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all).",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Minimum score to accept a detected person / keypoint.",
    )

    # ---- Visualization ----
    parser.add_argument(
        "--hide_bbox",
        dest="show_bbox",
        action="store_false",
        default=True,
        help="Do not draw person bounding boxes on the output video.",
    )
    parser.add_argument(
        "--show_confidence",
        action="store_true",
        default=False,
        help="Display per-keypoint confidence scores.",
    )

    # ---- Logging ----
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser


def resolve_device(device_str: str):
    """Parse the --device argument to a torch.device."""
    import torch

    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("run_pose_estimation")

    # ---- Device ----
    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    # ---- Build estimator ----
    logger.info("Initializing pose estimator (backend=%s)...", args.backend)
    estimator = create_pose_estimator(
        device=device,
        backend=args.backend,
        score_threshold=args.confidence_threshold,
        weights_path=args.weights,
        bbox_config=args.bbox_config,
        bbox_checkpoint=args.bbox_checkpoint,
        det_config=args.det_config,
        det_checkpoint=args.det_checkpoint,
    )
    logger.info("Pose estimator ready: %s", type(estimator).__name__)

    # ---- Video properties ----
    width, height, fps = get_video_properties(args.input)
    logger.info(
        "Input: %s  |  %dx%d @ %.1f fps", args.input, width, height, fps
    )

    # ---- Run pipeline ----
    pipeline = PoseInferencePipeline(estimator, max_frames=args.max_frames)
    visualizer = PoseVisualizer(
        output_path=args.output,
        confidence_threshold=args.confidence_threshold,
        show_bbox=args.show_bbox,
        show_confidence=args.show_confidence,
    )

    logger.info("Processing frames...")
    t_start = time.perf_counter()
    frame_count = 0
    total_persons = 0

    with visualizer:
        visualizer.open(width, height, fps)
        for frame_idx, frame_bgr, pose_result in pipeline.run(args.input):
            visualizer.process_frame(frame_bgr, pose_result)
            frame_count += 1
            total_persons += pose_result.num_persons
            if frame_count % 50 == 0:
                elapsed = time.perf_counter() - t_start
                logger.info(
                    "  Frame %d  |  %.1f fps  |  %d persons total",
                    frame_idx,
                    frame_count / elapsed,
                    total_persons,
                )

    elapsed = time.perf_counter() - t_start
    avg_fps = frame_count / elapsed if elapsed > 0 else float("inf")
    avg_persons = total_persons / frame_count if frame_count > 0 else 0

    logger.info("=" * 60)
    logger.info("Done.")
    logger.info("  Frames processed : %d", frame_count)
    logger.info("  Average throughput: %.1f fps", avg_fps)
    logger.info("  Avg persons/frame : %.1f", avg_persons)
    logger.info("  Output saved to   : %s", args.output)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
