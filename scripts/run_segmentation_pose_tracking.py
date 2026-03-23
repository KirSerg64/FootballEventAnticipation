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

    --sam_model          sam2.1_b.pt      SAM2 model weights (downloaded if absent)
    --det_model          yolo11x.pt       YOLO detection model
    --pose_model         yolo11x-pose.pt  YOLO pose model for keypoints (yolo backend only)
    --pose_backend       auto             Pose backend: auto|keypointrcnn|bboxmaskpose|yolo
    --max_frames         N                Process only the first N frames
    --conf               0.25             Detection confidence threshold
    --iou                0.3              IoU threshold for ID matching
    --mask_alpha         0.40             Segmentation overlay opacity
    --no_skeleton                         Disable skeleton rendering
    --no_ball                             Disable ball overlay
    --ball_debug                          Colour-code ball by detection source + show stats HUD
    --show_attractor                      Draw velocity vectors + vector-field attractor estimate
    --attractor_history  5                Frames of position history for velocity averaging
    --attractor_mode     velocity         Direction mode: 'velocity' or 'acceleration'
    --attractor_smooth   8.0              Kalman process-noise std (px/frame²); higher = smoother
    --attractor_max_stale 30              Stale frames before attractor marker disappears
    --attractor_source   bbox             Velocity source: bbox|keypoints|combined
    --kp_flow_backend    lk               Keypoint flow backend: lk|cotracker
    --kp_detect_interval 1                Re-detect pose every N frames; flow tracks between
    --cotracker_checkpoint               Optional CoTracker3 .pth checkpoint path
    --cotracker_model    cotracker3_online  CoTracker3 hub model: cotracker3_online|cotracker3_offline
    --ct_trail_len       0                Number of frames to draw CoTracker point trajectories (0=off)
    --attractor_dist_sigma 0.0            Gaussian distance-weighting sigma (px); 0=disabled
    --attractor_directional               Enable directional weighting (toward-anchor cosine)
    --export_json                         Export player_tracks.json + ball_track.json
    --redetect_interval  30               Re-run YOLO every N frames for new players
    --tracker            botsort          Primary tracker: botsort or bytetrack
    --max_age            30               Max frames a track survives without detection
    --no_homography                       Disable camera-motion compensation
    --team_colors                         Enable team classification (jersey colours)
    --team_classifier    siglip           Team classification backend: siglip (default) | hue
    --n_teams            2                Number of team clusters (2 or 3)
    --team_refit_interval 30             Refit team clusters every N frames (default 30)
    --ball_patch_size    32               Ball DCF MOSSE template patch size (px)
    --ball_search_radius 60               Ball DCF MOSSE search half-radius (px)
    --ball_psr_threshold 7.0             Ball DCF MOSSE PSR acceptance threshold
    --ball_conf          0.10            Ball YOLO confidence threshold (stage-1, lower than player conf)
    --ball_conf_roi      0.05            Ball YOLO confidence for ROI re-detection (stage-2, FRoG-MOT)
    --ball_tracker       dcf              Ball tracker backend: dcf|cotracker
    --ball_ct_redetect   15               CoTracker ball: YOLO re-anchor interval (frames)
    --ball_det_model     None             Dedicated ball detection model (ONNX/YOLO, e.g. weights/yolov26_ball_det.onnx)
    --ball_det_conf      0.25            Confidence threshold for the dedicated ball detector
    --no_attractor_use_ball              Disable ball-centric attractor (revert to pure vector-field mode)
    --codec              mp4v             FourCC codec for the output video

Player-only tracking (out-of-the-box, no custom model required)::

    --player_class_ids   0               Comma-separated YOLO class IDs to treat as players (default: 0 = COCO person).
                                          Use with a sport-specific model to exclude referees:
                                          e.g. --player_class_ids 0,1 (player + goalkeeper, not referee=2).
    --field_mask_filter                  Enable green-grass HSV field mask to discard off-pitch persons
                                          (spectators, coaches, camera operators). Zero training required.
    --field_hsv_lo       36,40,40        HSV lower bound for field mask (H,S,V in OpenCV scale)
    --field_hsv_hi       85,255,255      HSV upper bound for field mask
    --field_min_overlap  0.3             Min fraction of bbox foot-region on green pixels to keep the person
    --field_mask_interval 15             Recompute field mask every N frames

SAM 3 backend (replaces SAM 2 segmentation with text-prompt driven SAM 3)::

    --sam_backend        sam2            Segmentation backend: sam2 (default) | sam3
    --sam3_model         weights/sam3/sam3.pt   SAM3 checkpoint path
    --sam3_player_prompt "football player"       Text prompt for player detection
    --sam3_ball_prompt   "sports ball"           Text prompt for ball detection (empty string to disable)
    --sam3_field_prompt  ""                      Text prompt for field segmentation (empty = disabled)
    --sam3_score_thresh  0.30            Minimum SAM3 object confidence to accept a detection
    --sam3_float16                       Load SAM3 in float16 to halve GPU VRAM usage (~3 GiB vs ~6 GiB)
    --no_sam3_fallback                   Disable automatic SAM2 fallback on SAM3 failure (default: fallback enabled)

SAM 3 memory usage note::

    Frames are extracted one-by-one via OpenCV (CPU) into a temporary
    directory before being fed to SAM3.  SAM3 then reads frames from disk
    one at a time (offload_video_to_cpu=True), so GPU VRAM is proportional
    to the model size, not the video length.  This avoids the
    ``CUDA out of memory`` error that occurred when the full video was
    pre-loaded into GPU VRAM.

    When SAM3 encounters any execution error (import failure, RuntimeError,
    CUDA OOM, etc.) the pipeline automatically retries with SAM2 + YOLO so
    the job always produces annotated output.  Use --no_sam3_fallback to
    let SAM3 errors propagate instead.
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
    Sam3SegmentationTracker,
    Visualizer,
    associate_poses_with_tracks,
    TeamClassifier,
    SiglipTeamClassifier,
    create_team_classifier,
    PlayerVelocityTracker,
    KeypointVelocityTracker,
    estimate_attractor,
    AttractorSmoother,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

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
        "--sam_config", default="configs/sam2.1/sam2.1_b.yaml",
        help="SAM2 model config YAML path (default: configs/sam2.1/sam2.1_b.yaml)",
    )
    parser.add_argument(
        "--sam_checkpoint", default="weights/sam2.1/sam2.1_b.pt",
        help="SAM2 model checkpoint path (default: weights/sam2.1/sam2.1_b.pt)",
    )    
    parser.add_argument(
        "--det_model", default="yolo11x.pt",
        help="YOLO detection model (default: yolo11x.pt)",
    )
    parser.add_argument(
        "--pose_model", default="yolo11x-pose.pt",
        help="YOLO pose model (default: yolo11x-pose.pt; used only when --pose_backend yolo)",
    )
    parser.add_argument(
        "--pose_backend", default="auto",
        choices=["auto", "keypointrcnn", "bboxmaskpose", "yolo"],
        help=(
            "Pose estimation backend (default: auto).  "
            "'auto' tries KeypointRCNN first, then falls back to YOLO.  "
            "'keypointrcnn' uses torchvision ResNet-50 FPN (no extra install).  "
            "'bboxmaskpose' uses BBoxMaskPose ViTPose (requires --bbox_config, "
            "--bbox_checkpoint, --det_config, --det_checkpoint).  "
            "'yolo' uses the legacy YOLO pose model (--pose_model)."
        ),
    )
    # BBoxMaskPose-specific options
    parser.add_argument(
        "--bbox_config", default=None,
        help="BBoxMaskPose pose model config file (required for --pose_backend bboxmaskpose).",
    )
    parser.add_argument(
        "--bbox_checkpoint", default=None,
        help="BBoxMaskPose pose model checkpoint (required for --pose_backend bboxmaskpose).",
    )
    parser.add_argument(
        "--det_config", default=None,
        help="Person detector config for BBoxMaskPose (required for --pose_backend bboxmaskpose).",
    )
    parser.add_argument(
        "--det_checkpoint", default=None,
        help="Person detector checkpoint for BBoxMaskPose (required for --pose_backend bboxmaskpose).",
    )
    parser.add_argument(
        "--krcnn_weights", default=None,
        help="Optional local weights file for the KeypointRCNN backend.",
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
        "--iou", type=float, default=0.3,
        help="IoU threshold for Hungarian ID matching (default: 0.3)",
    )
    parser.add_argument(
        "--mask_alpha", type=float, default=0.40,
        help="Segmentation overlay opacity 0-1 (default: 0.40)",
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
        "--ball_debug", action="store_true",
        help=(
            "Enable ball detection debug visualisation: colour-code ball by source "
            "(green=DETECT, yellow-green=ROI, orange=MOSSE, red=PRED) and show a "
            "cumulative detection-rate HUD in the top-right corner."
        ),
    )
    parser.add_argument(
        "--show_attractor", action="store_true",
        help=(
            "Draw per-player velocity arrows and the action-focus attractor.  "
            "When ball detection is reliable (default), the attractor IS the ball "
            "position (shown as a gold crosshair target) and player velocity arrows "
            "are still drawn to visualise team pressure.  When the ball is lost the "
            "attractor falls back to the velocity-field convergence estimate "
            "(diamond marker).  Use --no_attractor_use_ball to disable the ball-centric "
            "mode and always show the velocity-field estimate."
        ),
    )
    parser.add_argument(
        "--no_attractor_use_ball", action="store_false", dest="attractor_use_ball",
        help=(
            "Disable ball-centric attractor mode.  When set, the attractor always "
            "uses the vector-field convergence estimate (old behaviour) even when "
            "ball detection is available.  Useful for comparing both modes."
        ),
    )
    parser.set_defaults(attractor_use_ball=True)
    parser.add_argument(
        "--attractor_history", type=int, default=5,
        help=(
            "Number of frames of position history used to compute per-player velocity "
            "when --show_attractor is enabled (default: 5)."
        ),
    )
    parser.add_argument(
        "--attractor_mode", default="velocity",
        choices=["velocity", "acceleration"],
        help=(
            "Direction mode for the vector-field attractor when --show_attractor is "
            "enabled.  'velocity' (default) uses player velocity vectors; "
            "'acceleration' uses the change in velocity — more reactive to sudden "
            "direction changes but requires ≥3 frames of history."
        ),
    )
    parser.add_argument(
        "--attractor_smooth", type=float, default=8.0,
        help=(
            "Kalman filter process-noise standard deviation (px/frame²) for the "
            "attractor smoother (default: 8.0).  Larger values allow the smoothed "
            "position to follow rapid changes more closely at the cost of less "
            "smoothing.  Set to 0 to disable smoothing."
        ),
    )
    parser.add_argument(
        "--attractor_max_stale", type=int, default=30,
        help=(
            "Maximum number of consecutive frames with no valid raw attractor "
            "estimate before the attractor marker disappears (default: 30).  "
            "During the hold period the marker is shown with a dashed outline "
            "and linearly decaying confidence."
        ),
    )
    # Attractor source and keypoint optical-flow settings
    parser.add_argument(
        "--attractor_source", default="bbox",
        choices=["bbox", "keypoints", "combined"],
        help=(
            "Velocity source for the attractor estimator when --show_attractor "
            "is enabled (default: bbox).  "
            "'bbox' uses bounding-box centre displacement (original method).  "
            "'keypoints' uses hip/knee keypoint centroids, optionally tracked "
            "with sparse optical flow between pose detections.  "
            "'combined' merges both; keypoint velocities take priority."
        ),
    )
    parser.add_argument(
        "--kp_flow_backend", default="lk",
        choices=["lk", "cotracker"],
        help=(
            "Sparse optical-flow backend used by the keypoint tracker when "
            "--attractor_source is 'keypoints' or 'combined' and "
            "--kp_detect_interval > 1 (default: lk).  "
            "'lk' uses Lucas-Kanade pyramidal flow (fast, CPU, no extra deps).  "
            "'cotracker' uses CoTracker3 online mode (accurate, GPU, requires "
            "'pip install cotracker')."
        ),
    )
    parser.add_argument(
        "--kp_detect_interval", type=int, default=1,
        help=(
            "Re-read pose keypoints from the estimator every N frames; "
            "between re-detections the selected optical-flow backend propagates "
            "the keypoints for smoother velocity estimates (default: 1, i.e. "
            "fresh keypoints every frame, flow backend not used).  "
            "Higher values reduce re-detection noise at the cost of drift."
        ),
    )
    parser.add_argument(
        "--cotracker_checkpoint", default=None,
        help=(
            "Optional local path to a CoTracker3 .pth checkpoint file.  "
            "If not provided, the default pretrained weights are downloaded "
            "automatically via torch.hub (requires internet on first run)."
        ),
    )
    parser.add_argument(
        "--cotracker_model", default="cotracker3_online",
        choices=["cotracker3_online", "cotracker3_offline"],
        help=(
            "CoTracker3 hub model to load via "
            "torch.hub.load('facebookresearch/co-tracker', MODEL) "
            "(default: cotracker3_online).  "
            "'cotracker3_online' uses a sliding-window online predictor — low "
            "latency, processes every step frames.  "
            "'cotracker3_offline' accumulates all frames since the last "
            "re-detection and runs batch inference on them — higher accuracy "
            "but results are only updated at each detect_interval boundary."
        ),
    )
    parser.add_argument(
        "--ct_trail_len", type=int, default=0,
        help=(
            "Number of past frames whose CoTracker3 point positions are drawn "
            "as a trajectory trail on each output frame (default: 0 = disabled).  "
            "Trails are colour-coded by player ID and fade from semi-transparent "
            "at the oldest position to opaque at the current position.  "
            "Requires --kp_flow_backend cotracker and "
            "--attractor_source keypoints|combined.  "
            "A value of 30 gives a ~1-second trail at 30 fps."
        ),
    )
    # Distance and directional weighting for the attractor
    parser.add_argument(
        "--attractor_dist_sigma", type=float, default=0.0,
        help=(
            "Gaussian distance-weighting sigma (pixels) for the attractor "
            "estimator (default: 0.0 = disabled).  When > 0, player rays are "
            "weighted by exp(-d²/2σ²) where d is the distance from the anchor "
            "(ball position when detected, otherwise previous attractor).  "
            "A value of 200 px works well for 1080p broadcast footage."
        ),
    )
    parser.add_argument(
        "--attractor_directional", action="store_true",
        help=(
            "Enable directional weighting for the attractor estimator.  "
            "Each player's weight is additionally multiplied by "
            "max(0.05, cos θ) where θ is the angle between the player's "
            "velocity/acceleration direction and the toward-anchor direction.  "
            "Players actively moving toward the centre of action contribute "
            "fully; players running away are strongly down-weighted.  "
            "Requires an anchor point (ball or previous attractor)."
        ),
    )
    parser.add_argument(
        "--export_json", action="store_true",
        help="Export player_tracks.json and ball_track.json to the output directory",
    )
    parser.add_argument(
        "--redetect_interval", type=int, default=30,
        help="Re-run YOLO detection every N frames to catch new players (default: 30; 0 disables)",
    )
    # Improvement E: tracker selection
    parser.add_argument(
        "--tracker", default="botsort",
        choices=["botsort", "bytetrack"],
        help="Primary tracker algorithm: botsort (default) or bytetrack",
    )
    # Improvement G: track lifecycle
    parser.add_argument(
        "--max_age", type=int, default=30,
        help="Max frames a track survives without a detection (default: 30)",
    )
    # Improvement D: camera-motion compensation
    parser.add_argument(
        "--no_homography", action="store_true",
        help="Disable camera-motion compensation (ORB + RANSAC homography)",
    )
    # Improvement F: team colour clustering
    parser.add_argument(
        "--team_colors", action="store_true",
        help="Enable jersey-colour team classification",
    )
    parser.add_argument(
        "--team_classifier", default="siglip", choices=["siglip", "hue"],
        help=(
            "Team classification backend (default: 'siglip').  "
            "'siglip' uses google/siglip-base-patch16-224 vision embeddings + scikit-learn "
            "KMeans for accurate team assignment that is robust to similar jersey colours.  "
            "Player segmentation masks are used to blank out the background before "
            "embedding for cleaner features.  Requires transformers, scikit-learn, torch.  "
            "'hue' uses the lightweight HSV hue K-means classifier (no extra dependencies).  "
            "Falls back to 'hue' automatically when SIGLIP dependencies are missing."
        ),
    )
    parser.add_argument(
        "--n_teams", type=int, default=2,
        help="Number of team clusters for --team_colors (default: 2; use 3 to include referee)",
    )
    parser.add_argument(
        "--team_refit_interval", type=int, default=30,
        help=(
            "Re-run K-means team clustering every N frames when --team_colors is enabled "
            "(default: 30)"
        ),
    )
    # Ball tracking: detection-first + MOSSE DCF correlation parameters
    parser.add_argument(
        "--ball_patch_size", type=int, default=32,
        help=(
            "Ball DCF: MOSSE template patch size (px, default: 32). "
            "Larger captures more context; smaller is faster."
        ),
    )
    parser.add_argument(
        "--ball_search_radius", type=int, default=60,
        help=(
            "Ball DCF: MOSSE search half-radius (px) when YOLO misses (default: 60). "
            "Increase for faster balls on wide-angle cameras."
        ),
    )
    parser.add_argument(
        "--ball_psr_threshold", type=float, default=7.0,
        help=(
            "Ball DCF: minimum Peak-to-Sidelobe Ratio for MOSSE acceptance (default: 7.0). "
            "Lower = accept noisier predictions; higher = more conservative."
        ),
    )
    parser.add_argument(
        "--ball_conf", type=float, default=0.10,
        help=(
            "Ball detection confidence threshold for the global YOLO pass (stage-1, default: 0.10). "
            "Lower than the player threshold (--conf) so motion-blurred fast balls are detected. "
            "BoT-SORT uses its own track_high_thresh for player tracks and is unaffected."
        ),
    )
    parser.add_argument(
        "--ball_conf_roi", type=float, default=0.05,
        help=(
            "Ball detection confidence for ROI-based re-detection (stage-2 / FRoG-MOT, default: 0.05). "
            "Applied only within the predicted ball region, so false-positive rate stays low."
        ),
    )
    parser.add_argument(
        "--ball_tracker", default="dcf",
        choices=["dcf", "cotracker"],
        help=(
            "Ball tracker backend (default: dcf).  "
            "'dcf' uses the MOSSE correlation-filter tracker (BallDCFTracker). "
            "'cotracker' uses CoTracker3 point tracking (BallCoTrackerTracker): "
            "takes the first YOLO detection as the initial anchor point and "
            "propagates the ball with CoTracker3 between detections, "
            "re-anchoring to YOLO every --ball_ct_redetect frames.  "
            "Reuses --cotracker_model, --cotracker_checkpoint, and --device."
        ),
    )
    parser.add_argument(
        "--ball_ct_redetect", type=int, default=15,
        help=(
            "CoTracker ball tracker: number of YOLO-confirmed detections "
            "between forced re-anchors (default: 15).  After this many "
            "confirmed YOLO hits the CoTracker3 query point is reset to the "
            "current YOLO position so the tracker stays locked after kicks. "
            "Only used when --ball_tracker cotracker."
        ),
    )
    parser.add_argument(
        "--ball_det_model", default=None,
        help=(
            "Optional path to a dedicated ball-detection model in ONNX or YOLO "
            "format (e.g. 'weights/yolov26_ball_det.onnx').  When provided, this "
            "model is used for both the global ball detection pass (stage-1) and "
            "the ROI re-detection pass (stage-2) instead of the main YOLO model.  "
            "The dedicated model must output class 0 as the ball class.  The main "
            "YOLO result is kept as a fallback for stage-1 if the dedicated model "
            "finds nothing."
        ),
    )
    parser.add_argument(
        "--ball_det_conf", type=float, default=0.25,
        help=(
            "Confidence threshold for the dedicated ball detector "
            "(--ball_det_model, default: 0.25).  Ignored when --ball_det_model "
            "is not set."
        ),
    )
    parser.add_argument(
        "--codec", default="mp4v",
        help=(
            "FourCC video codec for the output file (default: mp4v). "
            "Use 'avc1' for H.264 if supported by your OpenCV build."
        ),
    )
    # ── Player-only tracking ─────────────────────────────────────────────────
    parser.add_argument(
        "--player_class_ids", default="0",
        help=(
            "Comma-separated YOLO class IDs to treat as players (default: '0' = "
            "COCO person class).  Use this when you supply a sport-specific YOLO "
            "model that distinguishes players from referees.  For example, with a "
            "Roboflow football model that uses 0=player, 1=goalkeeper, 2=referee "
            "you can pass '--player_class_ids 0,1' to track players and goalkeepers "
            "but skip referees.  The ball class is managed separately and is not "
            "affected by this setting."
        ),
    )
    parser.add_argument(
        "--field_mask_filter", action="store_true",
        help=(
            "Enable green-grass HSV field mask filtering.  When active, a colour "
            "segmentation mask of the playing field is derived from each frame and "
            "used to discard person detections whose feet are not on the grass — "
            "eliminating spectators in the stands, coaches on the bench, camera "
            "operators, and other off-pitch persons.  No custom model or training is "
            "required.  Combine with --field_hsv_lo / --field_hsv_hi to tune the "
            "HSV range for artificial turf or unusual lighting conditions."
        ),
    )
    parser.add_argument(
        "--field_hsv_lo", default="36,40,40",
        help=(
            "HSV lower bound for the field mask as 'H,S,V' (OpenCV scale: "
            "H∈[0,180], S/V∈[0,255]; default: '36,40,40').  Used only when "
            "--field_mask_filter is set."
        ),
    )
    parser.add_argument(
        "--field_hsv_hi", default="85,255,255",
        help=(
            "HSV upper bound for the field mask as 'H,S,V' (default: '85,255,255'). "
            "Used only when --field_mask_filter is set."
        ),
    )
    parser.add_argument(
        "--field_min_overlap", type=float, default=0.3,
        help=(
            "Minimum fraction of the bounding-box foot region that must fall on "
            "green pixels to keep a tracked person (default: 0.3).  Lower values "
            "retain players near the sideline; higher values are more aggressive."
        ),
    )
    parser.add_argument(
        "--field_mask_interval", type=int, default=15,
        help=(
            "Recompute the field mask every N frames (default: 15).  "
            "Lower values adapt faster to camera panning or lighting changes; "
            "higher values are faster.  Used only when --field_mask_filter is set."
        ),
    )
    # ── SAM 3 backend ────────────────────────────────────────────────────────
    parser.add_argument(
        "--sam_backend", default="sam2", choices=["sam2", "sam3"],
        help=(
            "Segmentation backend to use for player masking (default: 'sam2').  "
            "'sam3' switches to the SAM 3 text-prompt-driven tracker "
            "(Sam3SegmentationTracker) which requires the 'sam3' Python package "
            "and a SAM3 checkpoint.  See weights/sam3/README.md for setup "
            "instructions.  When 'sam3' is selected the --det_model / --sam_model "
            "arguments are ignored and replaced by the --sam3_* flags below."
        ),
    )
    parser.add_argument(
        "--sam3_model", default="weights/sam3/sam3.pt",
        help=(
            "Path to the SAM3 checkpoint file (default: 'weights/sam3/sam3.pt').  "
            "Used only when --sam_backend sam3 is set."
        ),
    )
    parser.add_argument(
        "--sam3_player_prompt", default="football player",
        help=(
            "Text prompt describing the objects to track as players "
            "(default: 'football player').  SAM3 detects and tracks all instances "
            "of this concept across the video.  Used only with --sam_backend sam3."
        ),
    )
    parser.add_argument(
        "--sam3_ball_prompt", default="sports ball",
        help=(
            "Text prompt for the ball (default: 'sports ball').  Pass an empty "
            "string ('') to disable SAM3-based ball detection.  "
            "Used only with --sam_backend sam3."
        ),
    )
    parser.add_argument(
        "--sam3_field_prompt", default="",
        help=(
            "Text prompt for the playing field (default: '' = disabled).  "
            "When non-empty (e.g. 'football pitch'), SAM3 segments the field and "
            "the result is stored in TrackerState.field_mask and visualised "
            "as a lime-green boundary overlay.  "
            "Used only with --sam_backend sam3."
        ),
    )
    parser.add_argument(
        "--sam3_score_thresh", type=float, default=0.30,
        help=(
            "Minimum SAM3 object confidence score to accept a detection "
            "(default: 0.30).  Used only with --sam_backend sam3."
        ),
    )
    parser.add_argument(
        "--sam3_float16", action="store_true", default=False,
        help=(
            "Convert the SAM3 model to float16 immediately after loading to "
            "roughly halve its GPU VRAM footprint (~3 GiB vs ~6 GiB in float32). "
            "Recommended when GPU memory is limited or shared with other models. "
            "Used only with --sam_backend sam3."
        ),
    )
    parser.add_argument(
        "--no_sam3_fallback", action="store_true", default=False,
        help=(
            "Disable the automatic SAM2 + YOLO fallback that runs when SAM3 "
            "encounters an error (ImportError, RuntimeError, CUDA OOM, …).  "
            "By default the pipeline transparently retries with SAM2 so the job "
            "always produces output.  Pass this flag to let SAM3 errors propagate "
            "and crash the process — useful when debugging SAM3 itself."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _build_pose_estimator(args: argparse.Namespace):
    """Return a pose estimator object appropriate for *args.pose_backend*.

    Returns one of:

    * A ``pose_estimation.pose_model.BasePoseEstimator`` (new backends).
    * A YOLO model (legacy ``--pose_backend yolo``).

    The object is then passed to ``_run_pose_on_frame`` which handles both APIs.
    """
    backend = args.pose_backend

    if backend == "yolo":
        from ultralytics import YOLO
        logger.info("Loading YOLO pose model: %s", args.pose_model)
        return YOLO(args.pose_model)

    # New pose_estimation backends
    import torch
    from pose_estimation.pose_model import create_pose_estimator

    device = torch.device(
        "cuda" if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )

    try:
        estimator = create_pose_estimator(
            device=device,
            backend="keypointrcnn" if backend == "keypointrcnn" else
                    "bboxmaskpose" if backend == "bboxmaskpose" else "auto",
            score_threshold=args.conf,
            weights_path=getattr(args, "krcnn_weights", None),
            bbox_config=getattr(args, "bbox_config", None),
            bbox_checkpoint=getattr(args, "bbox_checkpoint", None),
            det_config=getattr(args, "det_config", None),
            det_checkpoint=getattr(args, "det_checkpoint", None),
        )
        logger.info(
            "Pose estimator ready: %s (device=%s)",
            type(estimator).__name__,
            device,
        )
        return estimator
    except Exception as exc:
        if backend == "auto":
            # Fall back to YOLO
            logger.warning(
                "pose_estimation backend unavailable (%s); falling back to YOLO pose model %s",
                exc,
                args.pose_model,
            )
            from ultralytics import YOLO
            return YOLO(args.pose_model)
        raise


def _run_pose_on_frame(pose_estimator, frame, args: argparse.Namespace):
    """Run pose inference for a single frame.

    Returns a pose result object compatible with :func:`associate_poses_with_tracks`.
    The result is either a YOLO pose result or a
    ``pose_estimation.pose_model.PoseResult`` — both are accepted by the
    updated association module.
    """
    # Detect backend type by duck-typing
    if hasattr(pose_estimator, "predict") and hasattr(pose_estimator, "num_persons"):
        # Should not normally happen (BasePoseEstimator doesn't have num_persons)
        return pose_estimator.predict(frame)

    # Check if it's the new pose_estimation BasePoseEstimator
    try:
        from pose_estimation.pose_model import BasePoseEstimator
        if isinstance(pose_estimator, BasePoseEstimator):
            return pose_estimator.predict(frame)
    except ImportError:
        pass

    # Legacy YOLO path
    result_list = pose_estimator.predict(
        frame,
        conf=args.conf,
        device=args.device,
        verbose=False,
    )
    return result_list[0] if result_list else None


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

    # -- Validate inputs ------------------------------------------------------
    if not os.path.isfile(args.input):
        logger.error("Input video not found: %s", args.input)
        sys.exit(1)

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Step 1: Segmentation + tracking (BoT-SORT + SAM2 or SAM3) ------------
    logger.info(
        "=== Step 1: Player segmentation and tracking (backend=%s) ===",
        args.sam_backend.upper(),
    )

    def _build_sam2_tracker() -> SegmentationTracker:
        """Return a fully-configured SAM2 + YOLO BoT-SORT tracker."""
        return SegmentationTracker(
            sam_model_config=args.sam_config,
            sam_model_checkpoint=args.sam_checkpoint,
            det_model_path=args.det_model,
            device=args.device,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            redetect_interval=args.redetect_interval,
            tracker=args.tracker,
            max_age=args.max_age,
            use_homography=not args.no_homography,
            ball_patch_size=args.ball_patch_size,
            ball_search_radius=args.ball_search_radius,
            ball_psr_threshold=args.ball_psr_threshold,
            ball_conf_threshold=args.ball_conf,
            ball_conf_roi=args.ball_conf_roi,
            ball_tracker_type=args.ball_tracker,
            ball_cotracker_model=args.cotracker_model,
            ball_cotracker_checkpoint=args.cotracker_checkpoint,
            ball_cotracker_device=args.device,
            ball_cotracker_redetect_interval=args.ball_ct_redetect,
            ball_det_model_path=args.ball_det_model,
            ball_det_conf=args.ball_det_conf,
            player_class_ids=[int(x) for x in args.player_class_ids.split(",") if x.strip()],
            field_mask_filter=args.field_mask_filter,
            field_hsv_lo=tuple(int(x) for x in args.field_hsv_lo.split(",")),  # type: ignore[arg-type]
            field_hsv_hi=tuple(int(x) for x in args.field_hsv_hi.split(",")),  # type: ignore[arg-type]
            field_min_overlap=args.field_min_overlap,
            field_mask_interval=args.field_mask_interval,
        )

    if args.sam_backend == "sam3":
        # SAM3 text-prompt-driven tracker with automatic SAM2 fallback
        logger.info(
            "SAM3 tracker: player='%s', ball='%s', field='%s'",
            args.sam3_player_prompt,
            args.sam3_ball_prompt or "(disabled)",
            args.sam3_field_prompt or "(disabled)",
        )
        try:
            tracker: SegmentationTracker | Sam3SegmentationTracker = Sam3SegmentationTracker(
                sam_model_config=args.sam_config,
                sam_model_checkpoint=args.sam_checkpoint,
                device=args.device,
                player_text_prompt=args.sam3_player_prompt,
                ball_text_prompt=args.sam3_ball_prompt or None,
                field_text_prompt=args.sam3_field_prompt or None,
                score_threshold=args.sam3_score_thresh,
                use_float16=args.sam3_float16,
            )
            seg_results = tracker.process_video(args.input, max_frames=args.max_frames)
        except Exception as sam3_exc:  # noqa: BLE001
            logger.warning(
                "SAM3 processing failed: %s\n"
                "  Error type : %s\n"
                "  Falling back to SAM2 + YOLO pipeline%s",
                sam3_exc,
                type(sam3_exc).__name__,
                "" if not getattr(args, "no_sam3_fallback", False)
                else "  (fallback disabled — re-raising)",
            )
            if getattr(args, "no_sam3_fallback", False):
                raise
            logger.info("Building SAM2 + YOLO fallback tracker …")
            tracker = _build_sam2_tracker()
            seg_results = tracker.process_video(args.input, max_frames=args.max_frames)
    else:
        # Default SAM2 + YOLO BoT-SORT tracker
        tracker = _build_sam2_tracker()
        seg_results = tracker.process_video(args.input, max_frames=args.max_frames)
    logger.info("Segmentation complete: %d frames", len(seg_results))

    # -- Step 2: Pose estimation -----------------------------------------------
    logger.info(
        "=== Step 2: Pose estimation (backend=%s) ===",
        args.pose_backend,
    )
    pose_estimator = _build_pose_estimator(args)

    # -- Team colour classifier (optional) ------------------------------------
    team_classifier: TeamClassifier | SiglipTeamClassifier | None = None
    if args.team_colors:
        _tc_backend = getattr(args, "team_classifier", "siglip")
        logger.info(
            "Team classification enabled (backend=%s, n_teams=%d)",
            _tc_backend,
            args.n_teams,
        )
        team_classifier = create_team_classifier(
            backend=_tc_backend,
            n_teams=args.n_teams,
            device=args.device,
        )

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        logger.error("Cannot re-open input video for pose estimation.")
        sys.exit(1)

    writer = _open_video_writer(args.output, cap, codec=args.codec)
    _show_ct_traj = args.ct_trail_len > 0 and args.kp_flow_backend == "cotracker"
    visualizer = Visualizer(
        mask_alpha=args.mask_alpha,
        show_skeleton=not args.no_skeleton,
        show_ball=not args.no_ball,
        show_ball_debug=args.ball_debug,
        show_attractor=args.show_attractor,
        show_ct_trajectories=_show_ct_traj,
    )

    # ── Vector-field attractor trackers (only allocated when needed) ──────────
    # bbox-based tracker (always created when attractor is enabled; used in
    # 'bbox' and 'combined' modes)
    bbox_vel_tracker: PlayerVelocityTracker | None = (
        PlayerVelocityTracker(history_len=args.attractor_history)
        if args.show_attractor and args.attractor_source in ("bbox", "combined")
        else None
    )
    # Decide whether CoTracker trajectories are needed.
    # They require the keypoint tracker with cotracker backend.
    _need_kp_tracker = (
        (args.show_attractor and args.attractor_source in ("keypoints", "combined"))
        or _show_ct_traj
    )
    # keypoint-based tracker (used in 'keypoints'/'combined' modes or for trajectories)
    kp_vel_tracker: KeypointVelocityTracker | None = (
        KeypointVelocityTracker(
            history_len=args.attractor_history,
            flow_backend=args.kp_flow_backend,
            detect_interval=args.kp_detect_interval,
            device=args.device,
            cotracker_checkpoint=args.cotracker_checkpoint,
            cotracker_model=args.cotracker_model,
            ct_trail_len=args.ct_trail_len,
        )
        if _need_kp_tracker
        else None
    )
    # Kalman smoother for the attractor position
    attractor_smoother: AttractorSmoother | None = (
        AttractorSmoother(
            process_noise_std=args.attractor_smooth,
            max_stale_frames=args.attractor_max_stale,
        )
        if args.show_attractor and args.attractor_smooth > 0
        else None
    )
    # Remember the last smoothed attractor point for distance-weighting anchor
    _prev_attractor_pt: tuple[float, float] | None = None

    # JSON export accumulators
    player_tracks_export: list[dict] = []
    ball_track_export: list[dict] = []

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    logger.info("=== Step 3-6: Association, team classification, visualization, output ===")
    for frame_idx, seg_result in enumerate(seg_results):
        ok, frame = cap.read()
        if not ok:
            break

        # -- Pose estimation for this frame -----------------------------------
        try:
            pose_result = _run_pose_on_frame(pose_estimator, frame, args)
        except Exception as exc:
            logger.warning("Pose estimation failed on frame %d: %s", frame_idx, exc)
            pose_result = None

        # -- Association: mask <-> keypoints (Hungarian) ----------------------
        player_tracks, ball_track = associate_poses_with_tracks(
            seg_result=seg_result,
            pose_result=pose_result,
            frame_shape=(frame_h, frame_w),
            iou_threshold=args.iou,
        )

        # -- Team colour classification ----------------------------------------
        if team_classifier is not None:
            for pt in player_tracks:
                crop = TeamClassifier.extract_torso_crop(frame, pt.bbox)
                # Pass the player mask to SIGLIP so background is blanked out;
                # HSV classifier ignores the mask keyword argument.
                team_classifier.update(pt.id, crop, mask=pt.mask)

            # Refit at configured interval; try fast assignment for new players otherwise
            if frame_idx % args.team_refit_interval == 0:
                team_classifier.fit()
            else:
                for pt in player_tracks:
                    if team_classifier.get_team(pt.id) is None:
                        team_classifier.assign_new(pt.id)

            # Assign labels to tracks
            for pt in player_tracks:
                pt.team_label = team_classifier.get_team(pt.id)

        # -- Vector-field attractor (optional) --------------------------------
        velocities = None
        attractor = None
        _kp_tracker_ran = False
        if args.show_attractor:
            # Collect velocity dicts from active trackers
            bbox_vels: dict = {}
            kp_vels: dict = {}

            if bbox_vel_tracker is not None:
                bbox_vels = bbox_vel_tracker.update(player_tracks)

            if kp_vel_tracker is not None and args.attractor_source in ("keypoints", "combined"):
                kp_vels = kp_vel_tracker.update(player_tracks, frame)
                _kp_tracker_ran = True

            # Merge velocity dicts according to attractor_source
            if args.attractor_source == "bbox":
                merged_vels = bbox_vels
            elif args.attractor_source == "keypoints":
                merged_vels = kp_vels
            else:  # combined: keypoints take priority, bbox fills gaps
                merged_vels = {**bbox_vels, **kp_vels}

            # Expose velocities for arrow visualisation (use whichever is active)
            velocities = merged_vels if merged_vels else None

            # Choose direction vectors based on attractor mode
            if args.attractor_mode == "acceleration":
                if bbox_vel_tracker is not None and args.attractor_source in ("bbox", "combined"):
                    bbox_acc = bbox_vel_tracker.get_accelerations()
                else:
                    bbox_acc = {}
                if kp_vel_tracker is not None and args.attractor_source in ("keypoints", "combined"):
                    kp_acc = kp_vel_tracker.get_accelerations()
                else:
                    kp_acc = {}

                if args.attractor_source == "bbox":
                    direction_vectors = bbox_acc
                elif args.attractor_source == "keypoints":
                    direction_vectors = kp_acc
                else:
                    direction_vectors = {**bbox_acc, **kp_acc}
            else:
                direction_vectors = merged_vels

            # Determine anchor point for distance / directional weighting.
            # Prefer the ball when detected; fall back to the previous smoothed
            # attractor position to avoid losing the weighting on missed frames.
            anchor_pt: tuple[float, float] | None = None
            ball_center_pt: tuple[float, float] | None = None
            if ball_track is not None:
                anchor_pt = ball_track.center
                if args.attractor_use_ball:
                    ball_center_pt = ball_track.center
            elif _prev_attractor_pt is not None:
                anchor_pt = _prev_attractor_pt

            raw_attractor = estimate_attractor(
                direction_vectors,
                frame_shape=(frame_h, frame_w),
                # Ball-centric fast-path: when ball is detected and
                # --attractor_use_ball is set (default), the expensive
                # vector-field computation is bypassed and the exact ball
                # position is returned with confidence=1.0.
                ball_center=ball_center_pt,
                anchor_point=anchor_pt,
                distance_sigma=args.attractor_dist_sigma,
                directional_weight=args.attractor_directional,
            )

            # Apply Kalman smoother (or use raw directly if smoothing disabled).
            # Ball-sourced estimates pass through without Kalman lag.
            if attractor_smoother is not None:
                attractor = attractor_smoother.update(
                    raw_attractor, frame_shape=(frame_h, frame_w)
                )
            else:
                attractor = raw_attractor

            # Cache the smoothed position for next frame's anchor fallback
            if attractor is not None:
                _prev_attractor_pt = attractor.point

        # Run the keypoint tracker for trajectory-only case (no attractor)
        if _show_ct_traj and not _kp_tracker_ran and kp_vel_tracker is not None:
            kp_vel_tracker.update(player_tracks, frame)

        # Collect CoTracker3 trajectories for visualisation
        ct_trajectories: dict | None = None
        if _show_ct_traj and kp_vel_tracker is not None:
            ct_trajectories = kp_vel_tracker.get_ct_trajectories() or None

        # -- Visualization ----------------------------------------------------
        annotated = visualizer.draw_frame(
            frame, player_tracks, ball_track,
            frame_idx=frame_idx,
            velocities=velocities,
            attractor=attractor,
            ct_trajectories=ct_trajectories,
            field_mask=seg_result.field_mask,
        )
        writer.write(annotated)

        # -- JSON export ------------------------------------------------------
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

    # -- Export JSON data -----------------------------------------------------
    if args.export_json:
        player_json_path = output_dir / "player_tracks.json"
        ball_json_path = output_dir / "ball_track.json"

        with open(player_json_path, "w", encoding="utf-8") as f:
            json.dump(player_tracks_export, f, indent=2)
        logger.info("Player tracks exported to: %s", player_json_path)

        with open(ball_json_path, "w", encoding="utf-8") as f:
            json.dump(ball_track_export, f, indent=2)
        logger.info("Ball track exported to: %s", ball_json_path)

    # -- Export team labels ---------------------------------------------------
    if team_classifier is not None and args.export_json:
        team_json_path = output_dir / "team_labels.json"
        with open(team_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {str(k): v for k, v in team_classifier.team_labels().items()},
                f,
                indent=2,
            )
        logger.info("Team labels exported to: %s", team_json_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(args)
