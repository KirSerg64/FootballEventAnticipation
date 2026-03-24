"""
run_sam2_tracking.py
--------------------
Minimal standalone script: SAM2 segmentation + YOLO BoT-SORT tracking only.

No pose estimation, no ball tracker, no team classifier — just player masks
and persistent IDs written to an annotated output video.

Usage
-----
    python scripts/run_sam2_tracking.py --input video.mp4

    python scripts/run_sam2_tracking.py \\
        --input  video.mp4 \\
        --output output/tracked.mp4 \\
        --sam_config    configs/sam2.1/sam2.1_hiera_l.yaml \\
        --sam_checkpoint checkpoints/sam2.1_hiera_large.pt \\
        --det_model     yolo11x.pt \\
        --device        cuda \\
        --conf          0.25 \\
        --max_frames    300
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sam2_tracking")

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
import colorsys

_GOLDEN_RATIO = 0.6180339887


def _id_to_bgr(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * _GOLDEN_RATIO) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _draw_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    mask_alpha: float = 0.40,
) -> np.ndarray:
    """Return an annotated copy of *frame* with coloured masks and IDs."""
    out = frame.copy()
    if detections is None or len(detections) == 0:
        return out

    tracker_ids = (
        detections.tracker_id
        if detections.tracker_id is not None
        else np.arange(len(detections))
    )

    overlay = out.copy()
    for i, (bbox, mask, tid) in enumerate(
        zip(detections.xyxy, detections.mask, tracker_ids)
    ):
        color = _id_to_bgr(int(tid))

        # -- Filled mask overlay -------------------------------------------
        if mask is not None and mask.any():
            overlay[mask] = color

        # -- Bounding box --------------------------------------------------
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # -- ID label ------------------------------------------------------
        label = f"#{int(tid)}"
        lx, ly = x1, max(y1 - 6, 14)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (lx, ly - th - bl), (lx + tw, ly + bl), color, -1)
        cv2.putText(
            out, label, (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, mask_alpha, out, 1.0 - mask_alpha, 0, out)
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _open_writer(output_path: str, cap: cv2.VideoCapture) -> cv2.VideoWriter:
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for: {output_path}")
    return writer


def run(args: argparse.Namespace) -> None:
    if not Path(args.input).is_file():
        logger.error("Input video not found: %s", args.input)
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    device = args.device

    # -- Step 1: First pass with YOLO BoT-SORT to get initial bounding boxes
    #    then prompt SAM2 on the first frame ---------------------------------
    logger.info("Loading YOLO model: %s", args.det_model)
    from ultralytics import YOLO
    yolo = YOLO(args.det_model)

    # -- Step 2: Build SAM2 camera predictor ---------------------------------
    logger.info("Loading SAM2: config=%s  checkpoint=%s", args.sam_config, args.sam_checkpoint)
    from sam2.build_sam import build_sam2_camera_predictor
    predictor = build_sam2_camera_predictor(
        args.sam_config,
        args.sam_checkpoint,
        device="cuda:1",
    )

    # Import SAM2Tracker (lives in the segmentation_tracking package)
    import importlib, os
    # Ensure the repo root is on sys.path when this script is run directly
    _repo_root = str(Path(__file__).resolve().parents[1])
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    from segmentation_tracking.sam2_tracker import SAM2Tracker
    sam_tracker = SAM2Tracker(predictor)

    # -- Step 3: Open video --------------------------------------------------
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", args.input)
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = args.max_frames or total
    writer = _open_writer(args.output, cap)

    logger.info("Processing video: %s  (%d frames → %d)", args.input, total, max_frames)

    _PERSON_CLS = 0
    prompted = False

    for frame_idx in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break

        # -- YOLO tracking pass (BoT-SORT, person class only) ----------------
        yolo_results = yolo.track(
            frame,
            persist=True,
            conf=args.conf,
            classes=[_PERSON_CLS],
            verbose=False,
            device=device,
        )

        # Build supervision Detections from YOLO output
        player_bboxes: list[np.ndarray] = []
        player_ids: list[int] = []
        if yolo_results and yolo_results[0].boxes is not None:
            r = yolo_results[0]
            for i in range(len(r.boxes)):
                bbox = r.boxes.xyxy[i].cpu().numpy().astype(np.float32)
                player_bboxes.append(bbox)
                tid = (
                    int(r.boxes.id[i].item())
                    if r.boxes.id is not None and i < len(r.boxes.id)
                    else i + 1
                )
                player_ids.append(tid)

        if not player_bboxes:
            writer.write(frame)
            if (frame_idx + 1) % 50 == 0:
                logger.info("Frame %d / %d — no detections", frame_idx + 1, max_frames)
            continue

        yolo_dets = sv.Detections(
            xyxy=np.array(player_bboxes, dtype=np.float32),
            tracker_id=np.array(player_ids, dtype=np.int32),
        )

        # -- SAM2: seed on first detection, then track -----------------------
        if not prompted:
            logger.info("Prompting SAM2 with %d players on frame %d", len(yolo_dets), frame_idx)
            sam_tracker.prompt_first_frame(frame, yolo_dets)
            prompted = True
            writer.write(frame)
            continue

        sam_dets = sam_tracker.track(frame, new_detections=None)

        # Remap SAM2 internal IDs back to YOLO BoT-SORT IDs via IoU matching
        sam_dets = _remap_ids(sam_dets, yolo_dets)

        annotated = _draw_frame(frame, sam_dets, mask_alpha=args.mask_alpha)
        writer.write(annotated)

        if (frame_idx + 1) % 50 == 0:
            logger.info("Frame %d / %d", frame_idx + 1, max_frames)

    cap.release()
    writer.release()
    logger.info("Done. Output: %s", args.output)


def _remap_ids(
    sam_dets: sv.Detections,
    yolo_dets: sv.Detections,
) -> sv.Detections:
    """Replace SAM2 internal sequential IDs with YOLO BoT-SORT IDs via IoU.

    SAM2 assigns its own 1-based object IDs that are internal to its session.
    YOLO BoT-SORT assigns stable IDs across the whole video.  We match the
    two sets by computing pairwise bbox IoU (Hungarian) and copy the BoT-SORT
    IDs into the SAM2 Detections object so the visualisation shows consistent
    IDs.
    """
    if (
        sam_dets is None
        or len(sam_dets) == 0
        or yolo_dets is None
        or len(yolo_dets) == 0
    ):
        return sam_dets

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    n_sam = len(sam_dets)
    n_yolo = len(yolo_dets)
    cost = np.zeros((n_sam, n_yolo), dtype=np.float32)
    for i in range(n_sam):
        for j in range(n_yolo):
            cost[i, j] = 1.0 - iou(sam_dets.xyxy[i], yolo_dets.xyxy[j])

    from scipy.optimize import linear_sum_assignment
    row_idx, col_idx = linear_sum_assignment(cost)

    new_ids = sam_dets.tracker_id.copy() if sam_dets.tracker_id is not None else np.arange(n_sam, dtype=np.int32)
    yolo_ids = yolo_dets.tracker_id if yolo_dets.tracker_id is not None else np.arange(n_yolo, dtype=np.int32)

    for r, c in zip(row_idx, col_idx):
        if cost[r, c] < 0.7:  # IoU > 0.3
            new_ids[r] = yolo_ids[c]

    return sv.Detections(
        xyxy=sam_dets.xyxy,
        mask=sam_dets.mask,
        tracker_id=new_ids,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM2 segmentation + YOLO BoT-SORT tracking — standalone script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument(
        "--output", default="output/sam2_tracking.mp4",
        help="Path to annotated output video",
    )
    parser.add_argument(
        "--sam_config", default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 model config YAML",
    )
    parser.add_argument(
        "--sam_checkpoint", default="checkpoints/sam2.1_hiera_large.pt",
        help="SAM2 model checkpoint",
    )
    parser.add_argument(
        "--det_model", default="yolo11x.pt",
        help="YOLO detection model for BoT-SORT",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Torch device: 'cuda', 'cuda:0', 'cuda:1', or 'cpu'",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="YOLO detection confidence threshold",
    )
    parser.add_argument(
        "--mask_alpha", type=float, default=0.40,
        help="Segmentation overlay opacity (0–1)",
    )
    parser.add_argument(
        "--max_frames", type=int, default=None,
        help="Maximum number of frames to process (default: all)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(_parse_args())
