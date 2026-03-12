"""
segmentation_model.py
---------------------
Player instance segmentation and persistent tracking via SAM2VideoPredictor.

Architecture
~~~~~~~~~~~~
1. A YOLO detection model (``yolo11x.pt``) scans the **first frame** to
   obtain bounding-box prompts for every visible player and the ball.
2. Those prompts seed the ``SAM2VideoPredictor`` (from the ultralytics
   ``ultralytics.models.sam.predict`` module). From frame 1 onwards, the
   predictor propagates the segmentation masks across the video entirely
   on its own, keeping each player's object-ID stable even through short
   occlusions.
3. The ball is tracked independently with YOLO on every frame because
   SAM2's memory-based propagation can occasionally lose a fast-moving
   small object; YOLO gives reliable ball detections per frame.
4. Re-detection runs every ``redetect_interval`` frames so that players
   who enter the scene after the video starts are picked up and given new
   persistent IDs.

Public API
~~~~~~~~~~
``SegmentationTracker.process_video(video_path, max_frames) → list[SegmentationResult]``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── COCO class indices ────────────────────────────────────────────────────────
_PERSON_CLS = 0
_BALL_CLS = 32  # sports ball


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SegmentationResult:
    """Per-frame output of :class:`SegmentationTracker`.

    Attributes
    ----------
    frame_index:
        0-based index of the video frame.
    player_ids:
        List of persistent player identifiers (one per detected player).
    player_masks:
        Binary masks, shape ``(H, W)``, dtype ``uint8`` (0/255), one per player.
    player_bboxes:
        Bounding boxes ``[x1, y1, x2, y2]`` (float32), one per player.
    ball_mask:
        Binary mask for the ball, or *None* if no ball was detected.
    ball_bbox:
        Bounding box for the ball ``[x1, y1, x2, y2]``, or *None*.
    ball_center:
        ``(cx, cy)`` pixel position of the ball centre, or *None*.
    """

    frame_index: int = 0
    player_ids: list[int] = field(default_factory=list)
    player_masks: list[np.ndarray] = field(default_factory=list)
    player_bboxes: list[np.ndarray] = field(default_factory=list)
    ball_mask: np.ndarray | None = None
    ball_bbox: np.ndarray | None = None
    ball_center: tuple[float, float] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationTracker:
    """Player and ball segmentation + tracking.

    Parameters
    ----------
    sam_model_path:
        Path to or name of the SAM2 model weights, e.g. ``"sam2.1_b.pt"``.
        The model is downloaded automatically by ultralytics on first use.
    det_model_path:
        Path to or name of the YOLO detection model used for initial player
        bounding-box prompts, e.g. ``"yolo11x.pt"``.
    device:
        Torch device string, ``"cuda"`` or ``"cpu"``.
    conf_threshold:
        Minimum detection confidence for YOLO.
    iou_threshold:
        IoU threshold used when matching new detections to known tracks during
        re-detection passes.
    redetect_interval:
        Re-run YOLO detection every *N* frames to handle players entering the
        scene after the start.  Set to ``0`` to disable re-detection.
    """

    def __init__(
        self,
        sam_model_path: str = "sam2.1_b.pt",
        det_model_path: str = "yolo11x.pt",
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        redetect_interval: int = 30,
    ) -> None:
        self.sam_model_path = sam_model_path
        self.det_model_path = det_model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.redetect_interval = redetect_interval

        self._detector = None   # lazy-loaded
        self._sam = None        # lazy-loaded SAM2VideoPredictor wrapper
        self._next_player_id: int = 1

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _get_detector(self):
        if self._detector is None:
            from ultralytics import YOLO
            logger.info("Loading YOLO detection model: %s", self.det_model_path)
            self._detector = YOLO(self.det_model_path)
        return self._detector

    def _get_sam_predictor(self):
        """Return an initialised SAM2VideoPredictor instance."""
        if self._sam is None:
            from ultralytics.models.sam.predict import SAM2VideoPredictor
            logger.info("Loading SAM2VideoPredictor: %s", self.sam_model_path)
            self._sam = SAM2VideoPredictor(
                overrides=dict(
                    model=self.sam_model_path,
                    device=self.device,
                    conf=self.conf_threshold,
                    task="segment",
                    mode="predict",
                    imgsz=1024,
                    save=False,
                    verbose=False,
                )
            )
        return self._sam

    # ── YOLO helpers ──────────────────────────────────────────────────────────

    def _detect_frame(
        self, frame: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray | None]:
        """Run YOLO detection and return (player_bboxes, ball_bbox)."""
        det = self._get_detector()
        results = det.predict(
            frame,
            conf=self.conf_threshold,
            classes=[_PERSON_CLS, _BALL_CLS],
            verbose=False,
        )[0]

        player_bboxes: list[np.ndarray] = []
        ball_bbox: np.ndarray | None = None
        best_ball_conf: float = -1.0

        for box in results.boxes:
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(np.float32)

            if cls == _PERSON_CLS:
                player_bboxes.append(xyxy)
            elif cls == _BALL_CLS and conf > best_ball_conf:
                ball_bbox = xyxy
                best_ball_conf = conf

        return player_bboxes, ball_bbox

    # ── Ball helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _segment_ball_from_bbox(
        frame: np.ndarray, bbox: np.ndarray
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Create a binary mask and centre point for the ball bounding box.

        We use a simple ellipse fill inside the bbox as a lightweight mask
        instead of running SAM2 per ball, keeping the pipeline fast.  Replace
        this method with a full SAM2 call if higher-quality ball masks are
        required.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w - 1), min(y2, h - 1)

        mask = np.zeros((h, w), dtype=np.uint8)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rx = max((x2 - x1) // 2, 1)
        ry = max((y2 - y1) // 2, 1)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        center = (float(cx), float(cy))
        return mask, center

    # ── IoU matching ──────────────────────────────────────────────────────────

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
        """Compute IoU between two ``[x1, y1, x2, y2]`` boxes."""
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0.0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> np.ndarray | None:
        """Derive a tight bounding box from a binary mask."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

    # ── Re-detection: assign IDs to new players ───────────────────────────────

    def _assign_new_player_ids(
        self,
        new_bboxes: list[np.ndarray],
        existing_bboxes: list[np.ndarray],
        existing_ids: list[int],
    ) -> list[int]:
        """
        Match *new_bboxes* against *existing_bboxes* by IoU.

        Returns a list of player IDs (same length as *new_bboxes*).
        Unmatched new bboxes get a fresh ID from ``self._next_player_id``.
        """
        assigned: list[int] = []
        used: set[int] = set()

        for nb in new_bboxes:
            best_iou = self.iou_threshold
            best_idx = -1
            for i, eb in enumerate(existing_bboxes):
                if i in used:
                    continue
                iou = self._bbox_iou(nb, eb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0:
                assigned.append(existing_ids[best_idx])
                used.add(best_idx)
            else:
                assigned.append(self._next_player_id)
                self._next_player_id += 1

        return assigned

    # ── Core processing via SAM2VideoPredictor ────────────────────────────────

    def process_video(
        self,
        video_path: str,
        max_frames: int | None = None,
    ) -> list[SegmentationResult]:
        """
        Process a video and return per-frame segmentation results.

        The method runs SAM2VideoPredictor over the entire video using
        bounding-box prompts obtained from YOLO on the first frame.  Ball
        segmentation is derived from per-frame YOLO detections.

        Parameters
        ----------
        video_path:
            Path to the input video file.
        max_frames:
            If set, process at most this many frames.

        Returns
        -------
        list[SegmentationResult]
            One :class:`SegmentationResult` per processed frame, in order.
        """
        self._next_player_id = 1

        # ── Read first frame to initialise ───────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        ok, first_frame = cap.read()
        if not ok:
            raise ValueError("Video is empty")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)
        cap.release()

        logger.info("Video: %s (%d frames to process)", video_path, total_frames)

        # ── Detect initial players on frame 0 ────────────────────────────────
        init_player_bboxes, _ = self._detect_frame(first_frame)
        if not init_player_bboxes:
            logger.warning(
                "No players detected in the first frame; the tracker will not track any players."
            )

        # Assign IDs for initially detected players
        init_ids: list[int] = []
        for _ in init_player_bboxes:
            init_ids.append(self._next_player_id)
            self._next_player_id += 1

        logger.info("Initialising SAM2VideoPredictor with %d player prompts",
                    len(init_player_bboxes))

        # ── Run SAM2VideoPredictor over the video ─────────────────────────────
        # The predictor maintains an inference_state internally and propagates
        # the masks frame-by-frame via memory-conditioned attention.
        predictor = self._get_sam_predictor()

        # Build the initial bbox prompt array (N×4, XYXY)
        init_bboxes_arr = (
            np.stack(init_player_bboxes, axis=0).astype(np.float32)
            if init_player_bboxes
            else None
        )

        # Stream results frame-by-frame from the predictor
        try:
            sam_stream = predictor.predict(
                source=video_path,
                bboxes=init_bboxes_arr,
                stream=True,
                verbose=False,
            )
            sam_results_all: list[Any] = []
            for i, r in enumerate(sam_stream):
                sam_results_all.append(r)
                if max_frames and i + 1 >= max_frames:
                    break
        except Exception as exc:
            logger.error(
                "SAM2VideoPredictor failed (%s). "
                "Falling back to YOLO-only segmentation.", exc
            )
            sam_results_all = []

        # ── Process frames: build SegmentationResult list ─────────────────────
        results: list[SegmentationResult] = []

        # Use a second video capture for ball detection and re-detection
        cap2 = cv2.VideoCapture(video_path)

        for frame_idx in range(total_frames):
            ok2, frame = cap2.read()
            if not ok2:
                break

            seg_result = SegmentationResult(frame_index=frame_idx)

            # ── Extract player masks from SAM2 stream ─────────────────────────
            if frame_idx < len(sam_results_all):
                sam_r = sam_results_all[frame_idx]
                seg_result = self._extract_player_info(
                    sam_r, frame_idx, init_ids, frame.shape, seg_result
                )
            else:
                # Fallback: YOLO detection + ID matching
                seg_result = self._fallback_detect(frame, frame_idx, results, seg_result)

            # ── Re-detection pass ─────────────────────────────────────────────
            if (
                self.redetect_interval > 0
                and frame_idx > 0
                and frame_idx % self.redetect_interval == 0
            ):
                seg_result = self._redetect_new_players(frame, seg_result)

            # ── Ball detection ────────────────────────────────────────────────
            _, ball_bbox = self._detect_frame(frame)
            if ball_bbox is not None:
                ball_mask, ball_center = self._segment_ball_from_bbox(frame, ball_bbox)
                seg_result.ball_mask = ball_mask
                seg_result.ball_bbox = ball_bbox
                seg_result.ball_center = ball_center

            results.append(seg_result)
            if (frame_idx + 1) % 50 == 0:
                logger.info("Processed %d / %d frames", frame_idx + 1, total_frames)

        cap2.release()
        logger.info("Finished processing %d frames", len(results))
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_player_info(
        self,
        sam_result: Any,
        frame_idx: int,
        init_ids: list[int],
        frame_shape: tuple[int, ...],
        seg_result: SegmentationResult,
    ) -> SegmentationResult:
        """Populate *seg_result* from a SAM2 result object for one frame."""
        if sam_result is None or sam_result.masks is None:
            return seg_result

        masks_tensor = sam_result.masks.data  # (N, H, W) bool/float
        orig_h, orig_w = frame_shape[:2]

        for obj_idx, mask_t in enumerate(masks_tensor):
            # Convert mask to uint8 numpy
            mask_np = mask_t.cpu().numpy().astype(np.uint8) * 255

            # Resize to original frame size if needed
            mh, mw = mask_np.shape
            if mh != orig_h or mw != orig_w:
                mask_np = cv2.resize(
                    mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )

            # Skip empty masks
            if not mask_np.any():
                continue

            bbox = self._mask_to_bbox(mask_np)
            if bbox is None:
                continue

            # Map SAM2 object index to a persistent player ID.
            # Objects within the initial prompt set use their pre-assigned IDs.
            # Any extra objects reported by SAM2 (rare) receive fresh IDs.
            if obj_idx < len(init_ids):
                pid = init_ids[obj_idx]
            else:
                pid = self._next_player_id
                self._next_player_id += 1

            seg_result.player_ids.append(pid)
            seg_result.player_masks.append(mask_np)
            seg_result.player_bboxes.append(bbox)

        return seg_result

    def _fallback_detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        previous_results: list[SegmentationResult],
        seg_result: SegmentationResult,
    ) -> SegmentationResult:
        """Fallback: use YOLO detection + ID matching when SAM2 stream is unavailable."""
        player_bboxes, _ = self._detect_frame(frame)

        existing_bboxes: list[np.ndarray] = []
        existing_ids: list[int] = []
        if previous_results:
            prev = previous_results[-1]
            existing_bboxes = prev.player_bboxes
            existing_ids = prev.player_ids

        assigned_ids = self._assign_new_player_ids(
            player_bboxes, existing_bboxes, existing_ids
        )
        h, w = frame.shape[:2]
        for pid, bbox in zip(assigned_ids, player_bboxes):
            mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            seg_result.player_ids.append(pid)
            seg_result.player_masks.append(mask)
            seg_result.player_bboxes.append(bbox)

        return seg_result

    def _redetect_new_players(
        self,
        frame: np.ndarray,
        seg_result: SegmentationResult,
    ) -> SegmentationResult:
        """Detect players not yet in the tracking state and assign fresh IDs."""
        new_bboxes, _ = self._detect_frame(frame)
        for nb in new_bboxes:
            matched = False
            for eb in seg_result.player_bboxes:
                if self._bbox_iou(nb, eb) > self.iou_threshold:
                    matched = True
                    break
            if not matched:
                pid = self._next_player_id
                self._next_player_id += 1
                h, w = frame.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                x1, y1, x2, y2 = nb.astype(int)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                seg_result.player_ids.append(pid)
                seg_result.player_masks.append(mask)
                seg_result.player_bboxes.append(nb)

        return seg_result
