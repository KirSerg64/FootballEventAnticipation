"""
segmentation_model.py
---------------------
Player instance segmentation and persistent tracking.

Architecture (updated)
~~~~~~~~~~~~~~~~~~~~~~
1. **BoT-SORT / ByteTrack** (via ``YOLO.track(persist=True)``) is the
   *primary tracker*.  It provides stable, persistent player IDs backed by a
   Kalman-filter motion model, globally-optimal Hungarian assignment, and
   configurable track-lifecycle management (``max_age``).

2. **SAM2VideoPredictor** is used as a *mask refiner*.  It is seeded with the
   initial YOLO bounding boxes on frame 0 and streams per-frame binary masks.
   These masks are matched to the BoT-SORT tracks via the Hungarian algorithm,
   so the final output combines SAM2's pixel-accurate mask quality with
   BoT-SORT's ID stability.

3. **Camera-motion compensation** (ORB + RANSAC homography) is applied in the
   YOLO-only fallback path and during re-detection to warp previous-frame
   bboxes before IoU matching, preventing ID switches caused by camera panning
   or zooming.

4. The **ball** is tracked with a **detection-first + MOSSE correlation**
   tracker (``BallDCFTracker``) extended with **FRoG-MOT inspired two-stage
   detection and motion-state aware prediction**.

   * **Zero-latency kick response**: YOLO detections are accepted immediately
     without Kalman gating or smoothing.
   * **Stage-1 detection**: global YOLO call with a dedicated
     ``ball_conf_threshold`` (default 0.10, much lower than the player
     threshold 0.25).  A fast-moving ball under motion blur has low YOLO
     confidence; the reduced threshold captures it without flooding the
     player tracker with false positives (BoT-SORT uses its own
     ``track_high_thresh = 0.25`` internally).
   * **Stage-2 detection** (FRoG-MOT stage-2 association): when the ball is
     not found globally, crop the predicted ROI and run YOLO at an even
     lower ``ball_conf_roi`` (default 0.05).  Restricting the search area
     eliminates most false positives and allows ultra-low confidence
     acceptance.
   * **Motion-state aware prediction** (FRoG-MOT motion-state model):
     ``BallDCFTracker`` classifies the ball as STATIC / IN_FLIGHT /
     HIGH_SPEED and selects the appropriate prediction: hold / median
     velocity / latest frame-to-frame displacement.
   * **MOSSE DCF gap filling** (Bolme et al., CVPR 2010): appearance-based
     correlation search when YOLO misses and the gap is short.
   * **Velocity extrapolation** as a tertiary fallback for longer gaps.

5. **Re-detection** every ``redetect_interval`` frames picks up players who
   enter the scene after frame 0.  All matching in the re-detection path also
   uses the Hungarian algorithm.

Public API
~~~~~~~~~~
``SegmentationTracker.process_video(video_path, max_frames) -> list[SegmentationResult]``
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from segmentation_tracking.ball_kalman import BallDCFTracker, BallKalmanFilter, BallCoTrackerTracker

logger = logging.getLogger(__name__)

# -- COCO class indices -------------------------------------------------------
_PERSON_CLS = 0
_BALL_CLS = 32  # sports ball

# -- Homography estimation constants -----------------------------------------
# Number of ORB keypoints to detect per frame for camera-motion estimation
_ORB_N_FEATURES = 500
# Maximum reprojection error (pixels) for RANSAC inlier classification
_HOMOGRAPHY_RANSAC_THRESH = 5.0

# -- Tracker YAML templates ---------------------------------------------------
_BOTSORT_TEMPLATE = (
    "tracker_type: botsort\n"
    "track_high_thresh: {conf:.3f}\n"
    "track_low_thresh: 0.100\n"
    "new_track_thresh: {conf:.3f}\n"
    "track_buffer: {max_age}\n"
    "match_thresh: 0.8\n"
    "fuse_score: true\n"
    "gmc_method: sparseOptFlow\n"
    "with_reid: false\n"
    "proximity_thresh: 0.5\n"
    "appearance_thresh: 0.25\n"
    "model_weights: osnet_x0_25_market.pt\n"
)

_BYTETRACK_TEMPLATE = (
    "tracker_type: bytetrack\n"
    "track_high_thresh: {conf:.3f}\n"
    "track_low_thresh: 0.100\n"
    "new_track_thresh: {conf:.3f}\n"
    "track_buffer: {max_age}\n"
    "match_thresh: 0.8\n"
    "fuse_score: false\n"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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
    ball_source: str = "none"
    """Source of the ball position for this frame.

    ``"detected"``  — stage-1 global YOLO detection.
    ``"roi"``       — stage-2 ROI YOLO detection (FRoG-MOT).
    ``"mosse"``     — MOSSE correlation filter gap-fill.
    ``"predicted"`` — velocity extrapolation (no appearance evidence).
    ``"none"``      — ball not visible / tracker not yet initialised.
    """


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class SegmentationTracker:
    """Player and ball segmentation + tracking.

    Parameters
    ----------
    sam_model_path:
        SAM2 model weights path or name (e.g. ``"sam2.1_b.pt"``).
    det_model_path:
        YOLO detection model for BoT-SORT tracking and initial SAM2 prompts.
    device:
        Torch device string: ``"cuda"`` or ``"cpu"``.
    conf_threshold:
        Minimum detection confidence for YOLO.
    iou_threshold:
        IoU threshold used in Hungarian matching: detections below this IoU
        are considered unmatched and receive a new ID.
    redetect_interval:
        Re-run YOLO every *N* frames to handle players entering after frame 0.
        Set to ``0`` to disable.
    tracker:
        Tracker algorithm: ``"botsort"`` (default) or ``"bytetrack"``.
    max_age:
        Maximum frames a track survives without a detection (maps to
        ``track_buffer`` in the tracker YAML).
    use_homography:
        When *True*, estimate frame-to-frame ORB+RANSAC homography and warp
        previous-frame bboxes before IoU matching in the fallback path to
        compensate for camera motion.
    ball_patch_size:
        MOSSE DCF template size (px).  Default ``32``.  Larger values capture
        more context around the ball; smaller values are faster.
    ball_search_radius:
        Half-side (px) of the MOSSE search window used when YOLO misses the
        ball.  Default ``60`` (120 px diameter).  Increase for a faster ball
        on a wide-angle camera.
    ball_psr_threshold:
        Minimum Peak-to-Sidelobe Ratio for a MOSSE result to be accepted.
        Default ``7.0`` (as recommended in the MOSSE paper).  Lower values
        accept noisier predictions; higher values are more conservative.
    ball_conf_threshold:
        YOLO confidence threshold used **specifically for ball detection** in
        the global pass.  Default ``0.10`` — considerably lower than the
        player threshold (``conf_threshold``) so that a fast-moving ball
        under motion blur (typical confidence 0.05–0.15) is still detected.
        BoT-SORT uses its own ``track_high_thresh`` for player track
        management and is unaffected by this lower value.
    ball_conf_roi:
        Confidence threshold for the **ROI-based secondary ball detection**
        pass (FRoG-MOT stage-2).  Default ``0.05``.  After restricting the
        search to the predicted ball region, false positives are rare even
        at this very low threshold.
    ball_tracker_type:
        Ball tracker implementation to use.  ``"dcf"`` (default) uses the
        MOSSE correlation-filter tracker (:class:`BallDCFTracker`).
        ``"cotracker"`` uses CoTracker3 point tracking
        (:class:`BallCoTrackerTracker`), which propagates the ball between
        YOLO detections using the same neural tracker as the keypoint flow
        backend.
    ball_cotracker_model:
        CoTracker3 hub model for the ball tracker (only used when
        *ball_tracker_type* is ``"cotracker"``).  ``"cotracker3_online"``
        (default) or ``"cotracker3_offline"``.
    ball_cotracker_checkpoint:
        Optional local path to a CoTracker3 ``.pth`` checkpoint for the
        ball tracker.  *None* uses the default pretrained weights.
    ball_cotracker_device:
        Torch device for the ball CoTracker3 model.  Defaults to ``"cuda"``.
    ball_cotracker_redetect_interval:
        Number of YOLO-confirmed ball detections between forced CoTracker3
        re-anchors.  Default ``15``.
    ball_det_model_path:
        Optional path to a dedicated ONNX ball-detection model (e.g.
        ``"weights/yolov26_ball_det.onnx"``).  When provided this model is
        used for **all** ball detection passes (global stage-1 and ROI
        stage-2) in place of the main YOLO model.  The main YOLO result is
        kept as a fallback for the global pass if the dedicated model finds
        nothing.  The dedicated model is expected to output class ``0`` as
        the ball class (single-class detector).  Loaded lazily via
        ``ultralytics.YOLO`` which natively supports ``.onnx`` files.
    ball_det_conf:
        Confidence threshold for the dedicated ball detector (global pass).
        Default ``0.25``.  Ignored when *ball_det_model_path* is *None*.
    """

    def __init__(
        self,
        sam_model_path: str = "sam2.1_b.pt",
        det_model_path: str = "yolo11x.pt",
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.3,
        redetect_interval: int = 30,
        tracker: str = "botsort",
        max_age: int = 30,
        use_homography: bool = True,
        ball_patch_size: int = 32,
        ball_search_radius: int = 60,
        ball_psr_threshold: float = 7.0,
        ball_conf_threshold: float = 0.10,
        ball_conf_roi: float = 0.05,
        ball_tracker_type: str = "dcf",
        ball_cotracker_model: str = "cotracker3_online",
        ball_cotracker_checkpoint: str | None = None,
        ball_cotracker_device: str = "cuda",
        ball_cotracker_redetect_interval: int = 15,
        ball_det_model_path: str | None = None,
        ball_det_conf: float = 0.25,
    ) -> None:
        self.sam_model_path = sam_model_path
        self.det_model_path = det_model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.redetect_interval = redetect_interval
        self.tracker = tracker.lower().replace(".yaml", "")
        self.max_age = max_age
        self.use_homography = use_homography
        self.ball_conf_threshold = ball_conf_threshold
        self.ball_conf_roi = ball_conf_roi
        self.ball_det_model_path = ball_det_model_path
        self.ball_det_conf = ball_det_conf

        self._detector = None                # lazy-loaded main YOLO model
        self._ball_det = None                # lazy-loaded dedicated ball YOLO/ONNX model
        self._sam = None                     # lazy-loaded SAM2VideoPredictor
        self._next_player_id: int = 1

        _btt = ball_tracker_type.lower()
        if _btt == "cotracker":
            self._ball_tracker: BallDCFTracker | BallCoTrackerTracker = BallCoTrackerTracker(
                redetect_interval=ball_cotracker_redetect_interval,
                hub_model=ball_cotracker_model,
                checkpoint=ball_cotracker_checkpoint,
                device=ball_cotracker_device,
            )
        else:
            self._ball_tracker = BallDCFTracker(
                patch_size=ball_patch_size,
                search_radius=ball_search_radius,
                psr_threshold=ball_psr_threshold,
            )

        # Path to customised tracker YAML written at init time
        self._tracker_config_path: str | None = None
        self._write_tracker_config()

    def __del__(self) -> None:
        """Remove the temporary tracker config file on garbage collection."""
        path = getattr(self, "_tracker_config_path", None)
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass

    # -- Tracker config -------------------------------------------------------

    def _write_tracker_config(self) -> None:
        """Write a customised BoT-SORT/ByteTrack YAML to a temp file."""
        template = (
            _BOTSORT_TEMPLATE if self.tracker == "botsort" else _BYTETRACK_TEMPLATE
        )
        config_str = template.format(
            conf=self.conf_threshold,
            max_age=self.max_age,
        )
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="tracker_")
        with os.fdopen(fd, "w") as fh:
            fh.write(config_str)
        self._tracker_config_path = path
        logger.debug("Tracker config written to %s", path)

    # -- Lazy model loading ---------------------------------------------------

    def _get_detector(self):
        if self._detector is None:
            from ultralytics import YOLO
            logger.info("Loading YOLO detection model: %s", self.det_model_path)
            self._detector = YOLO(self.det_model_path)
        return self._detector

    def _get_ball_detector(self):
        """Return the dedicated ball detector (ONNX or YOLO model).

        Loads ``self.ball_det_model_path`` lazily on first call via
        ``ultralytics.YOLO`` (which supports ``.onnx`` exports natively).
        Returns *None* when no dedicated model path was configured.
        """
        if self.ball_det_model_path is None:
            return None
        if self._ball_det is None:
            from ultralytics import YOLO
            logger.info(
                "Loading dedicated ball detection model: %s",
                self.ball_det_model_path,
            )
            self._ball_det = YOLO(self.ball_det_model_path)
        return self._ball_det

    def _detect_ball_with_dedicated(
        self,
        frame: np.ndarray,
        conf: float,
    ) -> np.ndarray | None:
        """Run the dedicated ball detector on *frame*.

        The dedicated model is expected to be a single-class detector with
        class ``0`` = ball.  Returns the ``[x1, y1, x2, y2]`` bbox of the
        highest-confidence detection, or *None* if nothing is found.
        """
        det = self._get_ball_detector()
        if det is None:
            return None
        try:
            results = det.predict(
                frame,
                conf=conf,
                verbose=False,
            )
        except Exception as exc:
            logger.debug("Dedicated ball detector failed: %s", exc)
            return None
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        best_conf = -1.0
        best_bbox: np.ndarray | None = None
        for box in results[0].boxes:
            c = float(box.conf[0].item())
            if c > best_conf:
                best_bbox = box.xyxy[0].cpu().numpy().astype(np.float32)
                best_conf = c
        return best_bbox

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

    # -- BoT-SORT tracking ----------------------------------------------------

    def _track_frame(
        self, frame: np.ndarray
    ) -> tuple[dict[int, np.ndarray], np.ndarray | None]:
        """Run one frame through BoT-SORT and detect the ball.

        A single YOLO forward pass covers both person tracking (class 0)
        and ball detection (class 32).  Person detections carry persistent
        BoT-SORT IDs; ball detections are returned as a raw bbox (the
        correlation tracker processes them in :meth:`_process_ball`).

        Returns
        -------
        player_tracks:
            ``{track_id: bbox_float32}`` for each confirmed person track.
        ball_bbox:
            ``[x1, y1, x2, y2]`` for the highest-confidence ball detection,
            or *None*.
        """
        det = self._get_detector()
        # Use the lower of the two thresholds so that low-confidence ball
        # detections pass through the YOLO forward pass.  BoT-SORT applies
        # its own track_high_thresh (= conf_threshold) for player track
        # creation/maintenance and is unaffected by the lower value.
        _eff_conf = min(self.conf_threshold, self.ball_conf_threshold)
        results = det.track(
            frame,
            persist=True,
            tracker=self._tracker_config_path,
            conf=_eff_conf,
            classes=[_PERSON_CLS, _BALL_CLS],
            verbose=False,
        )

        player_tracks: dict[int, np.ndarray] = {}
        ball_bbox: np.ndarray | None = None
        best_ball_conf = -1.0

        if results:
            r = results[0]
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    cls = int(r.boxes.cls[i].item())
                    bbox = r.boxes.xyxy[i].cpu().numpy().astype(np.float32)
                    conf = float(r.boxes.conf[i].item())
                    if cls == _PERSON_CLS:
                        if r.boxes.id is not None and i < len(r.boxes.id):
                            tid = int(r.boxes.id[i].item())
                            player_tracks[tid] = bbox
                    elif cls == _BALL_CLS and conf >= self.ball_conf_threshold and conf > best_ball_conf:
                        ball_bbox = bbox
                        best_ball_conf = conf

        return player_tracks, ball_bbox

    # -- YOLO helpers (fallback / re-detection only) --------------------------

    def _detect_frame(
        self, frame: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray | None]:
        """Run YOLO predict (no tracker) and return (player_bboxes, ball_bbox)."""
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

    # -- Ball helpers ---------------------------------------------------------

    def _detect_ball_in_roi(
        self,
        frame: np.ndarray,
        pred_cx: float,
        pred_cy: float,
        radius: int,
    ) -> np.ndarray | None:
        """FRoG-MOT stage-2: low-confidence ball detection within predicted ROI.

        When the global YOLO pass misses the ball (e.g. motion blur during a
        pass), this method crops the predicted region and runs YOLO again at
        ``self.ball_conf_roi`` (default 0.05).  Restricting the search area
        eliminates most false positives, making ultra-low confidence viable.

        Parameters
        ----------
        frame:
            Full BGR video frame.
        pred_cx, pred_cy:
            Centre of the predicted ball position (from the tracker).
        radius:
            Half-side (px) of the search region.  The ROI is a
            ``(2·radius × 2·radius)`` px crop centred on the prediction.

        Returns
        -------
        np.ndarray | None
            ``[x1, y1, x2, y2]`` in **frame coordinates** for the
            best ball detection inside the ROI, or *None* if not found.
        """
        h, w = frame.shape[:2]
        x1 = max(0, int(pred_cx - radius))
        y1 = max(0, int(pred_cy - radius))
        x2 = min(w, int(pred_cx + radius))
        y2 = min(h, int(pred_cy + radius))

        if x2 <= x1 or y2 <= y1 or radius < 10:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        det = self._get_ball_detector() or self._get_detector()
        # When using a dedicated ball model (single-class), class 0 is the ball.
        # When falling back to the main YOLO model, class 32 (COCO sports ball) is needed.
        classes_filter = [0] if self._get_ball_detector() is not None else [_BALL_CLS]
        try:
            results = det.predict(
                roi,
                conf=self.ball_conf_roi,
                classes=classes_filter,
                verbose=False,
            )
        except Exception as exc:
            logger.debug("ROI ball detection failed: %s", exc)
            return None

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        best_conf = -1.0
        best_bbox: np.ndarray | None = None
        for box in results[0].boxes:
            conf = float(box.conf[0].item())
            if conf > best_conf:
                xyxy = box.xyxy[0].cpu().numpy().astype(np.float32)
                # Translate ROI-local coords → frame coords
                best_bbox = np.array(
                    [xyxy[0] + x1, xyxy[1] + y1, xyxy[2] + x1, xyxy[3] + y1],
                    dtype=np.float32,
                )
                best_conf = conf

        if best_bbox is not None:
            logger.debug(
                "Ball ROI detection: conf=%.3f at (%.0f,%.0f) pred=(%.0f,%.0f)",
                best_conf,
                (best_bbox[0] + best_bbox[2]) / 2,
                (best_bbox[1] + best_bbox[3]) / 2,
                pred_cx, pred_cy,
            )
        return best_bbox

    @staticmethod
    def _segment_ball_from_center(
        frame: np.ndarray,
        cx: float,
        cy: float,
        bbox: np.ndarray | None,
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Create a binary ellipse mask centred on ``(cx, cy)``.

        Axes are derived from *bbox* when available; otherwise a fallback
        radius of 15 px is used so that a Kalman-predicted position still
        produces a visible mask.
        """
        h, w = frame.shape[:2]
        icx = int(round(max(0.0, min(cx, w - 1))))
        icy = int(round(max(0.0, min(cy, h - 1))))
        if bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            rx = max((x2 - x1) // 2, 1)
            ry = max((y2 - y1) // 2, 1)
        else:
            rx = ry = 15
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (icx, icy), (rx, ry), 0, 0, 360, 255, -1)
        return mask, (float(cx), float(cy))

    # -- Homography helpers ---------------------------------------------------

    @staticmethod
    def _estimate_homography(
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> np.ndarray | None:
        """Estimate the frame-to-frame homography using ORB + RANSAC.

        Returns a 3x3 matrix H such that a point in *prev_frame* maps to
        the corresponding point in *curr_frame* via ``cv2.perspectiveTransform``,
        or *None* if estimation fails.
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=_ORB_N_FEATURES)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 4:
            return None
        matches = sorted(matches, key=lambda m: m.distance)
        good = matches[: min(50, len(matches))]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, _HOMOGRAPHY_RANSAC_THRESH)
        return H

    @staticmethod
    def _warp_bboxes(
        bboxes: list[np.ndarray],
        H: np.ndarray | None,
        frame_shape: tuple[int, ...],
    ) -> list[np.ndarray]:
        """Warp bounding boxes through homography *H*.

        All four corners of each bbox are transformed; the resulting
        axis-aligned bbox is clamped to the frame boundaries.  When *H* is
        *None* the original boxes are returned unchanged.
        """
        if H is None or not bboxes:
            return bboxes
        fh, fw = frame_shape[:2]
        warped: list[np.ndarray] = []
        for bbox in bboxes:
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            corners = np.float32(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            ).reshape(-1, 1, 2)
            wc = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
            nx1 = float(np.clip(wc[:, 0].min(), 0, fw - 1))
            ny1 = float(np.clip(wc[:, 1].min(), 0, fh - 1))
            nx2 = float(np.clip(wc[:, 0].max(), 0, fw - 1))
            ny2 = float(np.clip(wc[:, 1].max(), 0, fh - 1))
            warped.append(np.array([nx1, ny1, nx2, ny2], dtype=np.float32))
        return warped

    # -- IoU and Hungarian matching -------------------------------------------

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

    def _hungarian_match_bboxes(
        self,
        bboxes_a: list[np.ndarray],
        bboxes_b: list[np.ndarray],
    ) -> list[tuple[int, int]]:
        """Globally optimal 1-to-1 assignment via the Hungarian algorithm.

        Builds a cost matrix ``cost[i, j] = 1 - IoU(a_i, b_j)`` and finds
        the minimum-cost assignment.  Only pairs whose IoU meets
        ``self.iou_threshold`` are included in the result.

        Returns
        -------
        list of ``(i, j)`` index pairs where ``bboxes_a[i]`` is matched to
        ``bboxes_b[j]``.
        """
        if not bboxes_a or not bboxes_b:
            return []
        n_a, n_b = len(bboxes_a), len(bboxes_b)
        cost = np.ones((n_a, n_b), dtype=np.float64)
        for i, a in enumerate(bboxes_a):
            for j, b in enumerate(bboxes_b):
                cost[i, j] = 1.0 - self._bbox_iou(a, b)
        row_ind, col_ind = linear_sum_assignment(cost)
        return [
            (int(r), int(c))
            for r, c in zip(row_ind, col_ind)
            if (1.0 - cost[r, c]) >= self.iou_threshold
        ]

    # -- Re-detection: assign IDs to new players ------------------------------

    def _assign_new_player_ids(
        self,
        new_bboxes: list[np.ndarray],
        existing_bboxes: list[np.ndarray],
        existing_ids: list[int],
        homography: np.ndarray | None = None,
        frame_shape: tuple[int, ...] | None = None,
    ) -> list[int]:
        """Match *new_bboxes* to *existing_bboxes* via Hungarian algorithm.

        Optionally warps *existing_bboxes* through *homography* before matching
        to compensate for camera motion (Improvement D).

        Returns a list of player IDs (same length as *new_bboxes*).
        Unmatched new bboxes receive a fresh ID from ``self._next_player_id``.
        """
        if not new_bboxes:
            return []
        if not existing_bboxes:
            ids: list[int] = []
            for _ in new_bboxes:
                ids.append(self._next_player_id)
                self._next_player_id += 1
            return ids

        if homography is not None and frame_shape is not None:
            warped_existing = self._warp_bboxes(existing_bboxes, homography, frame_shape)
        else:
            warped_existing = existing_bboxes

        n_new = len(new_bboxes)
        n_existing = len(warped_existing)
        cost = np.ones((n_new, n_existing), dtype=np.float64)
        for i, nb in enumerate(new_bboxes):
            for j, eb in enumerate(warped_existing):
                cost[i, j] = 1.0 - self._bbox_iou(nb, eb)

        row_ind, col_ind = linear_sum_assignment(cost)

        assigned: list[int | None] = [None] * n_new
        for r, c in zip(row_ind, col_ind):
            if (1.0 - cost[r, c]) >= self.iou_threshold:
                assigned[r] = existing_ids[c]

        for i in range(n_new):
            if assigned[i] is None:
                assigned[i] = self._next_player_id
                self._next_player_id += 1

        return assigned  # type: ignore[return-value]

    # -- SAM2 helpers ---------------------------------------------------------

    def _run_sam_stream(
        self,
        video_path: str,
        max_frames: int | None,
        init_player_bboxes: list[np.ndarray],
    ) -> list[Any]:
        """Stream SAM2VideoPredictor results for the whole video.

        Returns an ordered list of result objects (one per processed frame),
        or an empty list if the predictor fails.
        """
        predictor = self._get_sam_predictor()
        init_bboxes_arr = (
            np.stack(init_player_bboxes, axis=0).astype(np.float32)
            if init_player_bboxes
            else None
        )
        try:
            sam_stream = predictor.predict(
                source=video_path,
                bboxes=init_bboxes_arr,
                stream=True,
                verbose=False,
            )
            sam_results: list[Any] = []
            for i, r in enumerate(sam_stream):
                sam_results.append(r)
                if max_frames and i + 1 >= max_frames:
                    break
            return sam_results
        except Exception as exc:
            logger.error(
                "SAM2VideoPredictor failed (%s). Falling back to BoT-SORT bbox masks.",
                exc,
            )
            return []

    def _extract_sam_masks(
        self,
        sam_result: Any,
        frame_shape: tuple[int, ...],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Extract ``(masks, bboxes)`` lists from a SAM2 result object."""
        masks: list[np.ndarray] = []
        bboxes: list[np.ndarray] = []
        if sam_result is None or sam_result.masks is None:
            return masks, bboxes
        orig_h, orig_w = frame_shape[:2]
        for mask_t in sam_result.masks.data:
            mask_np = mask_t.cpu().numpy().astype(np.uint8) * 255
            mh, mw = mask_np.shape
            if mh != orig_h or mw != orig_w:
                mask_np = cv2.resize(
                    mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )
            if not mask_np.any():
                continue
            bbox = self._mask_to_bbox(mask_np)
            if bbox is None:
                continue
            masks.append(mask_np)
            bboxes.append(bbox)
        return masks, bboxes

    # -- Merge BoT-SORT tracks with SAM2 masks --------------------------------

    def _merge_bot_sam_results(
        self,
        bot_tracks: dict[int, np.ndarray],
        sam_masks: list[np.ndarray],
        sam_bboxes: list[np.ndarray],
        frame: np.ndarray,
        seg_result: SegmentationResult,
    ) -> SegmentationResult:
        """Combine BoT-SORT IDs with SAM2 masks via Hungarian matching.

        For each BoT-SORT track a matching SAM2 mask is sought.  Tracks with
        no sufficiently overlapping mask fall back to a rectangle mask.
        SAM2 masks with no matching BoT-SORT track are discarded.
        """
        track_ids = list(bot_tracks.keys())
        track_bboxes = [bot_tracks[tid] for tid in track_ids]

        if sam_bboxes:
            matches = self._hungarian_match_bboxes(track_bboxes, sam_bboxes)
            matched_tracks = {ti for ti, _ in matches}

            # Matched: BoT-SORT bbox + SAM2 mask
            for ti, si in matches:
                seg_result.player_ids.append(track_ids[ti])
                seg_result.player_masks.append(sam_masks[si])
                seg_result.player_bboxes.append(track_bboxes[ti])

            # Unmatched BoT-SORT tracks: rectangle mask
            for ti, (tid, bbox) in enumerate(zip(track_ids, track_bboxes)):
                if ti not in matched_tracks:
                    seg_result.player_ids.append(tid)
                    seg_result.player_masks.append(self._rect_mask(bbox, frame.shape))
                    seg_result.player_bboxes.append(bbox)
        else:
            # No SAM2 masks: rectangle masks for all BoT-SORT tracks
            for tid, bbox in zip(track_ids, track_bboxes):
                seg_result.player_ids.append(tid)
                seg_result.player_masks.append(self._rect_mask(bbox, frame.shape))
                seg_result.player_bboxes.append(bbox)

        return seg_result

    @staticmethod
    def _rect_mask(bbox: np.ndarray, frame_shape: tuple[int, ...]) -> np.ndarray:
        """Return a filled-rectangle binary mask for *bbox*."""
        fh, fw = frame_shape[:2]
        mask = np.zeros((fh, fw), dtype=np.uint8)
        x1 = max(int(bbox[0]), 0)
        y1 = max(int(bbox[1]), 0)
        x2 = min(int(bbox[2]), fw - 1)
        y2 = min(int(bbox[3]), fh - 1)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask

    # -- Fallback (no BoT-SORT results) ---------------------------------------

    def _fallback_detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        previous_results: list[SegmentationResult],
        seg_result: SegmentationResult,
        homography: np.ndarray | None = None,
    ) -> SegmentationResult:
        """YOLO-only fallback with Hungarian matching + homography compensation."""
        player_bboxes, _ = self._detect_frame(frame)
        existing_bboxes: list[np.ndarray] = []
        existing_ids: list[int] = []
        if previous_results:
            prev = previous_results[-1]
            existing_bboxes = prev.player_bboxes
            existing_ids = prev.player_ids
        assigned_ids = self._assign_new_player_ids(
            player_bboxes,
            existing_bboxes,
            existing_ids,
            homography=homography,
            frame_shape=frame.shape,
        )
        for pid, bbox in zip(assigned_ids, player_bboxes):
            seg_result.player_ids.append(pid)
            seg_result.player_masks.append(self._rect_mask(bbox, frame.shape))
            seg_result.player_bboxes.append(bbox)
        return seg_result

    def _redetect_new_players(
        self,
        frame: np.ndarray,
        seg_result: SegmentationResult,
        homography: np.ndarray | None = None,
    ) -> SegmentationResult:
        """Detect players not yet tracked and assign fresh IDs."""
        new_bboxes, _ = self._detect_frame(frame)
        if not new_bboxes:
            return seg_result

        existing = seg_result.player_bboxes
        existing_warped = (
            self._warp_bboxes(existing, homography, frame.shape)
            if homography is not None and existing
            else existing
        )

        for nb in new_bboxes:
            if any(self._bbox_iou(nb, eb) > self.iou_threshold for eb in existing_warped):
                continue  # already tracked
            pid = self._next_player_id
            self._next_player_id += 1
            seg_result.player_ids.append(pid)
            seg_result.player_masks.append(self._rect_mask(nb, frame.shape))
            seg_result.player_bboxes.append(nb)

        return seg_result

    # -- Ball processing ------------------------------------------------------

    def _process_ball(
        self,
        frame: np.ndarray,
        ball_bbox: np.ndarray | None,
        seg_result: SegmentationResult,
    ) -> SegmentationResult:
        """Update ball tracker and populate *seg_result* ball fields.

        Two-stage detection strategy (FRoG-MOT):

        **Stage 1 — Global YOLO** (performed in :meth:`_track_frame`):
            Ball detected at ``ball_conf_threshold`` (default 0.10).  Position
            accepted immediately — no gating, no smoothing.

        **Stage 2 — ROI YOLO** (FRoG-MOT stage-2 association):
            When Stage 1 misses, the tracker's ``predicted_position`` and
            ``adaptive_search_radius`` are used to crop the frame and re-run
            YOLO at ``ball_conf_roi`` (default 0.05).  False-positive rate is
            low because the search area is small.  A successful ROI detection
            is treated as a confirmed YOLO detection and updates the tracker.

        **Stage 3 — MOSSE / velocity extrapolation**:
            Fallback when both YOLO passes fail.
        """
        cx: float | None = None
        cy: float | None = None

        if ball_bbox is not None:
            # Stage 1: global YOLO detection accepted immediately
            raw_cx = float((ball_bbox[0] + ball_bbox[2]) / 2)
            raw_cy = float((ball_bbox[1] + ball_bbox[3]) / 2)
            cx, cy = self._ball_tracker.update(raw_cx, raw_cy, frame)
            seg_result.ball_source = "detected"
        elif self._ball_tracker.initialized:
            if self._ball_tracker.frames_since_detection < self.max_age:
                # Stage 2: ROI-based re-detection (FRoG-MOT stage-2 association)
                pred_pos = self._ball_tracker.predicted_position
                if pred_pos is not None:
                    roi_radius = self._ball_tracker.adaptive_search_radius
                    roi_bbox = self._detect_ball_in_roi(
                        frame, pred_pos[0], pred_pos[1], roi_radius
                    )
                    if roi_bbox is not None:
                        # Treat ROI detection as a confirmed detection
                        raw_cx = float((roi_bbox[0] + roi_bbox[2]) / 2)
                        raw_cy = float((roi_bbox[1] + roi_bbox[3]) / 2)
                        cx, cy = self._ball_tracker.update(raw_cx, raw_cy, frame)
                        ball_bbox = roi_bbox   # use for mask ellipse size
                        seg_result.ball_source = "roi"
                    else:
                        # Stage 3: MOSSE search / velocity extrapolation.
                        # For CoTracker3 tracker pass player bboxes so it can
                        # reject positions that drifted onto a player's foot.
                        if isinstance(self._ball_tracker, BallCoTrackerTracker):
                            cx, cy = self._ball_tracker.predict(
                                frame,
                                player_bboxes=seg_result.player_bboxes or None,
                            )
                        else:
                            cx, cy = self._ball_tracker.predict(frame)
                        seg_result.ball_source = self._ball_tracker.last_source

        if cx is not None and cy is not None:
            ball_mask, ball_center = self._segment_ball_from_center(
                frame, cx, cy, ball_bbox
            )
            seg_result.ball_mask = ball_mask
            seg_result.ball_bbox = ball_bbox  # None when position is predicted
            seg_result.ball_center = ball_center

        return seg_result

    # -- Core processing loop -------------------------------------------------

    def process_video(
        self,
        video_path: str,
        max_frames: int | None = None,
    ) -> list[SegmentationResult]:
        """Process a video and return per-frame segmentation results.

        Steps
        -----
        1. Run SAM2VideoPredictor over the video (seeded with YOLO detections
           from frame 0) to collect pixel-accurate masks.
        2. In a second pass, run BoT-SORT per frame to obtain stable player IDs.
        3. For each frame, Hungarian-match BoT-SORT bboxes to SAM2 mask bboxes
           to combine ID stability with mask quality.
        4. Apply ball detection-first correlation tracking (instant kick response).

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
        self._ball_tracker.reset()

        # -- Read first frame -------------------------------------------------
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

        # -- Detect initial players on frame 0 (seed for SAM2) ----------------
        init_player_bboxes, _ = self._detect_frame(first_frame)
        if not init_player_bboxes:
            logger.warning(
                "No players detected in the first frame; SAM2 will have no seeds."
            )
        logger.info(
            "Seeding SAM2VideoPredictor with %d player prompts from frame 0",
            len(init_player_bboxes),
        )

        # -- Pass 1: collect SAM2 masks for the whole video -------------------
        logger.info("=== SAM2 mask streaming pass ===")
        sam_results_all = self._run_sam_stream(video_path, max_frames, init_player_bboxes)
        logger.info("SAM2 pass complete: %d frame results", len(sam_results_all))

        # -- Pass 2: BoT-SORT tracking + merge --------------------------------
        logger.info(
            "=== BoT-SORT (%s) tracking + association pass ===", self.tracker
        )
        results: list[SegmentationResult] = []
        prev_frame: np.ndarray | None = None

        cap2 = cv2.VideoCapture(video_path)
        for frame_idx in range(total_frames):
            ok2, frame = cap2.read()
            if not ok2:
                break

            seg_result = SegmentationResult(frame_index=frame_idx)

            # Camera-motion compensation homography
            homography: np.ndarray | None = None
            if self.use_homography and prev_frame is not None:
                homography = self._estimate_homography(prev_frame, frame)

            # BoT-SORT tracking + ball detection (from main YOLO)
            bot_tracks, ball_bbox_raw = self._track_frame(frame)

            # Dedicated ball detector (YOLOv26 ONNX or similar).
            # Run in addition to the main YOLO call; preferred over the main
            # YOLO result when it finds a detection.  Falls back to the main
            # YOLO result if dedicated model returns nothing.
            if self.ball_det_model_path is not None:
                dedicated_bbox = self._detect_ball_with_dedicated(
                    frame, self.ball_det_conf
                )
                if dedicated_bbox is not None:
                    ball_bbox_raw = dedicated_bbox

            # SAM2 masks for this frame
            if frame_idx < len(sam_results_all):
                sam_masks, sam_bboxes = self._extract_sam_masks(
                    sam_results_all[frame_idx], frame.shape
                )
            else:
                sam_masks, sam_bboxes = [], []

            # Merge or fallback
            if bot_tracks:
                seg_result = self._merge_bot_sam_results(
                    bot_tracks, sam_masks, sam_bboxes, frame, seg_result
                )
            else:
                seg_result = self._fallback_detect(
                    frame, frame_idx, results, seg_result, homography
                )

            # Re-detection for late-entering players
            if (
                self.redetect_interval > 0
                and frame_idx > 0
                and frame_idx % self.redetect_interval == 0
            ):
                seg_result = self._redetect_new_players(frame, seg_result, homography)

            # Ball correlation tracker (detection-first + MOSSE gap fill)
            seg_result = self._process_ball(frame, ball_bbox_raw, seg_result)

            results.append(seg_result)
            prev_frame = frame

            if (frame_idx + 1) % 50 == 0:
                logger.info("Processed %d / %d frames", frame_idx + 1, total_frames)

        cap2.release()
        logger.info("Finished processing %d frames", len(results))
        return results
