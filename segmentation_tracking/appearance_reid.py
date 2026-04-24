"""
appearance_reid.py
------------------
Lightweight appearance-based re-identification for player tracking.

Maintains a per-track gallery of normalised HSV colour histograms extracted
from the torso region of each player bounding box.  When BoT-SORT issues a
brand-new track ID for a player who had disappeared (its ``track_buffer``
expired), ``AppearanceReIDMatcher.rematch`` compares the new detection's
histogram against the gallery and re-assigns the original ID if the cosine
similarity exceeds ``similarity_threshold``.

Design rationale
~~~~~~~~~~~~~~~~
* **HSV histograms** are robust to minor lighting changes and pose variation,
  cheap to compute (< 0.1 ms per crop on CPU), and capture jersey colour well.
* **Torso-only crop** (upper 15 %–60 % of bbox) removes the pitch background,
  feet, and ball from the appearance descriptor.
* **Exponential moving average** (α=0.7 old + 0.3 new) smoothly updates each
  active track's appearance model without storing many frames of history.
* **Gallery TTL** prunes lost tracks after ``gallery_ttl`` frames so stale
  entries do not cause false matches (default 90 frames ≈ 3 s at 30 fps).
* **Hungarian assignment** is used when multiple new IDs could match the same
  gallery entry; the globally optimal 1-to-1 assignment is selected.
* **YOLO backbone features** (optional): when the caller provides per-track
  L2-normalised FPN feature vectors extracted by
  :class:`~segmentation_tracking.yolo_features.YOLOFeatureExtractor`, these
  are stored alongside the HSV histogram and blended into the similarity score
  during re-matching.  The blend weight is controlled by ``yolo_feat_weight``.
  When YOLO features are unavailable for either gallery or candidate track, the
  matcher falls back to HSV-only similarity seamlessly.

Public API
~~~~~~~~~~
``AppearanceReIDMatcher(gallery_ttl, similarity_threshold, yolo_feat_weight)``
    Constructor.

``update_active(track_id, frame, bbox, yolo_feat=None)``
    Call every frame for each visible track to maintain its appearance model.
    ``yolo_feat`` is an optional L2-normalised float32 feature vector from
    :class:`~segmentation_tracking.yolo_features.YOLOFeatureExtractor`.

``notify_lost(lost_ids, last_bboxes)``
    Call when tracks disappear; moves their appearance into the gallery.

``rematch(new_track_ids, frame, bboxes, yolo_feats=None) -> dict[int, int]``
    Returns ``{new_id: old_id}`` for IDs that were successfully re-identified.
    ``yolo_feats`` is an optional ``{track_id: feature_vector}`` dict for the
    candidate new tracks.

``age_gallery(current_track_ids)``
    Ages gallery entries and removes expired ones.  Call at end of each frame.

``reset()``
    Clear all state (call at the start of each new video).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Histogram parameters
# ---------------------------------------------------------------------------

# 3-D HSV histogram: H × S × V bins
_H_BINS = 18     # Hue    (0–179 in OpenCV → 10° per bin)
_S_BINS = 16     # Saturation (0–255 → ~16 levels)
_V_BINS = 8      # Value / lightness (0–255 → ~32 levels)
_HIST_SIZE = [_H_BINS, _S_BINS, _V_BINS]
_HIST_RANGES = [0, 180, 0, 256, 0, 256]

# Torso crop fractions (skip head & feet)
_TORSO_TOP_FRAC = 0.15   # fraction from top of bbox where torso starts
_TORSO_BOT_FRAC = 0.60   # fraction from top of bbox where torso ends

# EMA weight for active-model updates (0.7 old + 0.3 new)
_EMA_ALPHA = 0.7


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _GalleryEntry:
    """Appearance record for a recently-lost track."""

    track_id: int
    histogram: np.ndarray            # normalised, unit-L2 flat float32 array
    last_bbox: np.ndarray            # [x1, y1, x2, y2] at time of disappearance
    frames_absent: int = 0           # incremented each frame the track is absent
    yolo_feat: np.ndarray | None = None  # optional L2-normalised YOLO feature vector


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class AppearanceReIDMatcher:
    """Re-identify players after occlusions using HSV appearance histograms.

    Optionally blends in YOLO backbone feature vectors (from
    :class:`~segmentation_tracking.yolo_features.YOLOFeatureExtractor`) to
    improve discrimination when two players share similar jersey colours.

    Parameters
    ----------
    gallery_ttl:
        Number of frames to retain a lost-track gallery entry before pruning.
        Set to ``≈ fps * max_occlusion_seconds`` (default ``90`` ≈ 3 s at 30 fps).
    similarity_threshold:
        Minimum cosine similarity ``[0, 1]`` required to accept a gallery
        match.  Higher is more conservative; default ``0.85``.
    yolo_feat_weight:
        Blend weight for YOLO feature similarity vs. HSV histogram similarity
        when both are available.  ``0.0`` → HSV only; ``1.0`` → YOLO only;
        default ``0.5`` (equal blend).  When YOLO features are absent for
        either gallery entry or candidate, the matcher silently falls back to
        HSV-only similarity regardless of this setting.
    """

    def __init__(
        self,
        gallery_ttl: int = 90,
        similarity_threshold: float = 0.85,
        yolo_feat_weight: float = 0.5,
    ) -> None:
        self.gallery_ttl = gallery_ttl
        self.similarity_threshold = similarity_threshold
        self.yolo_feat_weight = float(np.clip(yolo_feat_weight, 0.0, 1.0))

        # active_hists: running appearance model per currently-visible track
        self._active_hists: dict[int, np.ndarray] = {}
        # active_yolo_feats: latest YOLO feature vector per active track (optional)
        self._active_yolo_feats: dict[int, np.ndarray] = {}
        # gallery: old_track_id → GalleryEntry for recently-lost tracks
        self._gallery: dict[int, _GalleryEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state (call at the start of each new video)."""
        self._active_hists.clear()
        self._active_yolo_feats.clear()
        self._gallery.clear()

    def update_active(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: np.ndarray,
        yolo_feat: np.ndarray | None = None,
    ) -> None:
        """Update the rolling appearance model for an active track.

        Call this every frame for each currently-visible track **after**
        applying any ID remapping.

        Parameters
        ----------
        track_id:
            Confirmed player ID (after ReID remap).
        frame:
            Current BGR frame.
        bbox:
            ``[x1, y1, x2, y2]`` bounding box for the player.
        yolo_feat:
            Optional L2-normalised float32 YOLO backbone feature vector
            for this player in this frame, as produced by
            :class:`~segmentation_tracking.yolo_features.YOLOFeatureExtractor`.
            When provided, it is stored (replacing any previous value) and
            will be carried into the gallery when the track disappears.
        """
        hist = self._extract_histogram(frame, bbox)
        if hist is None:
            return
        prev = self._active_hists.get(track_id)
        if prev is None:
            self._active_hists[track_id] = hist
        else:
            # Exponential moving average: weight recent history more
            self._active_hists[track_id] = _EMA_ALPHA * prev + (1 - _EMA_ALPHA) * hist
            # Re-normalise after averaging
            norm = np.linalg.norm(self._active_hists[track_id])
            if norm > 1e-6:
                self._active_hists[track_id] /= norm

        # Store the latest YOLO feature (most recent frame overwrites previous)
        if yolo_feat is not None:
            self._active_yolo_feats[track_id] = yolo_feat

    def notify_lost(
        self,
        lost_ids: list[int],
        last_bboxes: dict[int, np.ndarray],
    ) -> None:
        """Move disappeared tracks into the gallery.

        Parameters
        ----------
        lost_ids:
            Track IDs that were visible in the previous frame but are absent
            in the current frame.
        last_bboxes:
            ``{track_id: bbox}`` of the tracks in their last visible frame.
        """
        for tid in lost_ids:
            hist = self._active_hists.pop(tid, None)
            yolo_feat = self._active_yolo_feats.pop(tid, None)
            if hist is None:
                continue
            bbox = last_bboxes.get(tid)
            if bbox is None:
                continue
            self._gallery[tid] = _GalleryEntry(
                track_id=tid,
                histogram=hist.copy(),
                last_bbox=bbox.copy(),
                yolo_feat=yolo_feat.copy() if yolo_feat is not None else None,
            )
            logger.debug("ReID gallery: track %d added (gallery size=%d)", tid, len(self._gallery))

    def rematch(
        self,
        new_track_ids: list[int],
        frame: np.ndarray,
        bboxes: dict[int, np.ndarray],
        yolo_feats: dict[int, np.ndarray] | None = None,
    ) -> dict[int, int]:
        """Match newly-issued track IDs against the lost-track gallery.

        Uses the Hungarian algorithm for globally-optimal 1-to-1 assignment,
        ensuring two new IDs are never mapped to the same gallery entry.

        Parameters
        ----------
        new_track_ids:
            IDs that appeared this frame but were absent in the previous
            frame (candidates for re-identification).
        frame:
            Current BGR frame.
        bboxes:
            ``{track_id: bbox}`` for all current tracks (used to extract
            histogram for new IDs).
        yolo_feats:
            Optional ``{track_id: feature_vector}`` mapping for new tracks,
            as produced by
            :class:`~segmentation_tracking.yolo_features.YOLOFeatureExtractor`.
            When provided for both the candidate and the gallery entry, the
            similarity is a weighted blend:
            ``(1 - yolo_feat_weight) * hsv_sim + yolo_feat_weight * yolo_sim``.

        Returns
        -------
        ``{new_id: old_id}`` mapping — apply this by replacing ``new_id``
        entries with ``old_id`` in the tracking results.  Only entries with
        blended similarity ≥ ``similarity_threshold`` are included.
        """
        if not new_track_ids or not self._gallery:
            return {}

        gallery_ids = list(self._gallery.keys())
        n_new = len(new_track_ids)
        n_gal = len(gallery_ids)

        # Build new-ID HSV histograms
        new_hists: list[np.ndarray | None] = []
        for nid in new_track_ids:
            bbox = bboxes.get(nid)
            new_hists.append(
                self._extract_histogram(frame, bbox) if bbox is not None else None
            )

        # Build combined similarity matrix
        sim_matrix = np.full((n_new, n_gal), -1.0, dtype=np.float64)
        w = self.yolo_feat_weight

        for i, (nid, hist) in enumerate(zip(new_track_ids, new_hists)):
            if hist is None:
                continue
            new_yolo = yolo_feats.get(nid) if yolo_feats else None
            for j, gid in enumerate(gallery_ids):
                entry = self._gallery[gid]
                hsv_sim = float(np.dot(hist, entry.histogram))

                # Blend in YOLO similarity when both sides have features
                if (
                    w > 0.0
                    and new_yolo is not None
                    and entry.yolo_feat is not None
                    and new_yolo.shape == entry.yolo_feat.shape
                ):
                    yolo_sim = float(np.dot(new_yolo, entry.yolo_feat))
                    sim = (1.0 - w) * hsv_sim + w * yolo_sim
                else:
                    sim = hsv_sim

                sim_matrix[i, j] = sim

        # Hungarian assignment on the cost matrix (cost = 1 - similarity)
        cost_matrix = 1.0 - sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        remap: dict[int, int] = {}
        for r, c in zip(row_ind, col_ind):
            sim = sim_matrix[r, c]
            if sim >= self.similarity_threshold:
                new_id = new_track_ids[r]
                old_id = gallery_ids[c]
                remap[new_id] = old_id
                logger.debug(
                    "ReID: new_id=%d → old_id=%d  (similarity=%.3f)",
                    new_id, old_id, sim,
                )

        return remap

    def age_gallery(self, current_track_ids: set[int]) -> None:
        """Age gallery entries and prune those that exceed ``gallery_ttl``.

        Re-identified tracks (whose old ID is now active again) are also
        removed from the gallery.  Call this at the end of each frame.

        Parameters
        ----------
        current_track_ids:
            Set of all currently-active track IDs (after any ReID remap).
        """
        expired: list[int] = []
        for gid, entry in self._gallery.items():
            if gid in current_track_ids:
                # Successfully re-identified; remove from gallery
                expired.append(gid)
                continue
            entry.frames_absent += 1
            if entry.frames_absent > self.gallery_ttl:
                expired.append(gid)
        for gid in expired:
            del self._gallery[gid]
            logger.debug("ReID gallery: track %d pruned (gallery size=%d)", gid, len(self._gallery))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_histogram(
        frame: np.ndarray,
        bbox: np.ndarray,
    ) -> np.ndarray | None:
        """Extract a normalised, unit-L2 HSV histogram from the torso region.

        Returns *None* when the crop is too small or the histogram is
        degenerate (all-zeros).
        """
        x1, y1, x2, y2 = bbox[:4].astype(int)
        h, w = frame.shape[:2]
        height = y2 - y1
        if height <= 0:
            return None

        # Torso crop: skip head and legs
        ty1 = max(0, y1 + int(height * _TORSO_TOP_FRAC))
        ty2 = max(ty1 + 1, min(h, y1 + int(height * _TORSO_BOT_FRAC)))
        x1c = max(0, x1)
        x2c = min(w, x2)

        if x2c <= x1c or ty2 <= ty1:
            return None

        crop = frame[ty1:ty2, x1c:x2c]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, _HIST_SIZE, _HIST_RANGES)
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L2)
        flat = hist.flatten().astype(np.float32)
        norm = float(np.linalg.norm(flat))
        if norm < 1e-6:
            return None
        return flat / norm
