"""
team_classifier.py
------------------
HSV K-means jersey-colour classifier for football player team assignment.

The classifier extracts the dominant hue from the upper-torso region of
each player crop and uses global K-means to group players into teams.  It
runs entirely on CPU with OpenCV and requires no additional ML model.

Typical usage
~~~~~~~~~~~~~
::

    classifier = TeamClassifier(n_teams=2)

    # Collect jersey-colour samples frame by frame
    for pt in player_tracks:
        crop = TeamClassifier.extract_torso_crop(frame, pt.bbox)
        classifier.update(pt.id, crop)

    # (Re-)fit periodically, e.g. every 30 frames
    if frame_idx % 30 == 0:
        classifier.fit()

    # Query the team label for each player
    for pt in player_tracks:
        pt.team_label = classifier.get_team(pt.id)
        # 0, 1 (or 2 for referee if n_teams=3), or None if not yet classified

Public API
~~~~~~~~~~
``TeamClassifier``
    ``update(player_id, crop_bgr)``
    ``fit() → bool``
    ``assign_new(player_id) → int | None``
    ``get_team(player_id) → int | None``
    ``team_labels() → dict[int, int]``
``TeamClassifier.extract_torso_crop(frame, bbox) → np.ndarray | None``
"""

from __future__ import annotations

import logging
from collections import defaultdict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Vertical fraction of the player bbox used as the torso ROI
_TORSO_TOP_FRAC = 0.15   # skip the very top (head)
_TORSO_BOT_FRAC = 0.60   # stop before the legs

# Minimum HSV saturation/value to count a pixel as "coloured jersey"
# (avoids grass, skin, and near-white background pixels)
_MIN_SAT = 30
_MIN_VAL = 30

# K-means termination criteria:
#   - stop after 200 iterations or when centres shift < 0.5° (hue units)
_KMEANS_MAX_ITER = 200
_KMEANS_EPSILON = 0.5

# Default minimum colour samples per player before contributing to clustering
_MIN_SAMPLES_DEFAULT = 5


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dominant_hue(crop_bgr: np.ndarray | None, n_bins: int = 16) -> float | None:
    """Return the dominant OpenCV hue (0–180°) for a BGR player crop.

    Dark and near-white pixels (skin, grass reflections, background) are
    excluded before the hue histogram is computed.  Returns *None* when the
    crop contains too few coloured pixels to produce a reliable estimate.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]
    mask = (s_chan > _MIN_SAT) & (v_chan > _MIN_VAL)
    hues = h_chan[mask]
    if len(hues) < 20:
        return None
    hist, edges = np.histogram(hues, bins=n_bins, range=(0, 180))
    peak_bin = int(np.argmax(hist))
    return float((edges[peak_bin] + edges[peak_bin + 1]) / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Public class
# ─────────────────────────────────────────────────────────────────────────────

class TeamClassifier:
    """Jersey-hue K-means classifier for team/referee assignment.

    Parameters
    ----------
    n_teams:
        Number of team clusters.  Use ``2`` for two opposing teams;
        ``3`` to also include a referee cluster.
    min_samples:
        Minimum number of hue observations per player before that player
        contributes to the K-means fit.
    """

    def __init__(
        self,
        n_teams: int = 2,
        min_samples: int = _MIN_SAMPLES_DEFAULT,
    ) -> None:
        self.n_teams = n_teams
        self.min_samples = min_samples

        # Raw hue samples collected per player ID
        self._hue_samples: dict[int, list[float]] = defaultdict(list)

        # Assigned team label per player ID (populated by fit())
        self._team_labels: dict[int, int] = {}

        # K-means cluster centres (hue values), set by fit()
        self._centres: np.ndarray | None = None

    # ── Data collection ───────────────────────────────────────────────────────

    def update(self, player_id: int, crop_bgr: np.ndarray | None) -> None:
        """Add a jersey-colour sample for *player_id* from a BGR *crop_bgr*.

        Silently skips frames where the crop is empty or has too few
        coloured pixels to extract a reliable hue.
        """
        hue = _dominant_hue(crop_bgr)
        if hue is not None:
            self._hue_samples[player_id].append(hue)

    # ── Clustering ────────────────────────────────────────────────────────────

    def fit(self) -> bool:
        """Run K-means on accumulated samples and assign team labels.

        Returns
        -------
        bool
            *True* if clustering succeeded; *False* if there are not
            enough qualified players (each needing ``min_samples``
            observations).
        """
        qualified = [
            pid
            for pid, samples in self._hue_samples.items()
            if len(samples) >= self.min_samples
        ]
        if len(qualified) < self.n_teams:
            logger.debug(
                "TeamClassifier.fit: only %d qualified players, need at least %d",
                len(qualified),
                self.n_teams,
            )
            return False

        # One robust representative hue per player (median reduces outliers)
        medians = np.array(
            [float(np.median(self._hue_samples[pid])) for pid in qualified],
            dtype=np.float32,
        ).reshape(-1, 1)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, _KMEANS_MAX_ITER, _KMEANS_EPSILON)
        _, labels, centres = cv2.kmeans(
            medians, self.n_teams, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        self._centres = centres.flatten()

        for pid, lbl in zip(qualified, labels.flatten()):
            self._team_labels[pid] = int(lbl)

        logger.debug(
            "TeamClassifier fitted: %d players → %d teams; hue centres=%s",
            len(qualified),
            self.n_teams,
            [f"{c:.1f}°" for c in self._centres],
        )
        return True

    def assign_new(self, player_id: int) -> int | None:
        """Assign *player_id* to the nearest existing cluster without refitting.

        Useful for classifying a new player that entered the scene after
        the last :meth:`fit` call.  Returns *None* if the player has fewer
        than ``min_samples`` observations or no clusters exist yet.
        """
        if self._centres is None:
            return None
        samples = self._hue_samples.get(player_id, [])
        if len(samples) < self.min_samples:
            return None
        median_hue = float(np.median(samples))
        distances = np.abs(self._centres - median_hue)
        lbl = int(np.argmin(distances))
        self._team_labels[player_id] = lbl
        return lbl

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_team(self, player_id: int) -> int | None:
        """Return the team label for *player_id*, or *None* if not yet classified."""
        return self._team_labels.get(player_id)

    def team_labels(self) -> dict[int, int]:
        """Return a ``{player_id: team_label}`` copy of the current mapping."""
        return dict(self._team_labels)

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def extract_torso_crop(
        frame: np.ndarray,
        bbox: np.ndarray,
    ) -> np.ndarray | None:
        """Crop the torso region of a player from *frame*.

        The torso is defined as the vertical fraction
        ``[_TORSO_TOP_FRAC, _TORSO_BOT_FRAC]`` of the bounding box height,
        spanning the full bbox width.

        Parameters
        ----------
        frame:
            BGR image array, shape ``(H, W, 3)``.
        bbox:
            Player bounding box ``[x1, y1, x2, y2]`` (float or int).

        Returns
        -------
        np.ndarray or None
            BGR crop of the torso region, or *None* if the region is invalid.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w - 1)
        y2 = min(y2, h - 1)
        bh = y2 - y1
        if bh <= 0 or x2 <= x1:
            return None
        t_y1 = y1 + int(bh * _TORSO_TOP_FRAC)
        t_y2 = y1 + int(bh * _TORSO_BOT_FRAC)
        if t_y2 <= t_y1:
            return None
        return frame[t_y1:t_y2, x1:x2].copy()
