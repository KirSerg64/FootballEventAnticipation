"""
team_classifier.py
------------------
Jersey-colour and SIGLIP-embedding team classifiers for football player
team assignment.

Two backends are provided:

``TeamClassifier`` (HSV K-means, CPU-only, no external ML model)
    Extracts the dominant hue from the upper-torso region of each player
    crop and clusters players with OpenCV K-means.  Fast and lightweight,
    but sensitive to similar jersey colours.

``SiglipTeamClassifier`` (SIGLIP embeddings + scikit-learn KMeans)
    Uses the ``google/siglip-base-patch16-224`` vision model from
    *transformers* to produce rich per-player crop embeddings.  The player
    segmentation mask can be supplied to blank out the background before
    embedding, which greatly improves accuracy on cluttered scenes.
    Requires ``transformers``, ``torch``, and ``scikit-learn``.

``create_team_classifier(backend, ...)``
    Factory that returns the correct classifier for the requested backend.
    Falls back gracefully to the HSV classifier if *transformers* is not
    installed when ``backend="siglip"`` is requested.

Typical usage
~~~~~~~~~~~~~
::

    clf = create_team_classifier("siglip")

    # Collect samples frame by frame
    for pt in player_tracks:
        crop = TeamClassifier.extract_torso_crop(frame, pt.bbox)
        mask = pt.mask  # optional: blanks out background for SIGLIP
        clf.update(pt.id, crop, mask=mask)

    # (Re-)fit periodically
    if frame_idx % 30 == 0:
        clf.fit()

    for pt in player_tracks:
        pt.team_label = clf.get_team(pt.id)

Public API
~~~~~~~~~~
Both classifiers share the same interface:

``update(player_id, crop_bgr, mask=None)``
``fit() → bool``
``assign_new(player_id) → int | None``
``get_team(player_id) → int | None``
``team_labels() → dict[int, int]``
``TeamClassifier.extract_torso_crop(frame, bbox) → np.ndarray | None``
``create_team_classifier(backend, **kwargs)``
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

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

# SIGLIP model identifier
_SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
# Minimum crops per player before contributing to SIGLIP fit
_SIGLIP_MIN_CROPS = 3
# Target size to resize each crop to before feeding SIGLIP
_SIGLIP_CROP_SIZE = 224


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


def _apply_mask_to_crop(
    crop_bgr: np.ndarray,
    mask: np.ndarray | None,
    bbox: np.ndarray | None = None,
    frame_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return *crop_bgr* with background pixels set to zero using *mask*.

    The *mask* may be a full-frame uint8 array (0/255) — in that case it is
    cropped to match the bbox region.  If *mask* is already the same spatial
    size as *crop_bgr* it is used directly.  Returns the original crop
    unchanged when mask is None or all-zero.
    """
    if mask is None or mask.size == 0:
        return crop_bgr
    ch, cw = crop_bgr.shape[:2]
    mh, mw = mask.shape[:2]
    if mh == ch and mw == cw:
        # mask already matches crop
        crop_mask = mask
    elif bbox is not None:
        # full-frame mask: slice to bbox
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        if frame_shape is not None:
            fh, fw = frame_shape
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, fw - 1), min(y2, fh - 1)
        region = mask[y1:y2, x1:x2]
        if region.shape[0] != ch or region.shape[1] != cw:
            region = cv2.resize(region, (cw, ch), interpolation=cv2.INTER_NEAREST)
        crop_mask = region
    else:
        # Can't align — resize mask to crop size
        crop_mask = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)

    if not (crop_mask > 0).any():
        return crop_bgr

    result = crop_bgr.copy()
    result[crop_mask == 0] = 0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HSV K-means classifier (original)
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

    def update(
        self,
        player_id: int,
        crop_bgr: np.ndarray | None,
        mask: np.ndarray | None = None,  # accepted for API compatibility; ignored by this backend
    ) -> None:
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


# ─────────────────────────────────────────────────────────────────────────────
# SIGLIP embedding classifier
# ─────────────────────────────────────────────────────────────────────────────

class SiglipTeamClassifier:
    """SIGLIP-embedding K-means team classifier.

    Uses the ``google/siglip-base-patch16-224`` vision encoder from
    *transformers* to produce dense per-player crop embeddings that capture
    jersey colour, texture, and shape simultaneously.  Clustering is then
    performed with ``sklearn.cluster.KMeans``.

    The player segmentation mask can optionally be passed to
    :meth:`update` to blank out the background before computing the
    embedding, reducing noise from grass, crowd, and advertising boards.

    Parameters
    ----------
    n_teams:
        Number of clusters (default 2 for two teams).
    min_crops:
        Minimum number of crops per player before they contribute to the
        :meth:`fit`.  Each call to :meth:`update` with a valid crop counts
        as one observation.
    device:
        PyTorch device string (e.g. ``"cuda"`` or ``"cpu"``).
    batch_size:
        Number of crops processed in one SIGLIP forward pass.
    model_id:
        HuggingFace model identifier for the vision encoder.
    """

    def __init__(
        self,
        n_teams: int = 2,
        min_crops: int = _SIGLIP_MIN_CROPS,
        device: str = "cpu",
        batch_size: int = 32,
        model_id: str = _SIGLIP_MODEL_ID,
    ) -> None:
        self.n_teams = n_teams
        self.min_crops = min_crops
        self.device = device
        self.batch_size = batch_size
        self.model_id = model_id

        # Stored crops per player {player_id: [crop_bgr, ...]}
        self._crops: dict[int, list[np.ndarray]] = defaultdict(list)

        # Assigned team label per player ID (populated by fit())
        self._team_labels: dict[int, int] = {}

        # KMeans cluster centres in embedding space (set by fit())
        self._cluster_centres: np.ndarray | None = None  # shape (n_teams, embed_dim)

        # Lazy-loaded model components
        self._model: Any | None = None
        self._processor: Any | None = None
        self._kmeans: Any | None = None

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _load_model(self) -> bool:
        """Load SIGLIP model and processor on first use.  Returns True on success."""
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import SiglipVisionModel, SiglipProcessor  # type: ignore[import]
        except ImportError as exc:
            logger.error(
                "SiglipTeamClassifier requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch.  Error: %s",
                exc,
            )
            return False
        try:
            logger.info("Loading SIGLIP vision model '%s' …", self.model_id)
            self._processor = SiglipProcessor.from_pretrained(self.model_id)
            self._model = SiglipVisionModel.from_pretrained(self.model_id)
            self._model = self._model.to(self.device)
            self._model.eval()
            logger.info("SIGLIP model loaded on device '%s'.", self.device)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load SIGLIP model: %s", exc)
            return False
        return True

    # ── Data collection ───────────────────────────────────────────────────────

    def update(
        self,
        player_id: int,
        crop_bgr: np.ndarray | None,
        mask: np.ndarray | None = None,
    ) -> None:
        """Queue a player crop for embedding.

        Parameters
        ----------
        player_id:
            Unique integer player identifier.
        crop_bgr:
            BGR numpy array of the player region (any size; will be resized).
        mask:
            Optional segmentation mask (uint8 0/255) aligned with or
            larger than *crop_bgr*.  When provided, background pixels
            are zeroed before embedding to reduce noise.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return
        if mask is not None:
            crop_bgr = _apply_mask_to_crop(crop_bgr, mask)
        # Resize to SIGLIP input resolution for consistent embeddings
        try:
            resized = cv2.resize(crop_bgr, (_SIGLIP_CROP_SIZE, _SIGLIP_CROP_SIZE))
        except cv2.error:
            return
        self._crops[player_id].append(resized)

    # ── Embedding extraction ──────────────────────────────────────────────────

    def _extract_embeddings(self, crops_bgr: list[np.ndarray]) -> np.ndarray | None:
        """Return (N, D) float32 SIGLIP embeddings for a list of BGR crops."""
        if not crops_bgr:
            return None
        if not self._load_model():
            return None
        try:
            import torch
            from PIL import Image  # type: ignore[import]
        except ImportError:
            return None

        all_embeds: list[np.ndarray] = []
        for i in range(0, len(crops_bgr), self.batch_size):
            batch_bgr = crops_bgr[i : i + self.batch_size]
            # Convert BGR → RGB PIL images
            pil_imgs = [
                Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                for c in batch_bgr
            ]
            inputs = self._processor(
                images=pil_imgs,
                return_tensors="pt",
                padding="max_length",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            # Mean-pool over patch tokens → (B, D)
            embeds = torch.mean(outputs.last_hidden_state, dim=1)
            all_embeds.append(embeds.cpu().numpy().astype(np.float32))
        return np.concatenate(all_embeds, axis=0)

    # ── Clustering ────────────────────────────────────────────────────────────

    def fit(self) -> bool:
        """Embed all accumulated crops and cluster with KMeans.

        Each qualified player is represented by the **mean embedding** of
        all their stored crops (robust to per-frame noise).

        Returns
        -------
        bool
            *True* if clustering succeeded; *False* otherwise.
        """
        try:
            from sklearn.cluster import KMeans  # type: ignore[import]
        except ImportError:
            logger.error(
                "SiglipTeamClassifier.fit requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )
            return False

        qualified = [
            pid
            for pid, crops in self._crops.items()
            if len(crops) >= self.min_crops
        ]
        if len(qualified) < self.n_teams:
            logger.debug(
                "SiglipTeamClassifier.fit: only %d qualified players, need at least %d",
                len(qualified),
                self.n_teams,
            )
            return False

        # One embedding per player (mean over all their crops)
        per_player_embeds: list[np.ndarray] = []
        for pid in qualified:
            embeds = self._extract_embeddings(self._crops[pid])
            if embeds is None:
                return False
            per_player_embeds.append(embeds.mean(axis=0))

        data = np.stack(per_player_embeds, axis=0)  # (N_players, D)

        kmeans = KMeans(
            n_clusters=self.n_teams,
            n_init=10,
            max_iter=300,
            random_state=42,
        )
        labels = kmeans.fit_predict(data)
        self._cluster_centres = kmeans.cluster_centers_
        self._kmeans = kmeans

        for pid, lbl in zip(qualified, labels):
            self._team_labels[pid] = int(lbl)

        logger.debug(
            "SiglipTeamClassifier fitted: %d players → %d teams",
            len(qualified),
            self.n_teams,
        )
        return True

    def assign_new(self, player_id: int) -> int | None:
        """Assign *player_id* to the nearest existing cluster without refitting.

        Returns *None* if the player has fewer than ``min_crops`` observations
        or if no clusters exist yet.
        """
        if self._cluster_centres is None or self._kmeans is None:
            return None
        crops = self._crops.get(player_id, [])
        if len(crops) < self.min_crops:
            return None
        embeds = self._extract_embeddings(crops)
        if embeds is None:
            return None
        mean_embed = embeds.mean(axis=0, keepdims=True)
        lbl = int(self._kmeans.predict(mean_embed)[0])
        self._team_labels[player_id] = lbl
        return lbl

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_team(self, player_id: int) -> int | None:
        """Return the team label for *player_id*, or *None* if not yet classified."""
        return self._team_labels.get(player_id)

    def team_labels(self) -> dict[int, int]:
        """Return a ``{player_id: team_label}`` copy of the current mapping."""
        return dict(self._team_labels)

    # ── Static helper (shared with TeamClassifier for convenience) ────────────

    @staticmethod
    def extract_torso_crop(
        frame: np.ndarray,
        bbox: np.ndarray,
    ) -> np.ndarray | None:
        """Same as :meth:`TeamClassifier.extract_torso_crop`."""
        return TeamClassifier.extract_torso_crop(frame, bbox)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_team_classifier(
    backend: str = "siglip",
    n_teams: int = 2,
    device: str = "cpu",
    **kwargs: Any,
) -> TeamClassifier | SiglipTeamClassifier:
    """Return a team classifier for the requested *backend*.

    Parameters
    ----------
    backend:
        ``"siglip"`` (default) — uses SIGLIP embeddings.
        ``"hue"`` — uses the lighter HSV hue K-means classifier.
    n_teams:
        Number of team clusters.
    device:
        PyTorch device for SIGLIP (ignored for ``"hue"`` backend).
    **kwargs:
        Extra keyword arguments forwarded to the classifier constructor.

    Returns
    -------
    TeamClassifier | SiglipTeamClassifier
        Ready-to-use classifier instance.

    Notes
    -----
    When ``backend="siglip"`` and *transformers* / *scikit-learn* are not
    installed the function falls back to the HSV classifier and logs a
    warning so the pipeline keeps running.
    """
    if backend == "hue":
        return TeamClassifier(n_teams=n_teams, **kwargs)

    # Try to confirm that the required dependencies are available
    try:
        import transformers  # noqa: F401
        from sklearn.cluster import KMeans  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        logger.warning(
            "SIGLIP team classifier requested but dependencies are missing (%s). "
            "Falling back to HSV hue classifier.  "
            "Install with: pip install transformers scikit-learn torch",
            exc,
        )
        return TeamClassifier(n_teams=n_teams)

    return SiglipTeamClassifier(n_teams=n_teams, device=device, **kwargs)
