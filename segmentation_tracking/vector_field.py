r"""
vector_field.py
---------------
Player velocity vector-field construction and attractor estimation.

Idea
~~~~
Players tend to move *towards* the ball.  The velocity vectors of all on-pitch
players therefore form a vector field whose **attractor** — the point that all
velocity rays collectively converge towards — approximates the ball position.

This module implements:

1. :class:`PlayerVelocityTracker`
   Maintains a sliding-window position history (and derived velocity history)
   for every tracked player using **bounding-box centres**.  On each frame:

   * *velocities* — ``{id: (cx, cy, vx, vy)}`` (displacement-based velocity).
   * *accelerations* via :meth:`get_accelerations` — ``{id: (cx, cy, ax, ay)}``
     (change in velocity, available once ≥3 frames of history exist).

2. :class:`KeypointVelocityTracker`
   Like :class:`PlayerVelocityTracker` but derives position from **pose
   keypoints** (hips + knees by default) rather than bounding-box centres.
   Supports two optional **sparse optical-flow backends** that propagate the
   keypoints between full pose detections:

   * ``"lk"`` — Lucas-Kanade pyramidal optical flow via ``cv2.calcOpticalFlowPyrLK``
     (fast, CPU-only, no extra dependencies).
   * ``"cotracker"`` — CoTracker3 online sliding-window transformer
     (higher accuracy, requires ``pip install cotracker`` and a GPU).

   A *detect_interval* parameter controls how often the underlying pose result
   is used to re-anchor the flow tracker, suppressing drift.

3. :func:`estimate_attractor`
   Given a collection of ``(cx, cy, dx, dy)`` records (velocity **or**
   acceleration), finds the least-squares **nearest point to the bundle of
   directed lines**.  Per-player weights combine:

   * *speed / acceleration magnitude* (always active).
   * Optional **distance weighting**: Gaussian decay from an anchor point
     (ball position or previous attractor) so nearby players dominate.
   * Optional **directional weighting**: cosine of the angle between the
     player's direction and the toward-anchor direction — players actively
     moving toward the centre of action contribute most.

4. :class:`AttractorSmoother`
   Wraps a lightweight constant-velocity 2-D Kalman filter to smooth the
   frame-to-frame attractor position estimate.  When no valid raw estimate is
   available (too few moving players), the smoother holds the last position
   with a linearly decaying confidence score rather than disappearing
   immediately.  After ``max_stale_frames`` consecutive frames without a valid
   raw estimate the smoother resets and returns *None* (marker disappears).

5. :class:`AttractorEstimate`
   Lightweight dataclass returned by :func:`estimate_attractor` and
   :meth:`AttractorSmoother.update`.

Algorithm — nearest point to a weighted set of lines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For player *i* with current position :math:`\mathbf{p}_i = (cx_i, cy_i)` and
unit direction :math:`\hat{d}_i = (dx_i, dy_i) / \|d_i\|`:

.. math::

    \\mathbf{X}^* = \\left(\\sum_i w_i (I - \\hat{d}_i \\hat{d}_i^T)\\right)^{-1}
                    \\sum_i w_i (I - \\hat{d}_i \\hat{d}_i^T) \\mathbf{p}_i

where :math:`w_i` is the combined per-player weight (see :func:`estimate_attractor`).

Attractor smoothing — constant-velocity 2-D Kalman filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
State: :math:`[x, y, \dot{x}, \dot{y}]^T`.
Process noise std ``q`` (px/frame²); measurement noise std ``r`` (px).
When a valid raw estimate exists it is used as the measurement.  When no
estimate is available the filter predicts forward without a measurement update,
and the held position's confidence decays linearly with stale frame count.

Public API
~~~~~~~~~~
``PlayerVelocityTracker``
    ``update(player_tracks) → dict[int, tuple[float,float,float,float]]``
    ``get_accelerations() → dict[int, tuple[float,float,float,float]]``
    ``reset()``

``KeypointVelocityTracker``
    ``update(player_tracks, frame) → dict[int, tuple[float,float,float,float]]``
    ``get_accelerations() → dict[int, tuple[float,float,float,float]]``
    ``reset()``

``estimate_attractor(vectors, min_magnitude, min_players, frame_shape,``
                    ``anchor_point, distance_sigma, directional_weight)``
    ``→ AttractorEstimate | None``

``AttractorSmoother``
    ``update(raw: AttractorEstimate | None) → AttractorEstimate | None``
    ``reset()``

``AttractorEstimate``
    ``.point``         – ``(x, y)`` estimated attractor position in pixels
    ``.confidence``    – 0–1 reliability score
    ``.n_players``     – number of players used in the raw estimate
    ``.mean_speed``    – mean direction-vector magnitude of contributing players
    ``.is_held``       – *True* when smoothed from a stale (no new raw) position
    ``.stale_frames``  – consecutive frames without a valid raw estimate
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO keypoint indices used for direction estimation
# ---------------------------------------------------------------------------
# Hips (11, 12) and knees (13, 14) are the most reliable indicators of a
# player's running direction: they sit at the body's centre of mass and are
# rigidly coupled to the locomotion axis.  Shoulders and arms are excluded
# because they rotate and swing independently of the running direction.
_COCO_DIRECTION_KP: tuple[int, ...] = (11, 12, 13, 14)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AttractorEstimate:
    """Result of :func:`estimate_attractor` or :meth:`AttractorSmoother.update`.

    Attributes
    ----------
    point:
        ``(x, y)`` pixel coordinates of the estimated attractor.
    confidence:
        Reliability score in [0, 1].  Increases with the number of
        contributing players and their speed/acceleration.  Decreases when
        all direction vectors are nearly parallel (ambiguous intersection),
        and decays linearly while the estimate is held (``is_held=True``).
    n_players:
        Number of players whose direction vector contributed to the raw
        estimate.  Zero when ``is_held=True`` and the smoother is predicting.
    mean_speed:
        Mean direction-vector magnitude (px/frame for velocity, px/frame² for
        acceleration) of the contributing players.
    is_held:
        *True* when the attractor position comes from the Kalman smoother's
        prediction rather than a fresh raw estimate.  The marker is drawn with
        a lighter style to indicate reduced certainty.
    stale_frames:
        Number of consecutive frames during which no valid raw estimate was
        available.  Used by the visualiser to fade the marker.
    """

    point: tuple[float, float]
    confidence: float
    n_players: int
    mean_speed: float
    is_held: bool = False
    stale_frames: int = 0


# ---------------------------------------------------------------------------
# PlayerVelocityTracker
# ---------------------------------------------------------------------------

class PlayerVelocityTracker:
    """Maintain per-player bounding-box–centre history and derive velocities
    and accelerations.

    Parameters
    ----------
    history_len:
        Length of the sliding position window (frames).  Velocity is
        estimated as the displacement from the oldest to the newest entry
        divided by the number of steps.  Defaults to 5.
    """

    def __init__(self, history_len: int = 5) -> None:
        if history_len < 1:
            raise ValueError("history_len must be at least 1")
        self._history_len = history_len
        # {player_id: deque of (cx, cy)}
        self._history: dict[int, deque[tuple[float, float]]] = {}
        # {player_id: deque of (vx, vy)} — velocity samples for acceleration
        self._vel_history: dict[int, deque[tuple[float, float]]] = {}
        # Most recently computed velocities (needed by get_accelerations)
        self._last_velocities: dict[int, tuple[float, float, float, float]] = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all per-player position and velocity histories."""
        self._history.clear()
        self._vel_history.clear()
        self._last_velocities.clear()

    def update(
        self,
        player_tracks: Sequence,
    ) -> dict[int, tuple[float, float, float, float]]:
        """Register current player positions and return velocity estimates.

        Also updates the internal velocity history used by
        :meth:`get_accelerations`.

        Parameters
        ----------
        player_tracks:
            Sequence of :class:`~segmentation_tracking.association.PlayerTrack`
            objects for the current frame.  Each track must have an ``id``
            attribute and a ``bbox`` attribute (``[x1, y1, x2, y2]``).

        Returns
        -------
        dict[int, tuple[float, float, float, float]]
            ``{player_id: (cx, cy, vx, vy)}`` for all players that have at
            least two position observations.  Players seen for the first time
            have their position recorded but are *not* included in the result.
        """
        # Remove IDs no longer in this frame
        current_ids = {pt.id for pt in player_tracks}
        stale = [pid for pid in self._history if pid not in current_ids]
        for pid in stale:
            del self._history[pid]
            self._vel_history.pop(pid, None)

        velocities: dict[int, tuple[float, float, float, float]] = {}

        for pt in player_tracks:
            bbox = pt.bbox
            cx = float((bbox[0] + bbox[2]) / 2)
            cy = float((bbox[1] + bbox[3]) / 2)

            if pt.id not in self._history:
                self._history[pt.id] = deque(maxlen=self._history_len)
                self._vel_history[pt.id] = deque(maxlen=self._history_len)

            buf = self._history[pt.id]
            buf.append((cx, cy))

            if len(buf) >= 2:
                # Velocity = displacement from oldest to newest / span
                oldest_cx, oldest_cy = buf[0]
                span = len(buf) - 1
                vx = (cx - oldest_cx) / span
                vy = (cy - oldest_cy) / span
                velocities[pt.id] = (cx, cy, vx, vy)
                # Record velocity sample for acceleration tracking
                self._vel_history[pt.id].append((vx, vy))

        self._last_velocities = velocities
        return velocities

    def get_accelerations(self) -> dict[int, tuple[float, float, float, float]]:
        """Return per-player acceleration estimates based on velocity history.

        Must be called *after* :meth:`update` has been called at least twice
        for a player (i.e. the velocity history has ≥2 entries).

        Returns
        -------
        dict[int, tuple[float, float, float, float]]
            ``{player_id: (cx, cy, ax, ay)}`` where ``(ax, ay)`` is the
            estimated acceleration in px/frame².  Only players with at least
            two velocity samples (≥3 position frames) are included.
        """
        accelerations: dict[int, tuple[float, float, float, float]] = {}
        for pid, vel_buf in self._vel_history.items():
            if len(vel_buf) < 2:
                continue
            if pid not in self._last_velocities:
                continue
            cx, cy, _vx, _vy = self._last_velocities[pid]
            oldest_vx, oldest_vy = vel_buf[0]
            newest_vx, newest_vy = vel_buf[-1]
            span = len(vel_buf) - 1
            ax = (newest_vx - oldest_vx) / span
            ay = (newest_vy - oldest_vy) / span
            accelerations[pid] = (cx, cy, ax, ay)
        return accelerations


# ---------------------------------------------------------------------------
# _CoTrackerState — internal CoTracker3 online-mode helper
# ---------------------------------------------------------------------------

class _CoTrackerState:
    """Wraps the CoTracker3 online predictor for streaming per-frame tracking.

    Not part of the public API.  Instantiated lazily by
    :class:`KeypointVelocityTracker` when ``flow_backend="cotracker"``.

    The online predictor processes frames in windows of ``2 * step`` frames
    (typically 16).  This class buffers incoming frames and fires inference
    every ``step`` new frames, returning the tracked positions for the most
    recent frame.  There is therefore a maximum latency of ``step`` frames
    (≈ 0.3 s at 25 fps) before the first output is available.
    """

    def __init__(self, checkpoint: str | None, device_str: str) -> None:
        import torch
        from cotracker.predictor import CoTrackerOnlinePredictor  # type: ignore[import]

        self._device = torch.device(
            device_str if torch.cuda.is_available() else "cpu"
        )
        self._predictor = (
            CoTrackerOnlinePredictor(checkpoint=checkpoint).to(self._device)
        )
        self._predictor.eval()

        # step / window come from the model; typical values: step=8, window=16
        try:
            self._step: int = int(self._predictor.step)
        except AttributeError:
            self._step = 8

        # Frame buffer: last 2*step BGR frames as (3, H, W) float tensors
        self._frame_buf: list = []
        self._max_buf: int = 3 * self._step + 1

        # Frames buffered since last inference call (for scheduling)
        self._pending: int = 0
        # Whether the first (initialisation) call has been made
        self._initialized: bool = False
        # (1, N, 3) float32 query tensor: [t_in_chunk, x, y]
        self._queries = None
        # Latest tracked positions: {flat_point_index → (x, y)}
        self.positions: dict[int, tuple[float, float]] = {}

    # ── Point-query management ────────────────────────────────────────────────

    def set_queries(
        self,
        player_tracks: Sequence,
        direction_kp_indices: tuple[int, ...],
        min_kp_score: float,
    ) -> list[tuple[int, int]]:
        """Register new query points from detected keypoints.

        Resets the predictor state so the next call will be a first-step.

        Returns
        -------
        list[tuple[int, int]]
            Point map: each entry is ``(player_id, kp_local_idx)`` for the
            corresponding flat point index in :attr:`positions`.
        """
        import torch

        queries: list[list[float]] = []
        point_map: list[tuple[int, int]] = []

        # Query at t=0 (the current frame) within the upcoming window
        for pt in player_tracks:
            if pt.keypoints is None:
                continue
            scores = getattr(pt, "keypoint_scores", None)
            for local_idx, kp_idx in enumerate(direction_kp_indices):
                if kp_idx >= len(pt.keypoints):
                    continue
                x = float(pt.keypoints[kp_idx, 0])
                y = float(pt.keypoints[kp_idx, 1])
                if x <= 0.0 and y <= 0.0:
                    continue
                score = (
                    float(scores[kp_idx])
                    if scores is not None and kp_idx < len(scores)
                    else 1.0
                )
                if score < min_kp_score:
                    continue
                queries.append([0.0, x, y])
                point_map.append((pt.id, local_idx))

        if queries:
            self._queries = torch.tensor(
                [queries], dtype=torch.float32, device=self._device
            )
        else:
            self._queries = None

        # Reset so the next call is treated as initialisation
        self._initialized = False
        self._pending = 0
        self.positions = {}
        return point_map

    # ── Per-frame stepping ────────────────────────────────────────────────────

    def add_frame(self, frame: np.ndarray) -> None:
        """Convert *frame* (BGR uint8) and append to the internal buffer."""
        import torch

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float()  # (3, H, W)
        self._frame_buf.append(t)
        if len(self._frame_buf) > self._max_buf:
            self._frame_buf = self._frame_buf[-self._max_buf:]
        self._pending += 1

    def maybe_run(self) -> bool:
        """Run inference if enough frames are pending.

        Returns *True* when inference was executed and :attr:`positions` was
        updated.
        """
        import torch

        if self._queries is None:
            return False

        need = self._step * 2 if not self._initialized else self._step
        if self._pending < need or len(self._frame_buf) < need:
            return False

        # Build video chunk: (1, T, 3, H, W)
        chunk_frames = self._frame_buf[-need:]
        video_chunk = (
            torch.stack(chunk_frames, dim=0).unsqueeze(0).to(self._device)
        )

        with torch.no_grad():
            pred_tracks, _ = self._predictor(
                video_chunk,
                is_first_step=not self._initialized,
                queries=self._queries if not self._initialized else None,
            )

        # pred_tracks: (1, T, N, 2) — take positions at the last frame
        last_pos = pred_tracks[0, -1].cpu().numpy()  # (N, 2)
        self.positions = {
            i: (float(last_pos[i, 0]), float(last_pos[i, 1]))
            for i in range(len(last_pos))
        }
        self._initialized = True
        self._pending = 0
        return True


# ---------------------------------------------------------------------------
# KeypointVelocityTracker
# ---------------------------------------------------------------------------

class KeypointVelocityTracker:
    """Derive per-player velocity from pose keypoints + sparse optical flow.

    Compared to :class:`PlayerVelocityTracker` (which uses bounding-box
    centres), this tracker uses body keypoints — specifically **hips and
    knees** — which are directly coupled to the player's locomotion axis.
    This eliminates artefacts caused by:

    * Camera panning (all bboxes translate uniformly).
    * Players pivoting in place (bbox barely moves, hips clearly rotate).
    * Players running toward the camera (bbox shrinks, keypoints shift).

    Between full pose detections the tracker optionally propagates keypoints
    via a sparse optical-flow backend to obtain smooth, low-latency velocity
    estimates:

    * ``"lk"`` — Lucas-Kanade pyramidal flow (``cv2.calcOpticalFlowPyrLK``).
      Fast CPU method, no extra dependencies.  Failed LK points fall back to
      the last known position so the tracker degrades gracefully.
    * ``"cotracker"`` — CoTracker3 online sliding-window transformer.
      Requires ``pip install cotracker`` and a CUDA-capable GPU.  Handles
      occlusions and large displacements better than LK.

    When ``detect_interval=1`` (the default) the flow backend is never used —
    fresh keypoints from the pose estimator are read every frame.  This is
    equivalent to :class:`PlayerVelocityTracker` but using keypoint centroids.

    Parameters
    ----------
    history_len:
        Length of the sliding position window (frames).  Velocity is
        estimated as the displacement from the oldest to the newest entry
        divided by the number of steps.  Defaults to 5.
    flow_backend:
        ``"lk"`` (Lucas-Kanade, default) or ``"cotracker"`` (CoTracker3).
        Ignored when ``detect_interval=1``.
    detect_interval:
        Re-read keypoints from the pose estimator every *N* frames and
        re-anchor the flow tracker.  Between re-detections the optical-flow
        backend propagates the last detected keypoints.  Setting this to 1
        (the default) disables the flow backend entirely.
    direction_kp_indices:
        COCO keypoint indices used to compute each player's direction
        centroid.  Defaults to hips (11, 12) and knees (13, 14).
    min_kp_score:
        Minimum keypoint confidence score.  Keypoints below this threshold
        (partially occluded / out-of-frame) are excluded from the centroid
        and from the flow tracker.  Defaults to 0.3.
    device:
        Torch device string used by the CoTracker3 backend.  Defaults to
        ``"cuda"``.
    cotracker_checkpoint:
        Optional local path to a CoTracker3 ``.pth`` checkpoint.  If *None*
        the default pretrained weights are downloaded automatically.
    """

    def __init__(
        self,
        history_len: int = 5,
        flow_backend: str = "lk",
        detect_interval: int = 1,
        direction_kp_indices: tuple[int, ...] = _COCO_DIRECTION_KP,
        min_kp_score: float = 0.3,
        device: str = "cuda",
        cotracker_checkpoint: str | None = None,
    ) -> None:
        if history_len < 1:
            raise ValueError("history_len must be at least 1")
        if flow_backend not in ("lk", "cotracker"):
            raise ValueError("flow_backend must be 'lk' or 'cotracker'")

        self._history_len = history_len
        self._flow_backend = flow_backend
        self._detect_interval = max(1, int(detect_interval))
        self._direction_kp = tuple(direction_kp_indices)
        self._min_kp_score = float(min_kp_score)
        self._device = device
        self._cotracker_checkpoint = cotracker_checkpoint

        # Sliding position / velocity history (same structure as PlayerVelocityTracker)
        self._history: dict[int, deque[tuple[float, float]]] = {}
        self._vel_history: dict[int, deque[tuple[float, float]]] = {}
        self._last_velocities: dict[int, tuple[float, float, float, float]] = {}

        # Internal frame counter used to determine detect vs flow frames
        self._frame_count: int = 0

        # ── LK state ──────────────────────────────────────────────────────────
        # Previous grayscale frame
        self._prev_gray: np.ndarray | None = None
        # Last set of LK points per player: {pid → (n_pts, 2) float32}
        self._prev_lk_pts: dict[int, np.ndarray] = {}
        # Last known keypoint centroid per player (fallback for flow failures)
        self._last_centroid: dict[int, tuple[float, float]] = {}

        # ── CoTracker3 state ───────────────────────────────────────────────────
        self._ct_state: _CoTrackerState | None = None
        # Mapping from flat CoTracker point index → (player_id, kp_local_idx)
        self._ct_point_map: list[tuple[int, int]] = []

    # ── Public ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all per-player history and flow state."""
        self._history.clear()
        self._vel_history.clear()
        self._last_velocities.clear()
        self._frame_count = 0
        self._prev_gray = None
        self._prev_lk_pts.clear()
        self._last_centroid.clear()
        self._ct_state = None
        self._ct_point_map.clear()

    def update(
        self,
        player_tracks: Sequence,
        frame: np.ndarray,
    ) -> dict[int, tuple[float, float, float, float]]:
        """Update the tracker and return velocity estimates.

        On *detect frames* (every ``detect_interval`` frames) the method reads
        keypoints directly from *player_tracks* and re-anchors the flow
        tracker.  On *flow frames* it runs LK or CoTracker3 to propagate the
        keypoints from the previous frame.

        Parameters
        ----------
        player_tracks:
            Sequence of :class:`~segmentation_tracking.association.PlayerTrack`
            objects for the current frame.  Each track must have ``id``,
            ``bbox``, and — on detect frames — ``keypoints`` (shape (17, 2))
            and optionally ``keypoint_scores`` (shape (17,)).
        frame:
            Current video frame (BGR, uint8), used by the optical-flow
            backend.

        Returns
        -------
        dict[int, tuple[float, float, float, float]]
            ``{player_id: (cx, cy, vx, vy)}`` for players with ≥2 position
            observations.  Players seen for the first time are not included.
        """
        is_detect = (self._frame_count % self._detect_interval == 0)
        self._frame_count += 1

        # Prune stale player IDs
        current_ids = {pt.id for pt in player_tracks}
        for pid in list(self._history):
            if pid not in current_ids:
                del self._history[pid]
                self._vel_history.pop(pid, None)
                self._prev_lk_pts.pop(pid, None)
                self._last_centroid.pop(pid, None)

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Determine centroid map for this frame
        if is_detect or self._detect_interval == 1:
            centroid_map = self._centroids_from_detections(player_tracks)
            if self._detect_interval > 1:
                # Re-anchor flow tracker
                if self._flow_backend == "lk":
                    self._lk_init(player_tracks, curr_gray)
                else:
                    self._ct_init(player_tracks, frame)
        elif self._flow_backend == "lk":
            centroid_map = self._lk_step(player_tracks, curr_gray)
        else:
            centroid_map = self._ct_step(player_tracks, frame)

        self._prev_gray = curr_gray

        # Update position / velocity history
        velocities: dict[int, tuple[float, float, float, float]] = {}

        for pid, (cx, cy) in centroid_map.items():
            if pid not in self._history:
                self._history[pid] = deque(maxlen=self._history_len)
                self._vel_history[pid] = deque(maxlen=self._history_len)

            buf = self._history[pid]
            buf.append((cx, cy))

            if len(buf) >= 2:
                oldest_cx, oldest_cy = buf[0]
                span = len(buf) - 1
                vx = (cx - oldest_cx) / span
                vy = (cy - oldest_cy) / span
                velocities[pid] = (cx, cy, vx, vy)
                self._vel_history[pid].append((vx, vy))

        self._last_velocities = velocities
        return velocities

    def get_accelerations(self) -> dict[int, tuple[float, float, float, float]]:
        """Return per-player acceleration estimates from velocity history.

        Must be called *after* :meth:`update` has been called at least twice
        for a player.

        Returns
        -------
        dict[int, tuple[float, float, float, float]]
            ``{player_id: (cx, cy, ax, ay)}`` — same format as
            :meth:`PlayerVelocityTracker.get_accelerations`.
        """
        accelerations: dict[int, tuple[float, float, float, float]] = {}
        for pid, vel_buf in self._vel_history.items():
            if len(vel_buf) < 2:
                continue
            if pid not in self._last_velocities:
                continue
            cx, cy, _vx, _vy = self._last_velocities[pid]
            oldest_vx, oldest_vy = vel_buf[0]
            newest_vx, newest_vy = vel_buf[-1]
            span = len(vel_buf) - 1
            ax = (newest_vx - oldest_vx) / span
            ay = (newest_vy - oldest_vy) / span
            accelerations[pid] = (cx, cy, ax, ay)
        return accelerations

    # ── Private helpers ────────────────────────────────────────────────────────

    def _kp_centroid(
        self,
        keypoints: np.ndarray | None,
        kp_scores: np.ndarray | None,
        bbox: np.ndarray,
    ) -> tuple[float, float]:
        """Weighted centroid of valid direction keypoints, or bbox centre."""
        if keypoints is not None and len(keypoints) > max(self._direction_kp):
            pts: list[tuple[float, float]] = []
            weights: list[float] = []
            for idx in self._direction_kp:
                x, y = float(keypoints[idx, 0]), float(keypoints[idx, 1])
                if x <= 0.0 and y <= 0.0:
                    continue
                score = (
                    float(kp_scores[idx])
                    if kp_scores is not None and idx < len(kp_scores)
                    else 1.0
                )
                if score < self._min_kp_score:
                    continue
                pts.append((x, y))
                weights.append(score)
            if pts:
                total_w = sum(weights)
                cx = sum(p[0] * w for p, w in zip(pts, weights)) / total_w
                cy = sum(p[1] * w for p, w in zip(pts, weights)) / total_w
                return float(cx), float(cy)

        # Fallback: bounding-box centre
        return float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2)

    def _centroids_from_detections(
        self,
        player_tracks: Sequence,
    ) -> dict[int, tuple[float, float]]:
        """Extract direction centroids from detected pose keypoints."""
        result: dict[int, tuple[float, float]] = {}
        for pt in player_tracks:
            cx, cy = self._kp_centroid(
                pt.keypoints,
                getattr(pt, "keypoint_scores", None),
                pt.bbox,
            )
            result[pt.id] = (cx, cy)
            self._last_centroid[pt.id] = (cx, cy)
        return result

    # ── LK helpers ────────────────────────────────────────────────────────────

    def _lk_init(self, player_tracks: Sequence, gray: np.ndarray) -> None:
        """Initialise per-player LK point sets from detected keypoints."""
        self._prev_lk_pts.clear()
        for pt in player_tracks:
            if pt.keypoints is None:
                continue
            scores = getattr(pt, "keypoint_scores", None)
            pts: list[tuple[float, float]] = []
            for idx in self._direction_kp:
                if idx >= len(pt.keypoints):
                    continue
                x, y = float(pt.keypoints[idx, 0]), float(pt.keypoints[idx, 1])
                if x <= 0.0 and y <= 0.0:
                    continue
                score = (
                    float(scores[idx])
                    if scores is not None and idx < len(scores)
                    else 1.0
                )
                if score < self._min_kp_score:
                    continue
                pts.append((x, y))
            if pts:
                self._prev_lk_pts[pt.id] = np.array(pts, dtype=np.float32)

    _LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def _lk_step(
        self,
        player_tracks: Sequence,
        curr_gray: np.ndarray,
    ) -> dict[int, tuple[float, float]]:
        """Run one LK step; return centroid map for all tracked players."""
        if self._prev_gray is None:
            # No previous frame yet — fall back to detections
            return self._centroids_from_detections(player_tracks)

        result: dict[int, tuple[float, float]] = {}

        for pt in player_tracks:
            pid = pt.id

            if pid not in self._prev_lk_pts or len(self._prev_lk_pts[pid]) == 0:
                # Newly appeared player: try detected keypoints
                cx, cy = self._kp_centroid(
                    pt.keypoints, getattr(pt, "keypoint_scores", None), pt.bbox
                )
                result[pid] = (cx, cy)
                self._last_centroid[pid] = (cx, cy)
                # Seed LK for this player on next flow frame
                if pt.keypoints is not None:
                    scores = getattr(pt, "keypoint_scores", None)
                    seed_pts: list[tuple[float, float]] = []
                    for idx in self._direction_kp:
                        if idx >= len(pt.keypoints):
                            continue
                        x, y = float(pt.keypoints[idx, 0]), float(pt.keypoints[idx, 1])
                        if x <= 0.0 and y <= 0.0:
                            continue
                        s = (
                            float(scores[idx])
                            if scores is not None and idx < len(scores)
                            else 1.0
                        )
                        if s < self._min_kp_score:
                            continue
                        seed_pts.append((x, y))
                    if seed_pts:
                        self._prev_lk_pts[pid] = np.array(seed_pts, dtype=np.float32)
                continue

            prev_pts = self._prev_lk_pts[pid].reshape(-1, 1, 2)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, curr_gray, prev_pts, None, **self._LK_PARAMS
            )

            if next_pts is None or status is None:
                if pid in self._last_centroid:
                    result[pid] = self._last_centroid[pid]
                continue

            good_mask = status.ravel().astype(bool)
            if not good_mask.any():
                if pid in self._last_centroid:
                    result[pid] = self._last_centroid[pid]
                continue

            good_pts = next_pts.reshape(-1, 2)[good_mask]
            cx = float(good_pts[:, 0].mean())
            cy = float(good_pts[:, 1].mean())
            result[pid] = (cx, cy)
            self._last_centroid[pid] = (cx, cy)
            # Propagate only successfully tracked points
            self._prev_lk_pts[pid] = good_pts

        return result

    # ── CoTracker3 helpers ────────────────────────────────────────────────────

    def _ensure_ct_state(self) -> bool:
        """Lazily initialise the CoTracker3 state; return *True* on success."""
        if self._ct_state is not None:
            return True
        try:
            self._ct_state = _CoTrackerState(
                self._cotracker_checkpoint, self._device
            )
            logger.info("CoTracker3 backend initialised (device=%s)", self._device)
            return True
        except Exception as exc:
            logger.warning(
                "CoTracker3 unavailable (%s); falling back to LK backend.", exc
            )
            self._flow_backend = "lk"
            return False

    def _ct_init(self, player_tracks: Sequence, frame: np.ndarray) -> None:
        """Re-initialise CoTracker3 queries from detected keypoints."""
        if not self._ensure_ct_state():
            self._lk_init(player_tracks, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            return
        assert self._ct_state is not None
        self._ct_point_map = self._ct_state.set_queries(
            player_tracks, self._direction_kp, self._min_kp_score
        )
        self._ct_state.add_frame(frame)
        self._ct_state.maybe_run()

    def _ct_step(
        self,
        player_tracks: Sequence,
        frame: np.ndarray,
    ) -> dict[int, tuple[float, float]]:
        """Run one CoTracker3 step; return centroid map."""
        if self._ct_state is None or not self._ct_point_map:
            return self._centroids_from_detections(player_tracks)

        self._ct_state.add_frame(frame)
        ran = self._ct_state.maybe_run()

        if not ran and not self._ct_state.positions:
            # Predictor not yet ready — fall back to last known centroids
            result: dict[int, tuple[float, float]] = {}
            for pt in player_tracks:
                if pt.id in self._last_centroid:
                    result[pt.id] = self._last_centroid[pt.id]
                else:
                    result[pt.id] = self._kp_centroid(
                        pt.keypoints, getattr(pt, "keypoint_scores", None), pt.bbox
                    )
            return result

        # Aggregate per-point positions into per-player centroids
        player_pts: dict[int, list[tuple[float, float]]] = {}
        for flat_idx, (pid, _) in enumerate(self._ct_point_map):
            if flat_idx in self._ct_state.positions:
                player_pts.setdefault(pid, []).append(
                    self._ct_state.positions[flat_idx]
                )

        result = {}
        for pid, pts in player_pts.items():
            cx = float(np.mean([p[0] for p in pts]))
            cy = float(np.mean([p[1] for p in pts]))
            result[pid] = (cx, cy)
            self._last_centroid[pid] = (cx, cy)

        # Players absent from CoTracker output: use last centroid or detection
        for pt in player_tracks:
            if pt.id not in result:
                if pt.id in self._last_centroid:
                    result[pt.id] = self._last_centroid[pt.id]
                else:
                    result[pt.id] = self._kp_centroid(
                        pt.keypoints, getattr(pt, "keypoint_scores", None), pt.bbox
                    )

        return result


# ---------------------------------------------------------------------------
# Attractor estimation
# ---------------------------------------------------------------------------

def estimate_attractor(
    vectors: dict[int, tuple[float, float, float, float]],
    min_magnitude: float = 1.5,
    min_players: int = 2,
    frame_shape: tuple[int, int] | None = None,
    # backward-compat alias kept for existing callers
    min_speed: float | None = None,
    # Distance and directional weighting (new)
    anchor_point: tuple[float, float] | None = None,
    distance_sigma: float = 0.0,
    directional_weight: bool = False,
    min_weight_floor: float = 0.05,
) -> AttractorEstimate | None:
    """Estimate the vector-field attractor from player direction rays.

    Works for *both* velocity-mode and acceleration-mode inputs.  In velocity
    mode the input is the output of :meth:`PlayerVelocityTracker.update`; in
    acceleration mode it is the output of
    :meth:`PlayerVelocityTracker.get_accelerations`.  Both share the same
    ``(cx, cy, dx, dy)`` tuple format.

    Each player whose direction magnitude exceeds *min_magnitude* contributes a
    directed ray (origin = player centre, direction = velocity / acceleration
    vector).  The per-player weight combines three terms:

    * **Speed weight**: the direction-vector magnitude (always active).
    * **Distance weight** (optional): Gaussian decay based on the player's
      distance from *anchor_point* (typically the ball position or the
      previous attractor estimate).  Players near the centre of action
      receive a weight ≈ 1; players far away are down-weighted.
    * **Directional weight** (optional): cosine of the angle between the
      player's direction vector and the toward-anchor direction.  Players
      actively moving toward the anchor contribute fully; perpendicular
      players contribute half; players running away receive *min_weight_floor*.

    Parameters
    ----------
    vectors:
        Mapping ``{player_id: (cx, cy, dx, dy)}`` — the last two components
        are the direction vector (velocity or acceleration).
    min_magnitude:
        Minimum direction-vector magnitude for a player to contribute.
        Stationary players or players with negligible direction vectors are
        excluded.  Defaults to 1.5.
    min_players:
        Minimum number of qualifying players required to compute the
        attractor.  Defaults to 2.
    frame_shape:
        Optional ``(height, width)`` used only for clamping the result to the
        visible frame area.  If *None*, no clamping is applied.
    min_speed:
        Alias for *min_magnitude* retained for backward compatibility with
        callers that used the original ``velocities``/``min_speed`` API.
        Use *min_magnitude* for new code.  If both are supplied, *min_speed*
        takes precedence.
    anchor_point:
        Optional ``(x, y)`` pixel coordinate used as the reference for
        distance and directional weighting.  Typically set to the current
        ball position or to the previous smoothed attractor estimate.
        If *None*, both distance and directional weighting are disabled
        regardless of the other flags.
    distance_sigma:
        Standard deviation (pixels) of the Gaussian distance weight.  Set to
        0.0 (default) to disable distance weighting.  A value of 200 px gives
        full weight to players within ≈ 200 px of the anchor and strongly
        suppresses players beyond 400–600 px.
    directional_weight:
        When *True* and *anchor_point* is provided, multiply each player's
        weight by ``max(min_weight_floor, cos θ)`` where ``θ`` is the angle
        between the player's direction and the toward-anchor direction.
        Defaults to *False*.
    min_weight_floor:
        Lower bound on the directional weight factor so that players running
        away from the anchor still contribute a small baseline signal rather
        than being zeroed out entirely.  Defaults to 0.05.

    Returns
    -------
    AttractorEstimate | None
        *None* when insufficient data is available or the linear system is
        numerically degenerate (all direction vectors nearly parallel).
    """
    # Backward-compat: honour the old keyword argument name
    if min_speed is not None:
        min_magnitude = min_speed

    # Collect rays with sufficient magnitude
    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    magnitudes: list[float] = []

    for cx, cy, dx, dy in vectors.values():
        mag = float(np.hypot(dx, dy))
        if mag < min_magnitude:
            continue
        origins.append(np.array([cx, cy], dtype=np.float64))
        directions.append(np.array([dx / mag, dy / mag], dtype=np.float64))
        magnitudes.append(mag)

    n = len(origins)
    if n < min_players:
        logger.debug(
            "Attractor: only %d / %d qualifying players (need %d)",
            n, len(vectors), min_players,
        )
        return None

    # ── Per-player combined weights ───────────────────────────────────────────
    # w_i = speed_i × distance_weight_i × directional_weight_i
    anchor_arr: np.ndarray | None = (
        np.array(anchor_point, dtype=np.float64)
        if anchor_point is not None
        else None
    )

    final_weights: list[float] = []
    for p, d_hat, mag in zip(origins, directions, magnitudes):
        w = mag

        # Distance weighting — Gaussian decay from anchor
        if anchor_arr is not None and distance_sigma > 0.0:
            d = float(np.linalg.norm(p - anchor_arr))
            w *= float(np.exp(-(d * d) / (2.0 * distance_sigma * distance_sigma)))

        # Directional weighting — favour players moving toward anchor
        if directional_weight and anchor_arr is not None:
            toward = anchor_arr - p
            toward_norm = float(np.linalg.norm(toward))
            if toward_norm > 1e-6:
                toward_hat = toward / toward_norm
                cos_theta = float(np.dot(d_hat, toward_hat))
                w *= max(float(min_weight_floor), cos_theta)

        final_weights.append(w)

    # Least-squares nearest point to a weighted bundle of lines
    # X* = (Σ w_i (I - d̂_i d̂_i^T))^{-1} Σ w_i (I - d̂_i d̂_i^T) p_i
    I2 = np.eye(2, dtype=np.float64)
    A = np.zeros((2, 2), dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)

    for p, d_hat, w in zip(origins, directions, final_weights):
        M = I2 - np.outer(d_hat, d_hat)
        A += w * M
        b += w * M @ p

    # Check condition number — degenerate when all directions are parallel
    cond = np.linalg.cond(A)
    if cond > 1e8:
        logger.debug("Attractor: degenerate system (cond=%.2e), skipping", cond)
        return None

    try:
        x_star = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        logger.debug("Attractor: singular system, skipping")
        return None

    ax, ay = float(x_star[0]), float(x_star[1])

    # Optionally clamp to frame boundaries
    if frame_shape is not None:
        frame_h, frame_w = frame_shape
        ax = float(np.clip(ax, 0, frame_w - 1))
        ay = float(np.clip(ay, 0, frame_h - 1))

    # Confidence:
    # - Scales up to 1 with more players (saturates at ~8)
    # - Reduced when system is ill-conditioned (all directions similar)
    n_conf = min(1.0, n / 8.0)
    cond_conf = max(0.0, 1.0 - np.log10(max(cond, 1.0)) / 8.0)
    confidence = float(n_conf * cond_conf)

    mean_speed = float(np.mean(magnitudes))

    logger.debug(
        "Attractor: (%.1f, %.1f) conf=%.2f n=%d mean_mag=%.1f",
        ax, ay, confidence, n, mean_speed,
    )

    return AttractorEstimate(
        point=(ax, ay),
        confidence=confidence,
        n_players=n,
        mean_speed=mean_speed,
    )


# ---------------------------------------------------------------------------
# Attractor smoother — constant-velocity 2-D Kalman filter
# ---------------------------------------------------------------------------

class AttractorSmoother:
    """Smooth the frame-to-frame attractor position with a Kalman filter.

    Uses a constant-velocity 2-D linear Kalman filter to suppress the rapid
    frame-to-frame position jumps that occur when only a few players contribute
    to the raw attractor estimate.

    When no valid raw estimate is available (e.g. because too few players are
    moving), the smoother *holds* the last position by running the Kalman
    predict step without a measurement update.  The returned estimate's
    ``is_held`` flag is set to *True* and ``confidence`` decays linearly with
    ``stale_frames``.  After ``max_stale_frames`` consecutive frames without a
    valid raw estimate the smoother resets and returns *None* until a new raw
    estimate becomes available.

    Parameters
    ----------
    process_noise_std:
        Standard deviation of the process noise (px/frame²) — controls how
        much the predicted trajectory can deviate from a straight line.
        Larger values → smoother but slower to follow sudden changes.
        Defaults to 8.0.
    measure_noise_std:
        Standard deviation of the measurement noise (px) applied to the raw
        attractor position.  Larger values → smoother, less reactive.
        Defaults to 25.0.
    max_stale_frames:
        Number of consecutive frames with no valid raw estimate after which
        the smoother resets and returns *None*.  Defaults to 30.
    """

    def __init__(
        self,
        process_noise_std: float = 8.0,
        measure_noise_std: float = 25.0,
        max_stale_frames: int = 30,
    ) -> None:
        if max_stale_frames < 1:
            raise ValueError("max_stale_frames must be at least 1")
        self._q = float(process_noise_std)
        self._r = float(measure_noise_std)
        self._max_stale = int(max_stale_frames)

        # Kalman matrices (constant-velocity model)
        dt = 1.0
        self._F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        self._Q = (self._q ** 2) * np.eye(4, dtype=np.float64)
        self._R = (self._r ** 2) * np.eye(2, dtype=np.float64)
        self._I4 = np.eye(4, dtype=np.float64)

        # Kalman state
        self._x: np.ndarray | None = None   # shape (4,)
        self._P: np.ndarray | None = None   # shape (4, 4)

        # Stale-frame counter and cached raw metadata
        self._stale_frames: int = 0
        self._last_raw: AttractorEstimate | None = None

    # ── Public ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset the Kalman filter state (e.g. on scene cut)."""
        self._x = None
        self._P = None
        self._stale_frames = 0
        self._last_raw = None

    def update(
        self,
        raw: AttractorEstimate | None,
        frame_shape: tuple[int, int] | None = None,
    ) -> AttractorEstimate | None:
        """Apply one Kalman filter step and return the smoothed estimate.

        Parameters
        ----------
        raw:
            Raw attractor estimate from :func:`estimate_attractor`, or *None*
            when no valid raw estimate is available this frame.
        frame_shape:
            Optional ``(height, width)`` for clamping the smoothed position.

        Returns
        -------
        AttractorEstimate | None
            Smoothed estimate.  *None* when the smoother is uninitialised
            (first frame with no raw estimate) or after exceeding
            ``max_stale_frames``.
        """
        if raw is not None:
            # ── New measurement available ─────────────────────────────────
            z = np.array([raw.point[0], raw.point[1]], dtype=np.float64)

            if self._x is None:
                # First initialisation
                self._x = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float64)
                self._P = np.diag([
                    self._r ** 2,
                    self._r ** 2,
                    (self._q * 5) ** 2,
                    (self._q * 5) ** 2,
                ]).astype(np.float64)
            else:
                # Predict
                self._x = self._F @ self._x
                self._P = self._F @ self._P @ self._F.T + self._Q

                # Update (standard Kalman equations)
                y = z - self._H @ self._x
                S = self._H @ self._P @ self._H.T + self._R
                K = self._P @ self._H.T @ np.linalg.inv(S)
                self._x = self._x + K @ y
                self._P = (self._I4 - K @ self._H) @ self._P

            self._stale_frames = 0
            self._last_raw = raw

        else:
            # ── No measurement — predict-only ────────────────────────────
            if self._x is None:
                # Never initialised → nothing to return
                return None

            self._stale_frames += 1
            if self._stale_frames > self._max_stale:
                logger.debug(
                    "AttractorSmoother: stale for %d frames, resetting",
                    self._stale_frames,
                )
                self.reset()
                return None

            # Predict forward
            self._x = self._F @ self._x
            self._P = self._F @ self._P @ self._F.T + self._Q

        # ── Build smoothed estimate ───────────────────────────────────────
        sx, sy = float(self._x[0]), float(self._x[1])

        if frame_shape is not None:
            fh, fw = frame_shape
            sx = float(np.clip(sx, 0, fw - 1))
            sy = float(np.clip(sy, 0, fh - 1))

        # Confidence: use raw confidence when available; decay linearly when held
        if raw is not None:
            base_conf = raw.confidence
        else:
            base_conf = (self._last_raw.confidence if self._last_raw is not None else 0.5)

        stale_factor = max(0.0, 1.0 - self._stale_frames / self._max_stale)
        confidence = float(base_conf * stale_factor) if self._stale_frames > 0 else base_conf

        return AttractorEstimate(
            point=(sx, sy),
            confidence=confidence,
            n_players=raw.n_players if raw is not None else 0,
            mean_speed=raw.mean_speed if raw is not None else 0.0,
            is_held=(self._stale_frames > 0),
            stale_frames=self._stale_frames,
        )
