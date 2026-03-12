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
   for every tracked player.  On each frame it returns:

   * *velocities* — ``{id: (cx, cy, vx, vy)}`` (displacement-based velocity).
   * *accelerations* via :meth:`get_accelerations` — ``{id: (cx, cy, ax, ay)}``
     (change in velocity, available once ≥3 frames of history exist).

2. :func:`estimate_attractor`
   Given a collection of ``(cx, cy, dx, dy)`` records (velocity **or**
   acceleration), finds the least-squares **nearest point to the bundle of
   directed lines**.  Each ray is weighted by the magnitude of the direction
   vector so that faster / harder-accelerating players contribute more.

3. :class:`AttractorSmoother`
   Wraps a lightweight constant-velocity 2-D Kalman filter to smooth the
   frame-to-frame attractor position estimate.  When no valid raw estimate is
   available (too few moving players), the smoother holds the last position
   with a linearly decaying confidence score rather than disappearing
   immediately.  After ``max_stale_frames`` consecutive frames without a valid
   raw estimate the smoother resets and returns *None* (marker disappears).

4. :class:`AttractorEstimate`
   Lightweight dataclass returned by :func:`estimate_attractor` and
   :meth:`AttractorSmoother.update`.

Algorithm — nearest point to a weighted set of lines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For player *i* with current position :math:`\mathbf{p}_i = (cx_i, cy_i)` and
unit direction :math:`\hat{d}_i = (dx_i, dy_i) / \|d_i\|`:

.. math::

    \\mathbf{X}^* = \\left(\\sum_i w_i (I - \\hat{d}_i \\hat{d}_i^T)\\right)^{-1}
                    \\sum_i w_i (I - \\hat{d}_i \\hat{d}_i^T) \\mathbf{p}_i

where :math:`w_i = \\|d_i\\|` (speed or acceleration magnitude) is the weight.

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

``estimate_attractor(vectors, min_magnitude, min_players, frame_shape)``
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

import numpy as np

logger = logging.getLogger(__name__)

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
# Attractor estimation
# ---------------------------------------------------------------------------

def estimate_attractor(
    vectors: dict[int, tuple[float, float, float, float]],
    min_magnitude: float = 1.5,
    min_players: int = 2,
    frame_shape: tuple[int, int] | None = None,
    # backward-compat alias kept for existing callers
    min_speed: float | None = None,
) -> AttractorEstimate | None:
    """Estimate the vector-field attractor from player direction rays.

    Works for *both* velocity-mode and acceleration-mode inputs.  In velocity
    mode the input is the output of :meth:`PlayerVelocityTracker.update`; in
    acceleration mode it is the output of
    :meth:`PlayerVelocityTracker.get_accelerations`.  Both share the same
    ``(cx, cy, dx, dy)`` tuple format.

    Each player whose direction magnitude exceeds *min_magnitude* contributes a
    directed ray (origin = player centre, direction = velocity / acceleration
    vector).  The function returns the least-squares nearest point to all such
    rays, weighted by each player's direction magnitude.

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

    # Least-squares nearest point to a weighted bundle of lines
    # X* = (Σ w_i (I - d̂_i d̂_i^T))^{-1} Σ w_i (I - d̂_i d̂_i^T) p_i
    I2 = np.eye(2, dtype=np.float64)
    A = np.zeros((2, 2), dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)

    for p, d_hat, w in zip(origins, directions, magnitudes):
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
