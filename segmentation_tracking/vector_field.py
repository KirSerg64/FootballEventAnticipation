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
   Maintains a sliding-window position history for every tracked player.
   On each frame it returns a mapping from player ID to the quadruple
   ``(cx, cy, vx, vy)`` representing the player's current centre and
   estimated velocity in pixels-per-frame.

2. :func:`estimate_attractor`
   Given a collection of ``(cx, cy, vx, vy)`` records, finds the
   least-squares **nearest point to the bundle of directed lines** —
   the classical computer-vision formula for ray-bundle intersection.
   Each ray is weighted by the player's speed so that players sprinting
   toward the ball contribute more than slow-moving or stationary ones.

3. :class:`AttractorEstimate`
   Lightweight dataclass returned by :func:`estimate_attractor`.

Algorithm — nearest point to a weighted set of lines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For player *i* with current position :math:`\mathbf{p}_i = (cx_i, cy_i)` and
unit velocity direction :math:`\hat{d}_i = (vx_i, vy_i) / \|v_i\|`:

.. math::

    \\mathbf{X}^* = \\left(\\sum_i w_i (I - \\hat{d}_i \\hat{d}_i^T)\\right)^{-1}
                    \\sum_i w_i (I - \\hat{d}_i \\hat{d}_i^T) \\mathbf{p}_i

where :math:`w_i = \\|v_i\\|` (player speed) is the weight.  This minimises
the sum of squared perpendicular distances from the attractor point to each
player's velocity ray.

The result is *None* when fewer than two players have a meaningful (non-zero)
velocity, or when the linear system is numerically degenerate (all velocities
are parallel → no unique intersection).

Public API
~~~~~~~~~~
``PlayerVelocityTracker``
    ``update(player_tracks) → dict[int, tuple[float,float,float,float]]``
    ``reset()``

``estimate_attractor(velocities, min_speed, min_players) → AttractorEstimate | None``

``AttractorEstimate``
    ``.point``         – ``(x, y)`` estimated attractor position in pixels
    ``.confidence``    – 0–1 reliability score
    ``.n_players``     – number of players used in the estimate
    ``.mean_speed``    – mean speed of contributing players (px/frame)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AttractorEstimate:
    """Result of :func:`estimate_attractor`.

    Attributes
    ----------
    point:
        ``(x, y)`` pixel coordinates of the estimated attractor.
    confidence:
        Reliability score in [0, 1].  Increases with the number of
        contributing players and their speed.  Decreases when all
        velocity vectors are nearly parallel (ambiguous intersection).
    n_players:
        Number of players whose velocity contributed to the estimate.
    mean_speed:
        Mean speed (px/frame) of those players.
    """

    point: tuple[float, float]
    confidence: float
    n_players: int
    mean_speed: float


# ---------------------------------------------------------------------------
# PlayerVelocityTracker
# ---------------------------------------------------------------------------

class PlayerVelocityTracker:
    """Maintain per-player bounding-box–centre history and derive velocities.

    Parameters
    ----------
    history_len:
        Length of the sliding position window (frames).  Velocity is
        estimated as the displacement over this many frames divided by
        the number of steps, i.e. an average speed from up to
        ``history_len`` recent observations.  Defaults to 5.
    """

    def __init__(self, history_len: int = 5) -> None:
        if history_len < 1:
            raise ValueError("history_len must be at least 1")
        self._history_len = history_len
        # {player_id: deque of (cx, cy)}
        self._history: dict[int, deque[tuple[float, float]]] = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all per-player position histories."""
        self._history.clear()

    def update(
        self,
        player_tracks: Sequence,
    ) -> dict[int, tuple[float, float, float, float]]:
        """Register current player positions and return velocity estimates.

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
            least two position observations and therefore a non-trivial
            velocity estimate.  Players seen for the first time have their
            position recorded but are *not* included in the result.
        """
        # Remove IDs no longer in this frame
        current_ids = {pt.id for pt in player_tracks}
        stale = [pid for pid in self._history if pid not in current_ids]
        for pid in stale:
            del self._history[pid]

        velocities: dict[int, tuple[float, float, float, float]] = {}

        for pt in player_tracks:
            bbox = pt.bbox
            cx = float((bbox[0] + bbox[2]) / 2)
            cy = float((bbox[1] + bbox[3]) / 2)

            if pt.id not in self._history:
                self._history[pt.id] = deque(maxlen=self._history_len)

            buf = self._history[pt.id]
            buf.append((cx, cy))

            if len(buf) >= 2:
                # Velocity = displacement from oldest to newest divided by span
                oldest_cx, oldest_cy = buf[0]
                span = len(buf) - 1
                vx = (cx - oldest_cx) / span
                vy = (cy - oldest_cy) / span
                velocities[pt.id] = (cx, cy, vx, vy)

        return velocities


# ---------------------------------------------------------------------------
# Attractor estimation
# ---------------------------------------------------------------------------

def estimate_attractor(
    velocities: dict[int, tuple[float, float, float, float]],
    min_speed: float = 1.5,
    min_players: int = 2,
    frame_shape: tuple[int, int] | None = None,
) -> AttractorEstimate | None:
    """Estimate the vector-field attractor from player velocity rays.

    Each player whose speed exceeds *min_speed* contributes a directed ray
    (origin = player centre, direction = velocity vector).  The function
    returns the least-squares nearest point to all such rays, weighted by
    each player's speed.

    Parameters
    ----------
    velocities:
        Output of :meth:`PlayerVelocityTracker.update` — a mapping
        ``{player_id: (cx, cy, vx, vy)}``.
    min_speed:
        Minimum speed (px/frame) for a player to contribute to the
        estimate.  Stationary or barely-moving players are excluded
        because their velocity direction is uninformative.
        Defaults to 1.5.
    min_players:
        Minimum number of qualifying players required to compute the
        attractor.  Defaults to 2.
    frame_shape:
        Optional ``(height, width)`` used only for clamping the result
        to the visible frame area.  If *None*, no clamping is applied.

    Returns
    -------
    AttractorEstimate | None
        *None* when insufficient data is available or the linear system
        is numerically degenerate (all velocity directions nearly parallel).
    """
    # Collect rays with sufficient speed
    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    speeds: list[float] = []

    for cx, cy, vx, vy in velocities.values():
        speed = float(np.hypot(vx, vy))
        if speed < min_speed:
            continue
        origins.append(np.array([cx, cy], dtype=np.float64))
        directions.append(np.array([vx / speed, vy / speed], dtype=np.float64))
        speeds.append(speed)

    n = len(origins)
    if n < min_players:
        logger.debug(
            "Attractor: only %d / %d qualifying players (need %d)",
            n, len(velocities), min_players,
        )
        return None

    # Least-squares nearest point to a weighted bundle of lines
    # X* = (Σ w_i (I - d̂_i d̂_i^T))^{-1} Σ w_i (I - d̂_i d̂_i^T) p_i
    I2 = np.eye(2, dtype=np.float64)
    A = np.zeros((2, 2), dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)

    for p, d_hat, w in zip(origins, directions, speeds):
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

    mean_speed = float(np.mean(speeds))

    logger.debug(
        "Attractor: (%.1f, %.1f) conf=%.2f n=%d mean_speed=%.1f",
        ax, ay, confidence, n, mean_speed,
    )

    return AttractorEstimate(
        point=(ax, ay),
        confidence=confidence,
        n_players=n,
        mean_speed=mean_speed,
    )
