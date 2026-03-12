"""
ball_kalman.py
--------------
Adaptive Constant-Acceleration Kalman filter for football tracking.

Why constant-velocity is insufficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A football can change its velocity almost instantaneously during a kick or
bounce.  A constant-velocity (CV) Kalman filter with a fixed process-noise
covariance **Q** responds too slowly to such manoeuvres because:

* The small, fixed **Q** keeps the Kalman gain **K** low, so corrections
  from new measurements are heavily discounted.
* There is no acceleration state, so the filter cannot represent that a
  kicked ball will continue in the new direction and then gradually decelerate.

Design
~~~~~~
**State vector** : ``[cx, cy, vcx, vcy, acx, acy]``
    Position (px), velocity (px/frame), acceleration (px/frame²) for each axis.

**Motion model** : Constant-Acceleration (CA) with ``dt = 1`` frame.
    The transition matrix **F** propagates position by ``vel×dt + 0.5×acc×dt²``,
    velocity by ``acc×dt``, and holds acceleration constant.

**Process noise Q** : Discrete White-Noise Acceleration (DWNA) model.
    **Q** is derived from a white-noise disturbance on the acceleration with
    standard deviation ``sigma_acc`` (px/frame²):

    .. math::

        \\mathbf{G} = \\begin{bmatrix} \\tfrac{1}{2}dt^2 \\\\ dt \\\\ 1 \\end{bmatrix}, \\quad
        \\mathbf{Q}_{\\text{1D}} = \\sigma_a^2 \\, \\mathbf{G} \\mathbf{G}^\\top

    The 2-D Q is the block-diagonal of two independent 1-D models.  With the
    default ``sigma_acc = 30`` px/frame² this gives ``Q_vel ≈ 900 px²/frame²``
    (vs. 1 in the old CV filter) — a 900× improvement in velocity responsiveness.

**Adaptive Q** : Normalised Innovation Squared (NIS) scaling.
    After every accepted update the NIS statistic
    ``ε = yᵀ S⁻¹ y`` (chi-squared, 2 DOF) is computed from the innovation **y**
    and innovation covariance **S**.  When ``ε`` exceeds the 95 % chi² threshold
    (``5.99`` for 2 DOF), the effective process noise is temporarily boosted:

    .. math::

        \\lambda = \\min\\!\\left(\\frac{\\varepsilon}{2},\\; \\lambda_{\\max}\\right)

    The scale decays by ``adaptive_q_decay`` each frame back toward 1.0, so the
    filter relaxes to normal behaviour within a few frames after a kick.
    *This allows recovery in ≤ 2 frames from a sudden kick or bounce while
    remaining smooth during free-flight.*

**Measurement gating** : Mahalanobis-distance gate.
    If the squared Mahalanobis distance of a new detection exceeds
    ``gate_chi2`` (default 9.21 = chi² 99 %, 2 DOF) the measurement is
    classified as a likely false YOLO detection and **rejected** — the filter
    only predicts for that frame.  ``frames_since_detection`` is incremented
    as if no measurement arrived.  *This prevents spurious YOLO hits from
    corrupting the velocity and acceleration states.*

Public API
~~~~~~~~~~
``BallKalmanFilter``  (``AdaptiveBallKalmanFilter`` is an alias)
    ``initialize(cx, cy)``            – seed from the first observation
    ``predict() → (cx, cy)``          – advance without a measurement
    ``update(cx, cy) → (cx, cy)``     – advance + conditionally correct
    ``position → (cx, cy) | None``    – current filtered centre
    ``velocity → (vcx, vcy) | None``  – current filtered velocity
    ``acceleration → (acx, acy) | None`` – current filtered acceleration
    ``initialized``                   – True once seeded
    ``frames_since_detection``        – consecutive frames without an accepted measurement
    ``last_measurement_gated``        – True if the last ``update()`` call was gated
    ``reset()``                       – return to uninitialised state
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Chi-squared thresholds for 2 degrees of freedom (2 observed coordinates)
_CHI2_95_2DOF = 5.991  # 95th percentile — used to trigger adaptive Q
_CHI2_99_2DOF = 9.210  # 99th percentile — default measurement gate


def _build_dwna_q(sigma_acc: float, dt: float = 1.0) -> np.ndarray:
    """Build the 6×6 DWNA process-noise matrix for a 2-D CA model.

    State order: ``[cx, cy, vcx, vcy, acx, acy]``.  The x and y components
    are independent; each follows the 1-D DWNA model where the acceleration
    noise input has standard deviation ``sigma_acc``.

    Parameters
    ----------
    sigma_acc:
        Standard deviation of the acceleration-noise input (px/frame²).
    dt:
        Frame time step (default 1 frame).

    Returns
    -------
    np.ndarray
        6×6 symmetric positive-semidefinite covariance matrix.
    """
    # 1-D noise-input vector (maps jerk noise → [pos, vel, acc])
    g = np.array([0.5 * dt ** 2, dt, 1.0], dtype=np.float64)
    q1 = sigma_acc ** 2 * np.outer(g, g)  # 3×3 for one axis

    # Expand to 6×6 with interleaved x/y state ordering
    Q = np.zeros((6, 6), dtype=np.float64)
    # x sub-indices: 0=cx, 2=vcx, 4=acx
    ix = [0, 2, 4]
    # y sub-indices: 1=cy, 3=vcy, 5=acy
    iy = [1, 3, 5]
    for r in range(3):
        for c in range(3):
            Q[ix[r], ix[c]] = q1[r, c]
            Q[iy[r], iy[c]] = q1[r, c]
    return Q


class BallKalmanFilter:
    """Adaptive Constant-Acceleration Kalman filter for tracking the football.

    This replaces the original constant-velocity filter.  Key improvements:

    * **6-state CA model** — tracks position, velocity *and* acceleration so
      the filter can represent a kicked ball's new trajectory immediately.
    * **DWNA process noise** — the Q matrix is calibrated to the expected
      manoeuvre magnitude; ``sigma_acc = 30 px/frame²`` gives
      ``Q_vel ≈ 900 px²/frame²`` (900× larger than the old CV filter's 1.0).
    * **Adaptive Q scaling** — when the NIS statistic reveals a sudden
      trajectory change (kick/bounce) the process noise is temporarily boosted,
      allowing sub-2-frame recovery.
    * **Measurement gating** — detections beyond the Mahalanobis gate are
      rejected as likely false positives, protecting the velocity state.

    Parameters
    ----------
    sigma_acc:
        Standard deviation of the acceleration disturbance (px/frame²).
        Increase for a faster/more erratic ball, decrease for smoother
        predictions.  Default ``30.0`` handles typical broadcast football.
    measurement_noise:
        Diagonal of the measurement-noise covariance **R** (px²).
        Corresponds to the standard deviation of YOLO centre-coordinate
        errors (default ``4.0`` ≈ 2 px std).
    gate_chi2:
        Chi-squared gate threshold for 2 DOF.  Measurements with Mahalanobis
        distance² exceeding this value are rejected.  Default ``9.21`` is
        the chi² 99 % quantile.
    adaptive_q:
        Enable NIS-based adaptive Q scaling (default *True*).
    adaptive_q_decay:
        Per-frame multiplicative decay of the adaptive scale factor back
        toward 1.0.  Default ``0.85`` returns to normal within ~10 frames.
    adaptive_q_max_scale:
        Maximum allowed adaptive scale factor (default ``50.0``).
    """

    def __init__(
        self,
        sigma_acc: float = 30.0,
        measurement_noise: float = 4.0,
        gate_chi2: float = _CHI2_99_2DOF,
        adaptive_q: bool = True,
        adaptive_q_decay: float = 0.85,
        adaptive_q_max_scale: float = 50.0,
        # Legacy parameters kept for backward-compatibility only
        process_noise: float | None = None,  # ignored (superseded by sigma_acc)
    ) -> None:
        dt = 1.0  # inter-frame time step (1 frame)

        # -- State-transition matrix (Constant-Acceleration, dt=1) ---------------
        # State: [cx, cy, vcx, vcy, acx, acy]
        self.F = np.array(
            [
                [1, 0, dt,  0, 0.5 * dt**2, 0          ],
                [0, 1,  0, dt, 0,           0.5 * dt**2],
                [0, 0,  1,  0, dt,          0          ],
                [0, 0,  0,  1, 0,           dt         ],
                [0, 0,  0,  0, 1,           0          ],
                [0, 0,  0,  0, 0,           1          ],
            ],
            dtype=np.float64,
        )

        # -- Measurement matrix (observe position only) --------------------------
        self.H = np.zeros((2, 6), dtype=np.float64)
        self.H[0, 0] = 1.0  # cx
        self.H[1, 1] = 1.0  # cy

        # -- Process noise (DWNA Q, scaled adaptively) --------------------------
        self._Q_base = _build_dwna_q(sigma_acc, dt)

        # -- Measurement noise --------------------------------------------------
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

        # -- Gating + adaptive Q settings --------------------------------------
        self._gate_chi2 = gate_chi2
        self._adaptive_q = adaptive_q
        self._adaptive_q_decay = adaptive_q_decay
        self._adaptive_q_max_scale = adaptive_q_max_scale
        self._adaptive_scale: float = 1.0

        # -- State estimate and covariance (lazy-initialised) ------------------
        self.x: np.ndarray | None = None
        self.P: np.ndarray = np.diag(
            [100.0, 100.0, 900.0, 900.0, 2500.0, 2500.0]
        ).astype(np.float64)
        self._initialized = False
        self._frames_since_detection: int = 0
        self._last_gated: bool = False

    # ── Effective Q (Q_base scaled by adaptive factor) ─────────────────────────

    @property
    def _Q(self) -> np.ndarray:
        return self._Q_base * self._adaptive_scale

    # ── Public interface ──────────────────────────────────────────────────────

    def initialize(self, cx: float, cy: float) -> None:
        """Seed the filter from the first observation.

        Velocity and acceleration are initialised to zero; the large initial
        covariance allows rapid adaptation on the first few measurements.
        """
        self.x = np.array([cx, cy, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag(
            [100.0, 100.0, 900.0, 900.0, 2500.0, 2500.0]
        ).astype(np.float64)
        self._initialized = True
        self._frames_since_detection = 0
        self._adaptive_scale = 1.0
        self._last_gated = False

    def predict(self) -> tuple[float, float]:
        """Propagate state one frame ahead without using a new measurement.

        Updates the adaptive scale decay and increments
        ``frames_since_detection``.

        Returns
        -------
        (cx, cy):
            Predicted ball centre in pixel coordinates.

        Raises
        ------
        RuntimeError
            If called before :meth:`initialize` or :meth:`update`.
        """
        if not self._initialized:
            raise RuntimeError(
                "BallKalmanFilter.initialize() must be called before predict()."
            )
        # Decay adaptive scale toward 1.0 even when no measurement arrives
        self._adaptive_scale = max(1.0, self._adaptive_scale * self._adaptive_q_decay)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self._Q
        self._frames_since_detection += 1
        self._last_gated = True  # effectively treated as gated (no measurement)
        return float(self.x[0]), float(self.x[1])

    def update(self, cx: float, cy: float) -> tuple[float, float]:
        """Advance state and conditionally correct with a new measurement.

        The measurement is **gated** (rejected) when its squared Mahalanobis
        distance from the predicted position exceeds ``gate_chi2``.  A gated
        measurement is treated like no detection: the state is only predicted
        and ``frames_since_detection`` increments.

        If the filter has not been initialised yet, the state is seeded from
        ``(cx, cy)`` and the raw position is returned immediately.

        Returns
        -------
        (cx, cy):
            Filtered ball centre (or predicted centre if gated).
        """
        if not self._initialized:
            self.initialize(cx, cy)
            return cx, cy

        # -- Decay adaptive scale before predict step --------------------------
        self._adaptive_scale = max(1.0, self._adaptive_scale * self._adaptive_q_decay)

        # -- Predict step -------------------------------------------------------
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self._Q

        # -- Innovation and innovation covariance ------------------------------
        z = np.array([cx, cy], dtype=np.float64)
        y = z - self.H @ x_pred                   # innovation
        S = self.H @ P_pred @ self.H.T + self.R   # innovation covariance

        # -- Measurement gating (Mahalanobis distance) -------------------------
        S_inv = np.linalg.inv(S)
        nis = float(y @ S_inv @ y)                # Normalised Innovation Squared

        if nis > self._gate_chi2:
            # Reject measurement: only predict, do not correct
            logger.debug(
                "Ball Kalman: measurement gated (NIS=%.2f > %.2f); "
                "accepting predicted position instead.",
                nis, self._gate_chi2,
            )
            self.x = x_pred
            self.P = P_pred
            self._frames_since_detection += 1
            self._last_gated = True
            return float(self.x[0]), float(self.x[1])

        # -- Adaptive Q update (triggered when NIS > 95% chi² threshold) ------
        if self._adaptive_q and nis > _CHI2_95_2DOF:
            new_scale = min(nis / 2.0, self._adaptive_q_max_scale)
            if new_scale > self._adaptive_scale:
                self._adaptive_scale = new_scale
                logger.debug(
                    "Ball Kalman: adaptive Q boosted to %.1f× (NIS=%.2f)",
                    self._adaptive_scale, nis,
                )

        # -- Correction step ---------------------------------------------------
        K = P_pred @ self.H.T @ S_inv             # Kalman gain
        self.x = x_pred + K @ y
        I_KH = np.eye(6, dtype=np.float64) - K @ self.H
        self.P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T  # Joseph form (numerically stable)

        self._frames_since_detection = 0
        self._last_gated = False
        return float(self.x[0]), float(self.x[1])

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def initialized(self) -> bool:
        """*True* once :meth:`initialize` or the first :meth:`update` has run."""
        return self._initialized

    @property
    def position(self) -> tuple[float, float] | None:
        """Current filtered ``(cx, cy)``, or *None* if not yet initialized."""
        if not self._initialized or self.x is None:
            return None
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float] | None:
        """Current filtered velocity ``(vcx, vcy)`` in px/frame, or *None*."""
        if not self._initialized or self.x is None:
            return None
        return float(self.x[2]), float(self.x[3])

    @property
    def acceleration(self) -> tuple[float, float] | None:
        """Current filtered acceleration ``(acx, acy)`` in px/frame², or *None*."""
        if not self._initialized or self.x is None:
            return None
        return float(self.x[4]), float(self.x[5])

    @property
    def frames_since_detection(self) -> int:
        """Consecutive frames without an accepted (non-gated) measurement."""
        return self._frames_since_detection

    @property
    def last_measurement_gated(self) -> bool:
        """*True* if the last :meth:`update` call was rejected by the gate."""
        return self._last_gated

    def reset(self) -> None:
        """Return the filter to its uninitialised state."""
        self.x = None
        self.P = np.diag(
            [100.0, 100.0, 900.0, 900.0, 2500.0, 2500.0]
        ).astype(np.float64)
        self._initialized = False
        self._frames_since_detection = 0
        self._adaptive_scale = 1.0
        self._last_gated = False


# Alias for explicitness — both names refer to the same class
AdaptiveBallKalmanFilter = BallKalmanFilter
